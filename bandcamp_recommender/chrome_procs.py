"""Ownership and cleanup of the Chrome processes this library starts.

Every driver gets its own profile dir, ``bcr-chrome-<owner pid>-<hex>``,
passed to Chrome as ``--user-data-dir`` and to chromedriver as an inert
``--bandcamp-recommender-profile`` switch (chromedriver ignores switches it
doesn't know). chromedriver, the browser and all its child processes then
name the library process that owns them (Chrome's crashpad handlers don't,
and exit with their browser), so cleanup asks "is the owner alive?" instead
of "who spawned this?":

* :func:`quit_driver` quits a driver, then kills whatever still runs on its
  profile and deletes the dir.
* :func:`reap_orphans` kills the Chrome of library processes that died
  (crash, SIGKILL, OOM, a restarted service whose Chrome escaped its cgroup
  into a snap scope), waits our own dead chromedriver children, and deletes
  stale profile dirs. It spares the Chrome of every live process (two
  processes using the library no longer kill each other's browsers) and
  anything without the marker (your own Chrome, Playwright, ...), except an
  unmarked chromedriver whose spawner is gone (reparented to init or the
  systemd user manager): left by an older library version, nothing can
  drive it anymore. Runs once per process on the first driver start;
  long-running callers can also call it at startup.

Why explicit profiles: without one, chromedriver makes a temp profile
(``.org.chromium.Chromium.scoped_dir.*``) that only a clean quit deletes.
Snap Chromium's /tmp is private (``/tmp/snap-private-tmp/snap.chromium/tmp``,
behind a root-only dir), so every killed Chrome leaked a ~140 MB profile
there that the user can't even reach (it filled a server disk). Now a killed
Chrome leaves only its ~12 KB singleton-socket and empty temp dirs there.
Snap Chromium can't write hidden dirs directly under $HOME either, so when
it is installed the profiles live in its own ``~/snap/chromium/common``.

Linux reads ``/proc``; other POSIX systems ask ``ps``.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

PROFILE_PREFIX = "bcr-chrome-"
CHROMEDRIVER_SWITCH = "--bandcamp-recommender-profile"
# Set on every driver DriverManager starts: the profile to release on quit.
PROFILE_ATTR = "_bandcamp_profile"

_NAME_RE = re.compile(r"bcr-chrome-(\d+)-[0-9a-f]{8}")
_MARK_RE = re.compile(_NAME_RE.pattern.encode())
# A live owner's dir that no process uses may be a driver starting right now.
# Older than this it is a leftover (e.g. the owner pid was reused after a reboot).
_LAUNCH_GRACE_S = 300.0
_KILL_WAIT_S = 2.0
# Where a process lands when its parent dies: init, or a subreaper such as
# the systemd user manager (NOT pid 1, so a PPID==1 check misses it). Matched
# by name, not pid: in a container pid 1 may be the library process itself.
_REAPER_COMMS = frozenset({"systemd", "init", "launchd", "tini", "dumb-init", "docker-init"})

_PROC = "/proc"
_SNAP_CHROMIUM = "/snap/bin/chromium"

_live_lock = threading.Lock()
_live: Dict[str, str] = {}  # profile name -> path, for drivers this process runs
_swept = False


class _Proc(NamedTuple):
    pid: int
    ppid: int
    state: str  # one letter; "Z" = zombie
    comm: str  # executable basename, lowercased
    cmdline: bytes  # only read for Chrome-like processes


def _debug() -> bool:
    return os.environ.get("BANDCAMP_DEBUG") == "1"


def profiles_root() -> Path:
    """``BANDCAMP_BROWSER_PROFILE_DIR``, else snap Chromium's own writable
    dir when it is installed, else next to the disk cache."""
    override = os.environ.get("BANDCAMP_BROWSER_PROFILE_DIR")
    if override:
        return Path(override)
    home = Path.home()
    if sys.platform.startswith("linux") and (
        os.path.exists(_SNAP_CHROMIUM) or (home / "snap" / "chromium").is_dir()
    ):
        return home / "snap" / "chromium" / "common" / "bandcamp_recommender" / "chrome-profiles"
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache_home) / "bandcamp_recommender" / "chrome-profiles"


def new_profile() -> str:
    """Create a fresh profile dir owned by this process; return its path.
    The first call per process reaps what dead processes left behind."""
    global _swept
    with _live_lock:
        first, _swept = not _swept, True
    if first:
        try:
            reap_orphans()
        except Exception:
            logger.warning("chrome orphan sweep failed", exc_info=_debug())
    name = f"{PROFILE_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
    path = profiles_root() / name
    # Registered before it exists, so a concurrent sweep never takes it.
    with _live_lock:
        _live[name] = str(path)
    try:
        path.mkdir(parents=True, mode=0o700)
    except OSError:
        with _live_lock:
            _live.pop(name, None)
        raise
    return str(path)


def chromedriver_args(profile: str) -> List[str]:
    """Service args that tag chromedriver with ``profile`` (it has no
    ``--user-data-dir`` of its own to carry the marker)."""
    return [f"{CHROMEDRIVER_SWITCH}={profile}"]


def release_profile(profile: str) -> None:
    """Kill whatever still runs on ``profile`` (a driver whose quit failed, a
    browser whose chromedriver died) and delete the dir."""
    name = os.path.basename(profile)
    if os.name != "nt":
        try:
            table = _processes()
            _kill([p.pid for p in _running_chrome(table) if _marker(p) == name])
            _reap_zombies(table, os.getpid())
        except Exception:
            logger.warning("cleanup of browser profile %s failed", name, exc_info=_debug())
    shutil.rmtree(profile, ignore_errors=True)
    with _live_lock:
        _live.pop(name, None)


def quit_driver(driver) -> None:
    """Quit ``driver`` and clean up after it, even if the quit fails."""
    try:
        driver.quit()
    except Exception:
        logger.warning("browser quit failed", exc_info=_debug())
    profile = getattr(driver, PROFILE_ATTR, None)
    if isinstance(profile, str):
        release_profile(profile)


def reap_orphans() -> int:
    """Kill Chrome and chromedriver whose owning library process is dead (or,
    for this process, whose driver is gone), wait our own defunct
    chromedriver children, and delete stale profile dirs. Returns the number
    of processes killed or waited. Never touches a live process's browser.
    No-op on Windows."""
    if os.name == "nt":
        return 0
    try:
        table = _processes()
    except Exception:
        logger.warning("chrome reaper: could not list processes", exc_info=_debug())
        return 0
    me = os.getpid()
    zombies = _reap_zombies(table, me)
    doomed = _orphans(table, me)
    killed = _kill(doomed)
    gone = set(doomed)
    in_use = {_marker(p) for p in _running_chrome(table) if p.pid not in gone}
    swept = _sweep(table, me, in_use)
    if killed or zombies or swept:
        logger.warning(
            "chrome reaper: killed %d orphan process(es), waited %d zombie(s), deleted %d stale profile(s)",
            killed, zombies, swept,
        )
    return killed + zombies


def _release_all() -> None:
    """At exit: nothing this process started may outlive it."""
    me = os.getpid()
    with _live_lock:
        paths = [path for name, path in _live.items() if _owner(name) == me]
    for path in paths:
        release_profile(path)


atexit.register(_release_all)


# ---- classification ---------------------------------------------------------


def _owner(name: str) -> Optional[int]:
    m = _NAME_RE.fullmatch(name)
    return int(m.group(1)) if m else None


def _marker(p: _Proc) -> Optional[str]:
    m = _MARK_RE.search(p.cmdline)
    return m.group(0).decode() if m else None


def _alive(table: Dict[int, _Proc], pid: int) -> bool:
    p = table.get(pid)
    return p is not None and p.state not in ("Z", "X")


def _running_chrome(table: Dict[int, _Proc]) -> List[_Proc]:
    return [p for p in table.values() if "chrom" in p.comm and p.state not in ("Z", "X")]


def _orphans(table: Dict[int, _Proc], me: int) -> List[int]:
    doomed: Dict[int, None] = {}
    for p in _running_chrome(table):
        name = _marker(p)
        if name is not None:
            owner = _owner(name)
            if owner == me:
                with _live_lock:
                    dead = name not in _live
            else:
                dead = not _alive(table, owner)
            if dead:
                doomed[p.pid] = None
        elif p.comm == "chromedriver" and _spawner_gone(table, p):
            doomed[p.pid] = None
            for d in _descendants(table, p.pid):
                if "chrom" in table[d].comm:
                    doomed[d] = None
    return list(doomed)


def _spawner_gone(table: Dict[int, _Proc], p: _Proc) -> bool:
    """A chromedriver is a child of the process driving it; once that process
    dies it is reparented to init or a subreaper."""
    parent = table.get(p.ppid)
    return parent is not None and parent.comm in _REAPER_COMMS


def _descendants(table: Dict[int, _Proc], root: int) -> List[int]:
    kids: Dict[int, List[int]] = {}
    for p in table.values():
        kids.setdefault(p.ppid, []).append(p.pid)
    out: List[int] = []
    seen = {root}
    stack = [root]
    while stack:
        for c in kids.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c)
                out.append(c)
                stack.append(c)
    return out


def _sweep(table: Dict[int, _Proc], me: int, in_use: set) -> int:
    try:
        entries = list(os.scandir(profiles_root()))
    except (OSError, RuntimeError):  # RuntimeError: no home directory
        return 0
    removed = 0
    now = time.time()
    for entry in entries:
        owner = _owner(entry.name)
        if owner is None or entry.name in in_use:
            continue
        with _live_lock:
            if entry.name in _live:
                continue
        if owner != me and _alive(table, owner):
            try:
                if now - entry.stat().st_mtime < _LAUNCH_GRACE_S:
                    continue
            except OSError:
                continue
        shutil.rmtree(entry.path, ignore_errors=True)
        removed += 1
    return removed


# ---- killing and waiting ----------------------------------------------------


def _reap_zombies(table: Dict[int, _Proc], me: int) -> int:
    """waitpid defunct Chrome-like children of THIS process. Selenium's
    Service doesn't always reap a chromedriver that died behind its back
    (cgroup OOM kill, crash): one zombie per dead driver, accumulating, and
    SIGKILL is a no-op on zombies.

    Only waits pids whose comm matches chromium AND whose parent is us,
    never waitpid(-1): a blanket wait could steal exit statuses from the
    caller's own subprocesses (ffmpeg, sox, ...) mid-flight."""
    reaped = 0
    for p in table.values():
        if p.state != "Z" or p.ppid != me or "chrom" not in p.comm:
            continue
        try:
            if os.waitpid(p.pid, os.WNOHANG)[0] == p.pid:
                reaped += 1
        except OSError:
            continue
    return reaped


def _kill(pids: Iterable[int]) -> int:
    killed = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger.warning("chrome reaper: no permission to kill pid %d", pid)
    if killed:
        _wait_gone(killed)
    return len(killed)


def _wait_gone(pids: Iterable[int], timeout: float = _KILL_WAIT_S) -> None:
    """Give SIGKILLed processes a moment to exit before their profile is
    deleted, waiting the ones that are our own children (a chromedriver).
    These pids are all Chrome-like, so this keeps the never-waitpid(-1) rule."""
    pending = set(pids)
    deadline = time.monotonic() + timeout
    while pending:
        for pid in list(pending):
            try:
                os.waitpid(pid, os.WNOHANG)
            except OSError:  # ChildProcessError: someone else's child
                pass
            if not _is_running(pid):
                pending.discard(pid)
        if not pending or time.monotonic() >= deadline:
            return
        time.sleep(0.05)


def _is_running(pid: int) -> bool:
    if os.path.isdir(_PROC):
        try:
            with open(os.path.join(_PROC, str(pid), "stat"), "rb") as f:
                stat = f.read().decode("utf-8", "replace")
        except OSError:
            return False
        return stat[stat.rfind(")") + 2:][:1] not in ("Z", "X")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---- process table ----------------------------------------------------------


def _processes() -> Dict[int, _Proc]:
    if os.path.isdir(_PROC):
        return _proc_table()
    return _ps_table()


def _proc_table() -> Dict[int, _Proc]:
    table: Dict[int, _Proc] = {}
    for entry in os.listdir(_PROC):
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(_PROC, entry, "stat"), "rb") as f:
                stat = f.read().decode("utf-8", "replace")
            # "pid (comm) state ppid ...": comm may itself contain spaces and
            # parens, so split at the LAST ')'.
            rparen = stat.rfind(")")
            comm = stat[stat.find("(") + 1:rparen].lower()
            state, ppid = stat[rparen + 2:].split()[:2]
            cmdline = b""
            if "chrom" in comm and state != "Z":
                with open(os.path.join(_PROC, entry, "cmdline"), "rb") as f:
                    cmdline = f.read()
            table[int(entry)] = _Proc(int(entry), int(ppid), state, comm, cmdline)
        except (OSError, ValueError):
            continue
    return table


def _run_ps(ps: str, fields: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [ps, "-axww", "-o", fields],
            capture_output=True, encoding="utf-8", errors="replace", timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _ps_table() -> Dict[int, _Proc]:
    ps = shutil.which("ps")
    procs = _run_ps(ps, "pid=,ppid=,stat=,comm=") if ps else None
    if procs is None:
        return {}
    args: Dict[int, str] = {}
    for line in (_run_ps(ps, "pid=,args=") or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            args[int(parts[0])] = parts[1]
    table: Dict[int, _Proc] = {}
    for line in procs.splitlines():
        parts = line.split(None, 3)  # comm (a full path on macOS) may hold spaces
        if len(parts) < 4 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid = int(parts[0])
        comm = os.path.basename(parts[3]).lower()
        cmdline = args.get(pid, "").encode() if "chrom" in comm else b""
        table[pid] = _Proc(pid, int(parts[1]), parts[2][:1], comm, cmdline)
    return table
