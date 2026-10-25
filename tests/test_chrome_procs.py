"""Ownership and cleanup of the library's Chrome (bandcamp_recommender.chrome_procs).

Runs against a fake /proc tree and fake drivers; never starts or kills a
real process."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from selenium.common.exceptions import WebDriverException

from bandcamp_recommender import browser, chrome_procs
from bandcamp_recommender.recommendations import driver_manager

ME = os.getpid()
SYSTEMD_USER = 986  # the systemd user manager: a subreaper, not pid 1
OTHER = 4242  # another live process using the library
DEAD = 999_990  # a library process that died (not in the table)


class FakeProc:
    """A /proc tree; SIGKILL removes the process, as its exit would.

    ``linger``: a killed process still shows as running for that many
    liveness polls, as a real one does for a moment after SIGKILL.
    ``zombie_on_kill``: it then stays in the table as a zombie (its parent
    hasn't waited it) instead of vanishing."""

    def __init__(self, root: Path):
        self.root = root
        self.killed: list = []
        self.wait_calls: list = []
        self.waited: list = []
        self.unkillable: set = set()
        self.linger = 0
        self.zombie_on_kill = False
        self._dying: dict = {}
        self._real_is_running = chrome_procs._is_running
        root.mkdir()

    def add(self, pid, comm, ppid, *argv, state="S"):
        d = self.root / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pid} {pid} 0 -1 4194560 0 0\n")
        (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
        return pid

    def kill(self, pid, sig):
        d = self.root / str(pid)
        if not d.exists():
            raise ProcessLookupError(pid)
        if pid in self.unkillable:
            raise PermissionError(pid)
        if sig == signal.SIGKILL:
            self.killed.append(pid)
            if self.linger:
                self._dying[pid] = self.linger
            else:
                self._exit(pid)

    def _exit(self, pid):
        d = self.root / str(pid)
        if self.zombie_on_kill:
            stat = (d / "stat").read_text()
            rparen = stat.rfind(")")
            (d / "stat").write_text(stat[:rparen + 2] + "Z" + stat[rparen + 3:])
        else:
            shutil.rmtree(d)

    def is_running(self, pid):
        if pid in self._dying:
            self._dying[pid] -= 1
            if not self._dying[pid]:
                del self._dying[pid]
                self._exit(pid)
        return self._real_is_running(pid)

    def waitpid(self, pid, options):
        self.wait_calls.append(pid)
        d = self.root / str(pid)
        if pid > 0 and d.exists() and f") Z {ME} " in (d / "stat").read_text():
            self.waited.append(pid)
            shutil.rmtree(d)
            return pid, 0
        raise ChildProcessError(pid)


@pytest.fixture
def proc(tmp_path, monkeypatch):
    fake = FakeProc(tmp_path / "proc")
    fake.add(1, "systemd", 0, "/sbin/init")
    fake.add(SYSTEMD_USER, "systemd", 1, "/usr/lib/systemd/systemd", "--user")
    fake.add(ME, "python3", SYSTEMD_USER, "python3", "-m", "pytest")
    fake.add(OTHER, "python3", SYSTEMD_USER, "python3", "app.py")
    monkeypatch.setattr(chrome_procs, "_PROC", str(fake.root))
    monkeypatch.setattr(chrome_procs.os, "kill", fake.kill)
    monkeypatch.setattr(chrome_procs.os, "waitpid", fake.waitpid)
    monkeypatch.setattr(chrome_procs, "_is_running", fake.is_running)
    return fake


def _profile(owner, token="aaaaaaaa"):
    return str(chrome_procs.profiles_root() / f"bcr-chrome-{owner}-{token}")


def _driver_tree(proc, base, owner_pid, parent, token="aaaaaaaa"):
    """What one driver looks like on snap Chromium (seen on the server):
    chromedriver -> browser -> renderer all carry the profile; the browser's
    double-forked crashpad handler (reparented to init) doesn't, and exits
    with it. Returns the marked pids."""
    profile = _profile(owner_pid, token)
    proc.add(base, "chromedriver", parent, "chromedriver", "--port=1",
             f"{chrome_procs.CHROMEDRIVER_SWITCH}={profile}")
    proc.add(base + 1, "chrome", base, "chrome", "--headless=new", f"--user-data-dir={profile}")
    proc.add(base + 2, "chrome", base + 1, "chrome", "--type=renderer", f"--user-data-dir={profile}")
    proc.add(base + 3, "chrome_crashpad", 1, "chrome_crashpad_handler", "--monitor-self")
    return [base, base + 1, base + 2]


# ---- the reaper -------------------------------------------------------------


def test_reaps_a_dead_owners_tree_and_spares_everyone_else(proc):
    dead_tree = _driver_tree(proc, 5000, DEAD, SYSTEMD_USER)
    _driver_tree(proc, 6000, OTHER, OTHER, token="bbbbbbbb")
    os.makedirs(_profile(DEAD))
    os.makedirs(_profile(OTHER, "bbbbbbbb"))
    proc.add(7001, "chrome", 1, "/opt/google/chrome/chrome")  # the user's own browser
    proc.add(7002, "chrome-headless", 1, "chrome-headless-shell",
             "--user-data-dir=/tmp/playwright_chromiumdev_profile-x")
    proc.add(7003, "bash", SYSTEMD_USER, "ls", _profile(DEAD))  # not Chrome

    assert chrome_procs.reap_orphans() == 3
    assert sorted(proc.killed) == dead_tree
    # Its profile goes in the same pass, once nothing runs on it.
    assert not os.path.exists(_profile(DEAD))
    assert os.path.isdir(_profile(OTHER, "bbbbbbbb"))


@pytest.mark.parametrize("zombie", [False, True], ids=["exits", "lingers-as-zombie"])
def test_profile_goes_once_its_killed_processes_are_gone(proc, zombie):
    """SIGKILL is asynchronous: right after it the process still runs. The
    profile must go once it is dead, also while it lingers as a zombie
    nobody waited yet."""
    os.makedirs(_profile(DEAD))
    tree = _driver_tree(proc, 5000, DEAD, SYSTEMD_USER)
    proc.linger = 3
    proc.zombie_on_kill = zombie

    assert chrome_procs.reap_orphans() == 3
    assert sorted(proc.killed) == tree
    assert not os.path.exists(_profile(DEAD))


def test_an_owner_that_died_unwaited_counts_as_dead(proc):
    owner = proc.add(4343, "python3", SYSTEMD_USER, "python3", "app.py", state="Z")
    tree = _driver_tree(proc, 5000, owner, SYSTEMD_USER)  # its children were reparented

    assert chrome_procs.reap_orphans() == 3
    assert sorted(proc.killed) == tree


def test_own_browser_is_spared_while_its_driver_lives(proc):
    profile = chrome_procs.new_profile()
    name = os.path.basename(profile)
    token = name.rsplit("-", 1)[1]
    tree = _driver_tree(proc, 5000, ME, ME, token=token)

    assert chrome_procs.reap_orphans() == 0
    assert proc.killed == []
    # The driver is gone (its quit failed): what still runs on it is a leak.
    chrome_procs._live.pop(name)
    assert chrome_procs.reap_orphans() == 3
    assert sorted(proc.killed) == tree


def test_unmarked_selenium_leftovers_only_on_request(proc):
    """What versions before profile markers leave behind. Not ours to judge by
    default (could be anyone's); DriverManager.reap_orphans, the old opt-in
    cold-start reaper, keeps reaching it."""
    proc.add(8001, "chromedriver", SYSTEMD_USER, "chromedriver", "--port=2")
    proc.add(8002, "chrome", 8001, "chrome", "--user-data-dir=/tmp/.org.chromium.Chromium.scoped_dir.x")
    proc.add(8003, "chrome", 8002, "chrome", "--type=renderer")
    proc.add(8101, "chromedriver", 1, "chromedriver", "--port=3")
    # Its chromedriver died (OOM): the browser was reparented.
    proc.add(8401, "chrome", SYSTEMD_USER, "chrome", "--user-data-dir=/tmp/.org.chromium.Chromium.scoped_dir.z")
    proc.add(8402, "chrome", 8401, "chrome", "--type=renderer")
    # Alive: driven by a live process.
    proc.add(8201, "chromedriver", OTHER, "chromedriver", "--port=4")
    proc.add(8202, "chrome", 8201, "chrome", "--user-data-dir=/tmp/.org.chromium.Chromium.scoped_dir.y")
    # PPID 1 but nobody's driver: the user's own browser.
    proc.add(8301, "chrome", 1, "/opt/google/chrome/chrome")
    proc.add(8302, "chrome_crashpad", 1, "chrome_crashpad_handler", "--monitor-self")

    assert chrome_procs.reap_orphans() == 0
    assert driver_manager.DriverManager.reap_orphans() == 6
    assert sorted(proc.killed) == [8001, 8002, 8003, 8101, 8401, 8402]


def test_in_a_container_pid_1_may_be_a_live_library_process(proc):
    shutil.rmtree(proc.root / "1")
    proc.add(1, "python3", 0, "python3", "app.py")
    proc.add(8001, "chromedriver", 1, "chromedriver", "--port=2")

    chrome_procs.reap_orphans(legacy=True)
    assert proc.killed == []


def test_an_unusable_process_list_reaps_nothing(proc, monkeypatch):
    root = chrome_procs.profiles_root()
    (root / f"bcr-chrome-{DEAD}-aaaaaaaa").mkdir(parents=True)
    (root / f"bcr-chrome-{OTHER}-bbbbbbbb").mkdir()
    monkeypatch.setattr(chrome_procs, "_processes", lambda: {})  # e.g. ps timed out

    assert chrome_procs.reap_orphans() == 0
    assert len(list(root.iterdir())) == 2


def test_profile_of_an_unkillable_process_is_kept(proc):
    root = chrome_procs.profiles_root()
    (root / f"bcr-chrome-{DEAD}-aaaaaaaa").mkdir(parents=True)
    tree = _driver_tree(proc, 5000, DEAD, SYSTEMD_USER)
    proc.unkillable.add(tree[1])

    chrome_procs.reap_orphans()
    assert (root / f"bcr-chrome-{DEAD}-aaaaaaaa").is_dir()


def test_waits_only_our_own_chromium_zombies(proc):
    proc.add(9001, "chromedriver", ME, state="Z")
    proc.add(9002, "ffmpeg", ME, state="Z")  # the caller's own subprocess
    proc.add(9003, "chromedriver", OTHER, state="Z")

    assert chrome_procs.reap_orphans() == 1
    assert proc.wait_calls == [9001]  # never waitpid(-1), never someone else's child
    assert proc.killed == []


def test_sweeps_stale_profiles_only(proc):
    root = chrome_procs.profiles_root()
    root.mkdir(parents=True)
    mine = os.path.basename(chrome_procs.new_profile())
    names = {
        "dead": f"bcr-chrome-{DEAD}-aaaaaaaa",
        "starting": f"bcr-chrome-{OTHER}-bbbbbbbb",
        "reused_pid": f"bcr-chrome-{OTHER}-cccccccc",
        "in_use": f"bcr-chrome-{OTHER}-dddddddd",
        "my_leftover": f"bcr-chrome-{ME}-eeeeeeee",
        "unrelated": "something-else",
    }
    for name in names.values():
        (root / name).mkdir()
    old = time.time() - 2 * chrome_procs._LAUNCH_GRACE_S
    for key in ("reused_pid", "in_use"):
        os.utime(root / names[key], (old, old))
    proc.add(6001, "chrome", OTHER, "chrome", f"--user-data-dir={root / names['in_use']}")

    chrome_procs.reap_orphans()
    left = {p.name for p in root.iterdir()}
    assert left == {mine, names["starting"], names["in_use"], names["unrelated"]}


def test_release_profile_keeps_a_dir_still_in_use(proc):
    profile = chrome_procs.new_profile()
    name = os.path.basename(profile)
    tree = _driver_tree(proc, 5000, ME, ME, token=name.rsplit("-", 1)[1])
    proc.unkillable.add(tree[1])

    chrome_procs.release_profile(profile)
    assert os.path.isdir(profile)
    assert name not in chrome_procs._live  # so the next sweep takes it once it's gone


def test_release_profile_kills_leftovers_and_deletes_the_dir(proc):
    profile = chrome_procs.new_profile()
    name = os.path.basename(profile)
    tree = _driver_tree(proc, 5000, ME, ME, token=name.rsplit("-", 1)[1])
    other = _driver_tree(proc, 6000, OTHER, OTHER, token="bbbbbbbb")

    chrome_procs.release_profile(profile)
    assert sorted(proc.killed) == tree
    assert not os.path.exists(profile)
    assert name not in chrome_procs._live
    assert all((proc.root / str(pid)).exists() for pid in other)


def test_release_profile_waits_our_dead_chromedriver(proc):
    """Its chromedriver died behind Selenium's back (OOM kill): a zombie
    child of ours, while the browser it started still runs on the profile."""
    profile = chrome_procs.new_profile()
    proc.add(5000, "chromedriver", ME, state="Z")
    proc.add(5001, "chrome", SYSTEMD_USER, "chrome", f"--user-data-dir={profile}")

    chrome_procs.release_profile(profile)
    assert proc.waited == [5000]
    assert proc.killed == [5001]
    assert not os.path.exists(profile)


def test_release_profile_without_a_process_list_leaves_it_to_the_next_reap(proc):
    profile = chrome_procs.new_profile()
    name = os.path.basename(profile)
    tree = _driver_tree(proc, 5000, ME, ME, token=name.rsplit("-", 1)[1])  # its quit failed

    with patch.object(chrome_procs, "_processes", return_value={}):  # e.g. ps timed out
        chrome_procs.release_profile(profile)
    assert proc.killed == []
    assert os.path.isdir(profile)  # nothing seen on it doesn't mean nothing runs on it
    assert name not in chrome_procs._live

    assert chrome_procs.reap_orphans() == 3  # e.g. before the next driver start
    assert sorted(proc.killed) == tree
    assert not os.path.exists(profile)


def test_quit_driver_cleans_up_even_when_quit_fails(proc):
    profile = chrome_procs.new_profile()
    driver = MagicMock()
    driver.quit.side_effect = WebDriverException("chromedriver is gone")
    setattr(driver, chrome_procs.PROFILE_ATTR, profile)

    chrome_procs.quit_driver(driver)
    driver.quit.assert_called_once()
    assert not os.path.exists(profile)


def test_a_forked_child_leaves_its_parents_browser_alone(proc, tmp_path):
    parents = tmp_path / f"bcr-chrome-{OTHER}-bbbbbbbb"
    parents.mkdir()
    tree = _driver_tree(proc, 6000, OTHER, OTHER, token="bbbbbbbb")
    driver = MagicMock()
    setattr(driver, chrome_procs.PROFILE_ATTR, str(parents))

    chrome_procs.quit_driver(driver)
    driver.quit.assert_not_called()  # quitting would end the parent's session
    assert parents.is_dir()
    assert proc.killed == []
    assert all((proc.root / str(pid)).exists() for pid in tree)


def test_at_exit_nothing_this_process_started_survives(proc):
    profile = chrome_procs.new_profile()
    tree = _driver_tree(proc, 5000, ME, ME, token=os.path.basename(profile).rsplit("-", 1)[1])
    # A forked child inherits the parent's registry; it must not kill the
    # parent's browser when it exits.
    chrome_procs._live[f"bcr-chrome-{OTHER}-bbbbbbbb"] = _profile(OTHER, "bbbbbbbb")
    other = _driver_tree(proc, 6000, OTHER, OTHER, token="bbbbbbbb")

    chrome_procs._release_all()
    assert sorted(proc.killed) == tree
    assert not os.path.exists(profile)
    assert all((proc.root / str(pid)).exists() for pid in other)


def test_every_driver_start_reaps_first(monkeypatch):
    """So whatever an earlier reap or release had to leave is retried, and
    the first Bandcamp request doesn't scan again."""
    monkeypatch.setattr(chrome_procs, "_swept", False)
    reap = MagicMock(return_value=0)
    monkeypatch.setattr(chrome_procs, "reap_orphans", reap)

    first = chrome_procs.new_profile()
    chrome_procs.new_profile()
    chrome_procs.sweep_once()
    assert reap.call_count == 2
    assert os.path.basename(first).startswith(f"bcr-chrome-{ME}-")
    assert stat.S_IMODE(os.stat(first).st_mode) == 0o700


def test_first_bandcamp_request_reaps_once_per_process(monkeypatch):
    """Chrome a SIGKILLed predecessor left behind is freed at startup, not on
    the (rare) first fallback."""
    from bandcamp_recommender import http as bchttp

    monkeypatch.setattr(chrome_procs, "_swept", False)
    reap = MagicMock(return_value=0)
    monkeypatch.setattr(chrome_procs, "reap_orphans", reap)
    bchttp.reset()
    try:
        bchttp._state()
        bchttp._state()
        bchttp.pacer()
    finally:
        bchttp.reset()
    assert reap.call_count == 1


def test_profiles_root(tmp_path, monkeypatch):
    monkeypatch.setattr(chrome_procs.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(chrome_procs, "_SNAP_CHROMIUM", str(tmp_path / "no-snap"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert chrome_procs.profiles_root() == tmp_path / "chrome-profiles"  # conftest override

    monkeypatch.delenv("BANDCAMP_BROWSER_PROFILE_DIR")
    monkeypatch.setattr(chrome_procs.sys, "platform", "darwin")
    assert chrome_procs.profiles_root() == tmp_path / "cache" / "bandcamp_recommender" / "chrome-profiles"
    monkeypatch.setattr(chrome_procs.sys, "platform", "linux")
    assert chrome_procs.profiles_root() == tmp_path / "cache" / "bandcamp_recommender" / "chrome-profiles"
    # Snap Chromium can't write hidden dirs under $HOME: use its own dir.
    (tmp_path / "home" / "snap" / "chromium").mkdir(parents=True)
    assert chrome_procs.profiles_root() == (
        tmp_path / "home" / "snap" / "chromium" / "common" / "bandcamp_recommender" / "chrome-profiles"
    )


def test_ps_fallback_without_proc(tmp_path, monkeypatch):
    """macOS: no /proc; comm is a full path with spaces."""
    helper = "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Helper (Renderer)"
    profile = _profile(DEAD)
    outputs = {
        "pid=,ppid=,stat=,comm=": (
            f"    1     0 Ss   /sbin/launchd\n"
            f"{ME:>5}     1 S    /usr/local/bin/python3\n"
            f"  501     1 S    /Users/me/.wdm/chromedriver\n"
            f"  502   501 S    /Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n"
            f"  503   502 S    {helper}\n"
            f"  600     1 S    /Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n"
            f"  700     1 S    /bin/zsh\n"
        ),
        "pid=,args=": (
            f"    1 /sbin/launchd\n"
            f"{ME:>5} python3 -m pytest\n"
            f"  501 /Users/me/.wdm/chromedriver --port=5 {chrome_procs.CHROMEDRIVER_SWITCH}={profile}\n"
            f"  502 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir={profile}\n"
            f"  503 {helper} --type=renderer --user-data-dir={profile}\n"
            f"  600 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n"
            f"  700 -zsh {profile}\n"
        ),
    }

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[cmd[-1]], stderr="")

    monkeypatch.setattr(chrome_procs, "_PROC", str(tmp_path / "no-proc"))
    monkeypatch.setattr(chrome_procs.shutil, "which", lambda name: "/bin/ps")
    monkeypatch.setattr(chrome_procs.subprocess, "run", run)
    table = chrome_procs._processes()
    assert table[503].comm == "google chrome helper (renderer)"
    assert table[502].state == "S" and table[502].ppid == 501
    assert table[700].cmdline == b""  # not Chrome: never inspected

    killed = []
    monkeypatch.setattr(chrome_procs.os, "kill", lambda pid, sig: killed.append(pid) if sig else None)
    monkeypatch.setattr(chrome_procs, "_wait_gone", lambda pids: None)
    assert chrome_procs.reap_orphans() == 3
    assert sorted(killed) == [501, 502, 503]  # not the user's Chrome (600), not the shell

    # Without command lines every profile would look unused: no table at all.
    monkeypatch.setattr(
        chrome_procs.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1 if cmd[-1] == "pid=,args=" else 0,
                                                      stdout=outputs[cmd[-1]], stderr=""),
    )
    assert chrome_procs._processes() == {}


# ---- every driver gets its own profile ---------------------------------------


def _fake_chrome():
    fake = MagicMock(name="FakeChrome")
    fake.current_url = "about:blank"
    fake.page_source = "<html></html>"
    return fake


def test_each_driver_gets_its_own_profile_and_chromedriver(proc):
    fakes = [_fake_chrome() for _ in range(3)]
    with patch.object(driver_manager.webdriver, "Chrome", side_effect=fakes) as chrome:
        dm = driver_manager.DriverManager()
        pool = dm.get_driver_pool(pool_size=3)
    assert pool.qsize() == 3

    profiles, services = [], []
    for call, fake in zip(chrome.call_args_list, fakes):
        options, service = call.kwargs["options"], call.kwargs["service"]
        profile = getattr(fake, chrome_procs.PROFILE_ATTR)
        assert f"--user-data-dir={profile}" in options.arguments
        assert f"{chrome_procs.CHROMEDRIVER_SWITCH}={profile}" in service.service_args
        assert os.path.isdir(profile)
        profiles.append(profile)
        services.append(service)
    assert len(set(profiles)) == 3
    assert len({id(s) for s in services}) == 3  # a shared Service only stops its last chromedriver

    dm.close()
    for fake, profile in zip(fakes, profiles):
        fake.quit.assert_called_once()
        assert not os.path.exists(profile)


def test_failed_launch_releases_its_profile(proc):
    with patch.object(driver_manager.webdriver, "Chrome", side_effect=WebDriverException("no chrome")):
        with pytest.raises(WebDriverException):
            driver_manager.DriverManager().create_driver()
    assert chrome_procs._live == {}
    assert list(chrome_procs.profiles_root().iterdir()) == []


def test_driver_that_dies_while_configured_is_cleaned_up(proc):
    fake = _fake_chrome()
    fake.set_page_load_timeout.side_effect = ConnectionError("chromedriver died")
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with pytest.raises(ConnectionError):
            driver_manager.DriverManager().create_driver()
    fake.quit.assert_called_once()
    assert chrome_procs._live == {}


def test_driver_manager_reap_orphans_delegates(proc):
    _driver_tree(proc, 5000, DEAD, SYSTEMD_USER)
    assert driver_manager.DriverManager.reap_orphans() == 3
    assert browser.reap_orphans is chrome_procs.reap_orphans


# ---- the shared browser closes itself when idle ------------------------------


@pytest.fixture
def shared_browser(monkeypatch):
    monkeypatch.setenv("BANDCAMP_MIN_REQUEST_INTERVAL", "0")
    browser.close()
    yield
    browser.close()


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_idle_browser_is_closed_and_cleaned_up(shared_browser, monkeypatch):
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0.2")
    fake = _fake_chrome()
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session() as driver:
            profile = getattr(driver, chrome_procs.PROFILE_ATTR)
    assert _wait_for(lambda: browser._driver is None)
    fake.quit.assert_called_once()
    assert not os.path.exists(profile)


def test_browser_in_use_is_not_closed(shared_browser, monkeypatch):
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0.1")
    fake = _fake_chrome()
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session():
            pass
        with browser.session():
            time.sleep(0.4)  # well past the idle timeout, but in use
            assert browser._driver is fake
            fake.quit.assert_not_called()
    assert _wait_for(lambda: browser._driver is None)


def test_a_huge_idle_timeout_does_not_kill_the_watcher(shared_browser, monkeypatch):
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "inf")
    fake = _fake_chrome()
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session():
            pass
    time.sleep(0.2)
    assert browser._idle_thread.is_alive()
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0.1")
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session():
            pass
    assert _wait_for(lambda: browser._driver is None)


def test_idle_close_can_be_disabled(shared_browser, monkeypatch):
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0")
    fake = _fake_chrome()
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session():
            pass
    time.sleep(0.3)
    assert browser._driver is fake
    assert browser.close_if_idle() is True
    fake.quit.assert_called_once()


def _watchers():
    return sum(t.name == "bandcamp-browser-idle" for t in threading.enumerate())


def test_idle_watch_survives_driver_restarts(shared_browser, monkeypatch):
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0.2")
    before = _watchers()
    fakes = [_fake_chrome(), _fake_chrome()]
    with patch.object(driver_manager.webdriver, "Chrome", side_effect=fakes):
        for fake in fakes:
            with browser.session():
                pass
            assert _wait_for(lambda: browser._driver is None)
            fake.quit.assert_called_once()
    assert _watchers() == max(before, 1)  # one per process, not one per driver


def test_idle_watch_survives_a_failed_close(shared_browser, monkeypatch):
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0.1")
    fake = _fake_chrome()
    real_quit = chrome_procs.quit_driver
    calls = []

    def flaky_quit(driver):
        calls.append(driver)
        if len(calls) == 1:
            raise RuntimeError("unexpected")
        real_quit(driver)

    monkeypatch.setattr(chrome_procs, "quit_driver", flaky_quit)
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session():
            pass
        assert _wait_for(lambda: len(calls) == 1)
        with browser.session():
            pass
    assert _wait_for(lambda: browser._driver is None)
    fake.quit.assert_called_once()


def test_a_gone_idle_watcher_is_restarted(shared_browser, monkeypatch):
    """A forked child inherits the Thread object, not the thread."""
    monkeypatch.setenv("BANDCAMP_BROWSER_IDLE_S", "0.1")
    monkeypatch.setattr(browser, "_idle_thread", threading.Thread(target=lambda: None))
    fake = _fake_chrome()
    with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
        with browser.session():
            pass
    assert browser._idle_thread.is_alive()
    assert _wait_for(lambda: browser._driver is None)
