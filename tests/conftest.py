import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_browser(tmp_path, monkeypatch):
    """No test touches real Chrome profiles, downloads a chromedriver, or
    reaps real processes: every driver start reaps, so the reaper sees a
    /proc holding only this process (tests needing more build their own)."""
    from bandcamp_recommender import chrome_procs

    monkeypatch.setenv("BANDCAMP_BROWSER_PROFILE_DIR", str(tmp_path / "chrome-profiles"))
    fake_driver = tmp_path / "chromedriver"
    fake_driver.write_text("")
    monkeypatch.setenv("CHROMEDRIVER", str(fake_driver))
    me = os.getpid()
    only_me = tmp_path / "proc-only-me" / str(me)
    only_me.mkdir(parents=True)
    (only_me / "stat").write_text(f"{me} (python3) R 1 {me} {me} 0 -1 4194560 0 0\n")
    monkeypatch.setattr(chrome_procs, "_PROC", str(only_me.parent))
    monkeypatch.setattr(chrome_procs, "_swept", True)
    monkeypatch.setattr(chrome_procs, "_live", {})
