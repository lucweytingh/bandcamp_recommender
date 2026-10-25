import pytest


@pytest.fixture(autouse=True)
def _hermetic_browser(tmp_path, monkeypatch):
    """No test touches real Chrome profiles, downloads a chromedriver, or
    reaps real processes on first driver start."""
    from bandcamp_recommender import chrome_procs

    monkeypatch.setenv("BANDCAMP_BROWSER_PROFILE_DIR", str(tmp_path / "chrome-profiles"))
    fake_driver = tmp_path / "chromedriver"
    fake_driver.write_text("")
    monkeypatch.setenv("CHROMEDRIVER", str(fake_driver))
    monkeypatch.setattr(chrome_procs, "_swept", True)
    monkeypatch.setattr(chrome_procs, "_live", {})
