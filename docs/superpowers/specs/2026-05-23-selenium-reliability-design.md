# Selenium / scraping reliability — design

**Date:** 2026-05-23
**Trigger:** Production radio worker (downstream consumer `bandcamp_suggestor`)
hung indefinitely on a Bandcamp page fetch. py-spy showed the asyncio worker
parked in a Selenium `driver.get(...)` call with no page-load timeout. The
unit was SIGKILL'd and on restart reaped 64 orphan chromium processes.

The downstream agent is fixing its own `_init_selenium_driver`. This spec
covers the parallel reliability hardening that needs to happen inside
`bandcamp_recommender` itself, so every caller (this one and future ones)
inherits the protection.

## Goals

1. No Selenium call can hang forever. Every `driver.get()` and
   `execute_async_script` is bounded by a wall-clock timeout.
2. No leaked chromedriver / chromium processes from a hung or errored call.
3. Existing silent failures become visible warnings (no behavior change for
   callers that rely on `[]` returns).
4. A reliability test suite that simulates the failure modes — instant 200,
   infinite hang, slow trickle, 403, connection reset — and asserts each
   surface returns under a wall-clock budget.

## Decisions

| ID | Decision | Override |
|----|----------|----------|
| A  | Page-load + script timeouts: **30 s** | `BANDCAMP_PAGE_LOAD_TIMEOUT`, `BANDCAMP_SCRIPT_TIMEOUT` |
| B  | Audio download timeout: **30 s** (was 300 s) | `BANDCAMP_AUDIO_TIMEOUT` |
| C  | Orphan reaper: **opt-in**, exposed as `DriverManager.reap_orphans()` | n/a |
| D  | Replace silent `except: pass` with `logger.warning(...)`. `exc_info=True` only when `BANDCAMP_DEBUG=1`. | `BANDCAMP_DEBUG` |
| E  | Retry policy: **1 retry** with a fresh driver, only on `selenium.common.exceptions.TimeoutException` or `WebDriverException`. Never on HTTP 4xx. | n/a |
| F  | Spec location: `docs/superpowers/specs/2026-05-23-selenium-reliability-design.md` | n/a |

## Scope

In scope:

- `bandcamp_recommender/recommendations/driver_manager.py`
- `bandcamp_recommender/recommendations/scraper.py` (`_fetch_page_with_selenium`)
- `bandcamp_recommender/recommendations/api.py` (`get_fan_id_from_page`, `_fetch_via_driver`)
- `bandcamp_recommender/recommendations/supporter_recommender.py` (`_get_supporter_*_with_driver`, `_get_supporter_items_curl_first` fallback path)
- `bandcamp_recommender/recommendations/bpm.py` (`_download_audio_bytes`)
- `tests/test_reliability.py` (new)

Out of scope:

- The downstream `bandcamp_suggestor` repo.
- Any feature work — recommendation quality, BPM detection algorithms, etc.
- Migrating off Selenium.
- Replacing the curl-first path (it already has a circuit breaker).

## Phases

### Phase 1 — Bound every Selenium hang

- Add `DriverManager._configure_driver(driver)` private helper. Calls
  `driver.set_page_load_timeout(_page_load_timeout())` and
  `driver.set_script_timeout(_script_timeout())`. Helpers read the env vars
  with 30 s defaults.
- Call `_configure_driver` from `init_driver`, `create_driver`, and inside
  the `get_driver_pool` loop right after each `webdriver.Chrome(...)`.
- `scraper._fetch_page_with_selenium`: wrap the body in `try/finally` so
  `dm.close()` is guaranteed to run even when `driver.get()` raises
  `TimeoutException`. Catch `TimeoutException` separately from `Exception`
  for clearer logging.
- `api.get_fan_id_from_page` and the two `_get_supporter_*_with_driver`
  methods already see `TimeoutException` once Phase 1 lands (the driver
  raises). No extra try/except needed — their existing `except Exception`
  already covers it. Phase 4 turns these into log warnings.

### Phase 2 — Pool hygiene + orphan reaper

- `DriverManager._is_driver_alive(driver)`: try `driver.current_url`, swallow
  `WebDriverException`, return `bool`.
- `SupporterRecommender._get_supporter_items_curl_first` finally block:
  after the worker returns the driver, if `_is_driver_alive` is False,
  `driver.quit()` it instead of `put`ing it back. The pool gets smaller
  (acceptable — it's a fallback path used only when curl fails).
- `DriverManager.reap_orphans()`: list local `chromedriver` and
  `chrome`/`chromium` processes whose parent PID is 1 (orphaned) and SIGKILL
  them. macOS + Linux only. Safe no-op on Windows. Returns the count reaped.
  Opt-in — callers run this on cold start. Not called automatically.

### Phase 3 — Audio download timeout

- `_download_audio_bytes`: change default `timeout` to 30 s. Read the URL
  in chunks with a wall-clock budget check between chunks so a slow-trickle
  response can't exceed the timeout.
- Surface env var `BANDCAMP_AUDIO_TIMEOUT`.

### Phase 4 — Observability

- Module-level `logger = logging.getLogger(__name__)` in every module
  touched.
- Every `except Exception: pass` / `return []` / `return None` that's
  swallowing a real error becomes
  `logger.warning("...", exc_info=os.environ.get("BANDCAMP_DEBUG") == "1")`.
- Targets:
  - `api.get_fan_id_from_page` (line 62)
  - `api._fetch_via_driver` (line 151)
  - `scraper._fetch_page_with_selenium` (line 224)
  - `scraper.extract_supporters`, `extract_item_id`, `extract_tags`
  - `supporter_recommender._get_supporter_purchases_with_driver` (line 660)
  - `supporter_recommender._get_supporter_wishlist_with_driver` (line 754)
  - `supporter_recommender._get_supporter_items_curl_first` (line 934)
- Behavior preserved — still returns `[] / None`.

### Phase 5 — Bounded retry

- Inside `_get_supporter_items_curl_first` Selenium fallback only:
  on `TimeoutException` / `WebDriverException` from the worker call,
  quit the driver, get a fresh one from the pool, retry once. If the
  retry also fails, return `[]` (current behavior).
- Cap retries at 1 — never recurse.

### Phase 6 — Reliability test suite

`tests/test_reliability.py` — pytest module.

Fixtures:

- `local_http`: a `pytest` fixture that starts a `ThreadingHTTPServer` on
  a random localhost port for the test, with handlers:
  - `/instant` → 200, body `b"<html><body>ok</body></html>"`
  - `/hang` → accept connection, never write a byte (blocks until client times out)
  - `/slow` → drip 1 byte / second forever
  - `/forbidden` → 403
  - `/reset` → accept then immediately close the socket
- `wall_clock_budget(seconds)`: a context manager that asserts the wrapped
  block completed inside `seconds`. Fails the test if exceeded.
- `fake_webdriver`: returns a `unittest.mock.MagicMock` shaped like
  `selenium.webdriver.Chrome` so we can assert `set_page_load_timeout`
  and `set_script_timeout` were called on every driver this package creates.

Tests:

1. `test_driver_manager_configures_timeouts_init_driver` — patches
   `webdriver.Chrome`, calls `DriverManager().init_driver()`, asserts both
   `set_page_load_timeout(30)` and `set_script_timeout(30)` were called.
2. `test_driver_manager_configures_timeouts_create_driver` — same for
   `create_driver`.
3. `test_driver_manager_configures_timeouts_pool` — same for every driver
   in `get_driver_pool(pool_size=3)`.
4. `test_driver_manager_respects_env_overrides` — sets
   `BANDCAMP_PAGE_LOAD_TIMEOUT=12`, asserts the value used.
5. `test_scraper_selenium_fallback_quits_on_exception` — patches
   `webdriver.Chrome` to a mock whose `get` raises `TimeoutException`.
   Calls `scraper._fetch_page_with_selenium("http://...")`. Asserts the
   returned value is `None` AND the mock's `quit()` was called.
6. `test_fetch_page_html_against_hang_curl_under_budget` — points
   `fetch_page_html` at `local_http /hang`, with `timeout=2`. Asserts it
   returns `None` inside a 3 s wall-clock budget (curl honours its own
   `--max-time`-style timeout via `subprocess.run(..., timeout=...)`).
7. `test_fetch_page_html_against_403` — `/forbidden` returns `None` and
   trips the curl breaker on repeated calls. Verified via
   `curl_breaker._read_state()`.
8. `test_audio_download_bounded_against_hang` — point
   `_download_audio_bytes` at `/hang` with `timeout=2`. Asserts None
   under a 3 s budget.
9. `test_audio_download_bounded_against_slow_trickle` — point at `/slow`
   with `timeout=2`. Asserts None under a 3 s budget (this is the test
   that fails without the chunked wall-clock check).
10. `test_supporter_items_retry_once_on_selenium_timeout` — mock the
    DriverManager so the first driver `.get()` raises `TimeoutException`
    and the second returns a parseable wishlist page. Assert exactly two
    drivers were used and the result is non-empty.
11. `test_reap_orphans_returns_count` — fork a child that execs `sleep`,
    re-parent to init, assert `reap_orphans()` finds and kills it.
    Skipped on Windows.

All tests must complete in ≤ 5 s wall-clock individually and ≤ 30 s for
the whole module.

### Phase 7 — Iterate

Run `pytest tests/test_reliability.py` 10 times consecutively. Any test
that fails or exceeds budget gets tightened. Final acceptance: 10
consecutive green runs.

## Non-goals (explicit)

- We are not adding async support to the recommender. Callers must continue
  to wrap in `asyncio.to_thread` if used from an event loop.
- We are not removing the `time.sleep(3)` in `_fetch_page_with_selenium` —
  it exists so Bandcamp's JS can hydrate `collectors-data`. We can tighten
  it later if a real test forces it.
- No new public methods on `SupporterRecommender`. All changes are internal
  or on `DriverManager`.

## Backwards compatibility

- Default behavior changes (timeouts kick in). Pre-existing callers see
  `TimeoutException`-then-swallowed-to-`[]` instead of forever-hang. Both
  are "graceful failure" — callers already treat `[]` as a normal outcome.
- The audio timeout drop (300 → 30 s) is the only user-visible change for
  consumers running the optional `bpm` extra. Documented in CHANGELOG as
  the bug fix it is.
