# Mood scorer eval harness

Compares N chill/party scoring strategies side-by-side over your Bandcamp
wishlist. Each scorer lives in its own worktree; the harness invokes
each via `uv run --project <worktree>` and renders a single HTML page.

## Run

```bash
uv run python scripts/eval_mood.py --user lucrw --seeds 3 --recs 4
open eval_output/index.html
```

Flags:

- `--user <name>` — Bandcamp username (default `lucrw`).
- `--seeds N` — chill + party seeds per scorer (default 3).
- `--recs N` — recommendations per seed (default 4).
- `--min-supporters N` — passed through to the recommender (default 2).
- `--refresh-wishlist` — bypass `.context/wishlist_<user>.json` cache.

## Adding a scorer

Each worktree must ship `bandcamp_recommender/eval/score_<name>.py`
implementing the shared contract:

```python
def score_items(items: list[dict]) -> dict[str, Optional[float]]:
    """Returns {item_id: mood in [-1.0 chill, 1.0 party] or None}."""
```

…plus a CLI: `python -m bandcamp_recommender.eval.score_<name> --in items.json --out scores.json`.

Register the worktree in `SCORERS` at the top of `scripts/eval_mood.py`.
Missing modules render as "not ready" — safe to re-run the harness as
each agent lands.

## Current scorer map

| Name | Worktree | Status |
|---|---|---|
| `mood_tags` (C) | `chengdu/` | ✅ shipped |
| `intensity` (B) | `kabul-v2/` | ⏳ in progress |
| `bpm` (A) | `baku/` | ⏳ in progress |

## Layout

- `scripts/eval_mood.py` — harness orchestrator.
- `.context/wishlist_<user>.json` — cached, hydrated wishlist (tags + preview URLs). Delete or pass `--refresh-wishlist` to re-scrape.
- `eval_output/data.json` — full eval state (items, scores, seeds, rec lists, per-rec scores).
- `eval_output/index.html` — self-contained, no build step. Drag into a browser.
