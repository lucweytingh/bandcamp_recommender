"""Mood-scorer evaluation harness.

Runs each scorer (living in its own worktree) over a user's Bandcamp
wishlist, picks chill/party seeds, fetches supporter-overlap
recommendations for each seed, scores those recs with the same scorer,
and renders a single HTML page where you can listen + inspect.

The harness never imports scorer code — each scorer is invoked as a
subprocess via ``uv run --project <worktree>`` so multiple
``bandcamp_recommender`` versions on disk don't collide. Communication
is two JSON files per scorer (items in / scores out) per the shared
contract.

Pipeline
========

1. Scrape ``bandcamp.com/<user>/wishlist`` once → ``wishlist_<user>.json``.
2. Hydrate each item with ``tags`` + ``audio_url`` (one page fetch per
   item, parallel) → ``items.json``.
3. For each configured scorer:
   a. Subprocess ``score_<name>.py --in items.json --out scores.json``.
   b. Pick top-N chill and top-N party items (most negative / positive).
   c. For each seed, ``SupporterRecommender.get_recommendations`` →
      candidate recs. Score those recs with the same scorer.
4. Write ``eval_output/data.json`` + ``eval_output/index.html``.

Scorer status is reported per-scorer: missing module → "not ready"
column. Re-run the harness once a worktree finishes shipping its
adapter and the column lights up.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

# Reuse existing scraper machinery from this workspace.
from bandcamp_recommender.recommendations.scraper import fetch_page_html


HARNESS_ROOT = Path(__file__).resolve().parent.parent
CONTEXT_DIR = HARNESS_ROOT / ".context"
OUTPUT_DIR = HARNESS_ROOT / "eval_output"

WORKTREES_ROOT = Path("/Users/lucw/conductor/workspaces/bandcamp_recommender")


@dataclass
class Scorer:
    """One scoring approach plugged into the harness.

    ``module`` is the dotted path inside its worktree's
    ``bandcamp_recommender`` package. ``worktree`` is the absolute path
    to the worktree root (where ``pyproject.toml`` lives).
    """

    name: str
    worktree: Path
    module: str
    status: str = "pending"  # "ok" | "not_ready" | "error"
    error: str = ""
    items_scored: Dict[str, Optional[float]] = field(default_factory=dict)
    seeds: Dict[str, List[str]] = field(default_factory=dict)  # {"chill": [...], "party": [...]}
    rec_scores: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)  # seed_id -> {rec_id: score}
    rec_lists: Dict[str, List[str]] = field(default_factory=dict)  # seed_id -> [rec_id, ...]


SCORERS = [
    Scorer(
        name="mood_tags",
        worktree=WORKTREES_ROOT / "chengdu",
        module="bandcamp_recommender.eval.score_mood_tags",
    ),
    Scorer(
        name="intensity",
        worktree=WORKTREES_ROOT / "kabul-v2",
        module="bandcamp_recommender.eval.score_intensity",
    ),
    Scorer(
        name="bpm",
        worktree=WORKTREES_ROOT / "baku",
        module="bandcamp_recommender.eval.score_bpm",
    ),
]


def blend_scores(
    inputs: List[Tuple[Dict[str, Optional[float]], float]],
) -> Dict[str, Optional[float]]:
    """N-way linear blend of score dicts on the same [-1, 1] scale.

    ``inputs`` is a list of ``(score_dict, weight)`` pairs. Weights need
    not sum to 1 — for each item they're renormalized over the subset of
    signals that actually produced a non-None score. This gives graceful
    degradation: a missing signal drops out and its weight is
    redistributed proportionally to the others, rather than poisoning
    the blend with a zero.

    Returns None for any item where every input was None.
    """
    if not inputs:
        return {}
    all_ids: set[str] = set()
    for scores, _ in inputs:
        all_ids |= set(scores)

    out: Dict[str, Optional[float]] = {}
    for iid in all_ids:
        present = [(scores[iid], w) for scores, w in inputs
                   if scores.get(iid) is not None]
        total_w = sum(w for _, w in present)
        if not present or total_w <= 0:
            out[iid] = None
            continue
        out[iid] = sum(v * w for v, w in present) / total_w
    return out


# -----------------------------------------------------------------------------
# Step 1+2: wishlist scrape + per-item hydration (tags + audio_url)
# -----------------------------------------------------------------------------


def _audio_url_from_trackinfo(trackinfo: List[Dict[str, Any]]) -> Optional[str]:
    """First mp3-128 / mp3-v0 URL from a Bandcamp trackinfo list."""
    for track in trackinfo or []:
        file_dict = track.get("file") or {}
        url = file_dict.get("mp3-128") or file_dict.get("mp3-v0")
        if url:
            return url
    return None


def _extract_tags_and_audio(html: str) -> Tuple[List[str], Optional[str]]:
    """Pull tag list + first preview audio URL from a single item-page HTML.

    Doing both off one fetch halves wishlist-hydration wall clock. Audio
    URL extraction mirrors ``bpm.extract_track_info`` — try
    ``data-tralbum`` first, fall back to the ``pagedata`` blob, since
    some Bandcamp pages only populate the latter.
    """
    soup = BeautifulSoup(html, features="html.parser")
    tag_links = soup.find_all("a", class_=re.compile("tag"))
    tags = [t.get_text(strip=True) for t in tag_links if t.get_text(strip=True)]

    audio_url: Optional[str] = None

    # Method 1: data-tralbum attribute (album / track pages with embedded player).
    tralbum_elem = soup.find(attrs={"data-tralbum": True})
    if tralbum_elem:
        try:
            blob = json.loads(tralbum_elem.get("data-tralbum") or "{}")
            audio_url = _audio_url_from_trackinfo(blob.get("trackinfo") or [])
        except (json.JSONDecodeError, AttributeError):
            pass

    # Method 2: pagedata blob (covers pages where data-tralbum is missing).
    if not audio_url:
        pagedata_elem = soup.find(id="pagedata")
        if pagedata_elem:
            try:
                pagedata = json.loads(pagedata_elem.get("data-blob") or "{}")
                tralbum_data = pagedata.get("tralbum_data") or {}
                audio_url = _audio_url_from_trackinfo(tralbum_data.get("trackinfo") or [])
            except (json.JSONDecodeError, AttributeError):
                pass

    return tags, audio_url


def _hydrate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("item_url") or ""
    if not url or "/album/" + str(item.get("item_id", "")) in url:
        # Placeholder URL from supporter_recommender — useless for hydration.
        item.setdefault("tags", [])
        item.setdefault("audio_url", None)
        return item
    html = fetch_page_html(url, timeout=12)
    if not html:
        item.setdefault("tags", [])
        item.setdefault("audio_url", None)
        return item
    tags, audio = _extract_tags_and_audio(html)
    item["tags"] = tags
    item["audio_url"] = audio
    return item


def scrape_wishlist(user: str, cache: Path, refresh: bool = False) -> List[Dict[str, Any]]:
    """Return a list of hydrated wishlist items, cached on disk by user."""
    if cache.exists() and not refresh:
        print(f"[wishlist] cache hit: {cache}")
        return json.loads(cache.read_text())

    print(f"[wishlist] scraping bandcamp.com/{user}/wishlist ...")
    from bandcamp_recommender.recommendations.supporter_recommender import (
        SupporterRecommender,
    )

    r = SupporterRecommender()
    item_ids = r._get_supporter_items_curl_first(user, "wishlist_data", first_page_only=False)
    print(f"[wishlist] got {len(item_ids)} items, hydrating tags + audio URLs ...")

    items: List[Dict[str, Any]] = []
    for item_id in item_ids:
        cached = r.item_cache.get(item_id) or {}
        items.append({
            "item_id": item_id,
            "item_url": cached.get("item_url", f"https://bandcamp.com/album/{item_id}"),
            "item_title": cached.get("item_title", "Unknown"),
            "band_name": cached.get("band_name", "Unknown"),
        })

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_hydrate_item, it): it for it in items}
        for fut in as_completed(futs):
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"[wishlist] hydrated {done}/{len(items)} ({time.time() - t0:.1f}s)")
            # Mutation already happened in the worker; .result() just re-raises.
            try:
                fut.result()
            except Exception as e:
                print(f"[wishlist]   hydrate error: {e}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, indent=2))
    print(f"[wishlist] cached → {cache}")
    return items


# -----------------------------------------------------------------------------
# Step 3: run each scorer as a subprocess + recommendation generation
# -----------------------------------------------------------------------------


def _run_scorer_subprocess(
    scorer: Scorer, items: List[Dict[str, Any]], tag: str
) -> Optional[Dict[str, Optional[float]]]:
    """Write items → temp file, invoke scorer's CLI in its worktree's venv, read scores back.

    Returns the parsed scores dict, or None on any failure (with
    ``scorer.status`` / ``scorer.error`` mutated for the HTML).
    """
    if not scorer.worktree.exists():
        scorer.status = "not_ready"
        scorer.error = f"worktree missing: {scorer.worktree}"
        return None
    module_path = scorer.worktree / Path(*scorer.module.split(".")).with_suffix(".py")
    if not module_path.exists():
        scorer.status = "not_ready"
        scorer.error = f"{scorer.module} not implemented yet in {scorer.worktree.name}"
        return None

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / f"items_{tag}.json"
        out_path = td_path / f"scores_{tag}.json"
        in_path.write_text(json.dumps(items))
        cmd = [
            "uv", "run", "--project", str(scorer.worktree),
            "python", "-m", scorer.module,
            "--in", str(in_path), "--out", str(out_path),
        ]
        print(f"[scorer:{scorer.name}] {' '.join(cmd[-6:])}")
        t0 = time.time()
        # Strip the harness's own VIRTUAL_ENV so uv resolves the scorer's
        # worktree venv instead of warning about and ignoring it.
        sub_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        try:
            # Run from the scorer's worktree so Python doesn't pick up
            # kabul-v1's ``bandcamp_recommender/`` from cwd ahead of the
            # worktree's installed package.
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200,
                env=sub_env, cwd=str(scorer.worktree),
            )
        except subprocess.TimeoutExpired:
            scorer.status = "error"
            scorer.error = "scorer timed out after 1200s"
            return None
        dt = time.time() - t0
        if res.returncode != 0:
            scorer.status = "error"
            scorer.error = f"exit {res.returncode}: {res.stderr[-400:].strip()}"
            print(f"[scorer:{scorer.name}] FAILED ({dt:.1f}s): {scorer.error}")
            return None
        if not out_path.exists():
            scorer.status = "error"
            scorer.error = "scorer produced no output file"
            return None
        try:
            scores = json.loads(out_path.read_text())
        except json.JSONDecodeError as e:
            scorer.status = "error"
            scorer.error = f"invalid scores.json: {e}"
            return None
        print(f"[scorer:{scorer.name}] done ({dt:.1f}s, {sum(1 for v in scores.values() if v is not None)}/{len(scores)} scored)")
        return scores


def _pick_seeds(scores: Dict[str, Optional[float]], n: int) -> Dict[str, List[str]]:
    """Top-N chill (most negative) and top-N party (most positive)."""
    ranked = [(iid, v) for iid, v in scores.items() if v is not None]
    chill = sorted(ranked, key=lambda x: x[1])[:n]
    party = sorted(ranked, key=lambda x: x[1], reverse=True)[:n]
    return {
        "chill": [iid for iid, _ in chill],
        "party": [iid for iid, _ in party],
    }


def _fetch_recs_for_seed(
    seed_url: str, max_recs: int, min_supporters: int
) -> List[Dict[str, Any]]:
    """Run the supporter-overlap recommender for one seed URL."""
    from bandcamp_recommender.recommendations.supporter_recommender import (
        SupporterRecommender,
    )

    r = SupporterRecommender()
    try:
        recs = r.get_recommendations(
            wishlist_item_url=seed_url,
            max_recommendations=max_recs,
            min_supporters=min_supporters,
        )
    except Exception as e:
        print(f"[recs] error for {seed_url}: {e}")
        return []
    finally:
        try:
            r.close()
        except Exception:
            pass
    return recs


def _materialize_rec_items(
    scorer: Scorer,
    items_by_id: Dict[str, Dict[str, Any]],
    seed_items: List[Dict[str, Any]],
    max_recs: int,
    min_supporters: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """For each seed, fetch recs and hydrate them (tags + audio_url).

    Returns ``(seed_id -> [rec dict, ...], rec_id -> rec dict)``. The
    second map exists so the renderer can resolve any rec_id back to a
    full item dict, and so re-encountering the same rec across seeds
    only costs one hydration.
    """
    per_seed: Dict[str, List[Dict[str, Any]]] = {}
    rec_pool: Dict[str, Dict[str, Any]] = {}

    for seed in seed_items:
        seed_id = seed["item_id"]
        recs = _fetch_recs_for_seed(
            seed["item_url"], max_recs=max_recs, min_supporters=min_supporters
        )
        if not recs:
            per_seed[seed_id] = []
            continue
        # Recs from `get_recommendations` carry item_url + title/band + tags
        # but not necessarily a preview audio URL. Hydrate any missing ones.
        for rec in recs:
            rid = str(rec.get("item_id") or rec.get("item_url") or "")
            if not rid:
                # No stable id — synthesize one from URL.
                rid = rec.get("item_url", "")
            rec["item_id"] = rid
            if rid not in rec_pool:
                if not rec.get("audio_url"):
                    _hydrate_item(rec)
                rec_pool[rid] = rec
        per_seed[seed_id] = [rec_pool[str(rec.get("item_id"))] for rec in recs if rec.get("item_id")]

    return per_seed, rec_pool


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_eval(
    user: str,
    seeds_per_mood: int,
    recs_per_seed: int,
    min_supporters: int,
    refresh_wishlist: bool,
) -> Dict[str, Any]:
    items = scrape_wishlist(user, CONTEXT_DIR / f"wishlist_{user}.json", refresh_wishlist)
    items_by_id = {it["item_id"]: it for it in items}

    # Aggregate everything we want to render into one big dict.
    data: Dict[str, Any] = {
        "user": user,
        "items": items,  # full hydrated wishlist
        "scorers": [],
    }

    rec_pool_all: Dict[str, Dict[str, Any]] = {}

    for scorer in SCORERS:
        scores = _run_scorer_subprocess(scorer, items, tag=scorer.name)
        if scores is None:
            data["scorers"].append({
                "name": scorer.name,
                "status": scorer.status,
                "error": scorer.error,
            })
            continue

        scorer.items_scored = scores
        scorer.status = "ok"
        scorer.seeds = _pick_seeds(scores, seeds_per_mood)

        # Look up seed item dicts (skip any seed that's missing from cache).
        seed_items_chill = [items_by_id[i] for i in scorer.seeds["chill"] if i in items_by_id]
        seed_items_party = [items_by_id[i] for i in scorer.seeds["party"] if i in items_by_id]

        # Generate recs for all seeds (chill + party) up front so we can
        # score them with one subprocess call.
        chill_recs_by_seed, chill_pool = _materialize_rec_items(
            scorer, items_by_id, seed_items_chill, recs_per_seed, min_supporters
        )
        party_recs_by_seed, party_pool = _materialize_rec_items(
            scorer, items_by_id, seed_items_party, recs_per_seed, min_supporters
        )
        rec_pool_all.update(chill_pool)
        rec_pool_all.update(party_pool)

        # Score the recs (one subprocess invocation for all of them at once).
        rec_items = list({**chill_pool, **party_pool}.values())
        rec_scores: Dict[str, Optional[float]] = {}
        if rec_items:
            rec_scores = _run_scorer_subprocess(scorer, rec_items, tag=f"{scorer.name}_recs") or {}

        # Stitch into per-seed score maps for the renderer.
        per_seed_scores: Dict[str, Dict[str, Optional[float]]] = {}
        per_seed_recs: Dict[str, List[str]] = {}
        for seed_id, recs in {**chill_recs_by_seed, **party_recs_by_seed}.items():
            ids = [str(r["item_id"]) for r in recs]
            per_seed_recs[seed_id] = ids
            per_seed_scores[seed_id] = {rid: rec_scores.get(rid) for rid in ids}

        scorer.rec_scores = per_seed_scores
        scorer.rec_lists = per_seed_recs

        data["scorers"].append({
            "name": scorer.name,
            "status": "ok",
            "scores": scores,
            "seeds": scorer.seeds,
            "rec_lists": scorer.rec_lists,
            "rec_scores": scorer.rec_scores,
            "coverage": sum(1 for v in scores.values() if v is not None) / max(1, len(scores)),
        })

    data["rec_pool"] = rec_pool_all
    return data


# -----------------------------------------------------------------------------
# HTML rendering
# -----------------------------------------------------------------------------


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mood scorer eval — {user}</title>
<style>
  :root {{
    --bg:#0f1115; --fg:#e8e8ea; --muted:#8a8d96; --card:#171a21;
    --chill:#5fa9ff; --party:#ff5f88; --warn:#ffb454; --ok:#7ee787;
  }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ margin:0 0 8px; font-size:20px; font-weight:600; }}
  h2 {{ margin:28px 0 12px; font-size:16px; font-weight:600; }}
  .sub {{ color:var(--muted); margin-bottom:24px; }}
  .scorers {{ display:grid; grid-template-columns:repeat({n_cols}, 1fr); gap:16px;
              align-items:start; }}
  .col {{ background:var(--card); border-radius:8px; padding:14px; min-width:0; }}
  .col h3 {{ margin:0 0 4px; font-size:15px; }}
  .col .meta {{ color:var(--muted); font-size:12px; margin-bottom:10px; }}
  .section {{ margin-top:14px; }}
  .section h4 {{ margin:0 0 6px; font-size:13px; font-weight:600;
                 text-transform:uppercase; letter-spacing:.04em; }}
  .section.chill h4 {{ color:var(--chill); }}
  .section.party h4 {{ color:var(--party); }}
  .seed {{ border-left:2px solid var(--muted); padding:6px 0 6px 10px;
           margin-bottom:14px; }}
  .seed.chill {{ border-color:var(--chill); }}
  .seed.party {{ border-color:var(--party); }}
  .row {{ display:flex; gap:8px; align-items:center; padding:4px 0; min-width:0; }}
  .row .body {{ flex:1; min-width:0; }}
  .row .title {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                 font-weight:500; }}
  .row .artist {{ color:var(--muted); font-size:12px;
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .score {{ font-variant-numeric:tabular-nums; font-size:11px;
            padding:1px 5px; border-radius:3px; background:#222630; }}
  .score.chill {{ color:var(--chill); }}
  .score.party {{ color:var(--party); }}
  .score.null {{ color:var(--muted); }}
  audio {{ height:24px; }}
  .seed-row {{ font-weight:600; }}
  .recs {{ margin:4px 0 0 12px; }}
  a {{ color:inherit; }} a:hover {{ text-decoration:underline; }}
  .badge {{ display:inline-block; padding:1px 6px; border-radius:3px; font-size:11px; }}
  .badge.ok {{ background:#1f3a26; color:var(--ok); }}
  .badge.warn {{ background:#3a2e1c; color:var(--warn); }}
  .badge.err {{ background:#3a1c1c; color:#ff7a7a; }}
  .empty {{ color:var(--muted); padding:24px; text-align:center;
            border:1px dashed #2a2e38; border-radius:8px; }}
  .tags {{ color:var(--muted); font-size:11px; margin-top:1px;
           white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
</style>
</head>
<body>
<h1>Mood scorer eval — bandcamp.com/{user}</h1>
<div class="sub">
  Wishlist: {n_items} items · {seeds_per_mood} chill + {seeds_per_mood} party
  seeds per scorer · {recs_per_seed} recs per seed.
</div>

<div class="scorers">
{columns}
</div>

<script>
  // No JS needed — everything's static.
</script>
</body>
</html>"""


def _score_pill(score: Optional[float]) -> str:
    if score is None:
        return '<span class="score null">·</span>'
    cls = "chill" if score < 0 else "party"
    return f'<span class="score {cls}">{score:+.2f}</span>'


def _row_html(item: Dict[str, Any], score: Optional[float], is_seed: bool = False) -> str:
    title = (item.get("item_title") or "Unknown").replace("<", "&lt;")
    artist = (item.get("band_name") or "Unknown").replace("<", "&lt;")
    url = item.get("item_url") or "#"
    audio = item.get("audio_url")
    audio_html = (
        f'<audio controls preload="none" src="{audio}"></audio>' if audio else ""
    )
    tags = item.get("tags") or []
    tags_html = (
        f'<div class="tags">{", ".join(tags[:6])}</div>' if tags else ""
    )
    cls = "row seed-row" if is_seed else "row"
    return (
        f'<div class="{cls}">'
        f'{_score_pill(score)}'
        f'<div class="body">'
        f'<div class="title"><a href="{url}" target="_blank">{title}</a></div>'
        f'<div class="artist">{artist}</div>'
        f'{tags_html}'
        f'</div>'
        f'{audio_html}'
        f'</div>'
    )


def _scorer_column_html(
    scorer_data: Dict[str, Any],
    items_by_id: Dict[str, Dict[str, Any]],
    rec_pool: Dict[str, Dict[str, Any]],
) -> str:
    name = scorer_data["name"]
    status = scorer_data["status"]
    if status != "ok":
        badge_cls = "warn" if status == "not_ready" else "err"
        msg = scorer_data.get("error") or status
        return (
            f'<div class="col">'
            f'<h3>{name}</h3>'
            f'<div class="meta"><span class="badge {badge_cls}">{status}</span></div>'
            f'<div class="empty">{msg}</div>'
            f'</div>'
        )

    scores = scorer_data["scores"]
    seeds = scorer_data["seeds"]
    rec_lists = scorer_data["rec_lists"]
    rec_scores = scorer_data["rec_scores"]
    coverage = scorer_data.get("coverage", 0)

    def _section(label: str, mood: str) -> str:
        seed_ids = seeds.get(mood, [])
        if not seed_ids:
            return f'<div class="section {mood}"><h4>{label}</h4><div class="empty">no seeds</div></div>'
        parts = [f'<div class="section {mood}"><h4>{label}</h4>']
        for sid in seed_ids:
            seed_item = items_by_id.get(sid)
            if not seed_item:
                continue
            parts.append(f'<div class="seed {mood}">')
            parts.append(_row_html(seed_item, scores.get(sid), is_seed=True))
            parts.append('<div class="recs">')
            for rid in rec_lists.get(sid, []):
                rec_item = rec_pool.get(rid) or items_by_id.get(rid)
                if not rec_item:
                    continue
                parts.append(_row_html(rec_item, rec_scores.get(sid, {}).get(rid)))
            parts.append('</div>')
            parts.append('</div>')
        parts.append('</div>')
        return "".join(parts)

    return (
        f'<div class="col">'
        f'<h3>{name}</h3>'
        f'<div class="meta">'
        f'<span class="badge ok">ok</span> · coverage {coverage:.0%} '
        f'({sum(1 for v in scores.values() if v is not None)}/{len(scores)})'
        f'</div>'
        f'{_section("Chill seeds → recs", "chill")}'
        f'{_section("Party seeds → recs", "party")}'
        f'</div>'
    )


def render_html(
    data: Dict[str, Any], seeds_per_mood: int, recs_per_seed: int
) -> str:
    items_by_id = {it["item_id"]: it for it in data["items"]}
    rec_pool = data.get("rec_pool", {})
    columns = "\n".join(
        _scorer_column_html(s, items_by_id, rec_pool) for s in data["scorers"]
    )
    return HTML_TEMPLATE.format(
        user=data["user"],
        n_items=len(data["items"]),
        n_cols=len(data["scorers"]),
        seeds_per_mood=seeds_per_mood,
        recs_per_seed=recs_per_seed,
        columns=columns,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="lucrw", help="Bandcamp username (default: lucrw)")
    p.add_argument("--seeds", type=int, default=3, help="seeds per mood per scorer")
    p.add_argument("--recs", type=int, default=4, help="recs per seed")
    p.add_argument("--min-supporters", type=int, default=2)
    p.add_argument("--refresh-wishlist", action="store_true",
                   help="bypass the wishlist cache and re-scrape")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = run_eval(
        user=args.user,
        seeds_per_mood=args.seeds,
        recs_per_seed=args.recs,
        min_supporters=args.min_supporters,
        refresh_wishlist=args.refresh_wishlist,
    )

    data_path = OUTPUT_DIR / "data.json"
    data_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"[eval] wrote {data_path}")

    html = render_html(data, args.seeds, args.recs)
    html_path = OUTPUT_DIR / "index.html"
    html_path.write_text(html)
    print(f"[eval] wrote {html_path}")
    print(f"[eval] open: file://{html_path}")


if __name__ == "__main__":
    main()
