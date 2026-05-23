"""Similarity eval: one source song, one rec pool, three rankings.

Different from ``eval_mood.py`` — that one picks chill/party seeds per
scorer and shows what each scorer thinks fits each mood. This one fixes
both the source and the candidate pool, then asks each scorer to rank
the *same* candidates by closeness to the source. That isolates the
ranking method as the only variable.

Pipeline
========

1. Pick a source song (``--source`` URL, or default to the first item
   of ``lucrw``'s wishlist).
2. Fetch supporter-overlap recs for the source ONCE
   (``min_supporters=1``, generous ``max_recommendations``).
3. Hydrate the source + recs (tags + ``audio_url``) — one fetch per item.
4. For each scorer, run a single subprocess on ``[source] + recs`` and
   compute ``distance = |score_source - score_candidate|`` per rec.
5. Render ``eval_output/similarity.html``: source on top, three columns
   each showing the same recs in that scorer's ranked order.

Reuses ``SCORERS``, ``_run_scorer_subprocess``, ``_hydrate_item``, and
``_fetch_recs_for_seed`` from ``eval_mood.py`` so the scorer plumbing
stays in one place.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the shared subprocess plumbing.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from eval_mood import (  # type: ignore
    HARNESS_ROOT,
    CONTEXT_DIR,
    OUTPUT_DIR,
    SCORERS,
    blend_scores,
    _fetch_recs_for_seed,
    _hydrate_item,
    _run_scorer_subprocess,
    scrape_wishlist,
)


def _build_source(source_url: Optional[str], user: str) -> Dict[str, Any]:
    """Either honour ``--source`` or fall back to the first cached wishlist item."""
    if source_url:
        # Synthesise an item dict; hydration fills tags + audio_url.
        item: Dict[str, Any] = {
            "item_id": source_url,
            "item_url": source_url,
            "item_title": source_url.rsplit("/", 1)[-1],
            "band_name": "(source)",
        }
        _hydrate_item(item)
        return item

    items = scrape_wishlist(user, CONTEXT_DIR / f"wishlist_{user}.json", refresh=False)
    if not items:
        raise SystemExit(f"no wishlist items for {user}; pass --source instead")
    return items[0]


def _wishlist_as_pool(user: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use the (already cached + hydrated) wishlist minus the source as the pool.

    Free, deterministic, and bypasses Bandcamp's supporter-API rate limits.
    Slightly different eval semantics: this asks "within my wishlist,
    which items does each scorer think are most similar to the source?"

    Excludes both by item_id and item_url so a ``--source`` URL that
    happens to match a wishlist album doesn't appear as a trivial
    ``distance=0`` self-match.
    """
    items = scrape_wishlist(user, CONTEXT_DIR / f"wishlist_{user}.json", refresh=False)
    source_id = str(source.get("item_id") or "")
    source_url = (source.get("item_url") or "").rstrip("/")
    return [
        it for it in items
        if str(it["item_id"]) != source_id
        and (it.get("item_url") or "").rstrip("/") != source_url
    ]


def _fetch_pool(source: Dict[str, Any], pool_size: int, min_supporters: int) -> List[Dict[str, Any]]:
    """Pull a generous candidate pool once. Hydrate any rec missing audio_url."""
    recs = _fetch_recs_for_seed(
        source["item_url"], max_recs=pool_size, min_supporters=min_supporters
    )
    if not recs:
        return []
    # Stabilise IDs + hydrate any rec lacking a preview URL.
    out: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for rec in recs:
        rid = str(rec.get("item_id") or rec.get("item_url") or "")
        if not rid or rid in seen_ids:
            continue
        seen_ids.add(rid)
        rec["item_id"] = rid
        if not rec.get("audio_url"):
            _hydrate_item(rec)
        out.append(rec)
    return out


def _rank_by_distance(
    source_id: str, scores: Dict[str, Optional[float]], rec_ids: List[str]
) -> List[Dict[str, Any]]:
    """Return [{item_id, score, distance, rank}] sorted ascending by distance.

    Recs the scorer couldn't score sink to the bottom in their original
    pool order.
    """
    src = scores.get(source_id)
    rated: List[Dict[str, Any]] = []
    unrated: List[Dict[str, Any]] = []
    for rid in rec_ids:
        cand = scores.get(rid)
        if src is None or cand is None:
            unrated.append({"item_id": rid, "score": cand, "distance": None})
        else:
            rated.append({
                "item_id": rid, "score": cand, "distance": abs(src - cand),
            })
    rated.sort(key=lambda r: r["distance"])
    out = rated + unrated
    for i, row in enumerate(out):
        row["rank"] = i + 1
    return out


def _hybrid_rmse_distance(
    source_id: str,
    candidate_id: str,
    inputs: List[Tuple[Dict[str, Optional[float]], float]],
) -> Optional[float]:
    """Weighted RMSE over per-scorer (source − candidate)² errors.

    Skips dimensions where either source or candidate has no score and
    renormalizes by the sum of weights of present dimensions. This is
    the correct hybrid distance — it prevents per-axis errors from
    cancelling out the way a "blend then distance" formula does.

    Returns None when no dimension has scores for both source and
    candidate.
    """
    weighted_sq: List[float] = []
    total_weight = 0.0
    for scores, w in inputs:
        s = scores.get(source_id)
        c = scores.get(candidate_id)
        if s is None or c is None:
            continue
        weighted_sq.append(w * (s - c) ** 2)
        total_weight += w
    if not weighted_sq or total_weight <= 0:
        return None
    return math.sqrt(sum(weighted_sq) / total_weight)


def _rank_by_hybrid_distance(
    source_id: str,
    rec_ids: List[str],
    inputs: List[Tuple[Dict[str, Optional[float]], float]],
    blended: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    """Hybrid ranking: distance = weighted RMSE over per-scorer errors.

    ``blended`` is the per-item blended mood score, kept around purely
    for display so the row can show what the hybrid thinks of the item
    in isolation. It does NOT participate in the ranking.
    """
    rated: List[Dict[str, Any]] = []
    unrated: List[Dict[str, Any]] = []
    for rid in rec_ids:
        dist = _hybrid_rmse_distance(source_id, rid, inputs)
        score = blended.get(rid)
        if dist is None:
            unrated.append({"item_id": rid, "score": score, "distance": None})
        else:
            rated.append({"item_id": rid, "score": score, "distance": dist})
    rated.sort(key=lambda r: r["distance"])
    out = rated + unrated
    for i, row in enumerate(out):
        row["rank"] = i + 1
    return out


def _find_scorer(data: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    return next(
        (s for s in data["scorers"] if s["name"] == name and s["status"] == "ok"),
        None,
    )


def _append_hybrid(
    data: Dict[str, Any],
    source_id: str,
    rec_ids: List[str],
    label: str,
    inputs: List[Tuple[Dict[str, Optional[float]], float]],
) -> None:
    """Synthesize a hybrid scorer column from already-computed score dicts.

    Ranking uses weighted RMSE over per-scorer (source − cand)² errors.
    That penalises every axis's disagreement positively, so candidates
    where intensity says +0.4 and bpm says −0.4 can't average to "close".
    The blended mood score is kept around purely for display.
    """
    blended = blend_scores(inputs)
    ranked = _rank_by_hybrid_distance(source_id, rec_ids, inputs, blended)
    data["scorers"].append({
        "name": label,
        "status": "ok",
        "source_score": blended.get(source_id),
        "scores": blended,
        "ranked": ranked,
    })


def _maybe_add_hybrids(
    data: Dict[str, Any],
    source_id: str,
    rec_ids: List[str],
    hybrid_weight: float,
    hybrid_all_weights: Tuple[float, float, float],
) -> None:
    """Append two synthetic 'hybrid' scorer columns built from the real scorers.

    Both blends use already-computed score dicts — no extra audio decode.
    The 2-way blend (intensity + bpm) renders whenever both audio scorers
    ran ok. The 3-way blend (tags + intensity + bpm) needs all three.
    """
    mood_tags = _find_scorer(data, "mood_tags")
    intensity = _find_scorer(data, "intensity")
    bpm = _find_scorer(data, "bpm")

    # 2-way: intensity + bpm
    if intensity and bpm:
        _append_hybrid(
            data, source_id, rec_ids,
            label=f"hybrid 2 (i={hybrid_weight:.1f} / bpm={1 - hybrid_weight:.1f})",
            inputs=[
                (intensity["scores"], hybrid_weight),
                (bpm["scores"], 1.0 - hybrid_weight),
            ],
        )

    # 3-way: tags + intensity + bpm
    if mood_tags and intensity and bpm:
        w_t, w_i, w_b = hybrid_all_weights
        _append_hybrid(
            data, source_id, rec_ids,
            label=f"hybrid 3 (tags={w_t:.1f} / i={w_i:.1f} / bpm={w_b:.1f})",
            inputs=[
                (mood_tags["scores"], w_t),
                (intensity["scores"], w_i),
                (bpm["scores"], w_b),
            ],
        )


def run_similarity(
    user: str,
    source_url: Optional[str],
    pool_size: int,
    min_supporters: int,
    pool_source: str,
    hybrid_weight: float,
    hybrid_all_weights: Tuple[float, float, float],
) -> Dict[str, Any]:
    source = _build_source(source_url, user)
    print(f"[source] {source.get('band_name')} - {source.get('item_title')}  ({source['item_url']})")

    if pool_source == "wishlist":
        pool = _wishlist_as_pool(user, source)
        print(f"[pool] using wishlist as pool: {len(pool)} items")
    else:
        print(f"[pool] fetching up to {pool_size} supporter-overlap recs (min_supporters={min_supporters}) ...")
        pool = _fetch_pool(source, pool_size, min_supporters)
        print(f"[pool] hydrated {len(pool)} recs")
        if not pool:
            raise SystemExit(
                "recommender returned no candidates. Bandcamp may be rate-limiting; "
                "retry later or use --pool-source wishlist."
            )

    items_for_scoring = [source] + pool
    rec_ids = [str(r["item_id"]) for r in pool]
    source_id = str(source["item_id"])

    data: Dict[str, Any] = {
        "user": user,
        "source": source,
        "pool": pool,
        "scorers": [],
    }

    for scorer in SCORERS:
        scores = _run_scorer_subprocess(scorer, items_for_scoring, tag=scorer.name)
        if scores is None:
            data["scorers"].append({
                "name": scorer.name, "status": scorer.status, "error": scorer.error,
            })
            continue
        ranked = _rank_by_distance(source_id, scores, rec_ids)
        data["scorers"].append({
            "name": scorer.name,
            "status": "ok",
            "source_score": scores.get(source_id),
            "scores": scores,
            "ranked": ranked,
        })

    _maybe_add_hybrids(data, source_id, rec_ids, hybrid_weight, hybrid_all_weights)
    return data


# -----------------------------------------------------------------------------
# HTML rendering
# -----------------------------------------------------------------------------


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scorer similarity eval — {source_title}</title>
<style>
  :root {{
    --bg:#0f1115; --fg:#e8e8ea; --muted:#8a8d96; --card:#171a21;
    --chill:#5fa9ff; --party:#ff5f88; --warn:#ffb454; --ok:#7ee787;
  }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ margin:0 0 4px; font-size:18px; font-weight:600; }}
  .sub {{ color:var(--muted); margin-bottom:18px; font-size:13px; }}
  .source {{ background:var(--card); padding:14px 16px; border-radius:8px;
             margin-bottom:20px; display:flex; gap:12px; align-items:center; }}
  .source .body {{ flex:1; min-width:0; }}
  .source .label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
                    letter-spacing:.06em; margin-bottom:2px; }}
  .source .title {{ font-size:16px; font-weight:600;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .source .artist {{ color:var(--muted); font-size:13px; margin-top:1px; }}
  .source .tags {{ color:var(--muted); font-size:11px; margin-top:4px;
                   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .source .scores {{ display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }}
  .source .score-chip {{ font-size:11px; padding:2px 8px; border-radius:3px;
                         background:#222630; font-variant-numeric:tabular-nums; }}
  /* Auto-flowing grid: each column gets at least 440px. On a typical
     1440-1800px screen you see 3 columns, with the rest wrapping
     underneath. Beats cramming 5 narrow columns into one row. */
  .scorers {{ display:grid;
              grid-template-columns:repeat(auto-fill, minmax(440px, 1fr));
              gap:14px; align-items:start; }}
  .col {{ background:var(--card); border-radius:8px; padding:14px; min-width:0; }}
  .col h3 {{ margin:0 0 2px; font-size:15px; }}
  .col .meta {{ color:var(--muted); font-size:11px; margin-bottom:10px; }}
  .row {{ display:flex; gap:10px; align-items:center; padding:7px 0;
          border-top:1px solid #20242c; min-width:0; }}
  .row:first-child {{ border-top:none; }}
  .row .rank {{ font-variant-numeric:tabular-nums; color:var(--muted);
                font-size:12px; width:22px; text-align:right; flex:none; }}
  .row .body {{ flex:1; min-width:0; }}
  .row .title {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                 font-weight:500; font-size:14px; }}
  .row .artist {{ color:var(--muted); font-size:12px;
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .row .tags {{ color:var(--muted); font-size:11px; margin-top:1px;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .dist {{ font-variant-numeric:tabular-nums; font-size:12px;
           padding:2px 6px; border-radius:3px; background:#222630;
           flex:none; }}
  .dist.null {{ color:var(--muted); }}
  /* Audio player on its own line below title block at narrow widths;
     stays inline when there's room. */
  audio {{ height:24px; max-width:180px; flex:none; }}
  a {{ color:inherit; }} a:hover {{ text-decoration:underline; }}
  .badge {{ display:inline-block; padding:1px 6px; border-radius:3px; font-size:11px; }}
  .badge.ok {{ background:#1f3a26; color:var(--ok); }}
  .badge.warn {{ background:#3a2e1c; color:var(--warn); }}
  .badge.err {{ background:#3a1c1c; color:#ff7a7a; }}
  .empty {{ color:var(--muted); padding:18px; text-align:center;
            border:1px dashed #2a2e38; border-radius:6px; font-size:12px; }}
</style>
</head>
<body>
<h1>Similarity ranking — same pool, three scorers</h1>
<div class="sub">
  {n_recs} candidates fetched once via supporter overlap.
  Each scorer ranks the same pool by ``|score(source) − score(candidate)|``
  (lower = closer to source).
</div>

<div class="source">
  <div class="body">
    <div class="label">Source</div>
    <div class="title"><a href="{source_url}" target="_blank">{source_title}</a></div>
    <div class="artist">{source_artist}</div>
    {source_tags_html}
    <div class="scores">{source_scores_html}</div>
  </div>
  {source_audio_html}
</div>

<div class="scorers">
{columns}
</div>
</body>
</html>"""


def _esc(s: Any) -> str:
    return str(s or "").replace("<", "&lt;")


def _audio_html(url: Optional[str], compact: bool = False) -> str:
    if not url:
        return ""
    return f'<audio controls preload="none" src="{url}"></audio>'


def _row_html(rec: Dict[str, Any], row_data: Dict[str, Any]) -> str:
    title = _esc(rec.get("item_title") or "Unknown")
    artist = _esc(rec.get("band_name") or "Unknown")
    url = rec.get("item_url") or "#"
    audio = rec.get("audio_url")
    tags = rec.get("tags") or []
    tags_html = (
        f'<div class="tags">{_esc(", ".join(tags[:5]))}</div>' if tags else ""
    )
    dist = row_data.get("distance")
    if dist is None:
        dist_pill = '<span class="dist null">·</span>'
    else:
        dist_pill = f'<span class="dist">{dist:.2f}</span>'
    rank = row_data.get("rank", "")
    return (
        f'<div class="row">'
        f'<div class="rank">{rank}</div>'
        f'{dist_pill}'
        f'<div class="body">'
        f'<div class="title"><a href="{url}" target="_blank">{title}</a></div>'
        f'<div class="artist">{artist}</div>'
        f'{tags_html}'
        f'</div>'
        f'{_audio_html(audio, compact=True)}'
        f'</div>'
    )


def _scorer_column(scorer_data: Dict[str, Any], pool_by_id: Dict[str, Dict[str, Any]]) -> str:
    name = scorer_data["name"]
    status = scorer_data["status"]
    if status != "ok":
        badge_cls = "warn" if status == "not_ready" else "err"
        msg = _esc(scorer_data.get("error") or status)
        return (
            f'<div class="col"><h3>{name}</h3>'
            f'<div class="meta"><span class="badge {badge_cls}">{status}</span></div>'
            f'<div class="empty">{msg}</div></div>'
        )
    ranked = scorer_data["ranked"]
    src_score = scorer_data.get("source_score")
    src_pill = f"source={src_score:+.2f}" if src_score is not None else "source=·"
    rows = [
        _row_html(pool_by_id[row["item_id"]], row)
        for row in ranked
        if row["item_id"] in pool_by_id
    ]
    return (
        f'<div class="col"><h3>{name}</h3>'
        f'<div class="meta"><span class="badge ok">ok</span> · {src_pill}</div>'
        + "".join(rows)
        + '</div>'
    )


def render_html(data: Dict[str, Any]) -> str:
    source = data["source"]
    pool = data["pool"]
    pool_by_id = {str(r["item_id"]): r for r in pool}
    columns = "\n".join(_scorer_column(s, pool_by_id) for s in data["scorers"])
    source_tags_html = (
        f'<div class="tags">{_esc(", ".join(source.get("tags", [])[:8]))}</div>'
        if source.get("tags") else ""
    )
    def _src_chip(s: Dict[str, Any]) -> str:
        v = s.get("source_score") if s.get("status") == "ok" else None
        val = f"{v:+.2f}" if v is not None else "·"
        return f'<span class="score-chip">{_esc(s["name"])}: {val}</span>'
    source_scores_html = "".join(_src_chip(s) for s in data["scorers"])
    return HTML_TEMPLATE.format(
        n_cols=len(data["scorers"]),
        n_recs=len(pool),
        source_url=source.get("item_url", "#"),
        source_title=_esc(source.get("item_title") or "Unknown"),
        source_artist=_esc(source.get("band_name") or "Unknown"),
        source_tags_html=source_tags_html,
        source_audio_html=_audio_html(source.get("audio_url")),
        source_scores_html=source_scores_html,
        columns=columns,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="lucrw")
    p.add_argument("--source", default=None,
                   help="Bandcamp item URL to use as the source (default: first wishlist item)")
    p.add_argument("--pool", type=int, default=25,
                   help="size of the supporter-overlap candidate pool (default 25)")
    p.add_argument("--min-supporters", type=int, default=1)
    p.add_argument("--pool-source", choices=("recs", "wishlist"), default="wishlist",
                   help="where the candidate pool comes from. "
                        "'wishlist' (default) re-uses the cached, hydrated wishlist "
                        "and is free + reliable. 'recs' fetches supporter-overlap "
                        "recommendations from Bandcamp (rate-limit prone).")
    p.add_argument("--hybrid-weight", type=float, default=0.5,
                   help="weight of intensity in the 2-way hybrid (default 0.5). "
                        "BPM gets the remaining 1-w. Column renders when both "
                        "intensity and bpm scorers succeed.")
    p.add_argument("--hybrid-all-weights", type=str, default="0.2,0.4,0.4",
                   help="comma-separated weights for the 3-way hybrid "
                        "(tags,intensity,bpm). Default '0.2,0.4,0.4' — tags "
                        "downweighted because of its coarse score distribution. "
                        "Weights are renormalized over signals that exist for "
                        "each item, so a missing signal degrades gracefully.")
    args = p.parse_args()

    try:
        hybrid_all = tuple(float(x) for x in args.hybrid_all_weights.split(","))
        assert len(hybrid_all) == 3
    except (ValueError, AssertionError):
        p.error("--hybrid-all-weights must be three comma-separated floats")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = run_similarity(
        user=args.user,
        source_url=args.source,
        pool_size=args.pool,
        min_supporters=args.min_supporters,
        pool_source=args.pool_source,
        hybrid_weight=args.hybrid_weight,
        hybrid_all_weights=hybrid_all,
    )

    data_path = OUTPUT_DIR / "similarity_data.json"
    data_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"[eval] wrote {data_path}")

    html_path = OUTPUT_DIR / "similarity.html"
    html_path.write_text(render_html(data))
    print(f"[eval] wrote {html_path}")
    print(f"[eval] open: file://{html_path}")


if __name__ == "__main__":
    main()
