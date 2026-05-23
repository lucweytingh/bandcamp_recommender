"""Vector-similarity eval: one source, one pool, one ranking, full feature breakdown.

Replaces the old per-scorer columns / hybrid blending approach with a
single feature-vector model. Each track is represented as a dict of
normalized scalar features (see :mod:`bandcamp_recommender.features`).
Similarity = weighted Euclidean over the intersection of features both
tracks have. No more blend-then-distance error cancellation, no more
hybrid 2 vs hybrid 3 confusion — there's just one distance.

The HTML page renders:

* the source song with its full feature vector
* the candidate pool ranked by closeness to the source
* per-row feature deltas so you can see *why* a track is close (or far)

Reuses the wishlist scrape / hydration / audio-URL discovery from the
old ``eval_mood.py`` since that part is still useful.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from eval_mood import (  # type: ignore
    CONTEXT_DIR,
    OUTPUT_DIR,
    _fetch_recs_for_seed,
    _hydrate_item,
    scrape_wishlist,
)

from bandcamp_recommender.features import (
    DEFAULT_WEIGHTS,
    FEATURE_RANGES,
    distance,
    extract_features,
    project_mood,
)


# -----------------------------------------------------------------------------
# Source + pool
# -----------------------------------------------------------------------------


def _build_source(source_url: Optional[str], user: str) -> Dict[str, Any]:
    if source_url:
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


def _wishlist_pool(user: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use the (cached + hydrated) wishlist minus the source as the pool."""
    items = scrape_wishlist(user, CONTEXT_DIR / f"wishlist_{user}.json", refresh=False)
    source_id = str(source.get("item_id") or "")
    source_url = (source.get("item_url") or "").rstrip("/")
    return [
        it for it in items
        if str(it["item_id"]) != source_id
        and (it.get("item_url") or "").rstrip("/") != source_url
    ]


def _supporter_pool(
    source: Dict[str, Any], pool_size: int, min_supporters: int
) -> List[Dict[str, Any]]:
    recs = _fetch_recs_for_seed(
        source["item_url"], max_recs=pool_size, min_supporters=min_supporters
    )
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rec in recs:
        rid = str(rec.get("item_id") or rec.get("item_url") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        rec["item_id"] = rid
        if not rec.get("audio_url"):
            _hydrate_item(rec)
        out.append(rec)
    return out


# -----------------------------------------------------------------------------
# Feature extraction across the source + pool
# -----------------------------------------------------------------------------


def _extract_one(item: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[float]]]:
    iid = str(item.get("item_id") or "")
    return iid, extract_features(item)


def _extract_pool_features(
    items: List[Dict[str, Any]], max_workers: int = 4
) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract feature vectors for every item, parallel where it helps.

    librosa isn't fully thread-safe (the bpm scorer crashed at workers=6
    on a 77-item wishlist). 4 workers is the sweet spot — fast enough
    that 60s audio decode isn't the wall clock, low enough to avoid the
    SIGBUS we used to hit.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not items:
        return out
    workers = max(1, min(max_workers, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_extract_one, item) for item in items]
        for fut in as_completed(futures):
            iid, features = fut.result()
            if iid:
                out[iid] = features
    return out


# -----------------------------------------------------------------------------
# Distance ranking
# -----------------------------------------------------------------------------


def _per_feature_delta(
    source_vec: Dict[str, Optional[float]],
    cand_vec: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """Signed (cand - source) delta per feature, or None when one side missing."""
    deltas: Dict[str, Optional[float]] = {}
    for key in DEFAULT_WEIGHTS:
        s = source_vec.get(key)
        c = cand_vec.get(key)
        if s is None or c is None:
            deltas[key] = None
        else:
            deltas[key] = c - s
    return deltas


def _rank(
    source_id: str,
    source_vec: Dict[str, Optional[float]],
    pool_vecs: Dict[str, Dict[str, Optional[float]]],
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    rated: List[Dict[str, Any]] = []
    unrated: List[Dict[str, Any]] = []
    for iid, vec in pool_vecs.items():
        dist = distance(source_vec, vec, weights)
        row = {
            "item_id": iid,
            "features": vec,
            "deltas": _per_feature_delta(source_vec, vec),
            "distance": dist,
        }
        (rated if dist is not None else unrated).append(row)
    rated.sort(key=lambda r: r["distance"])
    out = rated + unrated
    for i, row in enumerate(out):
        row["rank"] = i + 1
    return out


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_similarity(
    user: str,
    source_url: Optional[str],
    pool_size: int,
    min_supporters: int,
    pool_source: str,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    source = _build_source(source_url, user)
    print(f"[source] {source.get('band_name')} - {source.get('item_title')}  ({source['item_url']})")

    if pool_source == "wishlist":
        pool = _wishlist_pool(user, source)
        print(f"[pool] using wishlist as pool: {len(pool)} items")
    else:
        print(f"[pool] fetching up to {pool_size} supporter-overlap recs (min_supporters={min_supporters}) ...")
        pool = _supporter_pool(source, pool_size, min_supporters)
        print(f"[pool] hydrated {len(pool)} recs")
        if not pool:
            raise SystemExit(
                "recommender returned no candidates. Bandcamp may be rate-limiting; "
                "retry later or use --pool-source wishlist."
            )

    print(f"[features] extracting source vector ...")
    t0 = time.time()
    _, source_vec = _extract_one(source)
    print(f"[features] source done in {time.time() - t0:.1f}s")

    print(f"[features] extracting pool ({len(pool)} items) ...")
    t0 = time.time()
    pool_vecs = _extract_pool_features(pool)
    print(f"[features] pool done in {time.time() - t0:.1f}s")

    effective_weights = weights or DEFAULT_WEIGHTS
    ranked = _rank(str(source["item_id"]), source_vec, pool_vecs, effective_weights)

    return {
        "user": user,
        "source": source,
        "source_features": source_vec,
        "source_mood_projection": project_mood(source_vec),
        "pool": pool,
        "ranked": ranked,
        "weights": effective_weights,
    }


# -----------------------------------------------------------------------------
# HTML rendering
# -----------------------------------------------------------------------------


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Similarity — {source_title}</title>
<style>
  :root {{
    --bg:#0f1115; --fg:#e8e8ea; --muted:#8a8d96; --card:#171a21;
    --chill:#5fa9ff; --party:#ff5f88; --pos:#7ee787; --neg:#ff7a7a;
  }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ margin:0 0 6px; font-size:18px; font-weight:600; }}
  .sub {{ color:var(--muted); margin-bottom:16px; font-size:13px; }}

  .source {{ background:var(--card); padding:14px 16px; border-radius:8px;
             margin-bottom:18px; display:flex; gap:12px; align-items:flex-start; }}
  .source .body {{ flex:1; min-width:0; }}
  .source .label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
                    letter-spacing:.06em; margin-bottom:2px; }}
  .source .title {{ font-size:16px; font-weight:600;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .source .artist {{ color:var(--muted); font-size:13px; margin-top:1px; }}
  .source .tags {{ color:var(--muted); font-size:11px; margin-top:4px;
                   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

  .feature-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr));
                   gap:4px 12px; margin-top:8px; font-size:11px; }}
  .feature-grid .feat {{ display:flex; justify-content:space-between; gap:6px; }}
  .feature-grid .feat .k {{ color:var(--muted); }}
  .feature-grid .feat .v {{ font-variant-numeric:tabular-nums; }}
  .feature-grid .feat .v.null {{ color:var(--muted); }}

  audio {{ height:26px; max-width:200px; flex:none; }}

  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th, td {{ padding:8px 6px; text-align:left; vertical-align:top;
            border-top:1px solid #20242c; }}
  th {{ position:sticky; top:0; background:var(--bg);
        font-size:11px; font-weight:600; color:var(--muted);
        text-transform:uppercase; letter-spacing:.04em; border-top:none; }}
  td.rank {{ width:32px; text-align:right; color:var(--muted);
             font-variant-numeric:tabular-nums; }}
  td.dist {{ width:64px; font-variant-numeric:tabular-nums;
             padding:8px 8px; }}
  td.title {{ min-width:180px; max-width:280px; }}
  td.title .t {{ font-weight:500; white-space:nowrap; overflow:hidden;
                 text-overflow:ellipsis; }}
  td.title .a {{ color:var(--muted); font-size:12px; white-space:nowrap;
                 overflow:hidden; text-overflow:ellipsis; }}
  td.title .tg {{ color:var(--muted); font-size:11px; margin-top:1px;
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  td.audio {{ width:210px; }}
  td.deltas {{ font-size:11px; }}
  td.deltas .row {{ display:inline-block; padding:1px 6px; margin:1px 3px 1px 0;
                    border-radius:3px; background:#222630;
                    font-variant-numeric:tabular-nums; }}
  td.deltas .row .k {{ color:var(--muted); margin-right:4px; }}
  td.deltas .row .v.pos {{ color:var(--pos); }}
  td.deltas .row .v.neg {{ color:var(--neg); }}
  td.deltas .row .v.null {{ color:var(--muted); }}
  a {{ color:inherit; }} a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<h1>Similarity ranking — vector model</h1>
<div class="sub">
  Pool: <b>{n_pool} items</b> · distance = weighted Euclidean over the
  intersection of features both tracks have ·
  <a href="#weights">weights</a> tunable per-feature.
</div>

<div class="source">
  <div class="body">
    <div class="label">Source</div>
    <div class="title"><a href="{source_url}" target="_blank">{source_title}</a></div>
    <div class="artist">{source_artist}</div>
    {source_tags_html}
    <div class="label" style="margin-top:8px">Source features
        · projected mood: <b>{source_mood_str}</b></div>
    <div class="feature-grid">{source_feature_grid}</div>
  </div>
  {source_audio_html}
</div>

<table>
  <thead><tr>
    <th class="rank">#</th>
    <th class="dist">distance</th>
    <th class="title">track</th>
    <th class="audio">preview</th>
    <th class="deltas">per-feature Δ (candidate − source)</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>

<h3 id="weights" style="margin-top:32px;font-size:13px;">Active feature weights</h3>
<div class="feature-grid" style="max-width:600px">{weights_grid}</div>

</body>
</html>"""


def _esc(s: Any) -> str:
    """HTML-escape a value for use in text content or an attribute.

    Covers ``&`` first (so it doesn't double-escape the entities we add
    next), then ``<``/``>`` for text content, and ``"``/``'`` so the
    same helper is safe inside attribute values like ``href="..."``.
    """
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _audio_html(url: Optional[str]) -> str:
    if not url:
        return ""
    return f'<audio controls preload="none" src="{_esc(url)}"></audio>'


def _format_value(key: str, value: Optional[float]) -> str:
    if value is None:
        return '<span class="v null">·</span>'
    return f'<span class="v">{value:+.2f}</span>'


def _feature_grid_html(features: Dict[str, Optional[float]]) -> str:
    parts = []
    for key in DEFAULT_WEIGHTS:
        v = features.get(key)
        parts.append(
            f'<div class="feat"><span class="k">{key}</span>{_format_value(key, v)}</div>'
        )
    return "".join(parts)


def _delta_chip(key: str, delta: Optional[float]) -> str:
    if delta is None:
        return (
            f'<span class="row"><span class="k">{key}</span>'
            f'<span class="v null">·</span></span>'
        )
    cls = "pos" if delta > 0 else ("neg" if delta < 0 else "")
    return (
        f'<span class="row"><span class="k">{key}</span>'
        f'<span class="v {cls}">{delta:+.2f}</span></span>'
    )


def _row_html(rank: int, ranked_row: Dict[str, Any], pool_by_id: Dict[str, Dict[str, Any]]) -> str:
    item = pool_by_id.get(ranked_row["item_id"], {})
    title = _esc(item.get("item_title") or "Unknown")
    artist = _esc(item.get("band_name") or "Unknown")
    url = _esc(item.get("item_url") or "#")
    audio = _esc(item.get("audio_url")) if item.get("audio_url") else None
    tags = item.get("tags") or []
    tags_html = f'<div class="tg">{_esc(", ".join(tags[:6]))}</div>' if tags else ""
    dist = ranked_row.get("distance")
    dist_str = f'{dist:.3f}' if dist is not None else '—'
    deltas_html = "".join(
        _delta_chip(key, ranked_row["deltas"].get(key))
        for key in DEFAULT_WEIGHTS
    )
    return (
        f'<tr>'
        f'<td class="rank">{rank}</td>'
        f'<td class="dist">{dist_str}</td>'
        f'<td class="title">'
        f'<div class="t"><a href="{url}" target="_blank">{title}</a></div>'
        f'<div class="a">{artist}</div>'
        f'{tags_html}'
        f'</td>'
        f'<td class="audio">{_audio_html(audio)}</td>'
        f'<td class="deltas">{deltas_html}</td>'
        f'</tr>'
    )


def render_html(data: Dict[str, Any]) -> str:
    source = data["source"]
    pool_by_id = {str(it["item_id"]): it for it in data["pool"]}
    rows = "\n".join(
        _row_html(row["rank"], row, pool_by_id) for row in data["ranked"]
    )
    src_features = data["source_features"]
    weights = data["weights"]
    weights_grid = "".join(
        f'<div class="feat"><span class="k">{k}</span>'
        f'<span class="v">{w:.2f}</span></div>'
        for k, w in weights.items()
    )
    src_tags = source.get("tags") or []
    src_tags_html = (
        f'<div class="tags">{_esc(", ".join(src_tags[:10]))}</div>' if src_tags else ""
    )
    src_mood = data.get("source_mood_projection")
    src_mood_str = f"{src_mood:+.2f}" if src_mood is not None else "·"
    return HTML_TEMPLATE.format(
        n_pool=len(data["pool"]),
        source_url=_esc(source.get("item_url") or "#"),
        source_title=_esc(source.get("item_title") or "Unknown"),
        source_artist=_esc(source.get("band_name") or "Unknown"),
        source_tags_html=src_tags_html,
        source_audio_html=_audio_html(source.get("audio_url")),
        source_feature_grid=_feature_grid_html(src_features),
        source_mood_str=src_mood_str,
        rows=rows,
        weights_grid=weights_grid,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_weight_overrides(s: Optional[str]) -> Optional[Dict[str, float]]:
    if not s:
        return None
    overrides: Dict[str, float] = {}
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(f"--weights entry must be key=value, got {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_WEIGHTS:
            raise SystemExit(
                f"--weights key {key!r} not in feature set; valid keys: "
                f"{list(DEFAULT_WEIGHTS)}"
            )
        try:
            overrides[key] = float(value.strip())
        except ValueError as exc:
            raise SystemExit(f"--weights value for {key} not a float: {exc}") from exc
    if not overrides:
        return None
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(overrides)
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="lucrw")
    p.add_argument("--source", default=None,
                   help="Bandcamp item URL (default: first wishlist item)")
    p.add_argument("--pool", type=int, default=25,
                   help="supporter-overlap pool size (only used with --pool-source recs)")
    p.add_argument("--min-supporters", type=int, default=1)
    p.add_argument("--pool-source", choices=("recs", "wishlist"), default="wishlist")
    p.add_argument(
        "--weights",
        default=None,
        help="comma-separated key=value overrides on top of DEFAULT_WEIGHTS, "
             "e.g. --weights tag_mood=2.0,bpm_norm=0.0",
    )
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = _parse_weight_overrides(args.weights)
    data = run_similarity(
        user=args.user,
        source_url=args.source,
        pool_size=args.pool,
        min_supporters=args.min_supporters,
        pool_source=args.pool_source,
        weights=weights,
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
