"""Eval adapter for the audio-intensity scorer.

Implements the shared eval contract:
- ``score_items(items)`` returns ``{item_id: mood_score}`` in ``[-1.0, 1.0]``
  derived from the 0..1 audio intensity score via ``mood = 2*x - 1``
  (so 0.0 intensity → -1 chill, 1.0 intensity → +1 party). ``None`` when
  audio can't be fetched/decoded or the item has no ``audio_url``.
- ``__main__`` CLI reads an items.json (list of dicts) and writes a
  scores.json (mapping). Items need ``item_id`` + ``audio_url``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from bandcamp_recommender.recommendations.intensity import score_intensity


def _score_one(item: Dict) -> tuple[str, Optional[float]]:
    item_id = str(item.get("item_id") or item.get("id") or "")
    audio_url = item.get("audio_url")
    if not item_id or not audio_url:
        return item_id, None
    raw = score_intensity(audio_url)
    if raw is None:
        return item_id, None
    return item_id, 2.0 * raw - 1.0


def score_items(items: List[Dict], max_workers: int = 2) -> Dict[str, Optional[float]]:
    """Lightly-parallel intensity scoring (default workers=2, matches
    ``score_bpm`` since both paths share the librosa decode lock).
    Higher worker counts only overlap downloads — the actual decode
    is serialized inside :mod:`bpm._decode_audio_with_librosa`."""
    out: Dict[str, Optional[float]] = {}
    targets = [i for i in items if i.get("item_id")]
    if not targets:
        return out
    workers = max(1, min(max_workers, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_score_one, item) for item in targets]
        for fut in as_completed(futures):
            iid, score = fut.result()
            if iid:
                out[iid] = score
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    items = json.loads(args.inp.read_text())
    scores = score_items(items, max_workers=args.workers)
    args.out.write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
