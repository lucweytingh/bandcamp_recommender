"""Eval adapter for the BPM scorer.

Implements the shared eval contract:
- ``score_items(items)`` returns ``{item_id: mood_score}`` in ``[-1.0, 1.0]``,
  mapped from detected BPM via ``mood = clip((bpm - 110) / 50, -1, 1)``.
  So 60 BPM → -1 (chill), 110 BPM → 0, 160 BPM → +1 (party). Returns
  ``None`` when the audio can't be fetched/decoded or has no ``audio_url``.
- ``__main__`` CLI reads an items.json (list of dicts) and writes a
  scores.json (mapping). Items need ``item_id`` + ``audio_url``.

The mapping deliberately ignores octave-tolerance — the goal here is a
chill/party axis where 80 BPM downtempo should *look* low and 160 BPM
DnB should look high. The re-rank flag (``--bpm-match`` in the
recommender) is the right place for half/double-time tolerance.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from bandcamp_recommender.recommendations.bpm import detect_bpm


_BPM_CHILL = 60.0   # → -1
_BPM_PIVOT = 110.0  # →  0
_BPM_PARTY = 160.0  # → +1


def _bpm_to_mood(bpm: float) -> float:
    raw = (bpm - _BPM_PIVOT) / (_BPM_PARTY - _BPM_PIVOT)
    return max(-1.0, min(1.0, raw))


def _score_one(item: Dict) -> tuple[str, Optional[float]]:
    item_id = str(item.get("item_id") or item.get("id") or "")
    audio_url = item.get("audio_url")
    if not item_id or not audio_url:
        return item_id, None
    result = detect_bpm(audio_url, method="auto")
    if not result or not result.get("bpm"):
        return item_id, None
    return item_id, _bpm_to_mood(float(result["bpm"]))


def score_items(items: List[Dict], max_workers: int = 1) -> Dict[str, Optional[float]]:
    """Serial BPM scoring. librosa / audioread aren't thread-safe on
    macOS — even workers=2 sometimes crashes with SIGBUS. workers=1 is
    slower but reliable. Most decode time is spent inside Joe Sullivan
    (NumPy, GIL-released) so the parallelism win is small anyway."""
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
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    items = json.loads(args.inp.read_text())
    scores = score_items(items, max_workers=args.workers)
    args.out.write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
