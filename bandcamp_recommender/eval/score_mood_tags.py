"""Eval adapter for the tag-mood scorer.

Implements the shared eval contract:
- ``score_items(items)`` returns ``{item_id: mood_score}`` in ``[-1.0, 1.0]``
  where ``-1`` is fully chill and ``+1`` is fully party. ``None`` when no
  relevant tags overlap the lexicon.
- ``__main__`` CLI reads an items.json (list of dicts) and writes a
  scores.json (mapping). Used by the cross-worktree harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from bandcamp_recommender.recommendations.mood_tags import tag_mood_score


def score_items(items: List[Dict]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for item in items:
        item_id = str(item.get("item_id") or item.get("id") or "")
        if not item_id:
            continue
        out[item_id] = tag_mood_score(item.get("tags") or [])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", required=True, type=Path)
    args = p.parse_args()

    items = json.loads(args.inp.read_text())
    scores = score_items(items)
    args.out.write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
