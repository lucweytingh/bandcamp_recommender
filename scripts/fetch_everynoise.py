#!/usr/bin/env python3
"""One-shot fetcher: snapshot everynoise.com genre coordinates to CSV.

Run this manually to refresh the vendored mood prior. The output CSV is
loaded at import time by ``bandcamp_recommender.recommendations.mood_tags``.

Usage:
    python scripts/fetch_everynoise.py

The site (Glenn McDonald's everynoise.com) has no public API; the page is
~3MB of inline-styled divs. We pull it once, parse the ``top``/``left``
pixel coordinates and the genre label, and write a small CSV. Please run
this sparingly — it's a single GET, but be polite.
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.request
from pathlib import Path


URL = "https://everynoise.com/engenremap.html"
USER_AGENT = "Mozilla/5.0 (bandcamp-recommender mood prior; one-time vendor snapshot)"

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "bandcamp_recommender"
    / "recommendations"
    / "data"
    / "everynoise_genres.csv"
)

# Match: `top: NNNpx; left: NNNpx; font-size: NNN% ...">NAME<a class=navlink`.
# The font-size is everynoise's visual proxy for genre popularity
# (bigger = more popular). It's perceptually log-scaled.
GENRE_PATTERN = re.compile(
    r'top:\s*(\d+)px;\s*left:\s*(\d+)px;\s*font-size:\s*(\d+)%[^>]*"[^>]*>'
    r'([^<]+)<a class=navlink',
    re.IGNORECASE,
)


def fetch(url: str = URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse(html: str) -> list[tuple[str, int, int, int]]:
    rows: dict[str, tuple[int, int, int]] = {}
    for top_s, left_s, font_s, raw_name in GENRE_PATTERN.findall(html):
        name = raw_name.strip()
        if not name:
            continue
        # If a genre appears twice (shouldn't, but be defensive), keep the
        # first occurrence.
        rows.setdefault(name, (int(top_s), int(left_s), int(font_s)))
    return [(n, t, l, f) for n, (t, l, f) in rows.items()]


def write_csv(rows: list[tuple[str, int, int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["genre", "top", "left", "font_pct"])
        w.writerows(rows_sorted)


def main() -> int:
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch()
    rows = parse(html)
    if len(rows) < 1000:
        print(f"Refusing to write: parsed only {len(rows)} genres", file=sys.stderr)
        return 1
    write_csv(rows, DATA_PATH)
    tops = [t for _, t, _, _ in rows]
    fonts = [f for _, _, _, f in rows]
    print(
        f"Wrote {len(rows)} genres to {DATA_PATH.relative_to(Path.cwd())} "
        f"(top {min(tops)}..{max(tops)}, font {min(fonts)}%..{max(fonts)}%)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
