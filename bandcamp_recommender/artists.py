"""Artist identity: when two Bandcamp artist credits are the same act.

Bandcamp's ``band_name`` is free text. The same act shows up solo
("Siu Mata"), in collaborations ("Siu Mata & Amor Satyr", "Amor Satyr x Siu
Mata") and with different capitalisation, and compilations are credited to
generic labels ("Various Artists", "V/A"). Comparing raw strings misses the
first two and lumps every compilation together; compare token sets instead:
two credits are the same artist when their ``artist_tokens`` intersect.
"""

from __future__ import annotations

import re
from typing import Optional, Set

# Collaboration separators: "Siu Mata & Amor Satyr", "A, B", "A / B", "A x B",
# "A vs B", "A feat. B", "A ft B", "A + B", "A | B" all split into individual
# artists. The word connectors (x/vs/feat/ft) eat an optional trailing dot
# ("feat."). Deliberately NOT splitting on "and"/"with": too many band names
# contain them.
_ARTIST_SPLIT = re.compile(
    r"\s*(?:&|,|/|\||\+|\b(?:x|vs|feat|ft)\b\.?)\s*", re.IGNORECASE
)
# Generic compilation credits: dropped so every V/A comp doesn't collapse into
# one artist and block all compilations.
_ARTIST_STOPWORDS = {"various artists", "various", "v/a", "va", "unknown", "unknown artist"}


def artist_tokens(artist: Optional[str]) -> Set[str]:
    """The individual artists in a credit, lowercased.

    ``"Siu Mata"``, ``"Siu Mata & Amor Satyr"`` and ``"Amor Satyr & Siu
    Mata"`` all contain ``"siu mata"``, so artist-level dedup catches the same
    act across releases. Generic compilation credits yield an empty set (no
    artist to compare), as does an empty credit.
    """
    if not artist:
        return set()
    # Whole-string stopword check first, before splitting: otherwise "V/A"
    # would split on "/" into "v" and "a" and slip past the stopword list.
    if artist.strip().lower() in _ARTIST_STOPWORDS:
        return set()
    tokens = set()
    for part in _ARTIST_SPLIT.split(artist):
        p = part.strip().lower()
        if p and p not in _ARTIST_STOPWORDS:
            tokens.add(p)
    return tokens
