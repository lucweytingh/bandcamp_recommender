"""Artist identity: when two Bandcamp artist credits are the same act.

Bandcamp's ``band_name`` is free text. The same act shows up solo
("Siu Mata"), in collaborations ("Siu Mata & Amor Satyr", "Amor Satyr x Siu
Mata") and with different capitalisation, and compilations are credited to
generic labels ("Various Artists", "V/A"). Comparing raw strings misses the
first two and lumps every compilation together; compare token sets instead:
two credits are the same artist when their ``artist_tokens`` intersect.

Trade-off, deliberately: a false merge only skips a candidate, while a missed
collab airs the same act twice, so the rules lean to splitting. A separator
inside a band's own name splits it too ("Earth, Wind & Fire" is {earth, wind,
fire}, "AC/DC" is {ac, dc}, "Hall & Oates" is {hall, oates}), so it matches
releases by "Earth" or "DC". Not split: "and" / "with" (too many band names
contain them) and parentheses other than a feature credit ("Phase (UK)" is not
"UK", but "A (feat. B)" is A and B).
"""

from __future__ import annotations

import re
from typing import Optional, Set

# Collaboration separators: "Siu Mata & Amor Satyr", "A, B", "A / B", "A + B",
# "A | B", "A × B", "A x B", "A vs B", "A feat. B", "A featuring B", "A ft B"
# all split into individual artists. The word connectors only count between
# spaces (so "Malcolm X" and "X-Ray Spex" stay whole) and eat an optional
# trailing dot ("feat.").
_ARTIST_SPLIT = re.compile(
    r"\s*[&,/|+×]\s*|\s+(?:x|vs|featuring|feat|ft)\.?\s+", re.IGNORECASE
)
# "A (feat. B)" / "A (ft. B, C)" is unwrapped to "A feat. B" before the split.
_PAREN_FEAT = re.compile(r"\s*\(\s*((?:featuring|feat|ft)\.?\s+[^()]*?)\s*\)", re.IGNORECASE)
# Generic compilation credits: dropped so every V/A comp doesn't collapse into
# one artist and block all compilations.
_ARTIST_STOPWORDS = {
    "various artists", "various artist", "various", "v/a", "va", "v.a.",
    "unknown", "unknown artist",
}


def artist_tokens(artist: Optional[str]) -> Set[str]:
    """The individual artists in a credit, lowercased.

    ``"Siu Mata"``, ``"Siu Mata & Amor Satyr"`` and ``"Amor Satyr & Siu
    Mata"`` all contain ``"siu mata"``, so artist-level dedup catches the same
    act across releases. Generic compilation credits yield an empty set (no
    artist to compare), as does an empty credit. See the module docstring for
    the accepted over-splitting.
    """
    if not artist:
        return set()
    # Whole-string stopword check first, before splitting: otherwise "V/A"
    # would split on "/" into "v" and "a" and slip past the stopword list.
    if artist.strip().lower() in _ARTIST_STOPWORDS:
        return set()
    tokens = set()
    for part in _ARTIST_SPLIT.split(_PAREN_FEAT.sub(r" \1 ", artist)):
        p = part.strip().lower()
        if p and p not in _ARTIST_STOPWORDS:
            tokens.add(p)
    return tokens
