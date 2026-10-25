"""Artist identity: when two Bandcamp artist credits are the same act.

Bandcamp's ``band_name`` is free text. The same act shows up solo
("Siu Mata"), in collaborations ("Siu Mata & Amor Satyr", "Amor Satyr x Siu
Mata") and with different capitalisation, and compilations are credited to
generic labels ("Various Artists", "V/A"). Comparing raw strings misses the
first two and lumps every compilation together; compare token sets instead:
two credits are the same artist when their ``artist_tokens`` intersect.
Placeholder credits ("Various Artists", "Unknown Artist") are not an artist:
given the release URL they are identified by the release page's host, so one
label's placeholder releases count as one act and other labels' stay apart.

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
from urllib.parse import urlparse

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
# Placeholder credits name no artist: "Various Artists" on a compilation,
# "Unknown Artist" on an unattributed release. Compared lowercased with every
# punctuation run turned into one space (_is_placeholder), so "V.A", "VA.",
# "V/A" and "Various Artists." match too.
_PLACEHOLDERS = {
    "various artists", "various artist", "various", "va", "v a", "vv aa", "aa vv",
    "varios artistas", "multiple artists",
    "unknown", "unknown artist", "unknown artists", "anonymous", "n a",
}
# A credit made only of these (e.g. "-", "?", "...") is a placeholder too.
# Other symbol-only names can be real acts ("!!!" is a band).
_PLACEHOLDER_ONLY = re.compile(r"[\s\-\u2013\u2014_.,/\\|?*~]*")
# "V/A" inside a longer credit ("Various Artists / V/A", "V.A. / X") is read as
# one placeholder before the split, which would otherwise make it "v" and "a".
_VA_ABBREV = re.compile(r"(?<!\w)v\s*[/.]\s*a\b\.?", re.IGNORECASE)


def _is_placeholder(credit: str) -> bool:
    if _PLACEHOLDER_ONLY.fullmatch(credit):
        return True
    return re.sub(r"[\W_]+", " ", credit.lower()).strip() in _PLACEHOLDERS


def _page_identity(item_url: str) -> Set[str]:
    try:
        host = (urlparse(item_url).hostname or "").lower()
    except ValueError:  # a malformed netloc, e.g. "https://[bad/x"
        return set()
    if host.startswith("www."):
        host = host[4:]
    # bandcamp.com itself hosts no releases: it is the URL the library
    # synthesizes (bandcamp.com/album/<id>) for an item it never saw a URL
    # for, which must not merge every such item into one act.
    if not host or host == "bandcamp.com":
        return set()
    return {"@" + host}


def artist_tokens(artist: Optional[str], item_url: Optional[str] = None) -> Set[str]:
    """The individual artists in a credit, lowercased.

    ``"Siu Mata"``, ``"Siu Mata & Amor Satyr"`` and ``"Amor Satyr & Siu
    Mata"`` all contain ``"siu mata"``, so artist-level dedup catches the same
    act across releases. See the module docstring for the accepted
    over-splitting.

    Placeholder credits ("Various Artists", "Unknown Artist", "V/A", "-", "")
    name no artist. Without ``item_url`` they yield an empty set (nothing to
    compare). With it they yield the release's Bandcamp page host instead,
    e.g. ``{"@kcik.bandcamp.com"}``: one label or page releasing everything
    as "Unknown Artist" is one act for dedup (measured: three such releases
    aired back to back), while compilations from different pages stay
    distinct. The ``@`` prefix cannot collide with a real credit. The host is
    taken literally (minus a leading ``www.``): a label's custom domain and
    its ``*.bandcamp.com`` twin are two identities.
    """
    tokens = _credit_tokens(artist)
    if not tokens and item_url:
        return _page_identity(item_url)
    return tokens


def _credit_tokens(artist: Optional[str]) -> Set[str]:
    if not artist:
        return set()
    # Whole-string placeholder check first, before splitting: otherwise "V/A"
    # would split on "/" into "v" and "a".
    if _is_placeholder(artist):
        return set()
    tokens = set()
    credit = _PAREN_FEAT.sub(r" \1 ", _VA_ABBREV.sub("various artists", artist))
    for part in _ARTIST_SPLIT.split(credit):
        p = part.strip().lower()
        if p and not _is_placeholder(p):
            tokens.add(p)
    return tokens
