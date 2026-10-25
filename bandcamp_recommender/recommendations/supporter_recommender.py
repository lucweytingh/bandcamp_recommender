"""Main recommendation engine for Bandcamp based on supporter purchases."""

import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def _debug_exc_info() -> bool:
    return os.environ.get("BANDCAMP_DEBUG") == "1"


_DEFAULT_DRIVER_POOL = 5
_DEFAULT_TAG_WORKERS = 6
_DEFAULT_SUPPORTER_CONCURRENCY = 6


def _resolve_supporter_concurrency() -> int:
    """Resolve the per-call ThreadPoolExecutor cap for supporter fetches.

    Default 6. Downstream observation: even 5 was too aggressive when
    Bandcamp custom domains (e.g. artist-hosted *.com pages) start
    throttling per-IP. Env-overridable via ``BANDCAMP_SUPPORTER_CONCURRENCY``.
    """
    raw = os.environ.get("BANDCAMP_SUPPORTER_CONCURRENCY")
    if raw is None:
        return _DEFAULT_SUPPORTER_CONCURRENCY
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return _DEFAULT_SUPPORTER_CONCURRENCY


def _resolve_pool_size(total_supporters: int) -> int:
    """Pool size = min(BANDCAMP_DRIVER_POOL || 5, total_supporters)."""
    try:
        cap = int(os.environ.get("BANDCAMP_DRIVER_POOL", _DEFAULT_DRIVER_POOL))
    except ValueError:
        cap = _DEFAULT_DRIVER_POOL
    cap = max(1, cap)
    if total_supporters <= 0:
        return cap
    return min(cap, total_supporters)

_DEFAULT_BPM_RERANK_ALPHA = 0.05

# API `count` for a supporter's collection beyond the first page. 10000 =
# "everything" (the historical behaviour). Callers that want speed and less
# whale bias pass a smaller page (the feed is newest-first).
_DEFAULT_COLLECTION_PAGE_SIZE = 10000
_DEFAULT_SELENIUM_FALLBACK_BUDGET = 8
# Items on a fan page's first page (what ``first_page_only`` reads).
_FIRST_PAGE_SIZE = 20


def _resolve_selenium_fallback_budget() -> int:
    """Max supporters per call read through the (slow, single) fallback
    browser when Bandcamp challenges plain HTTP. Env
    ``BANDCAMP_SELENIUM_FALLBACK_BUDGET``."""
    try:
        return max(0, int(os.environ.get("BANDCAMP_SELENIUM_FALLBACK_BUDGET",
                                         _DEFAULT_SELENIUM_FALLBACK_BUDGET)))
    except ValueError:
        return _DEFAULT_SELENIUM_FALLBACK_BUDGET


_DEFAULT_FEATURE_WORKERS = 4


def _resolve_feature_workers() -> int:
    """Parallel feature extractions (download + analysis; the decode itself
    is serialized process-wide). Env ``BANDCAMP_FEATURE_WORKERS``; small
    CPU-bound hosts want ~their core count."""
    try:
        return max(1, int(os.environ.get("BANDCAMP_FEATURE_WORKERS", _DEFAULT_FEATURE_WORKERS)))
    except ValueError:
        return _DEFAULT_FEATURE_WORKERS


class _NoFanData(Exception):
    """A fan page that legitimately has no readable list (private, deleted,
    empty). Cached as empty for a day so it isn't re-fetched every call."""


def _resolve_bpm_rerank_alpha() -> float:
    """Resolve α for the BPM re-rank score from the env, defaulting to 0.05."""
    raw = os.environ.get("BANDCAMP_BPM_RERANK_ALPHA")
    if raw is None:
        return _DEFAULT_BPM_RERANK_ALPHA
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_BPM_RERANK_ALPHA


def _bpm_rerank_score(
    supporters_count: int,
    bpm_distance: Optional[float],
    alpha: float,
) -> float:
    """Combined re-rank score: ``supporters_count - α * bpm_distance``.

    Items with no detectable BPM (``bpm_distance is None``) receive no
    penalty so they keep their natural supporter-count ranking.
    """
    penalty = alpha * bpm_distance if bpm_distance is not None else 0.0
    return supporters_count - penalty


from bandcamp_recommender import fans
from bandcamp_recommender.artists import artist_tokens
from bandcamp_recommender.items import normalize_item_url
from bandcamp_recommender.recommendations.scraper import (
    extract_artist,
    extract_item_id,
    extract_supporters,
    extract_tags,
    fetch_page_html,
    supporter_fan_ids,
)
from bandcamp_recommender.recommendations.mood_tags import tag_mood_score
from bandcamp_recommender.recommendations.tags import calculate_tag_similarity, normalize_tag


def _cap_per_artist(
    ranked: List[tuple],
    tokens_of: Callable[[str], set],
    max_per_artist: int,
    limit: int,
) -> Tuple[List[tuple], int]:
    """The first ``limit`` of ``ranked`` (``(item_id, count)``, best first)
    keeping at most ``max_per_artist`` releases per artist token, and how many
    releases the cap skipped on the way. A release credited to several
    artists counts for each of them and is skipped once any of them is at the
    cap; credits without tokens ("Various Artists", "") are never capped."""
    kept: List[tuple] = []
    skipped = 0
    per_artist: Counter = Counter()
    for iid, cnt in ranked:
        if len(kept) >= limit:
            break
        tokens = tokens_of(iid)
        if any(per_artist[t] >= max_per_artist for t in tokens):
            skipped += 1
            continue
        per_artist.update(tokens)
        kept.append((iid, cnt))
    return kept, skipped


class SupporterRecommender:
    """Generates Bandcamp recommendations based on what supporters purchased."""

    def __init__(self, headless: bool = True):
        """Initialize the recommender.

        Args:
            headless: Ignored - Selenium always runs headless to prevent popup windows.
                     Kept for API compatibility.
        """
        self.item_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()
        self._budget_lock = Lock()
        self._collection_page_size = _DEFAULT_COLLECTION_PAGE_SIZE
        self._selenium_budget = _resolve_selenium_fallback_budget()
        # username -> fan_id from the seed page (one-call collection reads).
        self._fan_ids: Dict[str, int] = {}
        # What the last get_recommendations pool steps did (seed artist, drop
        # counts); get_similar_recommendations reports it in last_call_stats.
        self._pool_stats: Dict[str, Any] = {}

    def get_recommendations(
        self,
        wishlist_item_url: str,
        max_recommendations: int = 10,
        min_supporters: int = 2,
        progress_callback: Optional[Callable] = None,
        include_bpm: bool = False,
        bpm_method: str = "auto",
        bpm_duration: float = 60.0,
        bpm_match: bool = False,
        include_intensity: bool = False,
        intensity_duration: float = 60.0,
        include_mood_tag_score: bool = False,
        first_page_only: bool = False,
        hydrate_tags: bool = True,
        use_wishlist: bool = False,
        event_callback: Optional[Callable[[dict], None]] = None,
        collection_page_size: int = _DEFAULT_COLLECTION_PAGE_SIZE,
        candidate_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        exclude_seed_artist: bool = False,
        max_per_artist: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on supporter purchases.

        Args:
            wishlist_item_url: URL of the Bandcamp item to get recommendations for
            max_recommendations: Maximum number of recommendations to return
            min_supporters: Minimum number of supporters who must have purchased an item
            progress_callback: Optional callback function(status, current, total, estimated_seconds)
            include_bpm: If True, detect a BPM for each recommendation's first
                playable preview and attach ``bpm`` / ``bpm_confidence`` /
                ``bpm_method`` keys. Requires optional audio deps (e.g. librosa);
                items without a streamable preview are left without a BPM.
            bpm_method: BPM backend to use — ``"auto"`` (default),
                ``"joe_sullivan"``, or ``"librosa"``.
            bpm_duration: Seconds of audio to analyse per track (default 60).
            bpm_match: If True, expand the candidate pool to
                ``max(50, max_recommendations * 3)``, detect BPM for each
                candidate, and re-rank by
                ``supporters_count - α * octave_tolerant_bpm_distance``
                (α from ``BANDCAMP_BPM_RERANK_ALPHA``, default 0.05).
                Each returned dict gains a ``bpm_distance`` field (None
                when no BPM could be detected for that item, or when this
                flag is False). Implies ``include_bpm`` behavior for the
                returned items.
            include_intensity: If True, attach a 0..1 ``intensity`` score for
                each recommendation's first playable preview (RMS + onset rate
                + spectral centroid + crest factor). Items without a
                streamable preview get ``intensity = None``. Requires the
                same optional audio deps as ``include_bpm``.
            intensity_duration: Seconds of audio to analyse for the intensity
                score (default 60).
            include_mood_tag_score: If True, attach a ``mood_tag_score`` key
                to each recommendation. The score is in ``[-1, 1]`` from
                chill to party (see :mod:`mood_tags`), or ``None`` when no
                tag in the result matches the lexicon. Free of extra
                fetches — tags are already hydrated for the top-N.
            first_page_only: If True, read only each supporter's first
                collection page (~20 most-recent purchases) instead of
                paginating their entire collection via the API. Trades
                deep-overlap recall for a large latency win — a single
                supporter with a multi-thousand-item collection can cost
                10-20s of API pagination otherwise. Defaults to False
                (full collections, unchanged behavior).
            hydrate_tags: If True (default), fetch tags for the final
                top-N items. Set False when the caller hydrates tags
                out-of-band (e.g. from its own page scrape) and wants the
                candidate pool back without paying the per-item curl cost.
            use_wishlist: If True, draw candidates from the distinct union of
                each supporter's collection AND wishlist (deduped per
                supporter). Both lists come from the same ``/wishlist``
                pagedata, so with ``first_page_only`` the wishlist read is a
                page-cache hit (no extra network). Defaults to False
                (collection only, unchanged behavior).

            collection_page_size: How many items to read from each
                supporter's collection beyond the first page (the API feed is
                newest-first). Default reads everything (unchanged
                behaviour); a smaller page is much faster and damps "whale"
                collectors who own thousands of items.
            candidate_filter: Optional predicate on an item's metadata dict
                (``item_title``, ``band_name``, ``item_url``). Applied to the
                ranked candidates BEFORE the result is cut to
                ``max_recommendations``, so the caller's exclusions (e.g.
                recently played) don't waste result slots or downstream work.
                It sees every candidate ``exclude_seed_artist`` keeps,
                including ones ``max_per_artist`` then skips.
            exclude_seed_artist: If True, drop candidates by the seed's own
                artist (credits whose :func:`~bandcamp_recommender.artists.artist_tokens`
                intersect the seed page's artist, so collabs count) before
                ``candidate_filter`` and the cut. A seed's supporters mostly
                own its artist's other releases, which can otherwise fill the
                result. A compilation seed ("Various Artists") excludes
                nothing; a seed page without an artist excludes nothing and
                logs a warning. Default False (unchanged behaviour).
            max_per_artist: If set (>= 1), keep at most this many releases
                per artist token, best-ranked first, after ``candidate_filter``
                and before the cut, so a few artists' back catalogues can't
                fill the result. A collab counts for each of its artists;
                compilations are never capped. Default None (no cap).

        Returns:
            List of recommendation dictionaries with item_title, band_name, item_url, supporters_count
        """
        if max_per_artist is not None and max_per_artist < 1:
            raise ValueError(f"max_per_artist must be >= 1 or None, got {max_per_artist!r}")
        self._begin_call(collection_page_size)
        # Get supporters of the wishlist item
        if progress_callback:
            progress_callback("Extracting supporters from album page...", 0, 0, 0)
        supporters = extract_supporters(wishlist_item_url)
        self._fan_ids = supporter_fan_ids(wishlist_item_url)
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            if event_callback:
                event_callback({"type": "supporters", "supporters": [], "total": 0})
            return []

        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)

        if event_callback:
            event_callback({
                "type": "supporters",
                "supporters": list(supporters),
                "total": len(supporters),
            })

        # Get the original item ID to exclude it from recommendations
        if progress_callback:
            progress_callback("Extracting item ID...", 0, 0, 0)
        original_item_id = extract_item_id(wishlist_item_url)
        # Read now, moments after extract_supporters fetched the seed page, so
        # it's a page-cache hit (after a long supporter phase it may not be).
        if exclude_seed_artist:
            seed_artist = extract_artist(wishlist_item_url)
            if not seed_artist:
                logger.warning("exclude_seed_artist: no artist on the seed page %s, "
                               "excluding nothing", wishlist_item_url)
            self._pool_stats["seed_artist"] = seed_artist
        # URL-normalized seed key, computed once. Used to drop the seed from
        # the per-supporter supporter_done emit (the animation's cloud), the
        # same way it's dropped from the ranked results below. Cosmetic only —
        # ranking/counting still happen on the full purchase set.
        seed_key = normalize_item_url(wishlist_item_url)

        # Get purchases from all supporters (with metadata) - parallel processing
        all_purchases = []
        start_time = time.time()
        total_supporters = len(supporters)
        completed_count = 0
        completed_lock = Lock()

        # Per-id provenance: "collection" (owned) wins over "wishlist"
        # (wanted), accumulated across supporters under a lock so the
        # supporter_done emit can tag owned vs. wanted edges.
        item_src: Dict[str, str] = {}
        item_src_lock = Lock()

        # Curl-first: no driver pool init up front. Workers spin up Chrome
        # lazily only if curl falls over for a specific supporter.
        def fetch_supporter_purchases(supporter):
            """Fetch purchases (and optionally wishlist) for one supporter.

            With ``use_wishlist`` the candidate set is the distinct union of
            the supporter's collection and wishlist. Both lists live in the
            same ``/wishlist`` pagedata, so the second fetch is a page-cache
            hit (no extra network) in the ``first_page_only`` path. Dedupe is
            per-supporter so one person counts an item once even if it's both
            bought and wishlisted — keeps ``min_supporters`` counting people.
            """
            coll = self._get_supporter_items_curl_first(
                supporter, "collection_data", first_page_only=first_page_only
            )
            ids = list(coll)
            if use_wishlist:
                wish = self._get_supporter_items_curl_first(
                    supporter, "wishlist_data", first_page_only=first_page_only
                )
                with item_src_lock:
                    for iid in wish:
                        if iid not in coll:
                            item_src.setdefault(iid, "wishlist")
                ids = list(dict.fromkeys(ids + wish))
            with item_src_lock:
                for iid in coll:
                    item_src[iid] = "collection"
            return ids, supporter

        # Use ThreadPoolExecutor for parallel processing. Cap at 15 to avoid
        # hammering Bandcamp; curl handshakes are cheap so this is the limit
        # set by politeness, not by laptop resources.
        max_workers = min(_resolve_supporter_concurrency(), total_supporters) if total_supporters else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_purchases, supporter): supporter
                for supporter in supporters
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_supporter):
                try:
                    purchases, supporter = future.result()
                    with completed_lock:
                        all_purchases.extend(purchases)
                        completed_count += 1

                        if progress_callback:
                            elapsed = time.time() - start_time
                            avg_time_per_supporter = elapsed / completed_count if completed_count > 0 else 2.0
                            remaining = total_supporters - completed_count
                            estimated_seconds = avg_time_per_supporter * remaining
                            progress_callback(
                                f"Fetching items from supporter {completed_count}/{total_supporters} ({supporter})...",
                                completed_count,
                                total_supporters,
                                int(estimated_seconds)
                            )

                        if event_callback:
                            items_meta = []
                            for iid in purchases:
                                info = self.item_cache.get(iid) or {}
                                # Exclude the seed from the emitted cloud nodes
                                # too — by id and by normalized url — so the
                                # searched track never appears as a live graph
                                # node. The ranked results already drop it.
                                if iid == original_item_id or (
                                    normalize_item_url(info.get("item_url", "")) == seed_key
                                ):
                                    continue
                                items_meta.append({
                                    "id": iid,
                                    "title": info.get("item_title", ""),
                                    "band": info.get("band_name", ""),
                                    "src": item_src.get(iid, "collection"),
                                })
                            event_callback({
                                "type": "supporter_done",
                                "supporter": supporter,
                                "index": completed_count,
                                "total": total_supporters,
                                "items": items_meta,
                            })
                except Exception as e:
                    supporter = future_to_supporter[future]
                    print(f"Error processing {supporter}: {e}")
                    with completed_lock:
                        completed_count += 1

        # Count purchases and filter
        purchase_counter = Counter(all_purchases)
        # Remove the original item from recommendations
        if original_item_id:
            purchase_counter.pop(original_item_id, None)

        if progress_callback:
            progress_callback(
                f"Processing {len(all_purchases)} purchases from {completed_count} supporters...",
                total_supporters,
                total_supporters,
                0
            )

        if len(all_purchases) == 0:
            if progress_callback:
                progress_callback(
                    "Note: No purchases found. Collections are likely private and require authentication.",
                    total_supporters,
                    total_supporters,
                    0
                )

        # Filter by minimum supporters and get top items
        filtered_items = {
            item_id: count
            for item_id, count in purchase_counter.items()
            if count >= min_supporters
        }
        sorted_items = sorted(
            filtered_items.items(), key=lambda x: x[1], reverse=True
        )

        # URL-normalized backstop for the id-based pop above: extract_item_id
        # can fail (curl 403, page-structure shift) and return None, leaving
        # the seed in the counter — where it would rank #1 in its own
        # recommendations. Drop any candidate whose normalized item_url
        # matches the seed url (seed_key computed once near original_item_id).
        # Done before the slice so the result count stays at max_recommendations.
        sorted_items = [
            (iid, cnt) for (iid, cnt) in sorted_items
            if normalize_item_url((self.item_cache.get(iid) or {}).get("item_url", "")) != seed_key
        ]

        def tokens_of(iid: str) -> set:
            info = self.item_cache.get(iid) or {}
            return artist_tokens(info.get("band_name") or "", info.get("item_url"))

        # Before candidate_filter, so the caller's filter (which may keep
        # state) never sees the seed's own artist. It does see candidates
        # max_per_artist skips after it.
        if exclude_seed_artist:
            seed_tokens = artist_tokens(seed_artist, wishlist_item_url)
            before = len(sorted_items)
            if seed_tokens:
                sorted_items = [
                    (iid, cnt) for (iid, cnt) in sorted_items
                    if not tokens_of(iid) & seed_tokens
                ]
            self._pool_stats["seed_artist_dropped"] = before - len(sorted_items)

        if candidate_filter is not None:
            before = len(sorted_items)
            sorted_items = [
                (iid, cnt) for (iid, cnt) in sorted_items
                if candidate_filter(self.item_cache.get(iid) or {})
            ]
            self._pool_stats["filtered"] = before - len(sorted_items)

        pool_size = max(50, max_recommendations * 3) if bpm_match else max_recommendations
        if max_per_artist is None:
            top_items = sorted_items[:pool_size]
        else:
            # After the caller's filter, so a release it rejects doesn't use
            # up its artist's slots.
            top_items, self._pool_stats["artist_capped"] = _cap_per_artist(
                sorted_items, tokens_of, max_per_artist, pool_size
            )

        if event_callback:
            top_meta = []
            for iid, cnt in top_items:
                info = self.item_cache.get(iid) or {}
                top_meta.append({
                    "id": iid,
                    "item_url": info.get("item_url", ""),
                    "title": info.get("item_title", ""),
                    "band": info.get("band_name", ""),
                    "supporters_count": cnt,
                })
            event_callback({
                "type": "ranked",
                "min_supporters": min_supporters,
                "top": top_meta,
            })

        if progress_callback:
            progress_callback("Building recommendations...", total_supporters, total_supporters, 0)

        # Hydrate tags only for the final ranked items. This is the big win
        # vs. the old behavior where every item from every supporter triggered
        # a separate curl call for tags. Callers running a progressive UI
        # (boemketel's fast /recommend) skip this entirely — they hydrate
        # tags out-of-band from their own page scrape — so the candidate
        # pool can return before paying ~N curl round-trips.
        if hydrate_tags:
            self._hydrate_tags_for_items([item_id for item_id, _ in top_items])

        # Get item info and build recommendations
        recommendations = []
        for item_id, supporters_count in top_items:
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info["supporters_count"] = supporters_count
                recommendations.append(item_info)

        # ``bpm_match`` reuses this field for re-rank distance; pre-init so
        # the key always exists on returned dicts even when no audio runs.
        for rec in recommendations:
            rec["bpm_distance"] = None

        if include_mood_tag_score:
            for rec in recommendations:
                rec["mood_tag_score"] = tag_mood_score(rec.get("tags") or [])

        if (include_bpm or include_intensity) and recommendations:
            # Imported here so the optional audio stack is only loaded when
            # an audio detector is actually requested.
            from bandcamp_recommender.recommendations.intensity import (
                attach_audio_features,
                attach_intensities,
            )
            from bandcamp_recommender.recommendations.bpm import attach_bpms

            if progress_callback:
                progress_callback(
                    "Analyzing audio for recommendations...",
                    0,
                    len(recommendations),
                    0,
                )

            if include_bpm and include_intensity:
                # Single shared decode per track — runs the Joe Sullivan
                # BPM detector against the librosa-decoded buffer, so
                # ``bpm_method`` is implicitly joe_sullivan in this path.
                # If a caller pinned ``bpm_method="librosa"`` we still run
                # the shared path; the BPM value differs negligibly in
                # practice and we save the second download.
                attach_audio_features(
                    recommendations,
                    include_bpm=True,
                    include_intensity=True,
                    bpm_duration=bpm_duration,
                    intensity_duration=intensity_duration,
                    progress_callback=progress_callback,
                )
            elif include_bpm:
                attach_bpms(
                    recommendations,
                    method=bpm_method,
                    duration=bpm_duration,
                    progress_callback=progress_callback,
                )
            else:
                attach_intensities(
                    recommendations,
                    duration=intensity_duration,
                    progress_callback=progress_callback,
                )

        if bpm_match and recommendations:
            from bandcamp_recommender.recommendations.bpm import (
                attach_bpms,
                get_seed_bpm,
                octave_tolerant_bpm_distance,
            )

            if progress_callback:
                progress_callback(
                    "Detecting seed BPM...", 0, len(recommendations), 0
                )
            seed = get_seed_bpm(
                wishlist_item_url, method=bpm_method, duration=bpm_duration
            )
            seed_bpm = float(seed["bpm"]) if seed and seed.get("bpm") else None

            if seed_bpm is None:
                # No seed BPM means every item gets a 0 penalty — re-ranking
                # would be a no-op. Skip the expensive per-candidate BPM
                # detection and just trim to the requested size.
                recommendations = recommendations[:max_recommendations]
            else:
                # attach_bpms is idempotent — items with `bpm` already set (from the
                # include_bpm pass above) are skipped without a page fetch.
                if progress_callback:
                    progress_callback(
                        "Detecting BPMs for expanded candidate pool...",
                        0,
                        len(recommendations),
                        0,
                    )
                attach_bpms(
                    recommendations,
                    method=bpm_method,
                    duration=bpm_duration,
                    progress_callback=progress_callback,
                )

                alpha = _resolve_bpm_rerank_alpha()
                for rec in recommendations:
                    cand_bpm = rec.get("bpm")
                    if cand_bpm is not None:
                        rec["bpm_distance"] = octave_tolerant_bpm_distance(
                            seed_bpm, float(cand_bpm)
                        )
                    else:
                        rec["bpm_distance"] = None

                recommendations.sort(
                    key=lambda r: _bpm_rerank_score(
                        r.get("supporters_count", 0),
                        r.get("bpm_distance"),
                        alpha,
                    ),
                    reverse=True,
                )
                recommendations = recommendations[:max_recommendations]

        if progress_callback:
            progress_callback(
                f"Complete! Found {len(recommendations)} recommendations.",
                total_supporters,
                total_supporters,
                0
            )

        return recommendations

    def get_similar_recommendations(
        self,
        source_url: str,
        max_recommendations: int = 10,
        candidate_pool_size: int = 30,
        min_supporters: int = 1,
        feature_weights: Optional[Dict[str, float]] = None,
        intensity_duration: float = 60.0,
        bpm_duration: float = 60.0,
        diversify: Optional[str] = None,
        diversity_lambda: float = 0.5,
        vibe_tau: float = 0.15,
        realm_mult: float = 1.5,
        progress_callback: Optional[Callable] = None,
        collection_page_size: int = _DEFAULT_COLLECTION_PAGE_SIZE,
        candidate_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        exclude_seed_artist: bool = False,
        max_per_artist: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get recommendations ordered by feature-vector similarity to a source.

        End-to-end pipeline for a "more like this" call:

        1. Run the existing supporter-overlap recommender to get a
           ``candidate_pool_size`` pool of candidates (default 30).
        2. Hydrate the source song (tags + preview ``audio_url``) and
           any candidate that's missing one.
        3. Extract the full feature vector for source + every candidate
           (one download + decode per track, shared between intensity
           and BPM detection).
        4. Compute weighted-Euclidean distance from source to each
           candidate over the intersection of present features.
        5. Sort ascending and return the top ``max_recommendations``
           candidates.

        Each returned dict carries the usual supporter-overlap metadata
        (``item_title``, ``band_name``, ``item_url``, ``supporters_count``,
        ``tags``) plus:

        * ``audio_url``   — the preview URL we used for feature extraction
                            (``None`` if no streamable preview).
        * ``features``    — the full feature dict from
                            :func:`bandcamp_recommender.features.extract_features`,
                            including the raw ``bpm`` float for display /
                            beat-matching.
        * ``distance``    — weighted-Euclidean similarity distance to the
                            source (smaller = more similar; ``None`` only
                            when no feature is shared).

        ``feature_weights`` is passed straight through to ``distance`` —
        leave as ``None`` to use ``features.DEFAULT_WEIGHTS``, or override
        per-feature for the radio's mode-specific tuning.

        ``diversify`` opts into an in-vibe diversity re-rank applied to the
        distance-ranked pool before trimming to ``max_recommendations``: instead
        of the top-k (which cluster in style) it returns k items that are diverse
        in style while keeping the source's vibe (intensity within ``vibe_tau``)
        and staying related (``realm_mult``). Modes: ``"mmr"`` (relevance↔
        diversity via ``diversity_lambda`` 0..1), ``"maxmin"`` (max style spread),
        ``"stratified"`` (genre/artist coverage). ``None`` (the default) is the
        plain similarity ranking — fully backward-compatible; if unset it falls
        back to the ``BANDCAMP_DIVERSIFY`` env var. See
        :mod:`bandcamp_recommender.recommendations.diversity`.

        ``collection_page_size``, ``candidate_filter``, ``exclude_seed_artist``
        and ``max_per_artist`` are passed to :meth:`get_recommendations`; all
        apply before the ``candidate_pool_size`` cut, so every hydrated +
        decoded candidate is one the caller can use. ``last_call_stats["pool"]``
        reports what they did: ``seed_artist`` (None if unreadable) and
        ``seed_artist_dropped``, ``filtered``, ``artist_capped`` (releases
        skipped while filling the pool), each only when its option is set.

        Audio features describe each item's featured preview track (the one
        ``audio_url`` points at) and are cached on disk per (item, track).
        """
        # Local import keeps the optional audio stack out of cold starts
        # for the rest of the package.
        from bandcamp_recommender import http
        from bandcamp_recommender.features import (
            DEFAULT_WEIGHTS,
            distance as feature_distance,
            features_for_item,
        )

        weights = feature_weights or DEFAULT_WEIGHTS
        counters_before = http.snapshot()
        t0 = time.monotonic()

        if progress_callback:
            progress_callback("Fetching candidate pool via supporter overlap...", 0, 0, 0)

        candidates = self.get_recommendations(
            wishlist_item_url=source_url,
            max_recommendations=candidate_pool_size,
            min_supporters=min_supporters,
            progress_callback=progress_callback,
            collection_page_size=collection_page_size,
            candidate_filter=candidate_filter,
            exclude_seed_artist=exclude_seed_artist,
            max_per_artist=max_per_artist,
        )
        t_pool = time.monotonic() - t0

        # Backstop the ID-based filter in get_recommendations: when
        # extract_item_id can't read the source's tralbum_id (curl 403,
        # page structure shift) the seed slips through and ranks #1 with
        # near-zero feature distance to itself.
        source_key = normalize_item_url(source_url)
        candidates = [
            c for c in candidates
            if normalize_item_url(c.get("item_url", "")) != source_key
        ]

        if not candidates:
            if progress_callback:
                progress_callback("No candidates returned by supporter overlap.", 0, 0, 0)
            return []

        if progress_callback:
            progress_callback(
                "Extracting feature vectors...",
                0,
                len(candidates) + 1,  # +1 for source
                0,
            )
        t1 = time.monotonic()

        def _featurize(item: Dict[str, Any]) -> Dict[str, Any]:
            # One page read (shared short-lived page cache with the tag
            # hydration above) picks the featured preview; its audio
            # features come from the disk cache or one decode.
            try:
                vector, preview = features_for_item(
                    item["item_url"],
                    item.get("tags") or [],
                    intensity_duration=intensity_duration,
                    bpm_duration=bpm_duration,
                )
            except Exception:
                logger.warning("feature extraction failed for %s", item.get("item_url"),
                               exc_info=_debug_exc_info())
                vector, preview = {k: None for k in weights}, None
            item["features"] = vector
            item["audio_url"] = preview.get("audio_path") if preview else None
            item["preview_track_id"] = preview.get("track_id") if preview else None
            item["preview_title"] = preview.get("title") if preview else None
            return item

        source_item: Dict[str, Any] = {
            "item_url": source_url,
            "tags": extract_tags(source_url) or [],
        }
        workers = max(1, min(_resolve_feature_workers(), len(candidates) + 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_featurize, [source_item] + candidates))
        source_features = source_item["features"]
        t_features = time.monotonic() - t1

        # Distance + ordering. Items without any shared feature get
        # distance=None and sink to the bottom (in stable order).
        rated: List[Dict[str, Any]] = []
        unrated: List[Dict[str, Any]] = []
        for cand in candidates:
            d = feature_distance(source_features, cand["features"], weights)
            cand["distance"] = d
            (rated if d is not None else unrated).append(cand)
        rated.sort(key=lambda c: c["distance"])

        # Opt-in in-vibe diversity re-rank (applied to the feature-comparable
        # candidates; the unrated tail keeps its sink-to-bottom order). Default
        # off → plain similarity ranking. Falls back to the BANDCAMP_DIVERSIFY
        # env so it can be toggled without a code change, mirroring the BPM
        # re-rank knob.
        mode = diversify if diversify is not None else os.environ.get("BANDCAMP_DIVERSIFY")
        if mode:
            from bandcamp_recommender.recommendations.diversity import diversify_items
            rated = diversify_items(
                rated,
                source_features,
                mode=mode,
                lam=diversity_lambda,
                vibe_tau=vibe_tau,
                realm_mult=realm_mult,
            )

        ranked = (rated + unrated)[:max_recommendations]
        with_audio = sum(1 for c in candidates if (c.get("features") or {}).get("rms_mean") is not None)
        logger.info(
            "similar recs for %s: pool=%.1fs features=%.1fs candidates=%d with_audio=%d http=%s",
            source_url, t_pool, t_features, len(candidates), with_audio,
            {k: v for k, v in http.diff(counters_before).items() if v},
        )
        self.last_call_stats = {
            "pool_s": round(t_pool, 1),
            "features_s": round(t_features, 1),
            "candidates": len(candidates),
            "with_audio": with_audio,
            "http": http.diff(counters_before),
            "pool": dict(self._pool_stats),
        }
        if progress_callback:
            progress_callback(
                f"Returning top {len(ranked)} ordered by similarity.",
                len(candidates) + 1,
                len(candidates) + 1,
                0,
            )
        return ranked

    def _store_item_metadata(
        self,
        item_id_str: str,
        item_data: Dict[str, Any],
        extract_tags_flag: bool
    ):
        """Store item metadata in cache (thread-safe).

        Args:
            item_id_str: Item ID as string
            item_data: Item data dictionary
            extract_tags_flag: Ignored. Tag extraction is now deferred to
                _hydrate_tags_for_items so that we don't pay a curl-per-item
                cost for items that won't appear in the final results.
        """
        with self._cache_lock:
            if item_id_str not in self.item_cache:
                item_url = item_data.get("item_url", "")
                self.item_cache[item_id_str] = {
                    "item_title": item_data.get("item_title", "Unknown Title"),
                    "band_name": item_data.get("band_name", "Unknown Artist"),
                    "item_url": item_url or f"https://bandcamp.com/album/{item_id_str}",
                    "tags": [],
                }

    def _parse_collection_pagedata(
        self,
        pagedata: Dict[str, Any],
        data_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract first-page items + pagination state from a wishlist pagedata blob.

        data_key is either "collection_data" (purchases) or "wishlist_data".
        Returns None if the blob is missing the expected structure.
        """
        section = pagedata.get(data_key)
        if not isinstance(section, dict):
            return None
        cache_key = "collection" if data_key == "collection_data" else "wishlist"
        item_cache = pagedata.get("item_cache", {}).get(cache_key, {})
        first_page_item_ids: List[str] = []
        for key in section.get("sequence", []) + section.get("pending_sequence", []):
            item_data = item_cache.get(key)
            if not item_data:
                continue
            tralbum_id = item_data.get("tralbum_id")
            if not tralbum_id:
                continue
            item_id_str = str(tralbum_id)
            first_page_item_ids.append(item_id_str)
            # extract_tags_flag is ignored downstream; pass True for API parity.
            self._store_item_metadata(item_id_str, item_data, True)
        return {
            "first_page_item_ids": first_page_item_ids,
            "last_token": section.get("last_token", "") or "",
            "item_count": section.get("item_count", 0) or 0,
            "fan_id": pagedata.get("fan_data", {}).get("fan_id"),
        }

    # ---- supporter collections --------------------------------------------
    #
    # One supporter = their /wishlist fan page (pagedata: first ~20 items of
    # each list + pagination state) plus, for deeper reads, one API call.
    # Plain HTTP first; the shared browser only when Bandcamp serves its bot
    # challenge, and at most `_selenium_budget` supporters per call. Rate
    # limits and timeouts skip the supporter (losing one supporter's votes is
    # harmless; launching Chrome for it is not). Complete reads are cached on
    # disk (bandcamp_recommender.cache), so recurring seeds and overlapping
    # scenes stop re-downloading the same collections.

    _LIST_NAMES = {"collection_data": "collection", "wishlist_data": "wishlist"}
    _COMPLETE_FRACTION = fans.COMPLETE_FRACTION

    def _begin_call(self, collection_page_size: int = _DEFAULT_COLLECTION_PAGE_SIZE) -> None:
        """Per-call knobs + fallback budget (entry points call this first)."""
        self._collection_page_size = max(1, int(collection_page_size))
        self._fan_ids = {}
        self._pool_stats = {}
        with self._budget_lock:
            self._selenium_budget = _resolve_selenium_fallback_budget()

    def _take_selenium_budget(self) -> bool:
        with self._budget_lock:
            if self._selenium_budget <= 0:
                return False
            self._selenium_budget -= 1
            return True

    def _get_supporter_items_curl_first(
        self,
        username: str,
        data_key: str,
        first_page_only: bool = False,
    ) -> List[str]:
        """A supporter's item ids (collection or wishlist), metadata stored
        in ``item_cache``. Never raises; returns [] when the supporter can't
        be read right now (logged, not cached)."""
        from bandcamp_recommender import browser, cache as bc_cache, http
        from bandcamp_recommender.recommendations import curl_breaker

        depth = "first" if first_page_only else f"n{self._collection_page_size}"
        cached = bc_cache.get_supporter_items(username, data_key, depth)
        if cached is not None:
            http.count("supporter_cached")
            ids = []
            for iid, title, band, url in cached:
                self._store_item_metadata(
                    iid, {"item_title": title, "band_name": band, "item_url": url}, True
                )
                ids.append(iid)
            return ids

        try:
            try:
                if curl_breaker.should_skip_curl():
                    raise http.Challenged(f"breaker tripped: {username}")
                result = None
                if username in self._fan_ids:
                    http.count("supporter_api")
                    result = self._read_supporter_api(username, data_key, first_page_only)
                if result is None:
                    http.count("supporter_page")
                    result = self._read_supporter(username, data_key, first_page_only, None)
                ids, complete = result
                curl_breaker.record_outcome(success=True)
            except http.Challenged:
                if not curl_breaker.should_skip_curl():
                    curl_breaker.record_outcome(success=False)
                if not self._take_selenium_budget():
                    logger.warning("challenged; browser budget spent, skipping supporter %s", username)
                    return []
                with browser.session() as driver:
                    ids, complete = self._read_supporter(username, data_key, first_page_only, driver)
        except _NoFanData:
            bc_cache.put_supporter_items(username, data_key, depth, [], ttl_s=bc_cache.EMPTY_SUPPORTER_TTL_S)
            return []
        except http.FetchError as e:
            logger.warning("skipping supporter %s (%s): %s", username, data_key, e)
            return []
        except Exception:
            logger.warning("skipping supporter %s (%s)", username, data_key, exc_info=_debug_exc_info())
            return []

        if complete:
            with self._cache_lock:
                rows = [
                    [iid,
                     (self.item_cache.get(iid) or {}).get("item_title", ""),
                     (self.item_cache.get(iid) or {}).get("band_name", ""),
                     (self.item_cache.get(iid) or {}).get("item_url", "")]
                    for iid in ids
                ]
            # An empty list may be a soft failure: don't pin it for a week.
            bc_cache.put_supporter_items(
                username, data_key, depth, rows,
                ttl_s=bc_cache.SUPPORTER_TTL_S if rows else bc_cache.EMPTY_SUPPORTER_TTL_S,
            )
        return ids

    def _read_supporter_api(self, username: str, data_key: str, first_page_only: bool):
        """One API call when the seed page told us the supporter's fan_id: a
        token in the future returns the newest ``count`` items, the same head
        the fan page shows (verified), so the fan page fetch is skipped and a
        cold supporter costs one request instead of two. Returns
        ``(ids, True)``, or None when the API answers without an item list
        (the page path then decides private vs. broken). Raises
        ``http.FetchError`` subclasses like the page path."""
        count = _FIRST_PAGE_SIZE if first_page_only else self._collection_page_size
        data = fans.read_list_page(
            self._LIST_NAMES[data_key], self._fan_ids[username], fans.newest_first_token(), count,
            referer=f"https://bandcamp.com/{username}",
        )
        items = (data or {}).get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return None
        ids: List[str] = []
        seen = set()
        for item in items:
            tralbum_id = item.get("tralbum_id")
            if not tralbum_id:
                continue
            item_id_str = str(tralbum_id)
            if item_id_str in seen:
                continue
            seen.add(item_id_str)
            ids.append(item_id_str)
            self._store_item_metadata(item_id_str, item, True)
        return ids, True

    def _read_supporter(self, username: str, data_key: str, first_page_only: bool, driver) -> tuple:
        """``(ids, complete)`` for one supporter list over plain HTTP
        (``driver`` None) or inside the shared browser session. Raises
        ``http.Challenged`` / ``http.FetchError``, or ``_NoFanData`` for a
        private, deleted or empty fan page."""
        from bandcamp_recommender import browser, http

        url = fans.fan_page_url(username)
        try:
            if driver is None:
                html = http.get_html(url, timeout=20)
            else:
                html = browser.load(driver, url, None)
        except http.HTTPStatus as e:
            if e.status in (404, 410):
                raise _NoFanData(username) from e
            raise
        pagedata = fans.parse_pagedata(html)
        if pagedata is None:
            raise _NoFanData(username)
        parsed = self._parse_collection_pagedata(pagedata, data_key)
        if parsed is None:
            raise _NoFanData(username)

        ids = list(parsed["first_page_item_ids"])
        first_len = len(ids)
        item_count = parsed["item_count"]
        if first_page_only:
            return ids, True
        page_size = self._collection_page_size
        if parsed["last_token"] and parsed["fan_id"] and first_len < item_count:
            try:
                data = fans.read_list_page(
                    self._LIST_NAMES[data_key], parsed["fan_id"], parsed["last_token"], page_size,
                    referer=url, driver=driver,
                )
            except http.Challenged:
                raise
            except http.FetchError as e:
                # Keep the first page (better than nothing), never cache it.
                logger.warning("collection API failed for %s, using first page: %s", username, e)
                return ids, False
            seen = set(ids)
            for item in (data or {}).get("items", []) or []:
                tralbum_id = item.get("tralbum_id")
                if not tralbum_id:
                    continue
                item_id_str = str(tralbum_id)
                if item_id_str in seen:
                    continue
                seen.add(item_id_str)
                ids.append(item_id_str)
                self._store_item_metadata(item_id_str, item, True)
        expected = min(item_count, first_len + page_size)
        return ids, len(ids) >= self._COMPLETE_FRACTION * expected

    def _hydrate_tags_for_items(self, item_ids: List[str]) -> None:
        """Fetch tags for a small set of items in parallel and store in cache.

        Used only on the final ranked list (top-N) or on the unique candidate
        set in tag-similarity mode — never per supporter purchase. This is the
        single biggest reason the recommender is faster: we don't pay a curl
        round-trip per item, only per result we're actually going to return.
        """
        # Pick items that are in the cache but have no tags yet, and have a real URL.
        targets: List[str] = []
        with self._cache_lock:
            for item_id in item_ids:
                info = self.item_cache.get(item_id)
                if info and not info.get("tags") and info.get("item_url"):
                    # Skip placeholder URLs we synthesized in _store_item_metadata
                    if "/album/" + item_id not in info["item_url"]:
                        targets.append(item_id)

        if not targets:
            return

        def _fetch_one(item_id: str) -> None:
            with self._cache_lock:
                info = self.item_cache.get(item_id)
                url = info.get("item_url") if info else None
            if not url:
                return
            # Cached item metadata: the page is read at most once per month
            # per item instead of once per call (Bandcamp allows ~1 req/s).
            from bandcamp_recommender.items import item_meta

            tags = (item_meta(url) or {}).get("tags")
            if not tags:
                return
            with self._cache_lock:
                if item_id in self.item_cache:
                    self.item_cache[item_id]["tags"] = tags

        # Small bounded pool so we don't hammer Bandcamp.
        workers = min(_DEFAULT_TAG_WORKERS, len(targets))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_fetch_one, targets))

    def _get_item_info_from_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item info from tralbum_id using cache.

        Args:
            item_id: tralbum_id

        Returns:
            Dict with item_title, band_name, item_url, tags, or None if not in cache
        """
        return self.item_cache.get(item_id)

    def get_tag_similar_recommendations(
        self,
        item_url: str,
        max_recommendations: int = 10,
        min_similarity: float = 0.1,
        max_supporters: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on tag similarity.

        Explores supporters' collections and ranks items by tag similarity to the original.

        Args:
            item_url: URL of the Bandcamp item to get recommendations for
            max_recommendations: Maximum number of recommendations to return
            min_similarity: Minimum tag similarity score (0.0 to 1.0)
            max_supporters: Maximum number of supporters to fetch items from (None = all)
            progress_callback: Optional callback function(status, current, total, estimated_seconds)

        Returns:
            List of recommendation dictionaries with item_title, band_name, item_url,
            tags, similarity_score, and supporters_count
        """
        self._begin_call()
        # Get original item tags
        if progress_callback:
            progress_callback("Extracting tags from original item...", 0, 0, 0)
        original_tags = extract_tags(item_url)
        if not original_tags:
            # Try one more time in case of transient error
            original_tags = extract_tags(item_url)
            if not original_tags:
                if progress_callback:
                    progress_callback("No tags found for original item.", 0, 0, 0)
                return []

        if progress_callback:
            progress_callback(f"Found tags: {', '.join(original_tags)}", 0, 0, 0)

        original_item_id = extract_item_id(item_url)

        # Get supporters
        if progress_callback:
            progress_callback("Extracting supporters from page...", 0, 0, 0)
        supporters = extract_supporters(item_url)
        self._fan_ids = supporter_fan_ids(item_url)
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            return []

        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)

        # Limit number of supporters if specified
        if max_supporters and max_supporters < len(supporters):
            supporters = random.sample(supporters, max_supporters)
            if progress_callback:
                progress_callback(f"Using {len(supporters)} random supporters", len(supporters), len(supporters), 0)

        # Get all items from supporters' collections
        all_items = []
        start_time = time.time()
        total_supporters = len(supporters)
        completed_count = 0
        completed_lock = Lock()

        # Curl-first: skip driver pool init. Workers fall back to Selenium
        # only for supporters where curl fails (rare on a residential IP).
        if progress_callback:
            progress_callback(
                f"Fetching items from {total_supporters} supporters...",
                0,
                total_supporters,
                0
            )

        started: Dict[str, float] = {}

        def fetch_supporter_items(supporter):
            """Fetch items for a single supporter (thread-safe)."""
            started[supporter] = time.time()
            try:
                items = self._get_supporter_items_curl_first(
                    supporter, "collection_data"
                )
                return items, supporter, None
            except Exception as e:
                return [], supporter, f"Error fetching items: {str(e)[:50]}"

        max_workers = min(_resolve_supporter_concurrency(), total_supporters) if total_supporters else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_items, supporter): supporter
                for supporter in supporters
            }

            # Block on futures.wait instead of polling+sleep. The per-future
            # timeout counts from when the worker STARTS (recorded in the
            # worker): requests are paced, so a future can legitimately wait
            # in the queue far longer than it runs. Each request is already
            # bounded by its own wall-clock deadline.
            pending = set(future_to_supporter.keys())
            max_future_time = 150

            while pending:
                done, _still_pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

                for future in done:
                    supporter = future_to_supporter[future]
                    try:
                        items, supporter, error = future.result(timeout=1)
                        with completed_lock:
                            if error:
                                if progress_callback:
                                    progress_callback(
                                        f"Error from {supporter}: {error[:30]}... ({completed_count + 1}/{total_supporters})",
                                        completed_count + 1,
                                        total_supporters,
                                        0
                                    )
                            else:
                                all_items.extend(items)
                                if progress_callback:
                                    elapsed = time.time() - start_time
                                    avg_time = elapsed / completed_count if completed_count > 0 else 2.0
                                    remaining = total_supporters - completed_count
                                    estimated_seconds = avg_time * remaining
                                    progress_callback(
                                        f"Fetched {len(items)} items from {supporter} ({completed_count + 1}/{total_supporters})...",
                                        completed_count + 1,
                                        total_supporters,
                                        int(estimated_seconds)
                                    )
                            completed_count += 1
                    except Exception as e:
                        with completed_lock:
                            completed_count += 1
                            if progress_callback:
                                error_msg = str(e)[:50] if str(e) else "Unknown error"
                                progress_callback(
                                    f"Error from {supporter}: {error_msg}... ({completed_count}/{total_supporters})",
                                    completed_count,
                                    total_supporters,
                                    0
                                )

                # Sweep for futures that have exceeded max_future_time.
                now = time.time()
                timed_out = {
                    f for f in pending - done
                    if now - started.get(future_to_supporter[f], now) > max_future_time
                }
                for future in timed_out:
                    future.cancel()
                    with completed_lock:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(
                                f"Timeout from {future_to_supporter[future]} ({completed_count}/{total_supporters})...",
                                completed_count,
                                total_supporters,
                                0
                            )

                pending -= done
                pending -= timed_out

        if progress_callback:
            progress_callback("Calculating tag similarities...", total_supporters, total_supporters, 0)

        # Remove duplicates and original item
        unique_items = list(set(all_items))
        if original_item_id and original_item_id in unique_items:
            unique_items.remove(original_item_id)

        # Tag-similarity mode needs tags up front (they feed the score), but
        # we still defer the fetch to right here, instead of doing it inside
        # every per-supporter worker. Same total tag-fetches, but bounded and
        # batched so it doesn't compete with collection scrapes for drivers.
        if progress_callback:
            progress_callback(
                f"Fetching tags for {len(unique_items)} candidate items...",
                total_supporters,
                total_supporters,
                0,
            )
        self._hydrate_tags_for_items(unique_items)

        # Build tag frequency map for TF-IDF weighting
        tag_frequencies: Dict[str, int] = Counter()
        items_with_tags: Dict[str, List[str]] = {}

        for item_id in unique_items:
            item_info = self._get_item_info_from_id(item_id)
            if item_info and item_info.get('tags'):
                tags = item_info['tags']
                items_with_tags[item_id] = tags
                for tag in tags:
                    normalized = normalize_tag(tag)
                    tag_frequencies[normalized] += 1

        total_items = len(items_with_tags) if items_with_tags else 1

        # Calculate similarity scores
        item_similarities: Dict[str, float] = {}
        for item_id, candidate_tags in items_with_tags.items():
            similarity = calculate_tag_similarity(
                original_tags,
                candidate_tags,
                tag_frequencies,
                total_items
            )
            if similarity >= min_similarity:
                item_similarities[item_id] = similarity

        # Sort by similarity (descending)
        sorted_items = sorted(
            item_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_recommendations]

        # Build recommendations
        recommendations = []
        for item_id, similarity_score in sorted_items:
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info['similarity_score'] = similarity_score
                # Count how many supporters have this item
                supporters_count = all_items.count(item_id)
                item_info['supporters_count'] = supporters_count
                recommendations.append(item_info)

        if progress_callback:
            progress_callback(
                f"Complete! Found {len(recommendations)} tag-similar recommendations.",
                total_supporters,
                total_supporters,
                0
            )

        return recommendations

    def _get_supporters(self, item_url: str) -> List[str]:
        """Get list of supporter usernames from an item page.
        
        Wrapper method for backward compatibility with scripts.
        
        Args:
            item_url: URL of the Bandcamp item
            
        Returns:
            List of supporter usernames
        """
        return extract_supporters(item_url)

    def get_random_items(
        self,
        item_url: str,
        num_items: int,
        num_supporters: int = 20,
        use_wishlist: bool = False,
        min_overlap: Optional[int] = None,
        use_fallback: bool = False,
        progress_callback: Optional[Callable] = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Get random items from random supporters' collections.
        
        Args:
            item_url: URL of the Bandcamp item to get supporters from
            num_items: Number of random items to return
            num_supporters: Number of random supporters to check (default: 20)
            use_wishlist: If True, draw candidates from the distinct union of each
                supporter's collection AND wishlist (deduped per supporter), not just
                their purchases. Consistent with get_recommendations and the UI.
                Defaults to False (collection only, unchanged behavior).
            min_overlap: Only select items found in at least N supporters' collections (default: None, i.e., any item)
            use_fallback: If True and min_overlap is set, automatically reduce min_overlap if not enough items found
            progress_callback: Optional callback function(status, current, total, estimated_seconds)
            
        Returns:
            List of item dictionaries with item_title, band_name, item_url, tags, and overlap_count
        """
        self._begin_call()
        # Get supporters from the album
        if progress_callback:
            progress_callback("Extracting supporters from album page...", 0, 0, 0)
        supporters = extract_supporters(item_url)
        self._fan_ids = supporter_fan_ids(item_url)

        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            if event_callback:
                event_callback({"type": "supporters", "supporters": [], "total": 0})
            return []

        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)

        # Select random supporters
        if len(supporters) > num_supporters:
            selected_supporters = random.sample(supporters, num_supporters)
        else:
            selected_supporters = supporters
        
        if progress_callback:
            progress_callback(f"Checking {len(selected_supporters)} random supporters...", len(selected_supporters), len(selected_supporters), 0)

        if event_callback:
            event_callback({
                "type": "supporters",
                "supporters": list(selected_supporters),
                "total": len(selected_supporters),
            })

        # Seed identity, computed once BEFORE the fetch loop so the
        # supporter_done emit (inside the loop) can drop the seed from the
        # animation's cloud nodes — by id and by normalized url. The
        # id-based pop / url backstop on the counts still happens below
        # (after the loop); this is purely cosmetic for the emit.
        seed_id = extract_item_id(item_url)
        seed_key = normalize_item_url(item_url)

        # Get items from selected supporters
        all_items = []
        start_time = time.time()
        total_supporters = len(selected_supporters)
        completed_count = 0
        completed_lock = Lock()

        # Curl-first: no driver pool init up front.
        if progress_callback:
            progress_callback(
                f"Fetching items from {total_supporters} supporters...",
                0,
                total_supporters,
                0,
            )

        def fetch_supporter_items(supporter):
            """Fetch items for one supporter: collection, plus wishlist when
            ``use_wishlist`` is set (distinct union, deduped per supporter).
            Union — not replace — so the flag means "also mine wishlists, not
            just collections" consistently with get_recommendations and the UI.
            """
            started[supporter] = time.time()
            try:
                items = self._get_supporter_items_curl_first(supporter, "collection_data")
                if use_wishlist:
                    items = list(dict.fromkeys(
                        items + self._get_supporter_items_curl_first(supporter, "wishlist_data")
                    ))
                return items, supporter
            except Exception:
                return [], supporter

        started: Dict[str, float] = {}
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(_resolve_supporter_concurrency(), total_supporters) if total_supporters else 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_items, supporter): supporter
                for supporter in selected_supporters
            }
            
            # Block on futures.wait instead of polling+sleep. The per-future
            # timeout counts from when the worker STARTS (recorded in the
            # worker): requests are paced, so a future can legitimately wait
            # in the queue far longer than it runs. Each request is already
            # bounded by its own wall-clock deadline.
            pending = set(future_to_supporter.keys())
            max_future_time = 150

            while pending:
                done, _still_pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

                for future in done:
                    supporter = future_to_supporter[future]
                    try:
                        items, supporter = future.result(timeout=1)
                        with completed_lock:
                            all_items.extend(items)
                            completed_count += 1

                            if progress_callback:
                                elapsed = time.time() - start_time
                                avg_time = elapsed / completed_count if completed_count > 0 else 2.0
                                remaining = total_supporters - completed_count
                                estimated_seconds = avg_time * remaining
                                item_type = "purchases + wishlist" if use_wishlist else "purchases"
                                progress_callback(
                                    f"Fetched {len(items)} {item_type} from {supporter} ({completed_count}/{total_supporters})...",
                                    completed_count,
                                    total_supporters,
                                    int(estimated_seconds)
                                )

                            if event_callback:
                                items_meta = []
                                for iid in items:
                                    info = self.item_cache.get(iid) or {}
                                    # Drop the seed from the emitted cloud
                                    # nodes too — by id and by normalized url.
                                    if iid == seed_id or (
                                        normalize_item_url(info.get("item_url", "")) == seed_key
                                    ):
                                        continue
                                    items_meta.append({
                                        "id": iid,
                                        "title": info.get("item_title", ""),
                                        "band": info.get("band_name", ""),
                                        "src": "collection",
                                    })
                                event_callback({
                                    "type": "supporter_done",
                                    "supporter": supporter,
                                    "index": completed_count,
                                    "total": total_supporters,
                                    "items": items_meta,
                                })
                    except Exception:
                        with completed_lock:
                            completed_count += 1
                            if progress_callback:
                                progress_callback(
                                    f"Error from {supporter} ({completed_count}/{total_supporters})...",
                                    completed_count,
                                    total_supporters,
                                    0
                                )

                # Sweep for futures that have exceeded max_future_time.
                now = time.time()
                timed_out = {
                    f for f in pending - done
                    if now - started.get(future_to_supporter[f], now) > max_future_time
                }
                for future in timed_out:
                    future.cancel()
                    with completed_lock:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(
                                f"Timeout from {future_to_supporter[future]} ({completed_count}/{total_supporters})...",
                                completed_count,
                                total_supporters,
                                0
                            )

                pending -= done
                pending -= timed_out
        
        if not all_items:
            if progress_callback:
                progress_callback("No items found.", total_supporters, total_supporters, 0)
            return []
        
        # Seed id already extracted once before the fetch loop (seed_id) so
        # the supporter_done emit could exclude it; reuse it here to drop the
        # seed from the counts.
        original_item_id = seed_id

        # Count item occurrences (for min_overlap filtering)
        item_counts = Counter(all_items)

        # Remove the original item from counts
        if original_item_id and original_item_id in item_counts:
            item_counts.pop(original_item_id)

        # URL-normalized backstop for the id-based pop above: extract_item_id
        # can fail (curl 403, page-structure shift) and return None, leaving
        # the seed in the counts. Drop any id whose normalized item_url
        # matches the seed url, before min_overlap filtering and sampling
        # (seed_key computed once before the fetch loop).
        item_counts = Counter({
            iid: cnt for iid, cnt in item_counts.items()
            if normalize_item_url((self.item_cache.get(iid) or {}).get("item_url", "")) != seed_key
        })

        # Filter by min_overlap if specified, with fallback if enabled
        final_overlap = None
        if min_overlap is not None and min_overlap > 1:
            current_overlap = min_overlap
            filtered_items = {}
            
            # Try progressively lower overlap requirements if fallback is enabled
            while current_overlap >= 1:
                filtered_items = {
                    item_id: count
                    for item_id, count in item_counts.items()
                    if count >= current_overlap
                }
                
                # Check if we have enough items at this overlap level
                if filtered_items and len(filtered_items) >= num_items:
                    # Found enough items with current overlap requirement
                    final_overlap = current_overlap
                    if current_overlap < min_overlap and progress_callback:
                        progress_callback(
                            f"Found {len(filtered_items)} items with overlap >= {current_overlap} (fallback from {min_overlap})",
                            total_supporters,
                            total_supporters,
                            0
                        )
                    break
                
                # Not enough items found, try lower overlap if fallback enabled
                if use_fallback and current_overlap > 1:
                    if filtered_items:
                        # Some items found but not enough
                        if progress_callback:
                            progress_callback(
                                f"Found {len(filtered_items)} items with overlap >= {current_overlap} (need {num_items}), trying overlap >= {current_overlap - 1}...",
                                total_supporters,
                                total_supporters,
                                0
                            )
                    else:
                        # No items found at this level
                        if progress_callback:
                            progress_callback(
                                f"No items with overlap >= {current_overlap}, trying overlap >= {current_overlap - 1}...",
                                total_supporters,
                                total_supporters,
                                0
                            )
                    current_overlap -= 1
                else:
                    # No fallback or reached minimum
                    final_overlap = current_overlap
                    if filtered_items:
                        # Some items found but not enough and fallback disabled
                        if progress_callback:
                            progress_callback(
                                f"Found {len(filtered_items)} items with overlap >= {min_overlap} (need {num_items}).",
                                total_supporters,
                                total_supporters,
                                0
                            )
                        # Use what we have (will return fewer items than requested)
                        break
                    else:
                        # No items found
                        if progress_callback:
                            progress_callback(
                                f"No items found in at least {min_overlap} collections.",
                                total_supporters,
                                total_supporters,
                                0
                            )
                        return []
            
            item_counts = filtered_items
        elif min_overlap == 1:
            final_overlap = 1
        
        # Select random items
        unique_items = list(item_counts.keys())
        if len(unique_items) > num_items:
            selected_item_ids = random.sample(unique_items, num_items)
        else:
            selected_item_ids = unique_items
        
        if progress_callback:
            if final_overlap is not None and final_overlap != min_overlap:
                progress_callback(
                    f"Selected {len(selected_item_ids)} random items (using overlap >= {final_overlap}, requested >= {min_overlap}).",
                    total_supporters,
                    total_supporters,
                    0
                )
            else:
                progress_callback(f"Selected {len(selected_item_ids)} random items.", total_supporters, total_supporters, 0)

        if event_callback:
            top_meta = []
            for iid in selected_item_ids:
                info = self.item_cache.get(iid) or {}
                top_meta.append({
                    "id": iid,
                    "item_url": info.get("item_url", ""),
                    "title": info.get("item_title", ""),
                    "band": info.get("band_name", ""),
                    "supporters_count": item_counts.get(iid, 0),
                })
            event_callback({
                "type": "ranked",
                "min_supporters": (min_overlap or 1),
                "top": top_meta,
            })

        # Build result list with metadata
        results = []
        for item_id in selected_item_ids:
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info['overlap_count'] = item_counts.get(item_id, 0)
                if final_overlap is not None:
                    item_info['final_overlap'] = final_overlap
                results.append(item_info)
            else:
                # Fallback if metadata not in cache
                result_item = {
                    'item_id': item_id,
                    'item_title': 'Unknown Title',
                    'band_name': 'Unknown Artist',
                    'item_url': f"https://bandcamp.com/album/{item_id}",
                    'tags': [],
                    'overlap_count': item_counts.get(item_id, 0),
                }
                if final_overlap is not None:
                    result_item['final_overlap'] = final_overlap
                results.append(result_item)
        
        return results

    def close(self):
        """Nothing to release: the fallback browser is process-wide and quits
        itself once unused for ``BANDCAMP_BROWSER_IDLE_S`` (see
        bandcamp_recommender.browser; ``browser.close()`` quits it now).
        Quitting it here made the caller's next page read, often moments
        later, pay a fresh Chrome start."""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

