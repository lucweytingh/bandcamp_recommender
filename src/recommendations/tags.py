"""Tag extraction and similarity calculation utilities."""

from collections import Counter
from math import log
from typing import Dict, List, Optional

from .scraper import extract_tags


def normalize_tag(tag: str) -> str:
    """Normalize a tag for comparison (lowercase, strip, handle variations).
    
    Args:
        tag: Tag string to normalize
        
    Returns:
        Normalized tag string
    """
    # Lowercase and strip
    normalized = tag.lower().strip()
    
    # Handle common variations
    variations = {
        'uk': 'united kingdom',
        'u.k.': 'united kingdom',
        'usa': 'united states',
        'u.s.a.': 'united states',
    }
    
    return variations.get(normalized, normalized)


def calculate_tag_similarity(
    original_tags: List[str],
    candidate_tags: List[str],
    tag_frequencies: Optional[Dict[str, int]] = None,
    total_items: int = 1
) -> float:
    """Calculate sophisticated tag similarity score between two tag sets.
    
    Uses TF-IDF weighted Jaccard similarity with tag normalization.
    
    Args:
        original_tags: Tags from the original item
        candidate_tags: Tags from the candidate item
        tag_frequencies: Optional dict of tag -> frequency across all items (for TF-IDF)
        total_items: Total number of items (for TF-IDF calculation)
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not original_tags or not candidate_tags:
        return 0.0
    
    # Normalize tags
    original_set = {normalize_tag(t) for t in original_tags}
    candidate_set = {normalize_tag(t) for t in candidate_tags}
    
    # Calculate intersection and union
    intersection = original_set & candidate_set
    union = original_set | candidate_set
    
    if not union:
        return 0.0
    
    # Basic Jaccard similarity
    jaccard = len(intersection) / len(union)
    
    # If we have tag frequencies, use TF-IDF weighting
    if tag_frequencies and total_items > 1:
        # Calculate weighted similarity
        # Weight each matching tag by its inverse document frequency (IDF)
        # Rare tags that match are more significant
        weighted_score = 0.0
        total_weight = 0.0
        
        for tag in intersection:
            # IDF = log(total_items / (tag_frequency + 1))
            # +1 to avoid division by zero
            tag_freq = tag_frequencies.get(tag, 0)
            idf = log(total_items / (tag_freq + 1))
            weighted_score += idf
            total_weight += idf
        
        # Also weight non-matching tags (penalty for dissimilarity)
        for tag in union - intersection:
            tag_freq = tag_frequencies.get(tag, 0)
            idf = log(total_items / (tag_freq + 1))
            total_weight += idf
        
        # Normalize weighted score
        if total_weight > 0:
            weighted_jaccard = weighted_score / total_weight
            # Combine basic Jaccard with weighted score (weighted average)
            return 0.6 * jaccard + 0.4 * weighted_jaccard
    
    return jaccard

