"""
Pattern Feature Encoding
Converts PatternResult objects into binary flags and confidence scores for ML models.
"""
import numpy as np

# All 20 pattern types
PATTERN_TYPES = [
    "double_top", "double_bottom",
    "head_shoulders_top", "head_shoulders_bottom",
    "triple_top", "triple_bottom",
    "rounding_top", "rounding_bottom",
    "flag", "pennant", "measured_move",
    "ascending_triangle", "descending_triangle", "symmetrical_triangle",
    "rectangle", "cup_and_handle",
    "broadening_formation", "broadening_top_bottom",
    "diamond", "wedge",
]


def encode_pattern_features(pattern_results):
    """Encode pattern detection results into feature dict for ML models.

    Args:
        pattern_results: list of PatternResult dataclass instances

    Returns:
        dict with ~65 features (20 patterns * 3 fields + 5 summary fields)
    """
    features = {}

    # Per-pattern features
    detected_patterns = {}
    if pattern_results:
        for p in pattern_results:
            name = p.pattern_name
            if name not in detected_patterns or p.confidence > detected_patterns[name].confidence:
                detected_patterns[name] = p

    for ptype in PATTERN_TYPES:
        if ptype in detected_patterns:
            p = detected_patterns[ptype]
            features[f"{ptype}_detected"] = 1
            features[f"{ptype}_direction"] = 1 if p.direction == "bullish" else -1
            features[f"{ptype}_confidence"] = p.confidence
        else:
            features[f"{ptype}_detected"] = 0
            features[f"{ptype}_direction"] = 0
            features[f"{ptype}_confidence"] = 0.0

    # Summary features
    if pattern_results:
        features["any_pattern_detected"] = 1
        features["best_pattern_confidence"] = max(p.confidence for p in pattern_results)
        directions = [1 if p.direction == "bullish" else -1 for p in pattern_results]
        features["pattern_direction_consensus"] = np.sign(sum(directions))
        features["pattern_count"] = len(pattern_results)
        features["avg_pattern_confidence"] = np.mean([p.confidence for p in pattern_results])
    else:
        features["any_pattern_detected"] = 0
        features["best_pattern_confidence"] = 0.0
        features["pattern_direction_consensus"] = 0
        features["pattern_count"] = 0
        features["avg_pattern_confidence"] = 0.0

    return features
