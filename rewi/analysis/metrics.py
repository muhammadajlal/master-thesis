"""
Distance and error metrics for handwriting recognition evaluation.
"""

# Runtime Levenshtein (raw + normalized)
try:
    import Levenshtein as _lev_lib
    
    def lev_dist(a: str, b: str) -> int:
        """Compute Levenshtein (edit) distance between two strings."""
        return _lev_lib.distance(a, b)

except ImportError:
    def lev_dist(a: str, b: str) -> int:
        """
        Compute Levenshtein (edit) distance between two strings.
        
        Fallback pure-Python implementation when python-Levenshtein is not installed.
        """
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]


def normalized_lev_dist(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein distance (d / max(|a|, |b|)).
    
    Returns a value in [0.0, 1.0] where 0 means identical strings.
    """
    if a == b:
        return 0.0
    d = lev_dist(a, b)
    max_len = max(len(a), len(b), 1)
    return d / max_len


def character_error_rate(pred: str, label: str) -> float:
    """
    Compute character error rate (CER).
    
    CER = edit_distance(pred, label) / len(label)
    """
    if not label:
        return 0.0 if not pred else 1.0
    return lev_dist(pred, label) / len(label)
