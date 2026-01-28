"""
Tokenizer utilities and text normalization.
"""

import unicodedata


def normalize_text(text: str, *, lowercase: bool = True, nfkc: bool = True) -> str:
    """
    Normalize text for tokenization.
    
    Args:
        text: Input text string.
        lowercase: Whether to convert to lowercase.
        nfkc: Whether to apply Unicode NFKC normalization.
    
    Returns:
        Normalized text string.
    """
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    return text
