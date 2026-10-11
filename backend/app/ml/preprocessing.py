"""
Text Preprocessing Module for NewsMap.

This module provides reusable, deterministic functions for text cleaning and
normalization designed for unsupervised news topic discovery.

Design Decisions:
1. Whitespace Normalization: Collapses consecutive whitespace (spaces, tabs, newlines)
   into single spaces and strips leading/trailing margins.
2. Case Normalization: Lowercases text consistently by default, with an option to preserve case.
3. Meaningful Punctuation Preservation:
   - Intra-word hyphens (e.g. 'peer-to-peer', 'cost-cutting', 't-mobile') are preserved
     to avoid splitting compound concepts.
   - Intra-word apostrophes / contractions (e.g. "didn't", "won't", "company's") are preserved
     to retain critical semantic negation and context.
   - Intra-number periods (e.g. '3.8', '95.5') and currency symbols/percentages (e.g. '£5.7m', '15.4%')
     are preserved to maintain numeric integrity.
4. Formatting Noise Removal:
   - Structural noise (brackets, braces, parentheses, quotes, colons, semicolons, exclamation marks,
     ampersands, and symbol decorations) is replaced with whitespace to prevent token concatenation.
   - HTML entities are unescaped and HTML tags (if encountered) are safely stripped.
   - Unicode characters are normalized using NFKC standard form.
5. Vocabulary Integrity:
   - No aggressive stemming or lemmatization is applied in this initial version.
   - Stop words are NOT removed indiscriminately, keeping raw semantic flow intact.
"""

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional
import pandas as pd


def clean_text(
    text: Any,
    preserve_case: bool = False,
    preserve_numbers: bool = True,
) -> str:
    """
    Clean and normalize raw text for topic discovery and NLP pipelines.

    Args:
        text: Raw text string, or non-string object (e.g. float NaN, None, int).
        preserve_case: If True, skips lowercasing. Defaults to False.
        preserve_numbers: If True, preserves numeric digits and decimals. Defaults to True.

    Returns:
        Cleaned, normalized text string. Returns empty string for null/empty inputs.
    """
    # 1. Handle null, non-string, and empty inputs safely
    if text is None:
        return ""
    if isinstance(text, bool):
        return ""
    if isinstance(text, float):
        import math
        if math.isnan(text):
            return ""
    if not isinstance(text, str):
        if hasattr(text, "__iter__") and not isinstance(text, (bytes, bytearray)):
            return ""
        text = str(text)

    if not text.strip():
        return ""

    # 2. Unescape HTML entities (&amp; -> &, &pound; -> £, etc.)
    text = html.unescape(text)

    # 3. Strip HTML tags if present (e.g. <p>, <br/>)
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Unicode normalization (NFKC decomposes compatibility chars and recomposes canonically)
    text = unicodedata.normalize("NFKC", text)

    # 5. Standardize curly quotes and typographic dashes to ASCII equivalents
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
        .replace("—", "-")
        .replace("–", "-")
    )

    # 6. Lowercase normalization
    if not preserve_case:
        text = text.lower()

    # 7. Remove unwanted structural punctuation and formatting symbols
    # Replace characters like (), [], {}, <>, ,, ;, :, !, ?, ", #, *, +, =, ~, ^, _, /, \, |, @, &
    # with spaces so neighboring words do not glue together.
    # We place ']' first in the bracket character set to avoid escaping ambiguities.
    text = re.sub(r"[]()[{}<>,;:!?\"#*+=~^_/\\|@&[]", " ", text)

    # 8. Clean isolated/trailing hyphens, preserving intra-word hyphens (e.g. peer-to-peer)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"(?<![a-zA-Z0-9])-", " ", text)
    text = re.sub(r"-(?![a-zA-Z0-9])", " ", text)

    # 9. Clean isolated/trailing apostrophes, preserving intra-word apostrophes (e.g. didn't)
    text = re.sub(r"(?<![a-zA-Z])'", " ", text)
    text = re.sub(r"'(?![a-zA-Z])", " ", text)

    # 10. Clean isolated/sentence periods, preserving intra-number decimals (e.g. 3.8)
    text = re.sub(r"(?<!\d)\.", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)

    # 11. Optionally remove numeric tokens if explicitly requested
    if not preserve_numbers:
        text = re.sub(r"\b\d+(\.\d+)?%?\b", " ", text)

    # 12. Normalize whitespace (tabs, newlines, multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_series(
    series: pd.Series,
    preserve_case: bool = False,
    preserve_numbers: bool = True,
) -> pd.Series:
    """
    Apply text cleaning across a pandas Series.

    Args:
        series: Pandas Series containing text entries.
        preserve_case: Whether to preserve original casing.
        preserve_numbers: Whether to preserve numeric digits.

    Returns:
        Pandas Series of cleaned text strings.
    """
    return series.apply(
        lambda x: clean_text(
            x,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
        )
    )


def extract_tokens(text: str) -> List[str]:
    """
    Tokenize preprocessed text into whitespace-delimited words.

    Args:
        text: Preprocessed text string.

    Returns:
        List of individual tokens.
    """
    if not text:
        return []
    return text.split()


def get_text_statistics(text: str) -> Dict[str, int]:
    """
    Calculate summary character and word counts for a text string.

    Args:
        text: Text string.

    Returns:
        Dictionary containing character count and word count.
    """
    if not text:
        return {"char_count": 0, "word_count": 0}
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
    }
