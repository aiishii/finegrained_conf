"""Utility functions shared across experiment modules."""

import re
import math
import unicodedata
import string


def _fullwidth_to_halfwidth(s: str) -> str:
    """Convert fullwidth characters to halfwidth."""
    return unicodedata.normalize('NFKC', s)


_NUM_PAT = re.compile(r'\d+')


def _kan_num_to_arabic(m):
    """Convert Japanese fullwidth numbers to Arabic numerals."""
    jp2num = str.maketrans('０１２３４５６７８９', '0123456789')
    return m.group(0).translate(jp2num)


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Args:
        text: The text to normalize

    Returns:
        Normalized text in lowercase with punctuation removed
    """
    if not text:
        return ''
    t = _fullwidth_to_halfwidth(text.strip())
    if t.casefold().startswith(('はい', 'yes')):
        t = 'YES'
    elif t.casefold().startswith(('いいえ', 'no')):
        t = 'NO'

    t = _NUM_PAT.sub(_kan_num_to_arabic, t)
    t = re.sub(r'\s*[\(（].+?[\)）]\s*', '', t)
    t = re.sub(r'[『』「」]', '', t)
    t = t.translate(str.maketrans('', '', string.punctuation + '、。'))
    return ' '.join(t.split()).lower()


def is_same_ans(a1: str, a2: str) -> bool:
    """Check if two answers are semantically equivalent.

    Args:
        a1: First answer
        a2: Second answer

    Returns:
        True if answers are equivalent after normalization
    """
    return normalize_answer(a1) == normalize_answer(a2)


def entropy_nat(p_hat: dict[str, float]) -> float:
    """Calculate natural entropy of a probability distribution.

    Args:
        p_hat: Dictionary mapping items to their probabilities

    Returns:
        Natural entropy H = -Σ p log(p)
    """
    return -sum(p * math.log(p) for p in p_hat.values() if p > 0)
