"""Module for data preprocessing."""
import re
import string
from typing import Callable, List, Set

import stop_words


def word_tokenize(text: str) -> List[str]:
    """Tokenize text via text.split()."""
    return text.split()


def get_all_stopwords() -> Set[str]:
    """Get all stopwords from all languages."""
    stop_words_array: List[str] = []
    for lang in stop_words.AVAILABLE_LANGUAGES:
        stop_words_array.extend(stop_words.get_stop_words(lang))
    return set(stop_words_array)


def to_lower(text: str) -> str:
    """Make lowercase string."""
    return text.lower()


def remove_stopwords(text: str, stopwords: Set[str] = get_all_stopwords()) -> str:  # noqa: WPS404 B008
    """Remove stopwords from text."""
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)


def remove_punctuations(text: str, punctuations=None) -> str:
    r"""Remove punctations ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~’') from text."""
    if punctuations is None:
        punctuations = f"{string.punctuation}’"
    return text.translate(str.maketrans("", "", punctuations))


def remove_links(text: str) -> str:
    """Remove links from texts."""
    return re.sub(r"https?://\S+", "", text)


def remove_mentions(text: str) -> str:
    """Remove mentions (@someone) from text."""
    return re.sub(r"@\w+", "", text)


def remove_short_tobe(text: str) -> str:
    """Remove short form of verb to be ('re, 'm, etc)."""
    short_tobe = {"ll", "d", "s", "re", "t", "ve", "m"}
    quotes = {"‘", "’", "′", "'"}
    quotes_verbs = [quote + verb for verb in short_tobe for quote in quotes]
    pattern = "|".join([f"({cond})" for cond in quotes_verbs])
    return re.sub(pattern, " ", text)


PREPROCESSING_FUNCTIONS: List[Callable] = [
    to_lower,
    remove_links,
    remove_short_tobe,
    remove_mentions,
    remove_punctuations,
    remove_stopwords,
]
