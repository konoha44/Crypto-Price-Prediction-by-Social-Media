"""Module for data transformations."""
from typing import Callable, Generator, List, Union

import pandas as pd

from src.data.preprocessing import PREPROCESSING_FUNCTIONS


class TextDataTransformation:
    """Class for text transformation."""

    def __init__(self, functions: List[Callable] = PREPROCESSING_FUNCTIONS, **qwargs):
        self.functions = functions
        self.qwargs = qwargs

    def transform(self, texts: Union[pd.Series, List[str], str]) -> List[str]:
        """Apply transformation(s) for text(s).

        Parameters
        ----------
        texts : Union[pd.Series, List[str], str]
            raw text(s)

        Returns
        -------
        List[str]
            transformed text(s)
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.to_list()
        texts = list(self.preprocess_iter(texts))
        return texts

    def preprocess_iter(self, texts: Union[List[str], str]) -> Generator[str, None, None]:
        """Apply function(s) for input text(s) step-by-step.

        Parameters
        ----------
        texts : Union[List[str], str]
            input raw text(s)

        Yields
        ------
        Generator[str, None, None]
            preprocessed text
        """  # noqa: DAR301
        for text in texts:
            for function in self.functions:
                text = function(text)
            yield text
