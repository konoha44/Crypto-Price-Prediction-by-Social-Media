"""Functions for work with data."""
from typing import List, Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit

from src.data.transformation import TextDataTransformation
from src.models.artifacts import save_object


def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data."""
    twitter_text_transformer = TextDataTransformation()
    dataset["text"] = twitter_text_transformer.transform(dataset["text"])
    return dataset


def process_data(x_train, x_test) -> Tuple[csr_matrix, csr_matrix]:
    """Preprocess data."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    save_object(vectorizer, "vectorizer")

    return x_train_vec, x_test_vec


def compute_split_indexes(dataset: pd.DataFrame) -> List[List[int]]:
    """Compute indexes from TimeSeriesSplit."""
    index = find_enough_index(dataset)
    ts_split = TimeSeriesSplit(n_splits=dataset.shape[0] - index, gap=0)
    return list(ts_split.split(dataset))


def find_enough_index(dataset: pd.DataFrame) -> int:
    """Find a first index where number of unique targets more than 1."""
    for index in range(10, dataset.shape[0], 10):
        nunique = dataset[:index]["target"].nunique()
        if nunique > 1:
            enough_index = index
            break
    else:
        raise StopIteration("nunique is only one in dataset")
    return enough_index
