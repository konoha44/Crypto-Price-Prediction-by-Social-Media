"""Module for train model and make artifacts."""
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy import typing as npt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from src.data.transformation import TextDataTransformation
from src.models.artifacts import save_object

logger.add("make_dataset_{time}.log")


def load_data() -> pd.DataFrame:
    """Load data from disc."""
    return pd.read_pickle("./data/interim/dataset.pkl")


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


def compute_split_indexes(dataset: pd.DataFrame) -> List[List[int]]:
    """Compute indexes from TimeSeriesSplit."""
    index = find_enough_index(dataset)
    ts_split = TimeSeriesSplit(n_splits=dataset.shape[0] - index, gap=0)
    return list(ts_split.split(dataset))


def train_test_split(
    dataset: pd.DataFrame,
    train_indexes,
    test_indexes,
) -> Tuple[Any, Any, Any, Any]:
    """Split data for train and test."""
    x_train, x_test = dataset.iloc[train_indexes]["text"], dataset.iloc[test_indexes]["text"]
    y_train, y_test = dataset.iloc[train_indexes]["target"], dataset.iloc[test_indexes]["target"]
    return x_train, x_test, y_train, y_test


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


def train(x_train, y_train: pd.Series, x_test, *args, **qwargs) -> Tuple[Any, Any]:
    """Train, save, predict model for train,test data."""
    model = LogisticRegression(penalty="l1", solver="liblinear", **qwargs)
    model.fit(x_train, y_train)
    save_object(model, "model")

    train_preds = model.predict_proba(x_train)[:, 1]
    test_pred = model.predict_proba(x_test)[:, 1]
    return train_preds, test_pred


def bin_tr_by_fscore(
    preds: npt.NDArray[np.float32],
    targets: npt.NDArray[np.int8],
    step: float = 0.01,
) -> Tuple[np.float16, np.float64]:
    """Find binarization threshold by max fb_score.

    Parameters
    ----------
    preds : npt.NDArray[np.float32]
        model predictions
    targets : npt.NDArray[np.int8]
        target values
    step : float, optional
        Step for bin threshold selection, by default 0.01

    Returns
    -------
    Tuple[np.float16, np.float64]
        fb max score, binarization threshold
    """
    fb_list = [fbeta_score(targets, preds >= prob, beta=1) for prob in np.arange(0.5, step, -step)]
    fb_array = np.array(fb_list, dtype="float16")
    bin_treshhold = np.arange(0.5, step, -step)[fb_array.argmax()]
    return fb_array.max(), bin_treshhold


def train_pipeline(dataset: Optional[pd.DataFrame] = None, **args) -> pd.DataFrame:  # noqa: WPS210
    """Start general training pipeline.

    — Load data
    — Preprocess data
    — Time-Series CrosVal training pipeline
    — Compute metrics


    Parameters
    ----------
    dataset : Optional[pd.DataFrame], optional
        Raw dataset for preprocessiong, by default None

    Returns
    -------
    pd.DataFrame
        dataframe with metrics and predictions for a next day
    """
    if not dataset:
        dataset = load_data()
    dataset = preprocess_data(dataset)
    pred_true = []

    for train_indexes, test_indexes in tqdm(compute_split_indexes(dataset)):
        x_train, x_test, y_train, y_test = train_test_split(dataset, train_indexes, test_indexes)
        x_train_vec, x_test_vec = process_data(x_train, x_test)

        preds_train, preds_test = train(x_train_vec, y_train, x_test_vec)
        fb_max, fb_bin_tr = bin_tr_by_fscore(preds_train, y_train)
        save_object(float(fb_bin_tr), "bin_threshold")

        pred_true.append([preds_test[0], fb_bin_tr, preds_test[0] >= fb_bin_tr, y_test.iloc[0], fb_max])

    pred_true_df = pd.DataFrame(pred_true, columns=["pred", "bin_treshhold", "verdict", "target", "fb_max"])
    logger.info(pred_true_df)
    return pred_true_df


if __name__ == "__main__":
    train_pipeline()
