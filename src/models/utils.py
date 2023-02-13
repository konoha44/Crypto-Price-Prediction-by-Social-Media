"""Functions for work with models."""
from typing import Any, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.data.utils import compute_split_indexes, process_data
from src.models.artifacts import save_object
from src.models.metrics import bin_tr_by_fscore


def train_test_split_custom(
    dataset: pd.DataFrame,
    train_indexes,
    test_indexes,
) -> Tuple[Any, Any, Any, Any]:
    """Split data for train and test."""
    x_train, x_test = dataset.iloc[train_indexes]["text"], dataset.iloc[test_indexes]["text"]
    y_train, y_test = dataset.iloc[train_indexes]["target"], dataset.iloc[test_indexes]["target"]
    return x_train, x_test, y_train, y_test


def train(x_train, y_train: pd.Series, x_test, *args, **qwargs) -> Tuple[Any, Any]:
    """Train, save, predict model for train,test data."""
    model = LogisticRegression(penalty="l1", solver="liblinear", **qwargs)
    model.fit(x_train, y_train)
    save_object(model, "model")

    train_preds = model.predict_proba(x_train)[:, 1]
    test_pred = model.predict_proba(x_test)[:, 1]
    return train_preds, test_pred


def train_pipeline_iteration(  # noqa: WPS210
    prepared_dataset: pd.DataFrame,
    train_indexes: int,
    test_indexes: int,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Train and get result by one crosval step.

    Parameters
    ----------
    train_indexes : int
        index of the last observation for training
    test_indexes : int
        index of observation for testing

    Returns
    -------
    Tuple[Any, Any, Any, Any, Any]
        prediction, bin threshold, verdict, true_value, fb max value for training set
    """
    x_train, x_test, y_train, y_test = train_test_split_custom(prepared_dataset, train_indexes, test_indexes)
    x_train_vec, x_test_vec = process_data(x_train, x_test)

    preds_train, preds_test = train(x_train_vec, y_train, x_test_vec)
    fb_max, fb_bin_tr = bin_tr_by_fscore(preds_train, y_train)
    save_object(float(fb_bin_tr), "bin_threshold")

    return preds_test[0], fb_bin_tr, preds_test[0] >= fb_bin_tr, y_test.iloc[0], fb_max


def train_pipeline_all_iterations(dataset: pd.DataFrame) -> pd.DataFrame:
    """Train and get result by all crosval steps."""
    train_iter_results = []
    for train_indexes, test_indexes in tqdm(compute_split_indexes(dataset)):
        train_iter_results.append(train_pipeline_iteration(dataset, train_indexes, test_indexes))

    train_pred_results = pd.DataFrame(
        train_iter_results,
        columns=["pred", "bin_treshhold", "verdict", "target", "fb_max"],
    )
    return train_pred_results
