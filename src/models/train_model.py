"""Module for train model and make artifacts."""
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.utils import preprocess_data
from src.models.utils import train_pipeline_all_iterations

logger.add("train_model_{time}.log")


def load_data() -> pd.DataFrame:
    """Load data from disc."""
    return pd.read_pickle("./data/interim/dataset.pkl")


def train_pipeline(dataset: Optional[pd.DataFrame] = None, **args) -> pd.DataFrame:
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

    train_pred_results = train_pipeline_all_iterations(dataset)

    train_pred_results.to_csv("./models/train_pred_results.csv", index=False)
    logger.info(train_pred_results)

    return train_pred_results


if __name__ == "__main__":
    train_pipeline()
