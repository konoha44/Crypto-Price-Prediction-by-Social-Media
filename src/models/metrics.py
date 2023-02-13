"""Module for metrics."""
from typing import Tuple

import numpy as np
from numpy import typing as npt
from sklearn.metrics import fbeta_score


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
