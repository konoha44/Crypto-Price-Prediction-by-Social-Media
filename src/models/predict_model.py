"""Module for model inference (predict)."""
from typing import Tuple

import numpy as np

from src.data.transformation import TextDataTransformation
from src.models.artifacts import load_object


def preprocess(text: str) -> str:
    """Preprocess string for vectorizer.

    Parameters
    ----------
    text : str
        raw socia media text

    Returns
    -------
    str
        preprocessed text
    """
    twitter_text_transformer = TextDataTransformation()
    return twitter_text_transformer.transform(text)[0]


def model_inference(text: str) -> Tuple[np.float32, bool]:
    """Inference model for text input.

    Parameters
    ----------
    text : str
        raw socia media text

    Returns
    -------
    Tuple[np.float32, bool]
        model predictions, final verdict
    """
    model = load_object("model")
    vectorizer = load_object("vectorizer")
    bin_threshold = load_object("bin_threshold")

    vectorizer.preprocessor = preprocess
    input_data = vectorizer.transform([text])

    prediction = model.predict_proba(input_data)[0, 1]
    return prediction, prediction >= bin_threshold


if __name__ == "__main__":
    input_text = input("Enter text\n")  # noqa: WPS421
    if not input_text:
        input_text = "bitcoin is bad coin"
    model_result = model_inference(input_text)
    print(input_text, model_result)  # noqa: WPS421
