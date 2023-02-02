"""Module for working with different artifacts."""
from typing import Any

import dill


def save_object(artifact: Any, name: str) -> None:
    """Save python object to pickle file.

    Parameters
    ----------
    artifact : Any
        a Python object
    name : str
        name of file ('{name}.pkl')
    """
    with open(f"./models/{name}.pkl", "wb") as binary_file:
        dill.dump(artifact, binary_file)


def load_object(name: str) -> Any:
    """Load python object from pickle file.

    Parameters
    ----------
    name : str
        name of file ('{name}.pkl')

    Returns
    -------
    Any
        a Python object
    """
    with open(f"./models/{name}.pkl", "rb") as binary_file:
        binary_object = dill.load(binary_file)
    return binary_object
