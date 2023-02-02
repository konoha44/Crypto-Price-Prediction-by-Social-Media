"""Module for creating base source class."""

from time import ctime
from typing import Dict, Union

from loguru import logger

logger.add("source_step_{time}.log")


class BaseSource:
    """Base class for sources."""

    def __init__(self, creds: dict, **qwargs):
        self.creds = creds
        self.qwargs = qwargs

    def make_final_response(self, **final_response) -> Dict[str, Union[str, int, float]]:
        """Make prepared final responce."""
        final_response.setdefault("status", "ok")
        final_response.setdefault("datetime", ctime())
        final_response.setdefault("result", None)
        return final_response

    def log_step(self, message: str) -> None:
        """Log messages."""
        logger.info(f"{self.__class__} {message}")
