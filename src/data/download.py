"""Module for downloading text from social meadia."""
import pandas as pd
import yfinance as yf
from dotenv import dotenv_values

from src.sources.twitter import Twitter


# TODO remove hardcode for Twitter
def download_data_from_source(source: Twitter, username: str) -> pd.DataFrame:
    """Download data from source."""
    source.connect()
    return source.download(username)


def download_social_data(source_type: str, username: str) -> pd.DataFrame:
    """Download social media data by username.

    Parameters
    ----------
    source_type : str
        social media source ("twitter")
    username : str
        user name ("cz_binance")

    Returns
    -------
    pd.DataFrame
        downloaded data from source
    """
    config = dotenv_values()
    if source_type.lower() == "twitter":
        source = Twitter({"bearer_token": config["TWITTER_API_APP_BEARER_TOKEN"]})
    return download_data_from_source(source, username)


def download_trading_data(
    ticker: str,
    start_date: pd.Timestamp,
    interval: str = "1d",
    val_col: str = "Close",
) -> pd.DataFrame:
    """Download trading data from yfinance.

    Parameters
    ----------
    ticker : str
        pair of currencies (BTC-USD)
    start_date : pd.Timestamp
        data starts from this date
    interval : str, optional
        size of candles , by default "1d"
    val_col : str, optional
        useful collumn to modeling, by default "Close"

    Returns
    -------
    pd.DataFrame
        trading data
    """
    finance_data = yf.download(tickers=ticker, start=start_date, interval=interval)
    return finance_data[[val_col]].rename(columns={val_col: "value"})
