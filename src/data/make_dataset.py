"""Module: Make DataSet for model."""

import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src.data.download import download_social_data, download_trading_data

logger.add("make_dataset_{time}.log")


def compute_targets(soc_med_data: pd.DataFrame, trading_data: pd.DataFrame, target_fraq: float) -> pd.DataFrame:
    """Compute targets by social media and trading data."""
    soc_med_data["created_at_d"] = soc_med_data["created_at"].dt.floor("1d")
    soc_med_data["created_at_d_minus _1_day"] = soc_med_data["created_at_d"] - pd.to_timedelta("1d")

    trading_data["rel_diff"] = trading_data["value"].diff() / trading_data["value"].shift(1)
    # TODO change to not for drop
    trading_data["target"] = (trading_data["rel_diff"] <= target_fraq).astype("int8")
    trading_data = trading_data.reset_index()

    result_dataset = pd.merge(
        trading_data,
        soc_med_data[["text", "created_at_d_minus _1_day", "created_at"]],
        left_on="Date",
        right_on="created_at_d_minus _1_day",
    )
    return result_dataset.sort_values("created_at")


@click.command()
@click.option("--social_source", default="twitter", type=str, help="Social media source (only one)", show_default=True)
@click.option("-query", type=str, help="Social media query (only one @person)")
@click.option("-ticker", type=str, default="BTC-USD", help="Coin for trading data", show_default=True)
@click.option(
    "-target_fraq",
    type=float,
    default="-0.05",
    help="Define drop for relative price diff",
    show_default=True,
)
def main(social_source: str, query: str, ticker: str, target_fraq: float, **qwargs) -> pd.DataFrame:
    """Compute dataset for model.

    Runs data processing scripts to download data from social media and trading data to dataset for model training.

    """
    logger.info(f"Download social media data {social_source} {query}")
    social_media_data = download_social_data(social_source, query)
    trading_start_data = social_media_data["created_at"].min()
    logger.info(f"Download trading data for {ticker} from {trading_start_data} with parameters {qwargs}")
    trading_data = download_trading_data(ticker, trading_start_data, **qwargs)
    logger.info(f"Compute targets for data target_fraq={target_fraq}")
    dataset = compute_targets(social_media_data, trading_data, target_fraq)
    logger.info(f"Save data {dataset.shape} to './data/interim/dataset.pkl")
    dataset.to_pickle("./data/interim/dataset.pkl")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    dataset = main()
