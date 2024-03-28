"""
Docs:
1. https://pypi.org/project/yfinance/
2. https://python-yahoofinance.readthedocs.io/en/latest/
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas
import pandas as pd
import yfinance as yf
from pandas import DataFrame
from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket

_DATE_FORMAT = "%Y-%m-%d"
_DATE_TIME_FORMAT = f"{_DATE_FORMAT}T%X"


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    """
    The purpose of this class is to prevent too many requests to Yahoo API
    """

    def __getstate__(self):
        pass


def download_stock_history(ticker_names: str, period: str, session: Session):
    """
    :param ticker_names: Examples: "VOO APPL" or ["VOO", "APPL"]
    :param period: Examples: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    :param session: session object to be used for all requests
    :return: nothing, just writes Panda's data frame to CSV
    """
    df_list = []
    ticker_name_list = ticker_names.strip('"').split(" ")
    for ticker in ticker_name_list:
        data: DataFrame = yf.download(
            ticker, group_by="Ticker", period=period, actions=True, session=session
        )
        data.insert(loc=0, column="Date", value=data.index.strftime(_DATE_TIME_FORMAT))
        data.insert(loc=1, column="Ticker", value=ticker)

        df_list.append(data)

    df = pd.concat(df_list)
    df = df.map(_convert_nan_to_zero)

    file_name = make_file_name()

    output_dir = "output"
    _create_dir(output_dir)

    df.to_csv(f"{output_dir}/{file_name}.csv", index=False)

    df = df.rename(columns={
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjClose",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stockSplits",
        "Capital Gains": "capitalGains"
    }, errors="raise")
    df.to_json(f"{output_dir}/{file_name}.json", orient="records")

    _output_to_separate_files(df, output_dir)


def _output_to_separate_files(df: DataFrame, output_dir: str):
    """
    Output data frame to separate json files with timestamp appended at the beginning.
    """
    for timestamp, row in df.iterrows():
        timestamp: pandas.Timestamp
        timestamp_as_str = timestamp.strftime(_DATE_FORMAT)
        next_file_name = f"{timestamp_as_str}-stock_history.json"
        ticker = row.get("ticker")

        ticker_dir = f"{output_dir}/{ticker}"
        _create_dir(ticker_dir)

        row_data = row.to_dict()
        file_path = f"{ticker_dir}/{next_file_name}"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(row_data, file)


def _create_dir(directory_name: str):
    """
    Create directory if it does not exist.
    """
    is_exist = os.path.exists(directory_name)
    if not is_exist:
        os.makedirs(directory_name)


def _convert_nan_to_zero(value: float):
    """
    Prevent output nan value to json files.
    """
    if pd.isna(value):
        return 0
    return value


def make_file_name():
    """
    :param df: data frame ingested with Yahoo Finance data
    :return: None
    """
    zone_info = ZoneInfo("Europe/Minsk")
    now = datetime.now(zone_info)
    now_as_string = now.strftime("%Y-%m-%dT%H-%M-%S%Z")
    return f"stock_history-{now_as_string}"


def main():
    """
    Download stock history and output data to CSV and json files.
    """
    yf.enable_debug_mode()

    ticker_names_env = os.environ.get("TICKER_NAMES", "VOO")
    stock_history_period = os.environ.get("STOCK_HISTORY_PERIOD", "1mo")

    # max 2 requests per 5 seconds
    request_limit = 2
    requests_interval = Duration.SECOND * 5

    cached_limiter_session = CachedLimiterSession(
        limiter=Limiter(RequestRate(request_limit, requests_interval)),
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("./yfinance.cache"),
    )

    # show_tickers_info(ticker_names_env, stock_history_period, cached_limiter_session)
    download_stock_history(ticker_names_env, stock_history_period, cached_limiter_session)


if __name__ == "__main__":
    main()
