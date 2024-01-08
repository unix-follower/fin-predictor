import pandas as pd
import yfinance as yf
from pandas import DataFrame
from pyrate_limiter import Duration, RequestRate, Limiter
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
import os
from datetime import datetime
from zoneinfo import ZoneInfo


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


def show_tickers_info(ticker_names: str, period: str, session: Session):
    tickers = yf.Tickers(ticker_names, session)

    for ticker_name in tickers.tickers:
        ticker = tickers.tickers[ticker_name]

        hist = ticker.history(period=period)
        print(hist)

        print(ticker.history_metadata)

        print(ticker.actions)
        print(ticker.dividends)
        print(ticker.splits)
        print(ticker.capital_gains)

        ticker.get_shares_full(start="2022-01-01", end=None)

        print(ticker.income_stmt)
        print(ticker.quarterly_income_stmt)
        print(ticker.balance_sheet)
        print(ticker.quarterly_balance_sheet)
        print(ticker.cashflow)
        print(ticker.quarterly_cashflow)
        print(ticker.major_holders)
        print(ticker.institutional_holders)
        print(ticker.mutualfund_holders)
        print(ticker.earnings_dates)
        print(ticker.isin)
        print(ticker.options)
        print(ticker.news)

        try:
            opt = ticker.option_chain("2023-12-29")
            print(opt.calls)
            print(opt.puts)
        except ValueError:
            print(f"Option chain for {ticker_name} is not found. Skipping.")

    return tickers


def download_stock_history(ticker_names: str, period: str, session: Session):
    df_list = list()
    ticker_name_list = ticker_names.strip('"').split(" ")
    for ticker in ticker_name_list:
        data: DataFrame = yf.download(ticker, group_by="Ticker", period=period, actions=True, session=session)
        data.insert(loc=0, column="Ticker", value=ticker)

        df_list.append(data)

    df = pd.concat(df_list)

    zone_info = ZoneInfo("Europe/Minsk")
    now = datetime.now(zone_info)
    now_as_string = now.strftime("%Y-%m-%dT%H-%M-%S%Z")
    df.to_csv(f"stock_history-{now_as_string}.csv")


if __name__ == "__main__":
    yf.enable_debug_mode()

    ticker_names = os.environ.get("TICKER_NAMES", "VOO")
    stock_history_period = os.environ.get("STOCK_HISTORY_PERIOD", "10y")

    # max 2 requests per 5 seconds
    request_limit = 2
    requests_interval = Duration.SECOND * 5

    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(request_limit, requests_interval)),
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )

    # show_tickers_info(ticker_names, stock_history_period, session)
    download_stock_history(ticker_names, stock_history_period, session)
