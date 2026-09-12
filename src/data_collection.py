"""Fetch raw data: news headlines from FinMind and stock prices from yfinance."""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from FinMind.data import DataLoader


def collect_news(stock_id: str, start_date: str, end_date: str, output_dir: str = "data") -> str:
    """Download daily news headlines for a stock from FinMind, one day at a time.

    Resumes from the last saved date if output_path already exists, since FinMind's
    free tier is rate-limited to 300 requests/hour (12s sleep between requests).
    Returns the path to the resulting CSV.
    """
    dl = DataLoader()
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/news_{stock_id}.csv"

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) > 0:
            last_date = pd.to_datetime(existing["date"]).max().strftime("%Y-%m-%d")
            current = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        else:
            current = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        current = datetime.strptime(start_date, "%Y-%m-%d")

    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            daily_news = dl.taiwan_stock_news(stock_id=stock_id, start_date=date_str)
            if len(daily_news) > 0:
                daily_news.to_csv(
                    output_path, mode="a",
                    header=not os.path.exists(output_path),
                    index=False,
                )
            print(f"{stock_id} {date_str}: {len(daily_news)} 則")
            time.sleep(12)  # 300 req/hr free tier -> 3600/300 = 12s/request
        except Exception as e:
            print(f"{stock_id} {date_str} 失敗: {e}, 等待 60 秒後重試...")
            time.sleep(60)
            continue

        current += timedelta(days=1)

    return output_path


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker via yfinance, with a flat column index."""
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["date"] = df["Date"].dt.date
    df["return"] = df["Close"].pct_change()
    return df
