"""Merge news sentiment with price data and build features/labels for prediction."""

import pandas as pd


def build_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-headline sentiment scores into one row per calendar day."""
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    return (
        news_df.groupby("date")
        .agg(
            avg_sentiment=("sentiment", "mean"),
            news_count=("sentiment", "count"),
            positive_ratio=("sentiment", lambda x: (x == 1).mean()),
        )
        .reset_index()
    )


def merge_price_sentiment(daily_sentiment: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join daily sentiment with daily price/return on trading dates."""
    merged = pd.merge(
        daily_sentiment, price_df[["date", "Close", "return"]], on="date", how="inner"
    )
    return merged.dropna()


def add_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Build lagged price/sentiment features and the next-day direction label.

    All price features use `.shift(1)` so they only look at past data. Sentiment
    features use *today's* score to predict *tomorrow's* direction, which is valid
    because the news is published before tomorrow's open.
    """
    df = merged.copy().sort_values("date").reset_index(drop=True)

    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["ma5"] = df["Close"].shift(1).rolling(5).mean()
    df["volatility_5d"] = df["return"].shift(1).rolling(5).std()

    df["sentiment_today"] = df["avg_sentiment"]
    df["sentiment_lag1"] = df["avg_sentiment"].shift(1)
    df["news_count_today"] = df["news_count"]
    df["sentiment_ma3"] = df["avg_sentiment"].rolling(3).mean()

    df["target"] = (df["return"].shift(-1) > 0).astype(int)  # 1 = up, 0 = down

    return df.dropna().reset_index(drop=True)


FEATURES_NO_SENTIMENT = ["return_lag1", "return_lag2", "ma5", "volatility_5d"]
FEATURES_WITH_SENTIMENT = FEATURES_NO_SENTIMENT + [
    "sentiment_today", "sentiment_lag1", "news_count_today", "sentiment_ma3",
]


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split chronologically (not randomly) so the test set is strictly later in time."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]
