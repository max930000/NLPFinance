"""Next-day direction prediction models (Logistic Regression, Random Forest),
baselines, and paired t-tests comparing accuracy with vs. without sentiment features.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit


def run_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.Series) -> dict:
    """Majority-class and momentum (yesterday's direction) baselines."""
    majority_class = train_df["target"].mode()[0]
    majority_pred = [majority_class] * len(y_test)
    momentum_pred = (test_df["return_lag1"] > 0).astype(int)
    return {
        "majority_pred": majority_pred,
        "momentum_pred": momentum_pred,
        "Baseline(多數類別)": accuracy_score(y_test, majority_pred),
        "Baseline(動量)": accuracy_score(y_test, momentum_pred),
    }


def full_report(y_true, y_pred, name: str) -> dict:
    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "name":
            print(f"{k.capitalize()}: {v:.4f}")
    return metrics


def cross_validate(df: pd.DataFrame, features_a: list, features_b: list, target: str,
                    model_fn, n_splits: int = 5):
    """Time-series cross-validation comparing a feature set without (A) vs. with (B)
    sentiment features, for any sklearn-style classifier factory `model_fn()`.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_a, X_b, y = df[features_a], df[features_b], df[target]
    scores_a, scores_b = [], []

    for train_idx, test_idx in tscv.split(df):
        model_a = model_fn().fit(X_a.iloc[train_idx], y.iloc[train_idx])
        model_b = model_fn().fit(X_b.iloc[train_idx], y.iloc[train_idx])
        scores_a.append(accuracy_score(y.iloc[test_idx], model_a.predict(X_a.iloc[test_idx])))
        scores_b.append(accuracy_score(y.iloc[test_idx], model_b.predict(X_b.iloc[test_idx])))

    return scores_a, scores_b


def paired_ttest(scores_with_sentiment: list, scores_without_sentiment: list):
    """Paired t-test: is the sentiment-feature model's per-fold accuracy significantly
    different from the no-sentiment model's? Returns (t_stat, p_value)."""
    return ttest_rel(scores_with_sentiment, scores_without_sentiment)


def logistic_regression_factory():
    return LogisticRegression(max_iter=1000)


def random_forest_factory(n_estimators=100, max_depth=3, random_state=42):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)


def feature_importance(model, features: list) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
