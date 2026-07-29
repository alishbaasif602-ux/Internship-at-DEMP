"""
preprocessing.py
----------------
Data preprocessing utilities for time-series forecasting:
- loading
- missing value handling
- normalization
- sliding-window sequence creation
- train/test split (chronological, no shuffling)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_series(csv_path: str, value_col: str = "Close", date_col: str = "Date") -> pd.Series:
    """Load a CSV and return a single univariate time series indexed by date."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    series = df.set_index(date_col)[value_col]
    return series


def handle_missing_values(series: pd.Series) -> pd.Series:
    """Fill missing values via linear interpolation, then forward/backward fill
    any remaining edge NaNs."""
    series = series.interpolate(method="linear")
    series = series.ffill().bfill()
    return series


def normalize(series: pd.Series):
    """Scale values to [0, 1] using MinMaxScaler. Returns (scaled_array, scaler)."""
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled.flatten(), scaler


def create_sequences(data: np.ndarray, window_size: int):
    """
    Sliding-window sequence-to-one transformation.

    Given data = [x0, x1, x2, ..., xn], produces:
      X[i] = [x_i, x_{i+1}, ..., x_{i+window_size-1}]
      y[i] = x_{i+window_size}

    i.e. the model sees `window_size` past points and predicts the single
    next value (sequence-to-one prediction).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_test_split_sequences(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    """Chronological split -- NEVER shuffle time-series data, since that would
    leak future information into the training set."""
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def prepare_dataset(csv_path: str, window_size: int = 30, train_ratio: float = 0.8, value_col: str = "Close"):
    """End-to-end pipeline: load -> clean -> normalize -> window -> split."""
    series = load_series(csv_path, value_col=value_col)
    series = handle_missing_values(series)
    scaled, scaler = normalize(series)
    X, y = create_sequences(scaled, window_size)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, train_ratio)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "series": series,
    }
