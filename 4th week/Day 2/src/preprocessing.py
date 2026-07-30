"""
Data preprocessing utilities for the stock-price sequence modeling project.

Pipeline: load CSV -> handle missing values -> MinMax normalize the
'Close' price -> build sliding-window sequences -> chronological
train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_series(csv_path: str, date_col: str = "Date", value_col: str = "Close") -> pd.Series:
    """Load a stock CSV and return a cleaned, missing-value-free Close price series."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    series = df[value_col].astype(float)

    # Handle missing values: forward-fill (carry last known price), then
    # back-fill any leading gaps.
    n_missing = series.isna().sum()
    if n_missing > 0:
        series = series.ffill().bfill()

    return series


def make_sliding_windows(values: np.ndarray, window_size: int):
    """Turn a 1-D array into (X, y) sequence-to-one supervised pairs."""
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i: i + window_size])
        y.append(values[i + window_size])
    X = np.array(X, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def prepare_dataset(csv_path: str, window_size: int = 30, train_ratio: float = 0.8,
                     date_col: str = "Date", value_col: str = "Close"):
    """
    Full preprocessing pipeline used by the notebook.

    Returns a dict with:
        X_train, y_train, X_test, y_test : np.ndarray sequences
        scaler   : fitted MinMaxScaler (fit on the training portion only)
        series   : the cleaned raw Close-price pandas Series
    """
    series = load_series(csv_path, date_col=date_col, value_col=value_col)
    values = series.values.reshape(-1, 1)

    # Chronological split point on the *raw* series so the scaler is only
    # ever fit on information available at "training time".
    split_idx = int(len(values) * train_ratio)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:split_idx])
    scaled = scaler.transform(values).flatten()

    X_all, y_all = make_sliding_windows(scaled, window_size)

    # Re-derive the split boundary in windowed-sequence space.
    seq_split_idx = split_idx - window_size
    seq_split_idx = max(seq_split_idx, 1)

    X_train, X_test = X_all[:seq_split_idx], X_all[seq_split_idx:]
    y_train, y_test = y_all[:seq_split_idx], y_all[seq_split_idx:]

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "series": series,
    }
