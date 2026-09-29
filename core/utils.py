import numpy as np
import pandas as pd


def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b)) & (b != 0)
    out[mask] = a[mask] / b[mask]
    return out


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope_fn(arr):
        y = np.asarray(arr, dtype=float)
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return ((x - x_mean) * (y - y_mean)).sum() / denom

    return series.rolling(window).apply(slope_fn, raw=True)


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
