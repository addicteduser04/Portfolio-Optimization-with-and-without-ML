import yfinance as yf 
from datetime import datetime
from typing import Any, cast
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


# ==================== DATA FUNCTIONS ====================
def yahoo_close_map(tickers: dict[str, str], start: str = '2018-01-01', end: str | None = None):
    yf_any = cast(Any, yf)
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    frames: list[pd.Series] = []
    for name, yft in tickers.items():
        df = yf_any.Ticker(yft).history(start=start, end=end, auto_adjust=False)
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is not None:
            frames.append(df[col].rename(name))
    return pd.concat(frames, axis=1).ffill().dropna(how='all')

def fred_series_map(series_map: dict[str, str], start: str = '2018-01-01', end: str | None = None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    frames = []
    for name, code in series_map.items():
        try:
            s = pdr.DataReader(code, "fred", start, end).rename(columns={code: name})
            frames.append(s)
        except Exception as e:
            print(f"Failed to download {code}: {e}")
    if frames:
        df = pd.concat(frames, axis=1).ffill().dropna(how="all")
        return df
    return pd.DataFrame()

def weekly_last(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-FRI").last()

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().dropna(how="all")

def load_greeks(assets):
    gfile = "./../data/synthetic_greeks.csv"
    g = pd.read_csv(gfile).set_index("asset").reindex(assets).fillna(0.0)
    return g["DV01_usd_per_bp"].values, g["Vega_usd_per_volpt"].values
