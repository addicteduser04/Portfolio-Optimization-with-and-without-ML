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

# ==================== OPTIMIZATION ====================
def mean_cvar_weights(mu, scenarios, lam=10.0, alpha=0.975, dv01=None, vega=None, dv01_cap=None, vega_cap=None, leverage_cap=1.0):
    try:
        import cvxpy as cp
    except Exception:
        return None  # signal caller to fallback

    T, N = scenarios.shape
    w = cp.Variable(N)
    z = cp.Variable()          # VaR
    u = cp.Variable(T)         # slack for CVaR

    # portfolio losses per scenario
    losses = - scenarios @ w   # negative of return

    constraints = [u >= 0, u >= losses - z]
    # Leverage (L1) cap
    if leverage_cap is not None:
        constraints += [cp.norm1(w) <= leverage_cap]
    # DV01/Vega caps
    if dv01 is not None and dv01_cap is not None:
        constraints += [cp.abs(dv01 @ w) <= dv01_cap]
    if vega is not None and vega_cap is not None:
        constraints += [cp.abs(vega @ w) <= vega_cap]

    cvar = z + (1.0/((1-alpha)*T)) * cp.sum(u)
    objective = cp.Maximize(mu @ w - lam * cvar)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    if w.value is None:
        print("w.value is None")
        prob.solve(solver=cp.SCS, verbose=False)
    return None if w.value is None else np.array(w.value).reshape(-1)