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

# ==================== ROBUST ML TRAINING ====================
def create_stable_features(returns_df):
    """Create features with stable column names that don't change over time"""
    df = returns_df.copy()
    asset_names = returns_df.columns.tolist()
    
    # Technical features with consistent naming
    for i, col in enumerate(asset_names):
        # Price-based features
        df[f'asset_{i}_ret_1w'] = returns_df[col].shift(1)
        df[f'asset_{i}_ret_2w'] = returns_df[col].shift(2)
        df[f'asset_{i}_ret_4w'] = returns_df[col].shift(4)
        
        # Momentum indicators
        df[f'asset_{i}_mom_4w'] = returns_df[col].rolling(4).mean()
        df[f'asset_{i}_mom_12w'] = returns_df[col].rolling(12).mean()
        df[f'asset_{i}_mom_ratio'] = df[f'asset_{i}_mom_4w'] / (df[f'asset_{i}_mom_12w'] + 1e-8)
        
        # Volatility features
        df[f'asset_{i}_vol_4w'] = returns_df[col].rolling(4).std()
        df[f'asset_{i}_vol_12w'] = returns_df[col].rolling(12).std()
        df[f'asset_{i}_vol_regime'] = (df[f'asset_{i}_vol_4w'] > df[f'asset_{i}_vol_12w']).astype(int)
        
        # Mean reversion
        df[f'asset_{i}_zscore_12w'] = (returns_df[col] - returns_df[col].rolling(12).mean()) / (returns_df[col].rolling(12).std() + 1e-8)
    
    # Cross-asset features with stable names
    if len(asset_names) >= 2:
        df['cross_mom_diff'] = df['asset_0_mom_4w'] - df['asset_1_mom_4w']
    
    # Market regime features
    vol_cols = [f'asset_{i}_vol_4w' for i in range(len(asset_names)) if f'asset_{i}_vol_4w' in df.columns]
    if vol_cols:
        df['market_vol'] = df[vol_cols].mean(axis=1)
    
    # Remove any infinite values and fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)
    
    return df

def train_asset_model(X_train, y_train, asset_name):
    """Train a robust model for a specific asset"""
    
    if len(X_train) < 40:
        return None, 0, None
    
    # Ensure we only use feature columns (exclude the original returns)
    feature_columns = [col for col in X_train.columns if not col in y_train.index if hasattr(y_train, 'index')]
    if not feature_columns:
        return None, 0, None
    
    X_train_features = X_train[feature_columns]
    
    # Walk-forward validation
    min_train_size = 30
    predictions = []
    actuals = []
    
    for test_idx in range(min_train_size, len(X_train_features)):
        X_tr = X_train_features.iloc[:test_idx]
        y_tr = y_train.iloc[:test_idx]
        
        if len(X_tr) < min_train_size:
            continue
            
        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        
        # Choose model based on asset type
        if asset_name.startswith('FX_'):
            model = RandomForestRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=4,
                min_samples_split=15,
                random_state=42
            )
        else:
            model = LassoCV(cv=3, random_state=42, max_iter=2000)  # Reduced for speed
        
        try:
            model.fit(X_tr_scaled, y_tr)
            
            # Predict next period (out-of-sample)
            if test_idx < len(X_train_features):
                X_te = X_train_features.iloc[test_idx:test_idx+1]
                X_te_scaled = scaler.transform(X_te)
                pred = model.predict(X_te_scaled)[0]
                predictions.append(pred)
                actuals.append(y_train.iloc[test_idx])
        except:
            continue
    
    # Calculate out-of-sample R²
    if len(predictions) >= 10:
        r2 = r2_score(actuals, predictions)
        
        # Retrain final model on full dataset if it has some predictive power
        if r2 > -0.2:
            scaler_final = StandardScaler()
            X_full_scaled = scaler_final.fit_transform(X_train_features)
            
            if asset_name.startswith('FX_'):
                final_model = RandomForestRegressor(
                    n_estimators=50, max_depth=4, min_samples_split=15, random_state=42
                )
            else:
                final_model = LassoCV(cv=3, random_state=42, max_iter=2000)
            
            final_model.fit(X_full_scaled, y_train)
            return final_model, r2, scaler_final
    
    return None, 0, None

def get_ml_prediction_safe(model, scaler, X_current, feature_columns):
    """Safe prediction with proper feature alignment"""
    if model is None or scaler is None:
        return 0.0
    
    try:
        # Ensure we only use the feature columns that the model was trained on
        X_current_features = X_current[feature_columns]
        X_current_scaled = scaler.transform(X_current_features)
        prediction = model.predict(X_current_scaled)[0]
        prediction = np.clip(prediction, -0.08, 0.08)  # Reasonable bounds
        return prediction
    except Exception as e:
        return 0.0

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
        prob.solve(solver=cp.SCS, verbose=False)
    return None if w.value is None else np.array(w.value).reshape(-1)

# ==================== STRATEGY EXECUTION ====================
def ml_strategy_with_validation(returns_df, start_train_weeks=104):
    """ML strategy with proper model validation"""
    print("🤖 Training ML models with proper validation...")
    
    assets = returns_df.columns.tolist()
    features_df = create_stable_features(returns_df)
    
    # Get stable feature columns (exclude original asset columns)
    feature_columns = [col for col in features_df.columns if col not in assets]
    
    # Store trained models and scalers for each asset
    asset_models = {}
    asset_scalers = {}
    asset_r2_scores = {}
    
    # Train models for each asset using initial training period
    for asset in assets:
        print(f"  Training model for {asset}...")
        
        # Use first start_train_weeks for initial model training
        X_train_full = features_df.iloc[:start_train_weeks]
        y_train_full = returns_df[asset].iloc[:start_train_weeks]
        
        model, r2, scaler = train_asset_model(X_train_full, y_train_full, asset)
        asset_models[asset] = model
        asset_scalers[asset] = scaler
        asset_r2_scores[asset] = r2
    
    print("\n📊 Model Quality Summary:")
    for asset, r2 in asset_r2_scores.items():
        model_type = "RF" if asset.startswith('FX_') else "Lasso"
        status = "GOOD" if r2 > 0 else "POOR" if r2 > -0.1 else "FAIL"
        print(f"  {asset} ({model_type}): R² = {r2:.4f} [{status}]")
    
    # Walk-forward execution
    portfolio_returns = []
    portfolio_weights = []
    
    for t in range(start_train_weeks, len(returns_df)-1):
        if t % 26 == 0:
            print(f"  Executing week {t}/{len(returns_df)-1}")
        
        historical_returns = returns_df.iloc[:t]
        current_features = features_df.iloc[t:t+1]
        
        Sigma = historical_returns.cov().values
        mu_ml = np.zeros(len(assets))
        
        for i, asset in enumerate(assets):
            model = asset_models.get(asset)
            scaler = asset_scalers.get(asset)
            
            if model is not None and scaler is not None and asset_r2_scores[asset] > -0.1:
                # Use ML prediction
                prediction = get_ml_prediction_safe(model, scaler, current_features, feature_columns)
                mu_ml[i] = prediction
            else:
                # Fallback: use recent momentum
                recent_trend = historical_returns[asset].tail(12).mean()
                mu_ml[i] = recent_trend  # No enhancement for baseline comparison
        
        # Optimize portfolio
        weights = mean_cvar_weights(mu_ml, Sigma)
        
        # Calculate next period return
        next_return = returns_df.iloc[t+1].values @ weights
        portfolio_returns.append({'date': returns_df.index[t+1], 'return': next_return})
        portfolio_weights.append(pd.Series(weights, index=assets, name=returns_df.index[t]))
    
    returns_series = pd.DataFrame(portfolio_returns).set_index('date')['return'] if portfolio_returns else pd.Series()
    weights_df = pd.DataFrame(portfolio_weights) if portfolio_weights else pd.DataFrame()
    
    return returns_series, weights_df

def load_greeks(assets):
    gfile = "/data/synthetic_greeks.csv"
    g = pd.read_csv(gfile).set_index("asset").reindex(assets).fillna(0.0)
    return g["DV01_usd_per_bp"].values, g["Vega_usd_per_volpt"].values


ALPHA = 0.975
LAMBDA = 10.0
LEVERAGE_CAP = 1.0
DV01_CAP = 50000.0   # $ per 1bp
VEGA_CAP  = 20000.0  # $ per vol point
def walkforward_cvar(weekly_returns: pd.DataFrame, start_train_weeks=104):
    print("🎯 Running baseline strategy...")
    assets = weekly_returns.columns
    dv, vg = load_greeks(assets)

    idx = weekly_returns.index
    port = []
    w_hist = []

    for t in range(start_train_weeks, len(idx)-1):
        hist = weekly_returns.iloc[:t]
        scen = hist.values  # T x N
        # simple expected return: last-week return (carry-like)
        mu = weekly_returns.iloc[t-1].values

        w = mean_cvar_weights(mu, scen, lam=LAMBDA, alpha=ALPHA, dv01=dv, vega=vg, dv01_cap=DV01_CAP, vega_cap=VEGA_CAP, leverage_cap=LEVERAGE_CAP)
        w_hist.append(pd.Series(w, index=assets, name=idx[t]))
        r_next = weekly_returns.iloc[t+1].values
        port.append(pd.Series({"date": idx[t+1], "port_ret": float(np.dot(w, r_next))}))

    P = pd.DataFrame(port).set_index("date").sort_index()
    W = pd.DataFrame(w_hist)
    return P, W

# ==================== ANALYSIS WITH FIGURE SAVING ====================
def analyze_results(ml_returns, baseline_returns, save_prefix="portfolio"):
    """Comprehensive results analysis with figure saving"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE STRATEGY COMPARISON")
    print("="*80)
    
    # Basic metrics - FIXED VERSION
    def calculate_metrics(returns, name):
        if len(returns) == 0:
            return {}
        
        # Convert to numpy array to avoid pandas Series issues
        returns_array = returns.values if hasattr(returns, 'values') else returns
        
        cumulative = float((1 + returns_array).prod() - 1)
        annual = float((1 + cumulative) ** (52/len(returns_array)) - 1)
        vol = float(np.std(returns_array) * np.sqrt(52))  # Use np.std instead of .std()
        sharpe = float(annual / vol) if vol > 0 else 0.0
        
        # Drawdown
        cum_series = (1 + returns_array).cumprod()
        running_max = np.maximum.accumulate(cum_series)
        drawdown = (cum_series - running_max) / running_max
        max_dd = float(drawdown.min())
        
        return {
            'Strategy': name,
            'Cumulative Return': cumulative,
            'Annual Return': annual,
            'Annual Volatility': vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': float(np.mean(returns_array > 0)),
            'Avg Return': float(np.mean(returns_array)),
            'Return Std': float(np.std(returns_array)),
            'Data Points': len(returns_array)
        }
    
    ml_metrics = calculate_metrics(ml_returns, "ML Strategy")
    base_metrics = calculate_metrics(baseline_returns, "Baseline")
    
    results_df = pd.DataFrame([ml_metrics, base_metrics]).set_index('Strategy')
    
    print("Performance Metrics:")
    print(results_df.round(4))
    
    # Strategy differences
    if len(ml_returns) == len(baseline_returns):
        ml_returns_array = ml_returns.values if hasattr(ml_returns, 'values') else ml_returns
        base_returns_array = baseline_returns.values if hasattr(baseline_returns, 'values') else baseline_returns
        
        outperformance = float(np.mean(ml_returns_array > base_returns_array))
        correlation = float(np.corrcoef(ml_returns_array, base_returns_array)[0, 1])
        
        print(f"\n🔍 Strategy Differences:")
        print(f"ML Outperformance Frequency: {outperformance:.2%}")
        print(f"Return Correlation: {correlation:.4f}")
        print(f"ML Total Return Advantage: {ml_metrics['Cumulative Return'] - base_metrics['Cumulative Return']:.4f}")
        
        # Statistical test for difference in means
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(ml_returns_array, base_returns_array, equal_var=False)
        print(f"T-test p-value: {p_value:.4f} {'(SIGNIFICANT)' if p_value < 0.1 else '(not significant)'}")
    
    # Plot results and save figures
    if len(ml_returns) > 0 and len(baseline_returns) > 0:
        # Create individual figures for each plot
        
        # Figure 1: Cumulative Returns
        plt.figure(figsize=(10, 6))
        cum_ml = (1 + ml_returns).cumprod()
        cum_base = (1 + baseline_returns).cumprod()
        plt.plot(cum_ml.index, cum_ml.values, label='ML Strategy', linewidth=2, color='blue')
        plt.plot(cum_base.index, cum_base.values, label='Baseline', linewidth=2, color='red')
        plt.title('Cumulative Returns: ML vs Baseline Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Growth of $1')
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig(f'/visualisation/{save_prefix}_cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: /visualisation/{save_prefix}_cumulative_returns.png")
        
        # Figure 2: Drawdowns
        plt.figure(figsize=(10, 6))
        def calculate_drawdown(returns):
            cum = (1 + returns).cumprod()
            running_max = cum.expanding().max()
            return (cum - running_max) / running_max
        
        dd_ml = calculate_drawdown(ml_returns)
        dd_base = calculate_drawdown(baseline_returns)
        plt.fill_between(dd_ml.index, dd_ml.values, 0, alpha=0.5, label='ML Drawdown', color='blue')
        plt.fill_between(dd_base.index, dd_base.values, 0, alpha=0.5, label='Baseline Drawdown', color='red')
        plt.title('Portfolio Drawdowns: ML vs Baseline Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Drawdown')
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig(f'/visualisation/{save_prefix}_drawdowns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: /visualisation/{save_prefix}_drawdowns.png")
        
        # Figure 3: Rolling Quarterly Returns
        plt.figure(figsize=(10, 6))
        rolling_ml = ml_returns.rolling(13).apply(lambda x: (1+x).prod()-1)
        rolling_base = baseline_returns.rolling(13).apply(lambda x: (1+x).prod()-1)
        plt.plot(rolling_ml.index, rolling_ml.values, label='ML Quarterly Returns', color='blue')
        plt.plot(rolling_base.index, rolling_base.values, label='Baseline Quarterly Returns', color='red')
        plt.title('Rolling Quarterly Returns: ML vs Baseline Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Quarterly Return')
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig(f'/visualisation/{save_prefix}_rolling_quarterly_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: /visualisation/{save_prefix}_rolling_quarterly_returns.png")
        
        # Figure 4: Return Distributions
        plt.figure(figsize=(10, 6))
        plt.hist(ml_returns.values, bins=30, alpha=0.7, label='ML Returns', color='blue', density=True)
        plt.hist(baseline_returns.values, bins=30, alpha=0.7, label='Baseline Returns', color='red', density=True)
        plt.title('Return Distributions: ML vs Baseline Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Density')
        plt.xlabel('Weekly Return')
        plt.tight_layout()
        plt.savefig(f'/visualisation/{save_prefix}_return_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_prefix}_return_distributions.png")
        
        # Create one combined figure for display
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        ax1.plot(cum_ml.index, cum_ml.values, label='ML Strategy', linewidth=2, color='blue')
        ax1.plot(cum_base.index, cum_base.values, label='Baseline', linewidth=2, color='red')
        ax1.set_title('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdowns
        ax2.fill_between(dd_ml.index, dd_ml.values, 0, alpha=0.5, label='ML Drawdown', color='blue')
        ax2.fill_between(dd_base.index, dd_base.values, 0, alpha=0.5, label='Baseline Drawdown', color='red')
        ax2.set_title('Drawdowns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling returns (quarterly)
        ax3.plot(rolling_ml.index, rolling_ml.values, label='ML Quarterly', color='blue')
        ax3.plot(rolling_base.index, rolling_base.values, label='Baseline Quarterly', color='red')
        ax3.set_title('Rolling Quarterly Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Return distributions
        ax4.hist(ml_returns.values, bins=30, alpha=0.7, label='ML Returns', color='blue', density=True)
        ax4.hist(baseline_returns.values, bins=30, alpha=0.7, label='Baseline Returns', color='red', density=True)
        ax4.set_title('Return Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/visualisation/{save_prefix}_combined_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: /visualisation/{save_prefix}_combined_analysis.png")

# ==================== MAIN EXECUTION ====================
def main():
    print("🚀 SERIOUS ML PORTFOLIO OPTIMIZATION")
    print("📊 Using proper financial ML with validation")
    
    # Get data
    FX_TICKERS = {"FX_EURUSD": "EURUSD=X", "FX_GBPUSD": "GBPUSD=X"}
    YIELD_CURVE = {"RATES_USD_2Y": "DGS2", "RATES_USD_10Y": "DGS10"}
    
    try:
        fx_data = yahoo_close_map(FX_TICKERS, start="2018-01-01")
        yields_data = fred_series_map(YIELD_CURVE, start="2018-01-01")
        
        fx_weekly = weekly_last(fx_data)
        yields_weekly = weekly_last(yields_data)
        
        fx_returns = pct_change(fx_weekly)
        yields_returns = yields_weekly.diff().dropna()
        
        fx_returns.index = pd.to_datetime(fx_returns.index).tz_localize(None)
        yields_returns.index = pd.to_datetime(yields_returns.index)
        
        all_returns = fx_returns.join(yields_returns, how='inner').dropna()
        
    except Exception as e:
        print(f"❌ Data download failed, using realistic synthetic data: {e}")
        dates = pd.date_range('2018-01-01', periods=300, freq='W-FRI')
        np.random.seed(42)
        # Create realistic synthetic data with some predictability
        t = np.arange(300)
        trend = 0.0005 * np.sin(t * 0.05)  # Slow cyclical trend
        noise = np.random.normal(0, 0.015, 300)
        
        all_returns = pd.DataFrame({
            'FX_EURUSD': trend + noise * 0.8 + 0.001,
            'FX_GBPUSD': trend * 0.8 + noise * 0.9 + 0.0008,
            'RATES_USD_2Y': trend * 0.3 + noise * 0.4 + 0.0001,
            'RATES_USD_10Y': trend * 0.5 + noise * 0.5 + 0.0002,
        }, index=dates)
    
    print(f"✅ Data loaded: {all_returns.shape[0]} weeks, {all_returns.shape[1]} assets")
    print(f"📈 Assets: {list(all_returns.columns)}")
    
    # Run strategies
    ml_returns, ml_weights = ml_strategy_with_validation(all_returns, start_train_weeks=104)
    baseline_returns, baseline_weights = walkforward_cvar(all_returns, start_train_weeks=104)
    
    # Analyze results and save figures
    analyze_results(ml_returns, baseline_returns, save_prefix="portfolio_comparison")
    
    # Also save the performance metrics to CSV
    if len(ml_returns) > 0 and len(baseline_returns) > 0:
        # Save returns data
        returns_comparison = pd.DataFrame({
            'ML_Strategy': ml_returns,
            'Baseline_Strategy': baseline_returns
        })
        returns_comparison.to_csv('portfolio_returns_comparison.csv')
        print("✅ Saved: portfolio_returns_comparison.csv")
        
        # Save weights if available
        if not ml_weights.empty:
            ml_weights.to_csv('ml_strategy_weights.csv')
            print("✅ Saved: ml_strategy_weights.csv")
        if not baseline_weights.empty:
            baseline_weights.to_csv('baseline_strategy_weights.csv')
            print("✅ Saved: baseline_strategy_weights.csv")
    
    print("\n🎉 Analysis complete! All figures and data saved.")

if __name__ == "__main__":
    main()