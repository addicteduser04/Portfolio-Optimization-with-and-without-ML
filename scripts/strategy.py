import pandas as pd
import numpy as np
import warnings
from robust_ml_training import create_stable_features, train_asset_model, get_ml_prediction_safe
from optimization import mean_cvar_weights
from loaders import load_greeks
warnings.filterwarnings('ignore')

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

