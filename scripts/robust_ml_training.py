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