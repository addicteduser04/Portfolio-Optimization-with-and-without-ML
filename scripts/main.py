from loaders import yahoo_close_map, fred_series_map, weekly_last, pct_change
from strategy import walkforward_cvar,ml_strategy_with_validation
import pandas as pd
from Analyse import analyze_results
import numpy as np
def main():
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