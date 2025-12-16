
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
        #correlation = float(np.corrcoef(ml_returns_array, base_returns_array)[0, 1])
        
        print(f"\n🔍 Strategy Differences:")
        print(f"ML Outperformance Frequency: {outperformance:.2%}")
        #print(f"Return Correlation: {correlation:.4f}")
        print(f"ML Total Return Advantage: {ml_metrics['Cumulative Return'] - base_metrics['Cumulative Return']:.4f}")
        
        # Statistical test for difference in means
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(ml_returns_array, base_returns_array, equal_var=False)
    
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