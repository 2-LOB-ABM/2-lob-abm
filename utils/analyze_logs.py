"""
Analysis script for detailed simulation logs.
Automatically analyzes logged data to identify issues and patterns.
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path


def find_latest_logs(log_dir="simulation_logs"):
    """Find the most recent log files."""
    market_files = sorted(glob.glob(os.path.join(log_dir, "market_log_*.csv")), reverse=True)
    dealer_files = sorted(glob.glob(os.path.join(log_dir, "dealers_log_*.csv")), reverse=True)
    
    if not market_files or not dealer_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")
    
    return market_files[0], dealer_files[0]


def analyze_simulation_logs(market_file=None, dealer_file=None, log_dir="simulation_logs"):
    """
    Analyze simulation logs and generate a comprehensive report.
    
    Returns:
        dict: Analysis results with identified issues and statistics
    """
    # Find latest logs if not provided
    if market_file is None or dealer_file is None:
        market_file, dealer_file = find_latest_logs(log_dir)
    
    print(f"Analyzing logs:")
    print(f"  Market: {market_file}")
    print(f"  Dealers: {dealer_file}")
    
    # Load data
    market_df = pd.read_csv(market_file)
    dealer_df = pd.read_csv(dealer_file)
    
    analysis = {
        'market_file': market_file,
        'dealer_file': dealer_file,
        'n_steps': len(market_df),
        'n_dealers': dealer_df['dealer_id'].nunique(),
        'issues': [],
        'statistics': {},
        'recommendations': []
    }
    
    # ========== Market Analysis ==========
    print("\n=== Market Analysis ===")
    
    # Price analysis
    price_changes = market_df['price'].diff()
    price_volatility = price_changes.std()
    max_price_drop = price_changes.min()
    max_price_jump = price_changes.max()
    
    analysis['statistics']['price'] = {
        'initial': market_df['price'].iloc[0],
        'final': market_df['price'].iloc[-1],
        'min': market_df['price'].min(),
        'max': market_df['price'].max(),
        'volatility': price_volatility,
        'max_drop': max_price_drop,
        'max_jump': max_price_jump,
    }
    
    # Regime analysis
    stress_periods = (market_df['regime'] == 1).sum()
    stress_percentage = 100 * stress_periods / len(market_df)
    regime_switches = (market_df['regime'].diff() != 0).sum()
    
    analysis['statistics']['regime'] = {
        'stress_periods': int(stress_periods),
        'stress_percentage': stress_percentage,
        'regime_switches': int(regime_switches),
    }
    
    # Spread analysis
    avg_spread = market_df['spread'].mean()
    max_spread = market_df['spread'].max()
    spread_in_stress = market_df[market_df['regime'] == 1]['spread'].mean()
    spread_in_calm = market_df[market_df['regime'] == 0]['spread'].mean()
    
    analysis['statistics']['spread'] = {
        'avg': avg_spread,
        'max': max_spread,
        'in_stress': spread_in_stress,
        'in_calm': spread_in_calm,
    }
    
    # ========== Dealer Analysis ==========
    print("\n=== Dealer Analysis ===")
    
    # Reward analysis
    dealer_rewards = dealer_df.groupby('dealer_id')['reward'].agg(['mean', 'std', 'min', 'max', 'sum'])
    avg_reward = dealer_df['reward'].mean()
    negative_rewards_pct = 100 * (dealer_df['reward'] < 0).sum() / len(dealer_df)
    
    analysis['statistics']['rewards'] = {
        'avg': avg_reward,
        'std': dealer_df['reward'].std(),
        'min': dealer_df['reward'].min(),
        'max': dealer_df['reward'].max(),
        'negative_percentage': negative_rewards_pct,
        'per_dealer': dealer_rewards.to_dict('index'),
    }
    
    # Check for issues with rewards
    if negative_rewards_pct > 80:
        analysis['issues'].append({
            'severity': 'HIGH',
            'category': 'Rewards',
            'issue': f'Too many negative rewards: {negative_rewards_pct:.1f}%',
            'description': 'Most rewards are negative, indicating potential issues with reward calculation or excessive penalties.',
        })
        analysis['recommendations'].append('Review reward calculation: check if penalties (risk, inventory) are too high.')
    
    if avg_reward < -0.01:
        analysis['issues'].append({
            'severity': 'MEDIUM',
            'category': 'Rewards',
            'issue': f'Average reward is very negative: {avg_reward:.6f}',
            'description': 'Dealers are consistently losing money on average.',
        })
    
    # Strategy quality analysis
    quality_bs = dealer_df['quality_BS'].mean()
    quality_tfbs = dealer_df['quality_TFBS'].mean()
    quality_heston = dealer_df['quality_HESTON'].mean()
    
    analysis['statistics']['strategy_quality'] = {
        'BS': quality_bs,
        'TFBS': quality_tfbs,
        'HESTON': quality_heston,
    }
    
    # Check if quality is degrading
    quality_trend_bs = dealer_df.groupby('step')['quality_BS'].mean()
    quality_trend_tfbs = dealer_df.groupby('step')['quality_TFBS'].mean()
    quality_trend_heston = dealer_df.groupby('step')['quality_HESTON'].mean()
    
    if len(quality_trend_bs) > 10:
        quality_slope_bs = np.polyfit(range(len(quality_trend_bs)), quality_trend_bs, 1)[0]
        quality_slope_tfbs = np.polyfit(range(len(quality_trend_tfbs)), quality_trend_tfbs, 1)[0]
        quality_slope_heston = np.polyfit(range(len(quality_trend_heston)), quality_trend_heston, 1)[0]
        
        if quality_slope_bs < -1e-6 or quality_slope_tfbs < -1e-6 or quality_slope_heston < -1e-6:
            analysis['issues'].append({
                'severity': 'MEDIUM',
                'category': 'Strategy Quality',
                'issue': 'Strategy quality is degrading over time',
                'description': f'Quality slopes: BS={quality_slope_bs:.6f}, TFBS={quality_slope_tfbs:.6f}, HESTON={quality_slope_heston:.6f}',
            })
            analysis['recommendations'].append('Check if off-policy learning is working correctly for unused strategies.')
    
    # Hedging error analysis
    hedge_error_bs = dealer_df['hedge_error_BS'].mean()
    hedge_error_tfbs = dealer_df['hedge_error_TFBS'].mean()
    hedge_error_heston = dealer_df['hedge_error_HESTON'].mean()
    
    analysis['statistics']['hedge_errors'] = {
        'BS': hedge_error_bs,
        'TFBS': hedge_error_tfbs,
        'HESTON': hedge_error_heston,
    }
    
    # Check if HESTON has significantly lower errors (expected)
    if hedge_error_heston > hedge_error_bs * 0.8:
        analysis['issues'].append({
            'severity': 'MEDIUM',
            'category': 'Hedging Errors',
            'issue': 'HESTON errors are not significantly lower than BS',
            'description': f'HESTON error ({hedge_error_heston:.6f}) should be much lower than BS ({hedge_error_bs:.6f})',
        })
        analysis['recommendations'].append('Verify HESTON model implementation and hedging calculation.')
    
    # Strategy switching analysis
    switches = dealer_df['strategy_switched'].sum()
    switch_rate = 100 * switches / len(dealer_df)
    
    analysis['statistics']['switching'] = {
        'total_switches': int(switches),
        'switch_rate_percent': switch_rate,
    }
    
    if switch_rate > 50:
        analysis['issues'].append({
            'severity': 'LOW',
            'category': 'Strategy Switching',
            'issue': f'Very high switching rate: {switch_rate:.1f}%',
            'description': 'Dealers are switching strategies too frequently, may indicate instability.',
        })
        analysis['recommendations'].append('Consider increasing min_hold_time or adjusting switching thresholds.')
    
    # Strategy distribution analysis
    strategy_dist = dealer_df.groupby('step').agg({
        'current_strategy': lambda x: x.value_counts().to_dict()
    })
    
    final_strategies = dealer_df[dealer_df['step'] == dealer_df['step'].max()]['current_strategy'].value_counts()
    
    analysis['statistics']['strategy_distribution'] = {
        'final': final_strategies.to_dict(),
    }
    
    # Wealth analysis
    wealth_changes = dealer_df.groupby('dealer_id')['wealth'].agg(['first', 'last'])
    wealth_changes['change'] = wealth_changes['last'] - wealth_changes['first']
    avg_wealth_change = wealth_changes['change'].mean()
    
    analysis['statistics']['wealth'] = {
        'avg_change': avg_wealth_change,
        'per_dealer': wealth_changes.to_dict('index'),
    }
    
    if avg_wealth_change < -10:
        analysis['issues'].append({
            'severity': 'HIGH',
            'category': 'Wealth',
            'issue': f'Dealers are losing significant wealth: avg change = {avg_wealth_change:.2f}',
            'description': 'Dealers are consistently losing money over the simulation.',
        })
        analysis['recommendations'].append('Review pricing, hedging, and reward mechanisms.')
    
    # ========== Model Distribution Analysis ==========
    print("\n=== Model Distribution Analysis ===")
    
    if 'model_dist_BS' in market_df.columns:
        final_dist_bs = market_df['model_dist_BS'].iloc[-1]
        final_dist_tfbs = market_df['model_dist_TFBS'].iloc[-1]
        final_dist_heston = market_df['model_dist_HESTON'].iloc[-1]
        
        analysis['statistics']['model_distribution'] = {
            'final_BS': final_dist_bs,
            'final_TFBS': final_dist_tfbs,
            'final_HESTON': final_dist_heston,
        }
        
        # Check if one model dominates
        if final_dist_heston > 0.8:
            analysis['issues'].append({
                'severity': 'LOW',
                'category': 'Model Distribution',
                'issue': 'HESTON model dominates (>80%)',
                'description': 'This may be expected if HESTON performs significantly better.',
            })
        elif final_dist_bs > 0.8:
            analysis['issues'].append({
                'severity': 'MEDIUM',
                'category': 'Model Distribution',
                'issue': 'BS model dominates (>80%)',
                'description': 'BS model may be overperforming or switching mechanism may not be working correctly.',
            })
    
    # ========== Generate Report ==========
    print("\n=== Analysis Summary ===")
    print(f"Total steps: {analysis['n_steps']}")
    print(f"Number of dealers: {analysis['n_dealers']}")
    print(f"Issues found: {len(analysis['issues'])}")
    
    if analysis['issues']:
        print("\nIssues:")
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"  {i}. [{issue['severity']}] {issue['category']}: {issue['issue']}")
            print(f"     {issue['description']}")
    
    if analysis['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return analysis


def save_analysis_report(analysis, output_file="simulation_analysis_report.txt"):
    """Save analysis report to text file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIMULATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Market log: {analysis['market_file']}\n")
        f.write(f"Dealer log: {analysis['dealer_file']}\n")
        f.write(f"Steps: {analysis['n_steps']}\n")
        f.write(f"Dealers: {analysis['n_dealers']}\n\n")
        
        f.write("STATISTICS\n")
        f.write("-" * 80 + "\n")
        for category, stats in analysis['statistics'].items():
            f.write(f"\n{category.upper()}:\n")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
        
        f.write("\n\nISSUES FOUND\n")
        f.write("-" * 80 + "\n")
        if analysis['issues']:
            for i, issue in enumerate(analysis['issues'], 1):
                f.write(f"\n{i}. [{issue['severity']}] {issue['category']}\n")
                f.write(f"   Issue: {issue['issue']}\n")
                f.write(f"   Description: {issue['description']}\n")
        else:
            f.write("No issues found!\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        if analysis['recommendations']:
            for i, rec in enumerate(analysis['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("No specific recommendations.\n")
    
    print(f"\nAnalysis report saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation_logs"
    
    try:
        analysis = analyze_simulation_logs(log_dir=log_dir)
        save_analysis_report(analysis)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run a simulation first to generate logs.")
