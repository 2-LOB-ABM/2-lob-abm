"""
Run ABM simulation with options market and model switching.

Usage:
    python run_simulation.py           # Run simulation
    python run_simulation.py --seed 42 # Run with custom seed
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from models.abm_model import UnifiedABMModel
from options.pricing import PricingModel

# Output directory for plots
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def get_config():
    """Get simulation configuration."""
    return {
        "S0": 100.0,
        "dt": 0.001,
        "steps": 2000,
        "n_fund": 15,
        "n_chart": 15,
        "n_mm": 5,
        "n_noise": 70,
        "fundamental_price": 100.0,
        "tick_size": 0.1,
        "enable_options": True,
        "n_option_dealers": 25,
        "n_option_contracts": 3,
        "n_option_takers": 40,
        "option_taker_trade_prob": 0.08,
        "enable_model_switching": True,
        "switching_window": 50,
        "switching_threshold": 0.05,
        "dealer_model_distribution": {
            PricingModel.BS: 0.33,
            PricingModel.TFBS: 0.33,
            PricingModel.HESTON: 0.34
        },
        "p01": 0.005,
        "p10": 0.03,
        "shock_rate": 0.003,
        "shock_impact": 8.0,
        "enable_detailed_logging": True,
        "log_dir": "simulation_logs",
    }


def plot_results(model, prices):
    """Create visualization with detailed metrics."""
    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(9, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 0.5, 1, 1, 1, 1, 1, 1, 0.8])
    
    # Price path
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices, linewidth=1.5, color='black')
    ax1.fill_between(range(len(model.regime_log)), 
                     np.min(prices) * 0.99, np.max(prices) * 1.01,
                     where=np.array(model.regime_log) == 1, 
                     alpha=0.2, color='red', label='Stress regime')
    ax1.set_title("Spot Price", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Regime
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(model.regime_log, color='red', linewidth=1)
    ax2.fill_between(range(len(model.regime_log)), 0, 1, 
                     where=np.array(model.regime_log) == 1, 
                     alpha=0.3, color='red')
    ax2.set_title("Market Regime (0=Calm, 1=Stress)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Regime")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Model distribution
    if model.enable_options and len(model.dealer_model_distribution_log) > 0:
        bs_share = [d.get(PricingModel.BS, 0) for d in model.dealer_model_distribution_log]
        tfbs_share = [d.get(PricingModel.TFBS, 0) for d in model.dealer_model_distribution_log]
        heston_share = [d.get(PricingModel.HESTON, 0) for d in model.dealer_model_distribution_log]
        
        ax3 = fig.add_subplot(gs[2, :])
        ax3.fill_between(range(len(model.regime_log)), 0, 1, 
                         where=np.array(model.regime_log) == 1, 
                         alpha=0.15, color='red', label='Stress regime')
        ax3.plot(bs_share, label="BS", alpha=0.8, linewidth=2, color='blue')
        ax3.plot(tfbs_share, label="TFBS", alpha=0.8, linewidth=2, color='cyan')
        ax3.plot(heston_share, label="HESTON", alpha=0.8, linewidth=2, color='purple')
        ax3.set_title("Model Distribution", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Share")
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # Hedging errors
    if hasattr(model, 'dealer_hedge_errors_by_model'):
        bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
        tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
        heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
        
        target_len = len(model.regime_log)
        for errors in [bs_errors, tfbs_errors, heston_errors]:
            if errors and len(errors) < target_len:
                errors.extend([0.0] * (target_len - len(errors)))
        
        if bs_errors and len(bs_errors) == len(model.regime_log):
            ax4 = fig.add_subplot(gs[3, :])
            bs_errors_plot = [max(e, 1e-6) for e in bs_errors]
            tfbs_errors_plot = [max(e, 1e-6) for e in tfbs_errors]
            heston_errors_plot = [max(e, 1e-6) for e in heston_errors]
            
            ax4.fill_between(range(len(model.regime_log)), 1e-6, 1e3,
                             where=np.array(model.regime_log) == 1, 
                             alpha=0.15, color='red', label='Stress regime')
            ax4.plot(bs_errors_plot, label="BS Error", alpha=0.8, linewidth=1.5, color='blue')
            ax4.plot(tfbs_errors_plot, label="TFBS Error", alpha=0.8, linewidth=1.5, color='cyan')
            ax4.plot(heston_errors_plot, label="HESTON Error", alpha=0.8, linewidth=1.5, color='purple')
            ax4.set_title("Hedging Errors (log scale)", fontsize=14, fontweight='bold')
            ax4.set_xlabel("Step")
            ax4.set_ylabel("Error (log)")
            ax4.set_yscale('log')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
    
    # Rewards
    if hasattr(model, 'dealer_reward_log') and len(model.dealer_reward_log) > 0:
        reward_bs = [r.get(PricingModel.BS, 0.0) for r in model.dealer_reward_log]
        reward_tfbs = [r.get(PricingModel.TFBS, 0.0) for r in model.dealer_reward_log]
        reward_heston = [r.get(PricingModel.HESTON, 0.0) for r in model.dealer_reward_log]
        
        target_len = len(model.regime_log)
        for rewards in [reward_bs, reward_tfbs, reward_heston]:
            if len(rewards) < target_len:
                rewards.extend([0.0] * (target_len - len(rewards)))
        
        ax5 = fig.add_subplot(gs[4, :])
        ax5.fill_between(range(len(model.regime_log)), -200, 200,
                         where=np.array(model.regime_log) == 1, 
                         alpha=0.15, color='red', label='Stress regime')
        ax5.plot(reward_bs, label="BS Reward", alpha=0.8, linewidth=1.5, color='blue')
        ax5.plot(reward_tfbs, label="TFBS Reward", alpha=0.8, linewidth=1.5, color='orange')
        ax5.plot(reward_heston, label="HESTON Reward", alpha=0.8, linewidth=1.5, color='green')
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title("Reward by Model (r(t) = ΔP&L - λ·Risk - κ·Inventory)", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Reward")
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
    
    # Strategy quality
    if hasattr(model, 'dealer_strategy_quality_log') and len(model.dealer_strategy_quality_log) > 0:
        quality_bs = [q.get(PricingModel.BS, 0.0) for q in model.dealer_strategy_quality_log]
        quality_tfbs = [q.get(PricingModel.TFBS, 0.0) for q in model.dealer_strategy_quality_log]
        quality_heston = [q.get(PricingModel.HESTON, 0.0) for q in model.dealer_strategy_quality_log]
        
        target_len = len(model.regime_log)
        for qualities in [quality_bs, quality_tfbs, quality_heston]:
            if len(qualities) < target_len:
                qualities.extend([0.0] * (target_len - len(qualities)))
        
        ax6 = fig.add_subplot(gs[5, :])
        ax6.fill_between(range(len(model.regime_log)), -150, 20,
                         where=np.array(model.regime_log) == 1, 
                         alpha=0.15, color='red', label='Stress regime')
        ax6.plot(quality_bs, label="BS Quality", alpha=0.8, linewidth=1.5, color='blue')
        ax6.plot(quality_tfbs, label="TFBS Quality", alpha=0.8, linewidth=1.5, color='orange')
        ax6.plot(quality_heston, label="HESTON Quality", alpha=0.8, linewidth=1.5, color='green')
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_title("Strategy Quality (Q_s)", fontsize=14, fontweight='bold')
        ax6.set_xlabel("Step")
        ax6.set_ylabel("Quality Score")
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
    
    # Cumulative metrics
    if hasattr(model, 'dealer_hedge_errors_by_model'):
        bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
        tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
        heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
        
        target_len = len(model.regime_log)
        for errors in [bs_errors, tfbs_errors, heston_errors]:
            if errors and len(errors) < target_len:
                errors.extend([0.0] * (target_len - len(errors)))
        
        if bs_errors and len(bs_errors) == len(model.regime_log):
            bs_cumulative = np.cumsum(bs_errors)
            tfbs_cumulative = np.cumsum(tfbs_errors)
            heston_cumulative = np.cumsum(heston_errors)
            
            ax7 = fig.add_subplot(gs[6, 0])
            ax7.fill_between(range(len(model.regime_log)), 0, max(heston_cumulative) * 1.1,
                             where=np.array(model.regime_log) == 1, 
                             alpha=0.15, color='red', label='Stress regime')
            ax7.plot(bs_cumulative, label="BS cumulative", alpha=0.8, linewidth=2, color='blue')
            ax7.plot(tfbs_cumulative, label="TFBS cumulative", alpha=0.8, linewidth=2, color='cyan')
            ax7.plot(heston_cumulative, label="HESTON cumulative", alpha=0.8, linewidth=2, color='purple')
            ax7.set_title("Cumulative Hedging Errors", fontsize=12, fontweight='bold')
            ax7.set_xlabel("Step")
            ax7.set_ylabel("Cumulative Error")
            ax7.legend(loc='upper left')
            ax7.grid(True, alpha=0.3)
    
    if hasattr(model, 'dealer_reward_log') and len(model.dealer_reward_log) > 0:
        reward_bs = [r.get(PricingModel.BS, 0.0) for r in model.dealer_reward_log]
        reward_tfbs = [r.get(PricingModel.TFBS, 0.0) for r in model.dealer_reward_log]
        reward_heston = [r.get(PricingModel.HESTON, 0.0) for r in model.dealer_reward_log]
        
        target_len = len(model.regime_log)
        for rewards in [reward_bs, reward_tfbs, reward_heston]:
            if len(rewards) < target_len:
                rewards.extend([0.0] * (target_len - len(rewards)))
        
        bs_cumulative_reward = np.cumsum(reward_bs)
        tfbs_cumulative_reward = np.cumsum(reward_tfbs)
        heston_cumulative_reward = np.cumsum(reward_heston)
        
        ax8 = fig.add_subplot(gs[6, 1])
        ax8.fill_between(range(len(model.regime_log)), 
                         min(bs_cumulative_reward) * 1.1, max(heston_cumulative_reward) * 1.1,
                         where=np.array(model.regime_log) == 1, 
                         alpha=0.15, color='red', label='Stress regime')
        ax8.plot(bs_cumulative_reward, label="BS cumulative", alpha=0.8, linewidth=2, color='blue')
        ax8.plot(tfbs_cumulative_reward, label="TFBS cumulative", alpha=0.8, linewidth=2, color='orange')
        ax8.plot(heston_cumulative_reward, label="HESTON cumulative", alpha=0.8, linewidth=2, color='green')
        ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax8.set_title("Cumulative Rewards", fontsize=12, fontweight='bold')
        ax8.set_xlabel("Step")
        ax8.set_ylabel("Cumulative Reward")
        ax8.legend(loc='upper left')
        ax8.grid(True, alpha=0.3)
    
    # Correlation plot
    if (hasattr(model, 'dealer_hedge_errors_by_model') and 
        hasattr(model, 'dealer_reward_log') and len(model.dealer_reward_log) > 0):
        
        bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
        reward_bs = [r.get(PricingModel.BS, 0.0) for r in model.dealer_reward_log]
        
        min_len = min(len(bs_errors), len(reward_bs))
        if min_len > 50:
            window = min(500, min_len)
            errors_window = bs_errors[-window:]
            rewards_window = reward_bs[-window:]
            
            ax9 = fig.add_subplot(gs[7, 0])
            ax9.scatter(errors_window, rewards_window, alpha=0.3, s=10, color='blue')
            ax9.set_title("Correlation: Hedging Errors vs Rewards (BS)", fontsize=12, fontweight='bold')
            ax9.set_xlabel("Hedging Error")
            ax9.set_ylabel("Reward")
            ax9.grid(True, alpha=0.3)
            
            if len(errors_window) == len(rewards_window) and len(errors_window) > 10:
                correlation = np.corrcoef(errors_window, rewards_window)[0, 1]
                ax9.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax9.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Statistics panel
    ax10 = fig.add_subplot(gs[7, 1])
    ax10.axis('off')
    
    stats_text = "Statistics:\n\n"
    
    if model.enable_options and len(model.dealer_model_distribution_log) > 0:
        final_dist = model.dealer_model_distribution_log[-1]
        stats_text += f"Final Model Distribution:\n"
        stats_text += f"  BS: {final_dist.get(PricingModel.BS, 0.0):.1%}\n"
        stats_text += f"  TFBS: {final_dist.get(PricingModel.TFBS, 0.0):.1%}\n"
        stats_text += f"  HESTON: {final_dist.get(PricingModel.HESTON, 0.0):.1%}\n\n"
    
    if hasattr(model, 'dealer_hedge_errors_by_model'):
        bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
        tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
        heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
        if bs_errors:
            stats_text += f"Avg Hedging Errors:\n"
            stats_text += f"  BS: {np.mean(bs_errors):.6f}\n"
            stats_text += f"  TFBS: {np.mean(tfbs_errors):.6f}\n"
            stats_text += f"  HESTON: {np.mean(heston_errors):.6f}\n\n"
    
    if hasattr(model, 'dealer_reward_log') and len(model.dealer_reward_log) > 0:
        reward_bs = [r.get(PricingModel.BS, 0.0) for r in model.dealer_reward_log]
        reward_tfbs = [r.get(PricingModel.TFBS, 0.0) for r in model.dealer_reward_log]
        reward_heston = [r.get(PricingModel.HESTON, 0.0) for r in model.dealer_reward_log]
        stats_text += f"Avg Rewards:\n"
        stats_text += f"  BS: {np.mean(reward_bs):.6f}\n"
        stats_text += f"  TFBS: {np.mean(reward_tfbs):.6f}\n"
        stats_text += f"  HESTON: {np.mean(reward_heston):.6f}\n\n"
    
    stats_text += f"Stress periods: {sum(model.regime_log)} / {len(model.regime_log)}\n"
    stats_text += f"Stress %: {100 * sum(model.regime_log) / len(model.regime_log):.1f}%\n"
    
    ax10.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
              verticalalignment='center', transform=ax10.transAxes)
    
    plt.tight_layout()
    return fig


def print_statistics(model, prices):
    """Print simulation statistics."""
    print("\n" + "="*60)
    print("SIMULATION STATISTICS")
    print("="*60)
    
    print(f"\nPrice:")
    print(f"  Initial: {prices[0]:.2f}")
    print(f"  Final: {prices[-1]:.2f}")
    print(f"  Range: [{np.min(prices):.2f}, {np.max(prices):.2f}]")
    
    print(f"\nMarket regimes:")
    print(f"  Stress periods: {sum(model.regime_log)} / {len(model.regime_log)}")
    print(f"  Stress %: {100 * sum(model.regime_log) / len(model.regime_log):.1f}%")
    
    if model.enable_options and len(model.dealer_model_distribution_log) > 0:
        final_dist = model.dealer_model_distribution_log[-1]
        print(f"\nFinal model distribution:")
        print(f"  BS: {final_dist.get(PricingModel.BS, 0.0):.1%}")
        print(f"  TFBS: {final_dist.get(PricingModel.TFBS, 0.0):.1%}")
        print(f"  HESTON: {final_dist.get(PricingModel.HESTON, 0.0):.1%}")
    
    if hasattr(model, 'dealer_hedge_errors_by_model'):
        bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
        tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
        heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
        if bs_errors:
            print(f"\nAverage hedging errors:")
            print(f"  BS: {np.mean(bs_errors):.6f}")
            print(f"  TFBS: {np.mean(tfbs_errors):.6f}")
            print(f"  HESTON: {np.mean(heston_errors):.6f}")
    
    if hasattr(model, 'dealer_reward_log') and len(model.dealer_reward_log) > 0:
        reward_bs = [r.get(PricingModel.BS, 0.0) for r in model.dealer_reward_log]
        reward_tfbs = [r.get(PricingModel.TFBS, 0.0) for r in model.dealer_reward_log]
        reward_heston = [r.get(PricingModel.HESTON, 0.0) for r in model.dealer_reward_log]
        print(f"\nAverage rewards:")
        print(f"  BS: {np.mean(reward_bs):.6f} (std: {np.std(reward_bs):.6f})")
        print(f"  TFBS: {np.mean(reward_tfbs):.6f} (std: {np.std(reward_tfbs):.6f})")
        print(f"  HESTON: {np.mean(reward_heston):.6f} (std: {np.std(reward_heston):.6f})")


def main():
    parser = argparse.ArgumentParser(description="Run ABM simulation with options market")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: timestamp-based)")
    args = parser.parse_args()
    
    cfg = get_config()
    
    # Set seed
    if args.seed is None:
        seed = int(time.time() * 1000000) % (2**31)
    else:
        seed = args.seed
    
    print(f"Running simulation...")
    print(f"Seed: {seed}")
    print(f"Parameters: {cfg['n_option_dealers']} dealers, {cfg['n_option_takers']} takers, "
          f"trade_prob={cfg['option_taker_trade_prob']}, steps={cfg['steps']}")
    
    # Run simulation
    model = UnifiedABMModel(seed=seed, **cfg)
    prices = model.run()
    
    # Create visualization
    fig = plot_results(model, prices)
    
    # Save plot
    timestamp = int(time.time())
    filename = f"simulation_{timestamp}.png"
    filepath = PLOTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {filepath}")
    
    # Print statistics
    print_statistics(model, prices)
    
    return model, prices


if __name__ == "__main__":
    main()
