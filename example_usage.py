"""
Example usage of the unified ABM model with options market.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from models.abm_model import UnifiedABMModel
from options.pricing import PricingModel


def run_basic_example():
    """Run a basic simulation with options market."""
    
    # Configuration
    cfg = {
        "S0": 100.0,
        "dt": 0.001,
        "steps": 1000,
        "n_fund": 10,
        "n_chart": 10,
        "n_mm": 3,
        "n_noise": 50,
        "fundamental_price": 100.0,
        "tick_size": 0.1,  # Price step: 100.0 -> 100.1 instead of 100.00 -> 100.01
        "enable_options": True,
        "n_option_dealers": 5,
        "n_option_contracts": 3,
        "n_option_takers": 10,
        "enable_model_switching": True,
        "dealer_model_distribution": {
            PricingModel.BS: 0.33,
            PricingModel.TFBS: 0.33,
            PricingModel.HESTON: 0.34
        },
        # Regime switching parameters - more realistic: less frequent but longer stress periods
        "p01": 0.005,  # Reduced from 0.02: less frequent transitions to stress (calm -> stress)
        "p10": 0.03,   # Reduced from 0.10: slower recovery from stress (stress -> calm), making stress last longer
        "shock_rate": 0.003,  # Reduced from 0.01: less frequent exogenous shocks
    }
    
    # Run simulation with random seed for variability
    # Use microseconds for better randomness between runs
    seed = int(time.time() * 1000000) % (2**31)  # Use timestamp-based seed with microseconds
    print(f"Running simulation with seed={seed}...")
    model = UnifiedABMModel(seed=seed, **cfg)
    prices = model.run()
    
    # Print results
    print(f"\nSimulation completed!")
    print(f"Final price: {prices[-1]:.2f}")
    print(f"Price range: [{np.min(prices):.2f}, {np.max(prices):.2f}]")
    print(f"Total stress periods: {sum(model.regime_log)}")
    print(f"Stress percentage: {100 * sum(model.regime_log) / len(model.regime_log):.1f}%")
    
    # Model distribution over time
    if model.enable_options and len(model.dealer_model_distribution_log) > 0:
        final_dist = model.dealer_model_distribution_log[-1]
        print(f"\nFinal dealer model distribution:")
        for model_type in [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]:
            share = final_dist.get(model_type, 0.0)
            print(f"  {model_type.value}: {share:.1%}")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price path
    axes[0].plot(prices)
    axes[0].set_title("Price Path")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)
    
    # Regime
    axes[1].plot(model.regime_log, color='red', alpha=0.7)
    axes[1].set_title("Market Regime (0=calm, 1=stress)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Regime")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True)
    
    # Spread
    axes[2].plot(model.spread_log)
    axes[2].set_title("Bid-Ask Spread")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Spread")
    axes[2].grid(True)
    
    plt.tight_layout()
    timestamp = int(time.time())
    filename = f"simulation_results_{timestamp}.png"
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to {filename}")
    
    return model, prices


def run_model_switching_example():
    """Demonstrate model switching behavior."""
    
    cfg = {
        "S0": 100.0,
        "dt": 0.001,
        "steps": 2000,
        "n_fund": 10,
        "n_chart": 10,
        "n_mm": 3,
        "n_noise": 50,
        "fundamental_price": 100.0,
        "tick_size": 0.1,  # Price step: 100.0 -> 100.1 instead of 100.00 -> 100.01
        "enable_options": True,
        "n_option_dealers": 15,
        "n_option_takers": 20,
        "n_option_contracts": 3,
        "enable_model_switching": True,
        "switching_window": 10,
        "switching_threshold": 0.03,
        "option_taker_trade_prob": 0.04,  # Probability of option taker trading each step
        "debug": False,  # Set to True to see switching events
        # Regime switching parameters - more realistic: less frequent but longer stress periods
        "p01": 0.005,  # Reduced from 0.02: less frequent transitions to stress (calm -> stress)
        "p10": 0.03,   # Reduced from 0.10: slower recovery from stress (stress -> calm), making stress last longer
        "shock_rate": 0.003,  # Reduced from 0.01: less frequent exogenous shocks
    }
    
    # Use random seed for variability
    # Use microseconds for better randomness between runs
    seed = int(time.time() * 1000000) % (2**31)  # Use timestamp-based seed with microseconds
    print(f"Running model switching simulation with seed={seed}...")
    model = UnifiedABMModel(seed=seed, **cfg)
    prices = model.run()
    
    # Track model distribution over time
    if len(model.dealer_model_distribution_log) > 0:
        bs_share = [d.get(PricingModel.BS, 0) for d in model.dealer_model_distribution_log]
        tfbs_share = [d.get(PricingModel.TFBS, 0) for d in model.dealer_model_distribution_log]
        heston_share = [d.get(PricingModel.HESTON, 0) for d in model.dealer_model_distribution_log]
        
        # Calculate distribution by regime
        calm_bs = []
        calm_tfbs = []
        calm_heston = []
        stress_bs = []
        stress_tfbs = []
        stress_heston = []
        
        for i, (dist, regime) in enumerate(zip(model.dealer_model_distribution_log, model.regime_log)):
            if regime == 0:  # Calm
                calm_bs.append(dist.get(PricingModel.BS, 0))
                calm_tfbs.append(dist.get(PricingModel.TFBS, 0))
                calm_heston.append(dist.get(PricingModel.HESTON, 0))
            else:  # Stress
                stress_bs.append(dist.get(PricingModel.BS, 0))
                stress_tfbs.append(dist.get(PricingModel.TFBS, 0))
                stress_heston.append(dist.get(PricingModel.HESTON, 0))
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 14))
        
        # Model distribution over time with regime shading
        axes[0].fill_between(range(len(model.regime_log)), 0, 1, 
                            where=np.array(model.regime_log) == 1, 
                            alpha=0.2, color='red', label='Stress regime')
        axes[0].plot(bs_share, label="BS", alpha=0.8, linewidth=2)
        axes[0].plot(tfbs_share, label="TFBS", alpha=0.8, linewidth=2)
        axes[0].plot(heston_share, label="Heston", alpha=0.8, linewidth=2)
        axes[0].set_title("Dealer Model Distribution Over Time (shaded = stress regime)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Share")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Average distribution by regime
        if calm_bs:
            axes[1].bar(['Calm', 'Stress'], 
                       [np.mean(calm_bs), np.mean(stress_bs)],
                       label='BS', alpha=0.7, color='blue')
            axes[1].bar(['Calm', 'Stress'], 
                       [np.mean(calm_tfbs), np.mean(stress_tfbs)],
                       bottom=[np.mean(calm_bs), np.mean(stress_bs)],
                       label='TFBS', alpha=0.7, color='orange')
            axes[1].bar(['Calm', 'Stress'], 
                       [np.mean(calm_heston), np.mean(stress_heston)],
                       bottom=[np.mean(calm_bs) + np.mean(calm_tfbs), 
                              np.mean(stress_bs) + np.mean(stress_tfbs)],
                       label='Heston', alpha=0.7, color='green')
            axes[1].set_title("Average Model Distribution by Regime")
            axes[1].set_ylabel("Share")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim(0, 1)
        
        # Average hedging errors over time for each model
        if hasattr(model, 'dealer_hedge_errors_by_model'):
            bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
            tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
            heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
            
            # Pad shorter lists with zeros to match regime_log length
            target_len = len(model.regime_log)
            if bs_errors and len(bs_errors) < target_len:
                bs_errors = list(bs_errors) + [0.0] * (target_len - len(bs_errors))
            if tfbs_errors and len(tfbs_errors) < target_len:
                tfbs_errors = list(tfbs_errors) + [0.0] * (target_len - len(tfbs_errors))
            if heston_errors and len(heston_errors) < target_len:
                heston_errors = list(heston_errors) + [0.0] * (target_len - len(heston_errors))
            
            if bs_errors and len(bs_errors) == len(model.regime_log):
                max_error = max([max(bs_errors) if bs_errors else 0, 
                                max(tfbs_errors) if tfbs_errors else 0, 
                                max(heston_errors) if heston_errors else 0, 1e-6])
                axes[2].fill_between(range(len(model.regime_log)), 0, max_error, 
                                    where=np.array(model.regime_log) == 1, 
                                    alpha=0.15, color='red', label='Stress regime')
                axes[2].plot(bs_errors, label="BS errors", alpha=0.8, linewidth=1.5, color='blue')
                axes[2].plot(tfbs_errors, label="TFBS errors", alpha=0.8, linewidth=1.5, color='orange')
                axes[2].plot(heston_errors, label="Heston errors", alpha=0.8, linewidth=1.5, color='green')
                axes[2].set_title("Average Hedging Errors by Model Over Time")
                axes[2].set_xlabel("Step")
                axes[2].set_ylabel("Average Error")
                axes[2].legend(loc='upper right')
                axes[2].grid(True, alpha=0.3)
                if max_error > 0:
                    axes[2].set_yscale('log')  # Log scale for better visibility
            else:
                axes[2].text(0.5, 0.5, 'No hedging error data available', 
                            ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title("Average Hedging Errors by Model Over Time")
        else:
            axes[2].text(0.5, 0.5, 'Hedging error tracking not enabled', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title("Average Hedging Errors by Model Over Time")
        
        # Cumulative hedging errors (to see which model accumulates less error)
        if hasattr(model, 'dealer_hedge_errors_by_model'):
            bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
            tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
            heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
            
            # Pad shorter lists with zeros to match regime_log length
            target_len = len(model.regime_log)
            if bs_errors and len(bs_errors) < target_len:
                bs_errors = list(bs_errors) + [0.0] * (target_len - len(bs_errors))
            if tfbs_errors and len(tfbs_errors) < target_len:
                tfbs_errors = list(tfbs_errors) + [0.0] * (target_len - len(tfbs_errors))
            if heston_errors and len(heston_errors) < target_len:
                heston_errors = list(heston_errors) + [0.0] * (target_len - len(heston_errors))
            
            if bs_errors and len(bs_errors) == len(model.regime_log):
                bs_cumulative = np.cumsum(bs_errors)
                tfbs_cumulative = np.cumsum(tfbs_errors)
                heston_cumulative = np.cumsum(heston_errors)
                
                max_cumulative = max([max(bs_cumulative) if bs_cumulative.size > 0 else 0, 
                                     max(tfbs_cumulative) if tfbs_cumulative.size > 0 else 0, 
                                     max(heston_cumulative) if heston_cumulative.size > 0 else 0, 1e-6])
                axes[3].fill_between(range(len(model.regime_log)), 0, max_cumulative, 
                                    where=np.array(model.regime_log) == 1, 
                                    alpha=0.15, color='red', label='Stress regime')
                axes[3].plot(bs_cumulative, label="BS cumulative", alpha=0.8, linewidth=2, color='blue')
                axes[3].plot(tfbs_cumulative, label="TFBS cumulative", alpha=0.8, linewidth=2, color='orange')
                axes[3].plot(heston_cumulative, label="Heston cumulative", alpha=0.8, linewidth=2, color='green')
                axes[3].set_title("Cumulative Hedging Errors by Model")
                axes[3].set_xlabel("Step")
                axes[3].set_ylabel("Cumulative Error")
                axes[3].legend(loc='upper left')
                axes[3].grid(True, alpha=0.3)
            else:
                axes[3].text(0.5, 0.5, 'No hedging error data available', 
                            ha='center', va='center', transform=axes[3].transAxes)
                axes[3].set_title("Cumulative Hedging Errors by Model")
        else:
            axes[3].text(0.5, 0.5, 'Hedging error tracking not enabled', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title("Cumulative Hedging Errors by Model")
        
        # Regime timeline
        axes[4].plot(model.regime_log, color='red', alpha=0.7, linewidth=1)
        axes[4].fill_between(range(len(model.regime_log)), 0, 1, 
                            where=np.array(model.regime_log) == 1, 
                            alpha=0.3, color='red')
        axes[4].set_title("Market Regime (0=calm, 1=stress)")
        axes[4].set_xlabel("Step")
        axes[4].set_ylabel("Regime")
        axes[4].set_ylim(-0.1, 1.1)
        axes[4].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\nModel distribution statistics:")
        print(f"  Calm regime - BS: {np.mean(calm_bs):.1%}, TFBS: {np.mean(calm_tfbs):.1%}, Heston: {np.mean(calm_heston):.1%}")
        print(f"  Stress regime - BS: {np.mean(stress_bs):.1%}, TFBS: {np.mean(stress_tfbs):.1%}, Heston: {np.mean(stress_heston):.1%}")
        
        if hasattr(model, 'dealer_hedge_errors_by_model'):
            bs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.BS, [])
            tfbs_errors = model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [])
            heston_errors = model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [])
            if bs_errors:
                print(f"\nAverage hedging errors:")
                print(f"  BS: {np.mean(bs_errors):.6f}")
                print(f"  TFBS: {np.mean(tfbs_errors):.6f}")
                print(f"  Heston: {np.mean(heston_errors):.6f}")
                print(f"\nTotal cumulative errors:")
                print(f"  BS: {np.sum(bs_errors):.6f}")
                print(f"  TFBS: {np.sum(tfbs_errors):.6f}")
                print(f"  Heston: {np.sum(heston_errors):.6f}")
        
        plt.tight_layout()
        timestamp = int(time.time())
        filename = f"model_switching_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"Model switching plot saved to {filename}")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Unified ABM with Options Market - Example Usage")
    print("=" * 60)
    
    # Run basic example
    model, prices = run_basic_example()
    
    print("\n" + "=" * 60)
    print("Model Switching Example")
    print("=" * 60)
    
    # Run model switching example
    model2 = run_model_switching_example()

