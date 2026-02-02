"""
Experiments for testing hypotheses H1-H4:

H1: Endogenous model switching by dealers (learning/survival)
H2: Heterogeneous dealer models amplify endogenous volatility regimes
H3: Adaptive switching increases volatility amplitude and regime transitions
H4: Memory parameter (α) negatively correlated with option liquidity, positively with extreme moves
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from collections import defaultdict

from models.abm_model import UnifiedABMModel
from options.pricing import PricingModel


def run_h1_experiment(
    cfg,
    n_replications=10,
    seed0=1,
    switching_window=50,
    switching_threshold=0.05
):
    """
    H1: Test endogenous model switching by dealers.
    
    Measures:
    - Model distribution over time
    - Switching frequency
    - Performance improvement after switching
    - Regime-dependent model preferences
    """
    results = []
    
    for rep in range(n_replications):
        model = UnifiedABMModel(
            seed=seed0 + rep * 1000,
            enable_model_switching=True,
            switching_window=switching_window,
            switching_threshold=switching_threshold,
            **cfg
        )
        
        prices = model.run()
        
        # Extract model distribution over time
        model_dist = model.dealer_model_distribution_log
        
        # Calculate switching events
        switches = []
        prev_dist = None
        for dist in model_dist:
            if prev_dist is not None:
                for m in PricingModel:
                    if dist.get(m, 0) != prev_dist.get(m, 0):
                        switches.append({
                            "step": len(switches),
                            "model": m.value,
                            "from_share": prev_dist.get(m, 0),
                            "to_share": dist.get(m, 0)
                        })
            prev_dist = dist
        
        # Regime-dependent model distribution
        regime_model_dist = defaultdict(lambda: {PricingModel.BS: [], PricingModel.TFBS: [], PricingModel.HESTON: []})
        for i, regime in enumerate(model.regime_log):
            if i < len(model_dist):
                dist = model_dist[i]
                for m in PricingModel:
                    regime_model_dist[regime][m].append(dist.get(m, 0))
        
        results.append({
            "rep": rep,
            "prices": prices,
            "model_distribution": model_dist,
            "switches": switches,
            "regime_model_dist": dict(regime_model_dist),
            "regime_log": model.regime_log,
            "volatility_log": [model.market.realized_sigma(window=50) for _ in range(len(model.regime_log))]
        })
    
    return results


def run_h2_experiment(
    cfg,
    n_replications=10,
    seed0=1,
    compare_homogeneous=True
):
    """
    H2: Test if heterogeneous dealer models amplify volatility regimes.
    
    Compares:
    - Heterogeneous dealers (BS/TFBS/Heston mix) vs homogeneous (all BS)
    - Frequency/duration of stress episodes
    - Volatility clustering (autocorrelation)
    - Transition probabilities calm->stress
    """
    results_hetero = []
    results_homo = []
    
    # Heterogeneous case
    for rep in range(n_replications):
        model = UnifiedABMModel(
            seed=seed0 + rep * 1000,
            dealer_model_distribution={
                PricingModel.BS: 0.33,
                PricingModel.TFBS: 0.33,
                PricingModel.HESTON: 0.34
            },
            enable_model_switching=False,  # Fixed models for H2
            **cfg
        )
        prices = model.run()
        results_hetero.append({
            "rep": rep,
            "prices": prices,
            "regime_log": model.regime_log,
            "returns": model.market.log_returns,
            "volatility": [model.market.realized_sigma(window=50) for _ in range(len(model.regime_log))]
        })
    
    # Homogeneous case (all BS)
    if compare_homogeneous:
        for rep in range(n_replications):
            model = UnifiedABMModel(
                seed=seed0 + rep * 1000,
                dealer_model_distribution={PricingModel.BS: 1.0},
                enable_model_switching=False,
                **cfg
            )
            prices = model.run()
            results_homo.append({
                "rep": rep,
                "prices": prices,
                "regime_log": model.regime_log,
                "returns": model.market.log_returns,
                "volatility": [model.market.realized_sigma(window=50) for _ in range(len(model.regime_log))]
            })
    
    # Analyze results
    def analyze_regimes(regime_log):
        """Calculate regime statistics."""
        regime_array = np.array(regime_log)
        stress_episodes = []
        in_stress = False
        start = None
        
        for i, r in enumerate(regime_array):
            if r == 1 and not in_stress:
                start = i
                in_stress = True
            elif r == 0 and in_stress:
                stress_episodes.append(i - start)
                in_stress = False
        
        if in_stress:
            stress_episodes.append(len(regime_array) - start)
        
        return {
            "stress_frequency": len(stress_episodes) / len(regime_array) if len(regime_array) > 0 else 0,
            "mean_duration": np.mean(stress_episodes) if stress_episodes else 0,
            "total_stress_time": np.sum(regime_array) / len(regime_array) if len(regime_array) > 0 else 0
        }
    
    def calculate_autocorr(returns, max_lag=10):
        """Calculate autocorrelation of absolute returns."""
        abs_returns = np.abs(returns)
        autocorrs = []
        for lag in range(1, max_lag + 1):
            if len(abs_returns) > lag:
                corr = np.corrcoef(abs_returns[:-lag], abs_returns[lag:])[0, 1]
                autocorrs.append(corr if not np.isnan(corr) else 0.0)
        return autocorrs
    
    hetero_stats = []
    homo_stats = []
    
    for res in results_hetero:
        stats = analyze_regimes(res["regime_log"])
        stats["autocorr"] = calculate_autocorr(res["returns"])
        hetero_stats.append(stats)
    
    for res in results_homo:
        stats = analyze_regimes(res["regime_log"])
        stats["autocorr"] = calculate_autocorr(res["returns"])
        homo_stats.append(stats)
    
    return {
        "heterogeneous": results_hetero,
        "homogeneous": results_homo,
        "hetero_stats": hetero_stats,
        "homo_stats": homo_stats
    }


def run_h3_experiment(
    cfg,
    n_replications=10,
    seed0=1
):
    """
    H3: Test if adaptive switching increases volatility amplitude and regime transitions.
    
    Compares:
    - Adaptive switching vs fixed strategies
    - Volatility amplitude
    - Regime transition frequency
    """
    results_adaptive = []
    results_fixed = []
    
    # Adaptive switching
    for rep in range(n_replications):
        model = UnifiedABMModel(
            seed=seed0 + rep * 1000,
            enable_model_switching=True,
            dealer_model_distribution={
                PricingModel.BS: 0.33,
                PricingModel.TFBS: 0.33,
                PricingModel.HESTON: 0.34
            },
            **cfg
        )
        prices = model.run()
        results_adaptive.append({
            "rep": rep,
            "prices": prices,
            "regime_log": model.regime_log,
            "volatility": [model.market.realized_sigma(window=50) for _ in range(len(model.regime_log))],
            "model_distribution": model.dealer_model_distribution_log
        })
    
    # Fixed strategies
    for rep in range(n_replications):
        model = UnifiedABMModel(
            seed=seed0 + rep * 1000,
            enable_model_switching=False,
            dealer_model_distribution={
                PricingModel.BS: 0.33,
                PricingModel.TFBS: 0.33,
                PricingModel.HESTON: 0.34
            },
            **cfg
        )
        prices = model.run()
        results_fixed.append({
            "rep": rep,
            "prices": prices,
            "regime_log": model.regime_log,
            "volatility": [model.market.realized_sigma(window=50) for _ in range(len(model.regime_log))]
        })
    
    # Analyze
    def calculate_volatility_amplitude(vol_log):
        """Calculate volatility amplitude (max - min)."""
        vol_array = np.array(vol_log)
        return float(np.max(vol_array) - np.min(vol_array)) if len(vol_array) > 0 else 0.0
    
    def count_regime_transitions(regime_log):
        """Count number of regime transitions."""
        regime_array = np.array(regime_log)
        transitions = 0
        for i in range(1, len(regime_array)):
            if regime_array[i] != regime_array[i-1]:
                transitions += 1
        return transitions
    
    adaptive_stats = []
    fixed_stats = []
    
    for res in results_adaptive:
        adaptive_stats.append({
            "vol_amplitude": calculate_volatility_amplitude(res["volatility"]),
            "transitions": count_regime_transitions(res["regime_log"]),
            "mean_vol": np.mean(res["volatility"]),
            "std_vol": np.std(res["volatility"])
        })
    
    for res in results_fixed:
        fixed_stats.append({
            "vol_amplitude": calculate_volatility_amplitude(res["volatility"]),
            "transitions": count_regime_transitions(res["regime_log"]),
            "mean_vol": np.mean(res["volatility"]),
            "std_vol": np.std(res["volatility"])
        })
    
    # Statistical tests
    adaptive_amps = [s["vol_amplitude"] for s in adaptive_stats]
    fixed_amps = [s["vol_amplitude"] for s in fixed_stats]
    
    adaptive_trans = [s["transitions"] for s in adaptive_stats]
    fixed_trans = [s["transitions"] for s in fixed_stats]
    
    tstat_amp, pval_amp = ttest_rel(adaptive_amps, fixed_amps)
    tstat_trans, pval_trans = ttest_rel(adaptive_trans, fixed_trans)
    
    return {
        "adaptive": results_adaptive,
        "fixed": results_fixed,
        "adaptive_stats": adaptive_stats,
        "fixed_stats": fixed_stats,
        "test_amplitude": {"tstat": float(tstat_amp), "pval": float(pval_amp)},
        "test_transitions": {"tstat": float(tstat_trans), "pval": float(pval_trans)}
    }


def run_h4_experiment(
    cfg,
    alpha_values=[0.7, 0.8, 0.85, 0.9, 1.0],
    n_replications=10,
    seed0=1
):
    """
    H4: Test relationship between memory parameter (α) and option liquidity/extreme moves.
    
    Measures:
    - Option bid-ask spreads vs α
    - Option depth vs α
    - Frequency of extreme price moves vs α
    """
    results = []
    
    for alpha in alpha_values:
        for rep in range(n_replications):
            # Create dealers with TFBS using this alpha
            model = UnifiedABMModel(
                seed=seed0 + rep * 1000 + int(alpha * 100),
                dealer_model_distribution={PricingModel.TFBS: 1.0},
                enable_model_switching=False,
                **cfg
            )
            
            # Override alpha for TFBS dealers
            for agent in model.agents_list:
                if hasattr(agent, 'pricer') and hasattr(agent.pricer, 'alpha'):
                    agent.pricer.alpha = alpha
            
            prices = model.run()
            
            # Extract option market metrics
            option_spreads = {}
            option_depths = {}
            
            for contract_id in model.option_contracts:
                cid = contract_id.contract_id
                spreads = model.option_spreads_log.get(cid, [])
                depths = model.option_depths_log.get(cid, [])
                
                option_spreads[cid] = spreads
                option_depths[cid] = depths
            
            # Calculate extreme moves
            returns = np.array(model.market.log_returns)
            extreme_threshold = np.percentile(np.abs(returns), 95)
            extreme_moves = np.sum(np.abs(returns) > extreme_threshold)
            
            results.append({
                "alpha": alpha,
                "rep": rep,
                "prices": prices,
                "option_spreads": option_spreads,
                "option_depths": option_depths,
                "extreme_moves": extreme_moves,
                "mean_spread": {cid: np.nanmean(spreads) for cid, spreads in option_spreads.items()},
                "mean_depth": {cid: np.nanmean([d[0] + d[1] for d in depths]) for cid, depths in option_depths.items()}
            })
    
    # Aggregate by alpha
    alpha_summary = {}
    for alpha in alpha_values:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        alpha_summary[alpha] = {
            "mean_extreme_moves": np.mean([r["extreme_moves"] for r in alpha_results]),
            "mean_spread": np.mean([np.nanmean(list(r["mean_spread"].values())) for r in alpha_results]),
            "mean_depth": np.mean([np.nanmean(list(r["mean_depth"].values())) for r in alpha_results])
        }
    
    return {
        "results": results,
        "alpha_summary": alpha_summary
    }

