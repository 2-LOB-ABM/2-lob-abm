"""
Experiments for testing hypotheses H1-H4:

H1: Endogenous model switching by dealers (learning/survival)
H2: Heterogeneous dealer models amplify endogenous volatility regimes
H3: Adaptive switching increases volatility amplitude and regime transitions
H4: Memory parameter (α) negatively correlated with option liquidity, positively with extreme moves
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu, wilcoxon
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
    compare_homogeneous=True,
    max_lag=10,
    rv_window=20
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
    base_cfg = dict(cfg)
    base_cfg.pop("enable_model_switching", None)
    base_cfg.pop("dealer_model_distribution", None)
    
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
            **base_cfg
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
                **base_cfg
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
    def _safe_mean(x):
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else 0.0

    def _safe_autocorr(x, lag):
        arr = np.asarray(x, dtype=float)
        if arr.size <= lag:
            return 0.0
        x0 = arr[:-lag]
        x1 = arr[lag:]
        if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
            return 0.0
        corr = np.corrcoef(x0, x1)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    def _acf_series(x, max_lag_):
        return [_safe_autocorr(x, lag) for lag in range(1, max_lag_ + 1)]

    def _realized_variance_series(returns, window):
        arr = np.asarray(returns, dtype=float)
        if arr.size < window:
            return np.array([], dtype=float)
        sq = arr ** 2
        kernel = np.ones(window, dtype=float)
        rv = np.convolve(sq, kernel, mode="valid")
        return rv

    def _stress_episode_lengths(regime_array):
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
        return stress_episodes

    def analyze_regimes(regime_log, returns, max_lag_, rv_window_):
        """
        Calculate H2 regime statistics.
        Includes autocorrelation diagnostics for |r| and RV:
        persistent positive ACF on several lags indicates endogenous volatility clustering.
        """
        regime_array = np.array(regime_log)
        returns_array = np.asarray(returns, dtype=float)

        stress_episodes = _stress_episode_lengths(regime_array)

        # Transition probabilities.
        if regime_array.size > 1:
            prev = regime_array[:-1]
            nxt = regime_array[1:]
            calm_mask = prev == 0
            stress_mask = prev == 1
            calm_to_stress = np.sum(calm_mask & (nxt == 1))
            stress_to_calm = np.sum(stress_mask & (nxt == 0))
            n_calm = np.sum(calm_mask)
            n_stress = np.sum(stress_mask)
            p_calm_to_stress = float(calm_to_stress / n_calm) if n_calm > 0 else 0.0
            p_stress_to_calm = float(stress_to_calm / n_stress) if n_stress > 0 else 0.0
        else:
            p_calm_to_stress = 0.0
            p_stress_to_calm = 0.0

        abs_ret = np.abs(returns_array)
        abs_ret_acf = _acf_series(abs_ret, max_lag_)
        rv = _realized_variance_series(returns_array, rv_window_)
        rv_acf = _acf_series(rv, max_lag_) if rv.size > 0 else [0.0] * max_lag_

        return {
            "stress_frequency": len(stress_episodes) / len(regime_array) if len(regime_array) > 0 else 0,
            "mean_duration": np.mean(stress_episodes) if stress_episodes else 0,
            "n_stress_episodes": len(stress_episodes),
            "total_stress_time": np.sum(regime_array) / len(regime_array) if len(regime_array) > 0 else 0,
            "p_calm_to_stress": p_calm_to_stress,
            "p_stress_to_calm": p_stress_to_calm,
            "abs_ret_acf": abs_ret_acf,
            "rv_acf": rv_acf,
            "abs_ret_acf_mean": _safe_mean(abs_ret_acf),
            "rv_acf_mean": _safe_mean(rv_acf),
        }

    hetero_stats = []
    homo_stats = []
    
    for res in results_hetero:
        stats = analyze_regimes(
            res["regime_log"],
            res["returns"],
            max_lag_=max_lag,
            rv_window_=rv_window,
        )
        hetero_stats.append(stats)
    
    for res in results_homo:
        stats = analyze_regimes(
            res["regime_log"],
            res["returns"],
            max_lag_=max_lag,
            rv_window_=rv_window,
        )
        homo_stats.append(stats)

    def _metric_vec(stats_list, metric):
        vals = [float(s.get(metric, 0.0)) for s in stats_list]
        arr = np.asarray(vals, dtype=float)
        return arr[np.isfinite(arr)]

    def _compare_metric(metric, alternative="greater"):
        x = _metric_vec(hetero_stats, metric)
        y = _metric_vec(homo_stats, metric)
        if x.size == 0 or y.size == 0:
            return {
                "hetero_mean": float("nan"),
                "homo_mean": float("nan"),
                "delta": float("nan"),
                "ttest_pval": float("nan"),
                "mannwhitney_pval": float("nan"),
                "supported": False,
            }

        # Welch t-test (robust to unequal variances) + Mann-Whitney as nonparametric check.
        t_res = ttest_ind(x, y, equal_var=False, alternative=alternative)
        mw_res = mannwhitneyu(x, y, alternative=alternative)

        hetero_mean = float(np.mean(x))
        homo_mean = float(np.mean(y))
        delta = hetero_mean - homo_mean
        supported = (
            (delta > 0)
            and np.isfinite(t_res.pvalue)
            and np.isfinite(mw_res.pvalue)
            and (float(t_res.pvalue) < 0.05)
            and (float(mw_res.pvalue) < 0.05)
        )
        return {
            "hetero_mean": hetero_mean,
            "homo_mean": homo_mean,
            "delta": delta,
            "ttest_pval": float(t_res.pvalue),
            "mannwhitney_pval": float(mw_res.pvalue),
            "supported": bool(supported),
        }

    comparison = {
        "mean_stress_duration": _compare_metric("mean_duration", alternative="greater"),
        "stress_episode_frequency": _compare_metric("stress_frequency", alternative="greater"),
        "calm_to_stress_probability": _compare_metric("p_calm_to_stress", alternative="greater"),
        "abs_return_autocorr_mean": _compare_metric("abs_ret_acf_mean", alternative="greater"),
        "rv_autocorr_mean": _compare_metric("rv_acf_mean", alternative="greater"),
    }

    key_metrics = [
        "mean_stress_duration",
        "stress_episode_frequency",
        "calm_to_stress_probability",
        "abs_return_autocorr_mean",
        "rv_autocorr_mean",
    ]
    n_supported = sum(1 for k in key_metrics if comparison[k]["supported"])
    
    return {
        "heterogeneous": results_hetero,
        "homogeneous": results_homo,
        "hetero_stats": hetero_stats,
        "homo_stats": homo_stats,
        "comparison": comparison,
        "h2_supported": n_supported >= 3,
        "h2_support_count": n_supported,
        "h2_support_total": len(key_metrics),
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

