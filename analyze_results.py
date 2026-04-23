"""
Statistical analysis of simulation results.

Evaluates hypotheses (H1-H4) and provides quantitative assessment of model switching,
regime effects, and dealer behavior patterns.

Usage:
    python3 analyze_results.py --market simulation_logs/market_log_*.csv --dealers simulation_logs/dealers_log_*.csv
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_ind, mannwhitneyu


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null_mean: float
    p_value_two_sided: float
    n_perm: int


def circular_shift_perm_test(
    x: np.ndarray,
    regime: np.ndarray,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """
    Permutation test where the regime series is circularly shifted.
    This preserves regime autocorrelation / episode structure.
    """
    x = np.asarray(x, dtype=float)
    regime = np.asarray(regime, dtype=int)
    if x.shape != regime.shape:
        raise ValueError("x and regime must have same shape")
    if x.size < 10:
        raise ValueError("Not enough observations for permutation test")

    rng = rng or np.random.default_rng(0)

    def diff_in_means(reg: np.ndarray) -> float:
        calm = x[reg == 0]
        stress = x[reg == 1]
        if calm.size == 0 or stress.size == 0:
            return float("nan")
        return float(np.nanmean(stress) - np.nanmean(calm))

    obs = diff_in_means(regime)
    if not np.isfinite(obs):
        return PermutationResult(observed=obs, null_mean=float("nan"), p_value_two_sided=float("nan"), n_perm=0)

    shifts = rng.integers(0, x.size, size=int(n_perm))
    null = np.empty(len(shifts), dtype=float)
    for i, s in enumerate(shifts):
        null[i] = diff_in_means(np.roll(regime, int(s)))

    null = null[np.isfinite(null)]
    if null.size == 0:
        return PermutationResult(observed=obs, null_mean=float("nan"), p_value_two_sided=float("nan"), n_perm=0)

    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    return PermutationResult(observed=obs, null_mean=float(np.mean(null)), p_value_two_sided=float(p), n_perm=int(null.size))


def strategy_to_model_key(strategy: str) -> str | None:
    """Convert strategy name to model key."""
    s = str(strategy)
    if s == "BlackScholes":
        return "BS"
    if s == "TimeFractionalBS":
        return "TFBS"
    if s == "Heston":
        return "HESTON"
    return None


def analyze_regime_model_relationship(market_df: pd.DataFrame) -> dict:
    """Analyze relationship between regime and model distribution."""
    regime = market_df["regime"].astype(int).to_numpy()
    results = {}

    for model in ["BS", "TFBS", "HESTON"]:
        col = f"model_dist_{model}"
        if col not in market_df.columns:
            continue

        shares = market_df[col].astype(float).to_numpy()
        calm_shares = shares[regime == 0]
        stress_shares = shares[regime == 1]

        calm_mean = float(np.nanmean(calm_shares))
        stress_mean = float(np.nanmean(stress_shares))
        diff = stress_mean - calm_mean

        perm = circular_shift_perm_test(shares, regime, n_perm=3000)
        tstat, t_pval = ttest_ind(stress_shares, calm_shares, nan_policy='omit')
        ustat, u_pval = mannwhitneyu(stress_shares, calm_shares, alternative='two-sided', nan_policy='omit')

        results[model] = {
            "calm_mean": calm_mean,
            "stress_mean": stress_mean,
            "diff": diff,
            "perm_pval": perm.p_value_two_sided,
            "t_test_pval": float(t_pval),
            "mannwhitney_pval": float(u_pval),
            "calm_std": float(np.nanstd(calm_shares)),
            "stress_std": float(np.nanstd(stress_shares)),
        }

    return results


def analyze_switching_patterns(dealer_df: pd.DataFrame) -> dict:
    """Analyze switching patterns and optimality."""
    if "strategy_switched" not in dealer_df.columns:
        return {"n_switches": 0}

    sw = dealer_df[dealer_df["strategy_switched"].astype(int) == 1].copy()
    sw = sw[sw["current_strategy"].notna()]

    if len(sw) == 0:
        return {"n_switches": 0}

    switch_directions = {"BS": 0, "TFBS": 0, "HESTON": 0}
    for _, row in sw.iterrows():
        chosen = strategy_to_model_key(row["current_strategy"])
        if chosen:
            switch_directions[chosen] += 1

    total_sw = sum(switch_directions.values())

    # Optimality tests
    hedge_cols = {
        "BS": "hedge_error_BS",
        "TFBS": "hedge_error_TFBS",
        "HESTON": "hedge_error_HESTON",
    }
    qual_cols = {
        "BS": "quality_BS",
        "TFBS": "quality_TFBS",
        "HESTON": "quality_HESTON",
    }

    n_best_err, n_sw_err, rate_err = _pick_best_rate(sw, hedge_cols, better="min")
    n_best_q, n_sw_q, rate_q = _pick_best_rate(sw, qual_cols, better="max")

    p0 = 1.0 / 3.0
    p_err = float(binomtest(n_best_err, n_sw_err, p=p0, alternative="greater").pvalue) if n_sw_err else float("nan")
    p_q = float(binomtest(n_best_q, n_sw_q, p=p0, alternative="greater").pvalue) if n_sw_q else float("nan")

    return {
        "n_switches": len(sw),
        "switch_directions": switch_directions,
        "switch_direction_pct": {k: (v / total_sw * 100) if total_sw > 0 else 0 for k, v in switch_directions.items()},
        "chose_min_error": {"n": n_best_err, "total": n_sw_err, "rate": rate_err, "pval": p_err},
        "chose_max_quality": {"n": n_best_q, "total": n_sw_q, "rate": rate_q, "pval": p_q},
    }


def _pick_best_rate(
    sw_df: pd.DataFrame,
    metric_cols: Dict[str, str],
    better: str,
) -> Tuple[int, int, float]:
    """Check whether chosen strategy is best according to metric."""
    if better not in ("min", "max"):
        raise ValueError("better must be 'min' or 'max'")

    n_total = 0
    n_best = 0

    for _, r in sw_df.iterrows():
        chosen = strategy_to_model_key(r["current_strategy"])
        if chosen is None:
            continue

        vals = {}
        ok = True
        for m, col in metric_cols.items():
            if col not in sw_df.columns:
                continue
            v = r[col]
            if not np.isfinite(v):
                ok = False
                break
            vals[m] = float(v)

        if not ok or len(vals) < 3:
            continue

        n_total += 1
        if better == "min":
            best_m = min(vals, key=vals.get)
        else:
            best_m = max(vals, key=vals.get)
        if chosen == best_m:
            n_best += 1

    rate = (n_best / n_total) if n_total else float("nan")
    return n_best, n_total, float(rate)


def analyze_performance_metrics(market_df: pd.DataFrame, dealer_df: pd.DataFrame) -> dict:
    """Analyze performance metrics by model."""
    results = {}

    for model in ["BS", "TFBS", "HESTON"]:
        model_key = {"BS": "BlackScholes", "TFBS": "TimeFractionalBS", "HESTON": "Heston"}[model]
        model_dealers = dealer_df[dealer_df["current_strategy"] == model_key]

        if len(model_dealers) == 0:
            continue

        results[model] = {
            "avg_hedge_error": float(model_dealers[f"hedge_error_{model}"].mean()),
            "avg_reward": float(model_dealers["reward"].mean()),
            "avg_quality": float(model_dealers[f"quality_{model}"].mean()),
            "std_hedge_error": float(model_dealers[f"hedge_error_{model}"].std()),
            "std_reward": float(model_dealers["reward"].std()),
            "n_observations": len(model_dealers),
        }

    # Final distribution
    final_step = dealer_df["step"].max()
    final_dealers = dealer_df[dealer_df["step"] == final_step]
    final_strategies = final_dealers["current_strategy"].value_counts()
    total_final = len(final_dealers)

    results["final_distribution"] = {
        strategy: (count / total_final * 100) if total_final > 0 else 0
        for strategy, count in final_strategies.items()
    }

    return results


def analyze_h2_regime_persistence(
    market_df: pd.DataFrame,
    max_lag: int = 10,
    rv_window: int = 20,
) -> dict:
    """
    H2 diagnostics from a single run.
    We cannot run a strict hetero-vs-homo causal test from one log pair, but we can
    measure signatures of endogenous volatility regimes:
    - stress episode durations,
    - transition probabilities calm->stress and stress->calm,
    - autocorrelation persistence of |r| and rolling RV.
    """
    if "regime" not in market_df.columns or "price" not in market_df.columns:
        return {"available": False, "reason": "Missing regime/price columns"}

    regime = market_df["regime"].astype(int).to_numpy()
    prices = market_df["price"].astype(float).to_numpy()
    if prices.size < 5 or regime.size < 5:
        return {"available": False, "reason": "Not enough observations"}

    # Returns aligned to regime[1:].
    ret = np.diff(np.log(np.clip(prices, 1e-12, None)))
    regime_ret = regime[1:] if regime.size == prices.size else regime[:ret.size]

    def _stress_episodes(reg: np.ndarray) -> list[int]:
        out = []
        in_stress = False
        start = 0
        for i, r in enumerate(reg):
            if r == 1 and not in_stress:
                in_stress = True
                start = i
            elif r == 0 and in_stress:
                out.append(i - start)
                in_stress = False
        if in_stress:
            out.append(len(reg) - start)
        return out

    def _safe_autocorr(x: np.ndarray, lag: int) -> float:
        if x.size <= lag:
            return 0.0
        x0 = x[:-lag]
        x1 = x[lag:]
        if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
            return 0.0
        corr = np.corrcoef(x0, x1)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    def _acf(x: np.ndarray, k_max: int) -> list[float]:
        return [_safe_autocorr(x, k) for k in range(1, k_max + 1)]

    episodes = _stress_episodes(regime_ret)

    prev = regime_ret[:-1]
    nxt = regime_ret[1:]
    calm_mask = prev == 0
    stress_mask = prev == 1
    n_calm = int(np.sum(calm_mask))
    n_stress = int(np.sum(stress_mask))
    calm_to_stress = int(np.sum(calm_mask & (nxt == 1)))
    stress_to_calm = int(np.sum(stress_mask & (nxt == 0)))
    p_calm_to_stress = float(calm_to_stress / n_calm) if n_calm > 0 else 0.0
    p_stress_to_calm = float(stress_to_calm / n_stress) if n_stress > 0 else 0.0

    abs_ret = np.abs(ret)
    abs_acf = _acf(abs_ret, max_lag)
    rv = np.convolve(ret**2, np.ones(rv_window), mode="valid") if ret.size >= rv_window else np.array([], dtype=float)
    rv_acf = _acf(rv, max_lag) if rv.size else [0.0] * max_lag

    abs_persistent_lags = int(np.sum(np.asarray(abs_acf) > 0))
    rv_persistent_lags = int(np.sum(np.asarray(rv_acf) > 0))
    abs_acf_mean = float(np.mean(abs_acf)) if abs_acf else 0.0
    rv_acf_mean = float(np.mean(rv_acf)) if rv_acf else 0.0

    # Optional heterogeneity context from model shares (if available in logs).
    hetero_info = None
    share_cols = [c for c in ["model_dist_BS", "model_dist_TFBS", "model_dist_HESTON"] if c in market_df.columns]
    if len(share_cols) == 3:
        shares = market_df[share_cols].astype(float).to_numpy()
        # Simpson diversity index = 1 - sum(p_i^2), higher -> more heterogeneous mix.
        simpson = 1.0 - np.sum(np.clip(shares, 0.0, 1.0) ** 2, axis=1)
        hetero_info = {
            "mean_simpson_diversity": float(np.nanmean(simpson)),
            "calm_simpson_diversity": float(np.nanmean(simpson[market_df["regime"].to_numpy() == 0])),
            "stress_simpson_diversity": float(np.nanmean(simpson[market_df["regime"].to_numpy() == 1])),
        }

    signatures = {
        "stress_persistence": float(np.mean(episodes)) > 1.0 if episodes else False,
        "abs_ret_clustering": abs_persistent_lags >= max(2, max_lag // 3),
        "rv_clustering": rv_persistent_lags >= max(2, max_lag // 3),
    }
    h2_supported_single_run = sum(bool(v) for v in signatures.values()) >= 2

    return {
        "available": True,
        "mean_stress_duration": float(np.mean(episodes)) if episodes else 0.0,
        "median_stress_duration": float(np.median(episodes)) if episodes else 0.0,
        "n_stress_episodes": int(len(episodes)),
        "p_calm_to_stress": p_calm_to_stress,
        "p_stress_to_calm": p_stress_to_calm,
        "abs_ret_acf": abs_acf,
        "rv_acf": rv_acf,
        "abs_ret_acf_mean": abs_acf_mean,
        "rv_acf_mean": rv_acf_mean,
        "abs_ret_positive_lags": abs_persistent_lags,
        "rv_positive_lags": rv_persistent_lags,
        "heterogeneity": hetero_info,
        "signatures": signatures,
        "h2_supported_single_run": h2_supported_single_run,
    }


def analyze_h3_adaptation_instability(
    market_df: pd.DataFrame,
    dealer_df: pd.DataFrame,
    rv_window: int = 20,
) -> dict:
    """
    H3 diagnostics from a single run.
    Hypothesis statement is comparative (adaptive vs fixed), so here we estimate
    whether observed adaptive switching co-moves with instability signatures.
    """
    if "price" not in market_df.columns or "regime" not in market_df.columns:
        return {"available": False, "reason": "Missing market columns price/regime"}
    if "strategy_switched" not in dealer_df.columns or "step" not in dealer_df.columns:
        return {"available": False, "reason": "Missing dealer columns strategy_switched/step"}

    prices = market_df["price"].astype(float).to_numpy()
    regime = market_df["regime"].astype(int).to_numpy()
    if prices.size < 10:
        return {"available": False, "reason": "Not enough observations"}

    ret = np.diff(np.log(np.clip(prices, 1e-12, None)))
    abs_ret = np.abs(ret)
    rv = np.convolve(ret**2, np.ones(rv_window), mode="valid") if ret.size >= rv_window else np.array([], dtype=float)

    per_step = dealer_df.groupby("step", as_index=False)["strategy_switched"].mean().rename(
        columns={"strategy_switched": "switch_rate"}
    )
    switch_rate = per_step["switch_rate"].astype(float).to_numpy()

    # Align to returns timeline (returns correspond to steps >= 2 if step starts at 1).
    n = min(switch_rate.size, abs_ret.size, max(regime.size - 1, 0))
    if n < 10:
        return {"available": False, "reason": "Not enough aligned observations"}

    switch_rate = switch_rate[:n]
    abs_ret = abs_ret[:n]
    regime_ret = regime[1 : n + 1]
    rv_n = rv[:n] if rv.size else np.array([], dtype=float)

    transition = (regime_ret[1:] != regime_ret[:-1]).astype(float)
    switch_for_transition = switch_rate[1:]

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return 0.0
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        return float(c) if np.isfinite(c) else 0.0

    corr_switch_absret = _safe_corr(switch_rate, abs_ret)
    corr_switch_transition = _safe_corr(switch_for_transition, transition)
    corr_switch_rv = _safe_corr(switch_rate, rv_n) if rv_n.size else 0.0

    q75 = float(np.quantile(switch_rate, 0.75))
    high_switch = switch_rate >= q75
    low_switch = switch_rate < q75
    mean_absret_high = float(np.mean(abs_ret[high_switch])) if np.any(high_switch) else 0.0
    mean_absret_low = float(np.mean(abs_ret[low_switch])) if np.any(low_switch) else 0.0
    rv_high = float(np.mean(rv_n[high_switch[: rv_n.size]])) if rv_n.size and np.any(high_switch[: rv_n.size]) else 0.0
    rv_low = float(np.mean(rv_n[low_switch[: rv_n.size]])) if rv_n.size and np.any(low_switch[: rv_n.size]) else 0.0

    signatures = {
        "switching_present": float(np.mean(switch_rate)) > 0.0,
        "switching_vs_absret": corr_switch_absret > 0.05,
        "switching_vs_regime_transitions": corr_switch_transition > 0.05,
        "high_switch_higher_volatility": mean_absret_high > mean_absret_low,
        "high_switch_higher_rv": rv_high > rv_low if rv_n.size else False,
    }
    h3_supported_single_run = sum(bool(v) for v in signatures.values()) >= 3

    return {
        "available": True,
        "mean_switch_rate": float(np.mean(switch_rate)),
        "corr_switch_absret": corr_switch_absret,
        "corr_switch_rv": corr_switch_rv,
        "corr_switch_transition": corr_switch_transition,
        "mean_absret_high_switch": mean_absret_high,
        "mean_absret_low_switch": mean_absret_low,
        "mean_rv_high_switch": rv_high,
        "mean_rv_low_switch": rv_low,
        "signatures": signatures,
        "h3_supported_single_run": h3_supported_single_run,
    }


def analyze_h4_memory_liquidity(
    market_df: pd.DataFrame,
    dealer_df: pd.DataFrame,
) -> dict:
    """
    H4 diagnostics from a single run.
    Logs do not contain dealer-level alpha directly, so we use alpha_eff proxy:
    alpha_eff ~ TFBS prevalence (share/weight), which tracks effective memory-model presence.
    """
    required_market = {"price", "option_spread_avg", "option_depth_avg"}
    if not required_market.issubset(set(market_df.columns)):
        missing = sorted(required_market - set(market_df.columns))
        return {"available": False, "reason": f"Missing market columns: {missing}"}

    prices = market_df["price"].astype(float).to_numpy()
    if prices.size < 20:
        return {"available": False, "reason": "Not enough observations"}
    ret = np.diff(np.log(np.clip(prices, 1e-12, None)))
    abs_ret = np.abs(ret)
    extreme_thr = float(np.nanpercentile(abs_ret, 95))
    extreme = (abs_ret > extreme_thr).astype(float)
    if extreme.size == 0:
        return {"available": False, "reason": "Cannot compute extreme-move series"}

    # alpha_eff proxy priority:
    # 1) market model_dist_TFBS, 2) dealer mean weight_TFBS, 3) dealer TFBS strategy share.
    alpha_eff = None
    alpha_source = None
    if "model_dist_TFBS" in market_df.columns:
        alpha_eff = market_df["model_dist_TFBS"].astype(float).to_numpy()
        alpha_source = "market:model_dist_TFBS"
    elif "weight_TFBS" in dealer_df.columns and "step" in dealer_df.columns:
        alpha_eff = dealer_df.groupby("step")["weight_TFBS"].mean().astype(float).to_numpy()
        alpha_source = "dealer:mean_weight_TFBS"
    elif "current_strategy" in dealer_df.columns and "step" in dealer_df.columns:
        tfbs_share = (
            dealer_df.assign(is_tfbs=(dealer_df["current_strategy"] == "TimeFractionalBS").astype(float))
            .groupby("step")["is_tfbs"]
            .mean()
            .astype(float)
            .to_numpy()
        )
        alpha_eff = tfbs_share
        alpha_source = "dealer:tfbs_share"

    if alpha_eff is None:
        return {"available": False, "reason": "Cannot build alpha_eff proxy from logs"}

    spread = market_df["option_spread_avg"].astype(float).to_numpy()
    depth = market_df["option_depth_avg"].astype(float).to_numpy()
    liquidity_score = depth / (1.0 + np.maximum(spread, 0.0))

    n = min(alpha_eff.size, liquidity_score.size, extreme.size)
    if n < 20:
        return {"available": False, "reason": "Not enough aligned observations"}

    alpha_eff = alpha_eff[:n]
    spread = spread[:n]
    depth = depth[:n]
    liquidity_score = liquidity_score[:n]
    extreme = extreme[:n]

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return 0.0
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        return float(c) if np.isfinite(c) else 0.0

    corr_alpha_liquidity = _safe_corr(alpha_eff, liquidity_score)  # expected negative
    corr_alpha_spread = _safe_corr(alpha_eff, spread)              # expected positive (worse liquidity)
    corr_alpha_depth = _safe_corr(alpha_eff, depth)                # expected negative
    corr_alpha_extreme = _safe_corr(alpha_eff, extreme)            # expected positive

    q75 = float(np.quantile(alpha_eff, 0.75))
    high_alpha = alpha_eff >= q75
    low_alpha = alpha_eff < q75
    extreme_high = float(np.mean(extreme[high_alpha])) if np.any(high_alpha) else 0.0
    extreme_low = float(np.mean(extreme[low_alpha])) if np.any(low_alpha) else 0.0
    liq_high = float(np.mean(liquidity_score[high_alpha])) if np.any(high_alpha) else 0.0
    liq_low = float(np.mean(liquidity_score[low_alpha])) if np.any(low_alpha) else 0.0

    signatures = {
        "alpha_vs_liquidity_negative": corr_alpha_liquidity < -0.03,
        "alpha_vs_spread_positive": corr_alpha_spread > 0.03,
        "alpha_vs_depth_negative": corr_alpha_depth < -0.03,
        "alpha_vs_extremes_positive": corr_alpha_extreme > 0.03,
        "high_alpha_more_extremes": extreme_high > extreme_low,
    }
    h4_supported_single_run = sum(bool(v) for v in signatures.values()) >= 3

    return {
        "available": True,
        "alpha_eff_source": alpha_source,
        "corr_alpha_liquidity": corr_alpha_liquidity,
        "corr_alpha_spread": corr_alpha_spread,
        "corr_alpha_depth": corr_alpha_depth,
        "corr_alpha_extreme": corr_alpha_extreme,
        "extreme_rate_high_alpha": extreme_high,
        "extreme_rate_low_alpha": extreme_low,
        "liquidity_high_alpha": liq_high,
        "liquidity_low_alpha": liq_low,
        "signatures": signatures,
        "h4_supported_single_run": h4_supported_single_run,
    }


def print_analysis_report(market_df: pd.DataFrame, dealer_df: pd.DataFrame):
    """Print comprehensive analysis report."""
    print("=" * 80)
    print("SIMULATION RESULTS ANALYSIS")
    print("=" * 80)

    # Basic stats
    n_steps = len(market_df)
    n_dealers = dealer_df["dealer_id"].nunique()
    stress_periods = (market_df["regime"] == 1).sum()
    stress_pct = 100 * stress_periods / n_steps

    print(f"\nSimulation Overview:")
    print(f"  Steps: {n_steps}")
    print(f"  Dealers: {n_dealers}")
    print(f"  Stress periods: {stress_periods} ({stress_pct:.1f}%)")

    # Performance metrics
    print("\n" + "=" * 80)
    print("Performance Metrics by Model")
    print("=" * 80)

    perf = analyze_performance_metrics(market_df, dealer_df)
    for model in ["BS", "TFBS", "HESTON"]:
        if model in perf:
            print(f"\n{model}:")
            print(f"  Avg hedge error: {perf[model]['avg_hedge_error']:.2f} ± {perf[model]['std_hedge_error']:.2f}")
            print(f"  Avg reward: {perf[model]['avg_reward']:.2f} ± {perf[model]['std_reward']:.2f}")
            print(f"  Avg quality: {perf[model]['avg_quality']:.2f}")

    if "final_distribution" in perf:
        print(f"\nFinal Model Distribution:")
        for strategy, pct in perf["final_distribution"].items():
            print(f"  {strategy}: {pct:.1f}%")

    # Regime-model relationship
    print("\n" + "=" * 80)
    print("H1: Regime -> Model Share Relationship")
    print("=" * 80)

    regime_analysis = analyze_regime_model_relationship(market_df)

    for model in ["BS", "TFBS", "HESTON"]:
        if model in regime_analysis:
            r = regime_analysis[model]
            print(f"\n{model}:")
            print(f"  Calm:   {r['calm_mean']:.1%} ± {r['calm_std']:.3f}")
            print(f"  Stress: {r['stress_mean']:.1%} ± {r['stress_std']:.3f}")
            print(f"  Difference: {r['diff']:+.1%} (stress - calm)")
            print(f"  Permutation p-value: {r['perm_pval']:.4f}")

            if r['perm_pval'] < 0.05:
                print(f"  → SIGNIFICANT: {model} share differs between regimes")
            else:
                print(f"  → NOT SIGNIFICANT: difference could be random")

    # Switching analysis
    print("\n" + "=" * 80)
    print("H1: Switching Behavior Analysis")
    print("=" * 80)

    switch_analysis = analyze_switching_patterns(dealer_df)

    print(f"\nTotal switch events: {switch_analysis['n_switches']}")

    if switch_analysis['n_switches'] > 0:
        print(f"\nSwitch directions:")
        for model, pct in switch_analysis['switch_direction_pct'].items():
            print(f"  → {model}: {pct:.1f}%")

        print(f"\nOptimality Tests:")
        err_info = switch_analysis['chose_min_error']
        if err_info['total'] > 0:
            print(f"  Chose MIN hedge error: {err_info['n']}/{err_info['total']} = {err_info['rate']:.1%}")
            print(f"    (vs 33.3% random, p-value: {err_info['pval']:.4e})")
            if err_info['pval'] < 0.001:
                print(f"    → STRONG EVIDENCE: switching is performance-driven")
            elif err_info['pval'] < 0.05:
                print(f"    → MODERATE EVIDENCE: switching favors better models")
            else:
                print(f"    → WEAK EVIDENCE: switching may be random")

        qual_info = switch_analysis['chose_max_quality']
        if qual_info['total'] > 0:
            print(f"  Chose MAX quality: {qual_info['n']}/{qual_info['total']} = {qual_info['rate']:.1%}")
            print(f"    (vs 33.3% random, p-value: {qual_info['pval']:.4e})")
            if qual_info['pval'] < 0.001:
                print(f"    → STRONG EVIDENCE: switching is quality-driven")
            elif qual_info['pval'] < 0.05:
                print(f"    → MODERATE EVIDENCE: switching favors higher quality")
            else:
                print(f"    → WEAK EVIDENCE: quality doesn't predict switching")

    # H2 analysis
    print("\n" + "=" * 80)
    print("H2: Heterogeneity and Endogenous Volatility Regime Diagnostics")
    print("=" * 80)

    h2 = analyze_h2_regime_persistence(market_df, max_lag=10, rv_window=20)
    if not h2.get("available", False):
        print(f"\nH2 diagnostics unavailable: {h2.get('reason', 'unknown reason')}")
    else:
        print(f"\nStress episode duration:")
        print(f"  Mean:   {h2['mean_stress_duration']:.2f} steps")
        print(f"  Median: {h2['median_stress_duration']:.2f} steps")
        print(f"  Count:  {h2['n_stress_episodes']}")

        print(f"\nRegime transitions:")
        print(f"  P(calm -> stress):  {h2['p_calm_to_stress']:.4f}")
        print(f"  P(stress -> calm):  {h2['p_stress_to_calm']:.4f}")

        print(f"\nVolatility clustering (ACF, lags 1..10):")
        print(f"  Mean ACF |r|: {h2['abs_ret_acf_mean']:.4f} (positive lags: {h2['abs_ret_positive_lags']}/10)")
        print(f"  Mean ACF RV:  {h2['rv_acf_mean']:.4f} (positive lags: {h2['rv_positive_lags']}/10)")

        if h2.get("heterogeneity") is not None:
            hinfo = h2["heterogeneity"]
            print(f"\nDealer-model heterogeneity (Simpson diversity):")
            print(f"  Mean:   {hinfo['mean_simpson_diversity']:.4f}")
            print(f"  Calm:   {hinfo['calm_simpson_diversity']:.4f}")
            print(f"  Stress: {hinfo['stress_simpson_diversity']:.4f}")

        print(f"\nSingle-run H2 verdict:")
        if h2["h2_supported_single_run"]:
            print("  → SUPPORTED (diagnostic): persistent volatility clustering/regime signatures are present.")
        else:
            print("  → NOT SUPPORTED (diagnostic): clustering/persistence signatures are weak in this run.")
        print("  Note: strict causal H2 test requires hetero-vs-homogeneous replicated experiment.")

    # H3 analysis
    print("\n" + "=" * 80)
    print("H3: Adaptation Amplifies Instability (Diagnostics)")
    print("=" * 80)

    h3 = analyze_h3_adaptation_instability(market_df, dealer_df, rv_window=20)
    if not h3.get("available", False):
        print(f"\nH3 diagnostics unavailable: {h3.get('reason', 'unknown reason')}")
    else:
        print(f"\nSwitching activity:")
        print(f"  Mean per-step switch rate: {h3['mean_switch_rate']:.4f}")

        print(f"\nCo-movement with instability:")
        print(f"  Corr(switch_rate, |r|): {h3['corr_switch_absret']:.4f}")
        print(f"  Corr(switch_rate, RV): {h3['corr_switch_rv']:.4f}")
        print(f"  Corr(switch_rate, regime_transition): {h3['corr_switch_transition']:.4f}")

        print(f"\nHigh-switch vs low-switch states:")
        print(f"  Mean |r|: high={h3['mean_absret_high_switch']:.6f}, low={h3['mean_absret_low_switch']:.6f}")
        print(f"  Mean RV:  high={h3['mean_rv_high_switch']:.6f}, low={h3['mean_rv_low_switch']:.6f}")

        print(f"\nSingle-run H3 verdict:")
        if h3["h3_supported_single_run"]:
            print("  → SUPPORTED (diagnostic): adaptation/switching co-moves with instability signatures.")
        else:
            print("  → NOT SUPPORTED (diagnostic): adaptation-instability link is weak in this run.")
        print("  Note: strict H3 confirmation requires explicit adaptive-vs-fixed replicated comparison.")

    # H4 analysis
    print("\n" + "=" * 80)
    print("H4: Memory ↔ Liquidity (Diagnostics)")
    print("=" * 80)

    h4 = analyze_h4_memory_liquidity(market_df, dealer_df)
    if not h4.get("available", False):
        print(f"\nH4 diagnostics unavailable: {h4.get('reason', 'unknown reason')}")
    else:
        print(f"\nalpha_eff proxy source: {h4['alpha_eff_source']}")
        print(f"\nCorrelations:")
        print(f"  Corr(alpha_eff, liquidity_score): {h4['corr_alpha_liquidity']:.4f} (expected < 0)")
        print(f"  Corr(alpha_eff, option_spread):   {h4['corr_alpha_spread']:.4f} (expected > 0)")
        print(f"  Corr(alpha_eff, option_depth):    {h4['corr_alpha_depth']:.4f} (expected < 0)")
        print(f"  Corr(alpha_eff, extreme_moves):   {h4['corr_alpha_extreme']:.4f} (expected > 0)")

        print(f"\nHigh-alpha vs low-alpha states:")
        print(f"  Extreme-move rate: high={h4['extreme_rate_high_alpha']:.4f}, low={h4['extreme_rate_low_alpha']:.4f}")
        print(f"  Liquidity score:    high={h4['liquidity_high_alpha']:.4f}, low={h4['liquidity_low_alpha']:.4f}")

        print(f"\nSingle-run H4 verdict:")
        if h4["h4_supported_single_run"]:
            print("  → SUPPORTED (diagnostic): memory proxy is linked to lower liquidity and more extremes.")
        else:
            print("  → NOT SUPPORTED (diagnostic): memory-liquidity/extreme link is weak in this run.")
        print("  Note: strict H4 requires direct alpha sweeps/replications (run_h4_experiment).")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    heston_regime = regime_analysis.get("HESTON", {})
    if heston_regime:
        print(f"\nH1 (Stress → HESTON):")
        print(f"  Visual observation: HESTON share appears to increase during stress")
        print(f"  Quantitative result: {heston_regime.get('calm_mean', 0):.1%} (calm) vs {heston_regime.get('stress_mean', 0):.1%} (stress)")
        print(f"  Statistical significance: p = {heston_regime.get('perm_pval', 1):.4f}")
        if heston_regime.get('perm_pval', 1) < 0.05:
            print(f"  → CONFIRMED: Quantitative analysis supports visual observation")
        else:
            print(f"  → NOT CONFIRMED: Difference is not statistically significant")
            print(f"  → Visual pattern may be due to temporal correlation, not regime effect")

    if switch_analysis['n_switches'] > 0:
        err_info = switch_analysis['chose_min_error']
        print(f"\nH1 (Endogenous Switching):")
        print(f"  Dealers choose best model {err_info['rate']:.1%} of the time (vs 33.3% random)")
        print(f"  Statistical significance: p = {err_info['pval']:.4e}")
        if err_info['pval'] < 0.001:
            print(f"  → CONFIRMED: Strong evidence that switching is NOT random")
        else:
            print(f"  → NOT CONFIRMED: Switching may be random")

    if h2.get("available", False):
        print(f"\nH2 (Heterogeneity -> Regime Persistence, diagnostic):")
        print(f"  Mean stress duration: {h2['mean_stress_duration']:.2f}, P(calm->stress)={h2['p_calm_to_stress']:.4f}")
        print(f"  ACF means: |r|={h2['abs_ret_acf_mean']:.4f}, RV={h2['rv_acf_mean']:.4f}")
        if h2["h2_supported_single_run"]:
            print("  → SUPPORTED (diagnostic): positive multi-lag volatility persistence detected.")
        else:
            print("  → NOT SUPPORTED (diagnostic): persistence is not strong in this run.")
        print("  → For causal confirmation use replicated hetero-vs-homo experiment (run_h2_experiment).")

    if h3.get("available", False):
        print(f"\nH3 (Adaptation -> Instability, diagnostic):")
        print(
            f"  Corrs: |r|={h3['corr_switch_absret']:.4f}, RV={h3['corr_switch_rv']:.4f}, "
            f"transitions={h3['corr_switch_transition']:.4f}"
        )
        if h3["h3_supported_single_run"]:
            print("  → SUPPORTED (diagnostic): higher switching aligns with higher instability.")
        else:
            print("  → NOT SUPPORTED (diagnostic): weak switching-instability alignment.")
        print("  → For causal confirmation use adaptive-vs-fixed replicated experiment (run_h3_experiment).")

    if h4.get("available", False):
        print(f"\nH4 (Memory ↔ Liquidity, diagnostic):")
        print(
            f"  Corr(alpha_eff, liquidity)={h4['corr_alpha_liquidity']:.4f}, "
            f"Corr(alpha_eff, extremes)={h4['corr_alpha_extreme']:.4f}"
        )
        if h4["h4_supported_single_run"]:
            print("  → SUPPORTED (diagnostic): memory proxy relates to poorer liquidity and more extremes.")
        else:
            print("  → NOT SUPPORTED (diagnostic): weak memory-liquidity/extreme relationship.")
        print("  → For causal confirmation use direct alpha sweep replications (run_h4_experiment).")


def _extract_log_timestamp(path: Path) -> str | None:
    """Extract trailing timestamp token from log filename stem."""
    token = path.stem.split("_")[-1]
    return token if token.isdigit() else None


def _resolve_log_pair(market_args: list[str], dealer_args: list[str]) -> Tuple[Path, Path]:
    """
    Resolve one market/dealer log pair.
    If multiple files are provided, choose the latest common timestamp.
    """
    market_paths = [Path(p).expanduser().resolve() for p in market_args]
    dealer_paths = [Path(p).expanduser().resolve() for p in dealer_args]

    if len(market_paths) == 1 and len(dealer_paths) == 1:
        return market_paths[0], dealer_paths[0]

    market_by_ts = {}
    for p in market_paths:
        ts = _extract_log_timestamp(p)
        if ts is not None:
            market_by_ts[ts] = p

    dealer_by_ts = {}
    for p in dealer_paths:
        ts = _extract_log_timestamp(p)
        if ts is not None:
            dealer_by_ts[ts] = p

    common_ts = sorted(set(market_by_ts) & set(dealer_by_ts))
    if common_ts:
        latest_ts = common_ts[-1]
        return market_by_ts[latest_ts], dealer_by_ts[latest_ts]

    # Fallback when no matching timestamps exist.
    market_latest = max(market_paths, key=lambda p: p.stat().st_mtime)
    dealer_latest = max(dealer_paths, key=lambda p: p.stat().st_mtime)
    return market_latest, dealer_latest


def main():
    ap = argparse.ArgumentParser(description="Statistical analysis of simulation results")
    ap.add_argument("--market", required=True, nargs="+", help="Path(s) to market_log_*.csv")
    ap.add_argument("--dealers", required=True, nargs="+", help="Path(s) to dealers_log_*.csv")
    args = ap.parse_args()

    market_path, dealers_path = _resolve_log_pair(args.market, args.dealers)

    market_df = pd.read_csv(market_path)
    dealer_df = pd.read_csv(dealers_path)

    print(f"Market log: {market_path}")
    print(f"Dealers log: {dealers_path}\n")

    print_analysis_report(market_df, dealer_df)


if __name__ == "__main__":
    main()
