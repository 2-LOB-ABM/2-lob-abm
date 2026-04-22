"""
Statistical analysis of simulation results.

Evaluates hypotheses (H1-H4) and provides quantitative assessment of model switching,
regime effects, and dealer behavior patterns.

Usage:
    python analyze_results.py --market simulation_logs/market_log_*.csv --dealers simulation_logs/dealers_log_*.csv
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


def main():
    ap = argparse.ArgumentParser(description="Statistical analysis of simulation results")
    ap.add_argument("--market", required=True, help="Path to market_log_*.csv")
    ap.add_argument("--dealers", required=True, help="Path to dealers_log_*.csv")
    args = ap.parse_args()

    market_path = Path(args.market).expanduser().resolve()
    dealers_path = Path(args.dealers).expanduser().resolve()

    market_df = pd.read_csv(market_path)
    dealer_df = pd.read_csv(dealers_path)

    print(f"Market log: {market_path}")
    print(f"Dealers log: {dealers_path}\n")

    print_analysis_report(market_df, dealer_df)


if __name__ == "__main__":
    main()
