"""Supplementary experiments for H-PIOS study.

Implements three supplementary analyses from ANALYSIS_REPORT Part 13:
  Supp 1: Adaptive Threshold OFS (Override-Aware Ground Truth)
  Supp 2: C2 Ablation (persona removal for TAF comparison)
  Supp 4: Override decay coefficient (k) sensitivity analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from CORE_MODELS_models import (
    MasterEngineConfig,
    MarketDataPayload,
    MarketRegime,
    OrchestratorOutput,
)
from CORE_ENGINE_core import GraphOrchestrator
from EXP_ENGINES import TextPersonaEngine
from EXP_RUNNER import (
    load_wdss,
    load_master_configs,
    load_optimized_weights,
    apply_optimized_weights,
    build_market_data_payload,
    compute_simple_ensemble_for_c2,
)
from EXP_METRICS import compute_ofs, compute_cdi

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "EXP_RESULTS"


# ═══════════════════════════════════════════════════
# Supp 1: Adaptive Threshold OFS
# ═══════════════════════════════════════════════════

def run_supp1_adaptive_threshold(
    c1_fair_results: List[Dict],
    c2_results: List[Dict],
    c3_fair_results: List[Dict],
) -> Dict[str, Any]:
    """Recompute OFS using an adaptive threshold derived from C1_fair score distribution."""

    all_scores = []
    for r in c1_fair_results:
        if "error" in r:
            continue
        for score in r.get("expert_scores", {}).values():
            all_scores.append(score)

    if not all_scores:
        return {"error": "No C1_fair scores available"}

    adaptive_threshold = float(np.median(all_scores))
    mean_score = float(np.mean(all_scores))
    std_score = float(np.std(all_scores))

    logger.info(
        "C1_fair score distribution: median=%.4f, mean=%.4f, std=%.4f (n=%d)",
        adaptive_threshold, mean_score, std_score, len(all_scores),
    )

    report = {
        "adaptive_threshold": round(adaptive_threshold, 4),
        "c1_fair_distribution": {
            "median": round(adaptive_threshold, 4),
            "mean": round(mean_score, 4),
            "std": round(std_score, 4),
            "n_scores": len(all_scores),
            "min": round(min(all_scores), 4),
            "max": round(max(all_scores), 4),
        },
        "ofs_comparison": {},
    }

    datasets = {
        "C1_fair": c1_fair_results,
        "C2": c2_results,
        "C3_fair": c3_fair_results,
    }

    for cond_name, data in datasets.items():
        cond_filter = cond_name if cond_name != "C2" else "C2"
        ofs_fixed = compute_ofs(data, threshold=0.5, condition_filter=cond_filter)
        ofs_adaptive = compute_ofs(data, threshold=adaptive_threshold, condition_filter=cond_filter)

        report["ofs_comparison"][cond_name] = {
            "fixed_0.5": {
                "ofs_score": ofs_fixed["ofs_score"],
                "matches": ofs_fixed["matches"],
                "total": ofs_fixed["total"],
                "per_expert_ofs": ofs_fixed.get("per_expert_ofs", {}),
            },
            f"adaptive_{adaptive_threshold:.3f}": {
                "ofs_score": ofs_adaptive["ofs_score"],
                "matches": ofs_adaptive["matches"],
                "total": ofs_adaptive["total"],
                "per_expert_ofs": ofs_adaptive.get("per_expert_ofs", {}),
            },
        }

    return report


# ═══════════════════════════════════════════════════
# Supp 2: C2 Ablation
# ═══════════════════════════════════════════════════

C2_ABLATIONS = {
    "c2_no_graham": {
        "description": "Remove Graham persona from C2",
        "exclude": ["VAL_GRAHAM_001"],
    },
    "c2_no_taleb": {
        "description": "Remove Taleb persona from C2",
        "exclude": ["RSK_TALEB_001"],
    },
    "c2_no_risk": {
        "description": "Remove all RSK_* personas from C2",
        "exclude": ["RSK_TALEB_001", "RSK_SHANNON_001", "RSK_THORP_001"],
    },
}

C2_ABLATION_PREDICTIONS = {
    "c2_no_graham": {
        "metric": "val_domain_mean_on_A3",
        "direction": "decrease",
        "description": "Removing Graham should decrease Value domain mean score on A3_TOXIC",
    },
    "c2_no_taleb": {
        "metric": "ensemble_on_high_skew",
        "direction": "increase",
        "description": "Removing Taleb persona should increase ensemble on A3/A4",
    },
    "c2_no_risk": {
        "metric": "score_variance",
        "direction": "increase",
        "description": "Removing all Risk personas should increase score variance",
    },
}


def run_supp2_c2_ablation(
    wdss: Dict[str, Any],
    text_persona_engine: TextPersonaEngine,
    resume: bool = False,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Run C2 ablation experiments and compute C2-TAF.
    
    Each ablation is saved incrementally so partial progress survives crashes.
    With resume=True, completed ablations are loaded from disk instead of re-run.
    """
    scenario_ids = list(wdss["scenarios"].keys())
    scenario_map = {
        sid: (s["asset"], s["macro"]) for sid, s in wdss["scenarios"].items()
    }

    all_results: List[Dict] = []
    partial_dir = RESULTS_DIR / "supp2_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    total_llm_calls = 0
    for abl_name, abl_spec in C2_ABLATIONS.items():
        n_experts = 12 - len(abl_spec["exclude"])
        total_llm_calls += len(scenario_ids) * n_experts

    logger.info(
        "C2 Ablation: %d ablations x %d scenarios = %d experiment runs, "
        "actual LLM API calls = %d",
        len(C2_ABLATIONS), len(scenario_ids),
        len(C2_ABLATIONS) * len(scenario_ids), total_llm_calls,
    )

    for abl_name, abl_spec in C2_ABLATIONS.items():
        partial_path = partial_dir / f"{abl_name}.json"

        if resume and partial_path.exists():
            logger.info("RESUME: loading cached %s from %s", abl_name, partial_path)
            with open(partial_path, "r", encoding="utf-8") as f:
                abl_results = json.load(f)
            all_results.extend(abl_results)
            continue

        n_experts = 12 - len(abl_spec["exclude"])
        logger.info(
            "C2 ablation: %s (%s) — %d scenarios x %d experts = %d LLM calls",
            abl_name, abl_spec["description"],
            len(scenario_ids), n_experts, len(scenario_ids) * n_experts,
        )

        abl_results = []
        for i, sid in enumerate(scenario_ids, 1):
            md = build_market_data_payload(wdss, sid)
            expert_results = text_persona_engine.execute_all_experts(
                market_data=md,
                current_regime=md.current_regime,
                exclude_experts=abl_spec["exclude"],
            )
            expert_scores = {nid: r.normalized_score for nid, r in expert_results.items()}
            ensemble = compute_simple_ensemble_for_c2(expert_results)

            abl_results.append({
                "scenario_id": sid,
                "condition": f"C2_ablation_{abl_name}",
                "ablation": abl_name,
                "repetition": 1,
                "expert_scores": expert_scores,
                "ensemble_signal": ensemble,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(
                "  %s [%d/%d] %s ensemble=%.4f",
                abl_name, i, len(scenario_ids), sid, ensemble,
            )

        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(abl_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved partial: %s (%d results)", partial_path, len(abl_results))

        all_results.extend(abl_results)

    c2_taf = _compute_c2_taf(all_results, scenario_map)
    return all_results, c2_taf


def _compute_c2_taf(
    c2_abl_results: List[Dict],
    scenario_map: Dict[str, Tuple[str, str]],
) -> Dict[str, Any]:
    """Compute TAF-equivalent for C2 ablations using C2 baseline."""

    c2_baseline_path = RESULTS_DIR / "main_results_FULL.json"
    if not c2_baseline_path.exists():
        return {"error": "No C2 baseline found"}

    with open(c2_baseline_path, "r", encoding="utf-8") as f:
        all_main = json.load(f)

    c2_baseline = [r for r in all_main if r.get("condition") == "C2" and "error" not in r]
    c2_baseline_avg = _average_repetitions(c2_baseline)

    correct = 0
    total = 0
    details = []

    for abl_name, pred in C2_ABLATION_PREDICTIONS.items():
        abl_data = [r for r in c2_abl_results if r.get("ablation") == abl_name]
        if not abl_data:
            details.append({"ablation": abl_name, "result": "SKIPPED", "correct": False})
            continue

        actual_dir = _evaluate_c2_ablation_direction(
            c2_baseline_avg, abl_data, pred, scenario_map,
        )
        is_correct = actual_dir == pred["direction"]
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "ablation": abl_name,
            "predicted_direction": pred["direction"],
            "actual_direction": actual_dir,
            "description": pred["description"],
            "correct": is_correct,
        })

    taf_score = correct / total if total > 0 else 0.0
    return {
        "c2_taf_score": round(taf_score, 4),
        "correct": correct,
        "total": total,
        "details": details,
    }


def _average_repetitions(results: List[Dict]) -> List[Dict]:
    """Average C2 results across repetitions to get one entry per scenario."""
    by_scenario: Dict[str, List[Dict]] = {}
    for r in results:
        by_scenario.setdefault(r["scenario_id"], []).append(r)

    averaged = []
    for sid, reps in by_scenario.items():
        avg_scores: Dict[str, float] = {}
        all_experts = set()
        for rep in reps:
            all_experts.update(rep.get("expert_scores", {}).keys())
        for eid in all_experts:
            vals = [rep["expert_scores"][eid] for rep in reps if eid in rep.get("expert_scores", {})]
            avg_scores[eid] = sum(vals) / len(vals) if vals else 0.0

        avg_ensemble = sum(r["ensemble_signal"] for r in reps) / len(reps)
        averaged.append({
            "scenario_id": sid,
            "condition": "C2",
            "expert_scores": avg_scores,
            "ensemble_signal": avg_ensemble,
        })
    return averaged


def _evaluate_c2_ablation_direction(
    baseline: List[Dict],
    ablated: List[Dict],
    prediction: Dict,
    scenario_map: Dict,
) -> str:
    """Evaluate actual direction of change for a C2 ablation."""
    metric = prediction["metric"]

    if metric == "val_domain_mean_on_A3":
        a3_scenarios = [sid for sid, (a, m) in scenario_map.items() if a == "A3_TOXIC"]

        def val_mean(data: List[Dict]) -> float:
            vals = []
            for r in data:
                if r["scenario_id"] not in a3_scenarios:
                    continue
                for eid, sc in r.get("expert_scores", {}).items():
                    if eid.startswith("VAL_"):
                        vals.append(sc)
            return np.mean(vals) if vals else 0.0

        return "decrease" if val_mean(ablated) < val_mean(baseline) else "increase"

    elif metric == "ensemble_on_high_skew":
        skew_assets = ["A3_TOXIC", "A4_HYPE"]
        skew_scenarios = [sid for sid, (a, m) in scenario_map.items() if a in skew_assets]

        def ens_mean(data: List[Dict]) -> float:
            vals = [r["ensemble_signal"] for r in data if r["scenario_id"] in skew_scenarios]
            return np.mean(vals) if vals else 0.0

        return "increase" if ens_mean(ablated) > ens_mean(baseline) else "decrease"

    elif metric == "score_variance":
        def variance(data: List[Dict]) -> float:
            all_ens = [r["ensemble_signal"] for r in data]
            return float(np.var(all_ens)) if all_ens else 0.0

        return "increase" if variance(ablated) > variance(baseline) else "decrease"

    return "unknown"


# ═══════════════════════════════════════════════════
# Supp 4: Override k Sensitivity
# ═══════════════════════════════════════════════════

def run_supp4_k_sensitivity(
    configs: List[MasterEngineConfig],
    wdss: Dict[str, Any],
    k_values: List[float],
) -> Dict[str, Any]:
    """Run C1 with different override decay coefficients k."""
    scenario_ids = list(wdss["scenarios"].keys())
    report: Dict[str, Any] = {"k_values": k_values, "per_k": {}}

    for k in k_values:
        logger.info("Running k=%.1f sensitivity (%d scenarios)", k, len(scenario_ids))
        results = []
        for sid in scenario_ids:
            md = build_market_data_payload(wdss, sid)
            orch = GraphOrchestrator(configs)
            output: OrchestratorOutput = orch.resolve_signals(
                market_data=md,
                nlp_data=None,
                extra_context={"_variant_no_spg": True, "_variant_override_k": k},
            )
            expert_scores = {nid: r.normalized_score for nid, r in output.node_results.items()}
            results.append({
                "scenario_id": sid,
                "condition": f"C1_k{k:.1f}",
                "repetition": 1,
                "expert_scores": expert_scores,
                "ensemble_signal": output.ensemble_signal,
                "override_active": output.override_active,
            })

        ensembles = [r["ensemble_signal"] for r in results]
        all_scores = []
        for r in results:
            all_scores.extend(r.get("expert_scores", {}).values())

        non_taleb_scores = []
        for r in results:
            for eid, sc in r.get("expert_scores", {}).items():
                if eid != "RSK_TALEB_001":
                    non_taleb_scores.append(sc)

        ofs_result = compute_ofs(results, threshold=0.5, condition_filter=f"C1_k{k:.1f}")
        cdi_result = compute_cdi(results, condition_filter=f"C1_k{k:.1f}")

        report["per_k"][f"k={k:.1f}"] = {
            "k": k,
            "ofs_score": ofs_result["ofs_score"],
            "cdi_score": cdi_result.get("cdi_score", 0.0),
            "mean_abs_corr": cdi_result.get("mean_abs_correlation", 0.0),
            "ensemble_mean": round(float(np.mean(ensembles)), 4),
            "ensemble_std": round(float(np.std(ensembles)), 4),
            "non_taleb_score_mean": round(float(np.mean(non_taleb_scores)), 4) if non_taleb_scores else 0.0,
            "non_taleb_score_std": round(float(np.std(non_taleb_scores)), 4) if non_taleb_scores else 0.0,
            "n_override_active": sum(1 for r in results if r.get("override_active")),
            "per_expert_ofs": ofs_result.get("per_expert_ofs", {}),
            "results": results,
        }
        logger.info(
            "  k=%.1f: OFS=%.4f CDI=%.4f ensemble_mean=%.4f non_taleb_mean=%.4f",
            k, ofs_result["ofs_score"], cdi_result.get("cdi_score", 0),
            np.mean(ensembles), np.mean(non_taleb_scores) if non_taleb_scores else 0,
        )

    return report


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="H-PIOS Supplementary Experiments")
    parser.add_argument("--supp", nargs="+", choices=["1", "2", "4", "all"], default=["all"])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip C2 ablations that already have saved partial results")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    run_1 = "1" in args.supp or "all" in args.supp
    run_2 = "2" in args.supp or "all" in args.supp
    run_4 = "4" in args.supp or "all" in args.supp

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    wdss = load_wdss()
    base_configs = load_master_configs()
    opt_weights = load_optimized_weights()
    configs = apply_optimized_weights(base_configs, opt_weights)

    full_report: Dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # --- Supp 4: k sensitivity (deterministic, fast) ---
    if run_4:
        logger.info("=" * 50)
        logger.info("SUPP 4: Override k Sensitivity")
        logger.info("=" * 50)
        k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        supp4 = run_supp4_k_sensitivity(configs, wdss, k_values)
        supp4_clean = {
            "k_values": supp4["k_values"],
            "per_k": {
                kname: {k2: v2 for k2, v2 in kdata.items() if k2 != "results"}
                for kname, kdata in supp4["per_k"].items()
            },
        }
        full_report["supp4_k_sensitivity"] = supp4_clean

        out4 = RESULTS_DIR / f"supp4_k_sensitivity_{ts}.json"
        with open(out4, "w", encoding="utf-8") as f:
            json.dump(supp4_clean, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Supp 4 saved: %s", out4)

    # --- Supp 1: Adaptive threshold (uses existing data) ---
    if run_1:
        logger.info("=" * 50)
        logger.info("SUPP 1: Adaptive Threshold OFS")
        logger.info("=" * 50)

        fair_path = RESULTS_DIR / "supplementary_fair_results.json"
        full_path = RESULTS_DIR / "main_results_FULL.json"

        if not fair_path.exists() or not full_path.exists():
            logger.error("Missing required result files for Supp 1")
        else:
            with open(fair_path, "r", encoding="utf-8") as f:
                fair_data = json.load(f)
            with open(full_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)

            c1_fair = [r for r in fair_data if r.get("condition") == "C1_fair"]
            c2_data = [r for r in full_data if r.get("condition") == "C2"]
            c3_fair = [r for r in fair_data if r.get("condition") == "C3_fair"]

            supp1 = run_supp1_adaptive_threshold(c1_fair, c2_data, c3_fair)
            full_report["supp1_adaptive_threshold"] = supp1

            out1 = RESULTS_DIR / f"supp1_adaptive_threshold_{ts}.json"
            with open(out1, "w", encoding="utf-8") as f:
                json.dump(supp1, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Supp 1 saved: %s", out1)

    # --- Supp 2: C2 ablation (LLM calls, slow) ---
    if run_2:
        logger.info("=" * 50)
        logger.info("SUPP 2: C2 Ablation")
        logger.info("=" * 50)
        engine = TextPersonaEngine()
        c2_abl_results, c2_taf = run_supp2_c2_ablation(wdss, engine, resume=args.resume)

        out2_results = RESULTS_DIR / f"supp2_c2_ablation_results_{ts}.json"
        with open(out2_results, "w", encoding="utf-8") as f:
            json.dump(c2_abl_results, f, indent=2, ensure_ascii=False, default=str)

        full_report["supp2_c2_ablation"] = c2_taf

        out2_taf = RESULTS_DIR / f"supp2_c2_taf_{ts}.json"
        with open(out2_taf, "w", encoding="utf-8") as f:
            json.dump(c2_taf, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Supp 2 saved: %s, %s", out2_results, out2_taf)

    # --- Save combined report ---
    combined_path = RESULTS_DIR / f"supplementary_report_{ts}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Combined supplementary report: %s", combined_path)

    # --- Print summary ---
    _print_summary(full_report)


def _print_summary(report: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("  SUPPLEMENTARY EXPERIMENTS SUMMARY")
    print("=" * 70)

    supp1 = report.get("supp1_adaptive_threshold")
    if supp1:
        print(f"\n--- Supp 1: Adaptive Threshold OFS ---")
        print(f"  Adaptive threshold: {supp1.get('adaptive_threshold', '?')}")
        print(f"  C1_fair distribution: median={supp1['c1_fair_distribution']['median']}, "
              f"mean={supp1['c1_fair_distribution']['mean']}, std={supp1['c1_fair_distribution']['std']}")
        print(f"\n  {'Condition':<15} {'OFS@0.5':>10} {'OFS@adaptive':>14}")
        print("  " + "-" * 42)
        for cond, data in supp1.get("ofs_comparison", {}).items():
            fixed = data.get("fixed_0.5", {}).get("ofs_score", "?")
            adaptive_key = [k for k in data.keys() if k.startswith("adaptive_")][0]
            adaptive = data[adaptive_key].get("ofs_score", "?")
            print(f"  {cond:<15} {fixed:>10} {adaptive:>14}")

    supp2 = report.get("supp2_c2_ablation")
    if supp2:
        print(f"\n--- Supp 2: C2 Ablation TAF ---")
        print(f"  C2 TAF Score: {supp2.get('c2_taf_score', '?')} ({supp2.get('correct', '?')}/{supp2.get('total', '?')})")
        for d in supp2.get("details", []):
            status = "O" if d.get("correct") else "X"
            print(f"  {status} {d['ablation']}: predicted={d.get('predicted_direction')}, actual={d.get('actual_direction')}")

    supp4 = report.get("supp4_k_sensitivity")
    if supp4:
        print(f"\n--- Supp 4: Override k Sensitivity ---")
        print(f"  {'k':>5} {'OFS':>8} {'CDI':>8} {'Ens_mean':>10} {'NonTaleb':>10}")
        print("  " + "-" * 48)
        for kname, kdata in supp4.get("per_k", {}).items():
            print(f"  {kdata['k']:5.1f} {kdata['ofs_score']:8.4f} {kdata['cdi_score']:8.4f} "
                  f"{kdata['ensemble_mean']:10.4f} {kdata['non_taleb_score_mean']:10.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
