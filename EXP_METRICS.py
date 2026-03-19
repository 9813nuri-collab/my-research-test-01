"""Evaluation metrics for H-PIOS Executable Ontology study.

Implements 3 core metrics defined in EXP_PLAN.txt Section 4:
  - OFS (Ontology Faithfulness Score): Do agents follow the ontology rules?
  - CDI (Cross-Expert Differentiation Index): Do different experts judge differently?
  - TAF (Targeted Ablation Fidelity): Do components cause predicted effects when removed?
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
GROUND_TRUTH_PATH = BASE_DIR / "EXP_GROUND_TRUTH.json"
WDSS_PATH = BASE_DIR / "EXP_DATA_wdss.json"


def _load_ground_truth() -> Dict[str, Any]:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_wdss() -> Dict[str, Any]:
    with open(WDSS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _scenario_to_asset_macro(wdss: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Map scenario_id -> (asset_key, macro_key)."""
    return {
        sid: (sdata["asset"], sdata["macro"])
        for sid, sdata in wdss["scenarios"].items()
    }


# ═══════════════════════════════════════════════════
# Metric 1: OFS — Ontology Faithfulness Score
# ═══════════════════════════════════════════════════

def compute_ofs(
    results: List[Dict[str, Any]],
    threshold: float = 0.5,
    condition_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Ontology Faithfulness Score.

    For each (expert, scenario) pair:
      - Look up expected direction from ground truth (HIGH/LOW)
      - Binarize actual score at threshold: score >= threshold → HIGH, else LOW
      - OFS = match_count / total_count (excluding N/A)

    Args:
        results: List of experiment result dicts (from EXP_RUNNER).
        threshold: Binary threshold for score (default 0.5).
        condition_filter: If set, only compute for this condition (e.g., "C1").

    Returns:
        Dict with ofs_score, details, and per-expert breakdown.
    """
    gt = _load_ground_truth()
    wdss = _load_wdss()
    scenario_map = _scenario_to_asset_macro(wdss)

    asset_expectations = gt.get("asset_expectations", {})
    macro_expectations = gt.get("macro_expectations", {})
    expert_type = gt.get("expert_type", {})
    asset_dependent = set(expert_type.get("asset_dependent", []))
    macro_dependent = set(expert_type.get("macro_dependent", []))

    filtered = results
    if condition_filter:
        filtered = [r for r in results if r.get("condition") == condition_filter]

    matches = 0
    total = 0
    details: List[Dict[str, Any]] = []
    per_expert_correct: Dict[str, int] = {}
    per_expert_total: Dict[str, int] = {}

    for result in filtered:
        if "error" in result:
            continue

        scenario_id = result["scenario_id"]
        expert_scores = result.get("expert_scores", {})

        if scenario_id not in scenario_map:
            continue
        asset_key, macro_key = scenario_map[scenario_id]

        for expert_id, actual_score in expert_scores.items():
            expected = _lookup_expected(
                expert_id, asset_key, macro_key,
                asset_expectations, macro_expectations,
                asset_dependent, macro_dependent,
            )

            if expected is None or expected == "N/A":
                continue

            actual_direction = "HIGH" if actual_score >= threshold else "LOW"
            is_match = actual_direction == expected

            if is_match:
                matches += 1
            total += 1

            per_expert_correct.setdefault(expert_id, 0)
            per_expert_total.setdefault(expert_id, 0)
            per_expert_total[expert_id] += 1
            if is_match:
                per_expert_correct[expert_id] += 1

            details.append({
                "scenario_id": scenario_id,
                "expert_id": expert_id,
                "expected": expected,
                "actual_score": round(actual_score, 4),
                "actual_direction": actual_direction,
                "match": is_match,
            })

    ofs_score = matches / total if total > 0 else 0.0

    per_expert_ofs = {}
    for eid in per_expert_total:
        et = per_expert_total[eid]
        ec = per_expert_correct.get(eid, 0)
        per_expert_ofs[eid] = round(ec / et, 4) if et > 0 else 0.0

    return {
        "ofs_score": round(ofs_score, 4),
        "matches": matches,
        "total": total,
        "threshold": threshold,
        "condition": condition_filter,
        "per_expert_ofs": per_expert_ofs,
        "details": details,
    }


def _lookup_expected(
    expert_id: str,
    asset_key: str,
    macro_key: str,
    asset_exp: Dict,
    macro_exp: Dict,
    asset_dep: set,
    macro_dep: set,
) -> Optional[str]:
    """Look up the expected direction for an expert in a scenario."""
    if expert_id in asset_dep:
        expert_data = asset_exp.get(expert_id, {})
        asset_entry = expert_data.get(asset_key, {})
        return asset_entry.get("expected")
    elif expert_id in macro_dep:
        expert_data = macro_exp.get(expert_id, {})
        macro_entry = expert_data.get(macro_key, {})
        return macro_entry.get("expected")
    return None


# ═══════════════════════════════════════════════════
# Metric 2: CDI — Cross-Expert Differentiation Index
# ═══════════════════════════════════════════════════

def compute_cdi(
    results: List[Dict[str, Any]],
    condition_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Cross-Expert Differentiation Index.

    Build a matrix of expert score vectors across scenarios,
    compute the pairwise Pearson correlation matrix,
    and return CDI = 1 - mean(|off-diagonal correlations|).

    Args:
        results: List of experiment result dicts.
        condition_filter: If set, only compute for this condition.

    Returns:
        Dict with cdi_score, correlation matrix, and details.
    """
    filtered = results
    if condition_filter:
        filtered = [r for r in results if r.get("condition") == condition_filter]

    filtered = [r for r in filtered if "error" not in r and r.get("expert_scores")]

    if not filtered:
        return {"cdi_score": 0.0, "error": "No valid results"}

    all_experts = set()
    all_scenarios = set()
    for r in filtered:
        all_experts.update(r["expert_scores"].keys())
        all_scenarios.add(r["scenario_id"])

    experts = sorted(all_experts)
    scenarios = sorted(all_scenarios)

    if len(experts) < 2:
        return {"cdi_score": 0.0, "error": "Need at least 2 experts"}

    score_by_exp_scen: Dict[str, Dict[str, List[float]]] = {
        e: {s: [] for s in scenarios} for e in experts
    }
    for r in filtered:
        sid = r["scenario_id"]
        for eid, score in r["expert_scores"].items():
            score_by_exp_scen[eid][sid].append(score)

    score_matrix = np.zeros((len(experts), len(scenarios)))
    for i, eid in enumerate(experts):
        for j, sid in enumerate(scenarios):
            scores = score_by_exp_scen[eid][sid]
            score_matrix[i, j] = np.mean(scores) if scores else 0.5

    std_devs = np.std(score_matrix, axis=1)
    constant_mask = std_devs < 1e-10
    if np.all(constant_mask):
        return {
            "cdi_score": 0.0,
            "n_experts": len(experts),
            "n_scenarios": len(scenarios),
            "note": "All experts have constant scores across scenarios",
        }

    n = len(experts)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if constant_mask[i] or constant_mask[j]:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
            else:
                r_val = np.corrcoef(score_matrix[i], score_matrix[j])[0, 1]
                if np.isnan(r_val):
                    r_val = 0.0
                corr_matrix[i, j] = r_val
                corr_matrix[j, i] = r_val

    off_diagonal = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diagonal.append(abs(corr_matrix[i, j]))

    cdi_score = 1.0 - np.mean(off_diagonal)

    corr_dict = {}
    for i, ei in enumerate(experts):
        for j, ej in enumerate(experts):
            if i < j:
                corr_dict[f"{ei}_vs_{ej}"] = round(float(corr_matrix[i, j]), 4)

    return {
        "cdi_score": round(float(cdi_score), 4),
        "n_experts": len(experts),
        "n_scenarios": len(scenarios),
        "mean_abs_correlation": round(float(np.mean(off_diagonal)), 4),
        "expert_list": experts,
        "pairwise_correlations": corr_dict,
        "condition": condition_filter,
    }


# ═══════════════════════════════════════════════════
# Metric 3: TAF — Targeted Ablation Fidelity
# ═══════════════════════════════════════════════════

ABLATION_PREDICTIONS = {
    "no_graham": {
        "metric": "val_domain_mean_on_A3",
        "direction": "decrease",
        "description": "Removing Graham should decrease Value domain mean score on A3_TOXIC (deep value signal lost)",
    },
    "no_taleb": {
        "metric": "ensemble_on_high_skew",
        "direction": "increase",
        "description": "Removing Taleb should increase ensemble score on A3/A4 (brake removed)",
    },
    "no_risk": {
        "metric": "score_variance",
        "direction": "increase",
        "description": "Removing all Risk nodes should increase score variance (dampening removed)",
    },
    "no_edges": {
        "metric": "graham_score_on_A3",
        "direction": "increase",
        "description": "Removing edges should increase Graham's A3 score (Munger→Graham discount removed)",
    },
    "flat_ensemble": {
        "metric": "ensemble_extremity",
        "direction": "decrease",
        "description": "Flat ensemble should make extreme scenario scores closer to center",
    },
}


def compute_taf(
    baseline_results: List[Dict[str, Any]],
    ablation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Targeted Ablation Fidelity.

    For each ablation type, check if the predicted directional change
    actually occurred compared to the C1 baseline.

    Args:
        baseline_results: C1 (Full Ontology) results.
        ablation_results: Results from ablation experiments.

    Returns:
        Dict with taf_score and per-ablation details.
    """
    wdss = _load_wdss()
    scenario_map = _scenario_to_asset_macro(wdss)

    baseline_c1 = [r for r in baseline_results if r.get("condition") == "C1" and "error" not in r]

    abl_by_type: Dict[str, List[Dict]] = {}
    for r in ablation_results:
        if "error" in r:
            continue
        abl_name = r.get("ablation", "")
        abl_by_type.setdefault(abl_name, []).append(r)

    correct = 0
    total = 0
    details: List[Dict] = []

    for abl_name, prediction in ABLATION_PREDICTIONS.items():
        abl_data = abl_by_type.get(abl_name, [])
        if not abl_data:
            details.append({
                "ablation": abl_name,
                "prediction": prediction["description"],
                "result": "SKIPPED — no ablation data",
                "correct": False,
            })
            continue

        actual_direction = _evaluate_ablation_direction(
            baseline_c1, abl_data, prediction, scenario_map,
        )

        is_correct = actual_direction == prediction["direction"]
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "ablation": abl_name,
            "predicted_direction": prediction["direction"],
            "actual_direction": actual_direction,
            "description": prediction["description"],
            "correct": is_correct,
        })

    taf_score = correct / total if total > 0 else 0.0

    return {
        "taf_score": round(taf_score, 4),
        "correct": correct,
        "total": total,
        "details": details,
    }


def _evaluate_ablation_direction(
    baseline: List[Dict],
    ablated: List[Dict],
    prediction: Dict,
    scenario_map: Dict,
) -> str:
    """Evaluate the actual direction of change for a specific ablation."""
    metric_type = prediction["metric"]

    if metric_type == "val_domain_mean_on_A3":
        return _compare_val_domain_A3(baseline, ablated, scenario_map)
    elif metric_type == "ensemble_on_high_skew":
        return _compare_ensemble_high_skew(baseline, ablated, scenario_map)
    elif metric_type == "score_variance":
        return _compare_score_variance(baseline, ablated)
    elif metric_type == "graham_score_on_A3":
        return _compare_graham_A3(baseline, ablated, scenario_map)
    elif metric_type == "ensemble_extremity":
        return _compare_ensemble_extremity(baseline, ablated, scenario_map)
    else:
        return "unknown"


def _compare_val_domain_A3(
    baseline: List[Dict], ablated: List[Dict], scenario_map: Dict,
) -> str:
    """Compare Value domain mean score on A3 scenarios."""
    a3_scenarios = [sid for sid, (a, m) in scenario_map.items() if a == "A3_TOXIC"]

    def val_mean(results: List[Dict]) -> float:
        scores = []
        for r in results:
            if r["scenario_id"] in a3_scenarios:
                for eid, s in r.get("expert_scores", {}).items():
                    if eid.startswith("VAL_"):
                        scores.append(s)
        return np.mean(scores) if scores else 0.0

    base_val = val_mean(baseline)
    abl_val = val_mean(ablated)
    return "decrease" if abl_val < base_val else "increase"


def _compare_ensemble_high_skew(
    baseline: List[Dict], ablated: List[Dict], scenario_map: Dict,
) -> str:
    """Compare ensemble score on A3/A4 (high SKEW) scenarios."""
    high_skew = [
        sid for sid, (a, m) in scenario_map.items()
        if a in ("A3_TOXIC", "A4_HYPE")
    ]

    def ens_mean(results: List[Dict]) -> float:
        vals = [
            r["ensemble_signal"] for r in results
            if r["scenario_id"] in high_skew and "ensemble_signal" in r
        ]
        return np.mean(vals) if vals else 0.0

    base_val = ens_mean(baseline)
    abl_val = ens_mean(ablated)
    return "increase" if abl_val > base_val else "decrease"


def _compare_score_variance(
    baseline: List[Dict], ablated: List[Dict],
) -> str:
    """Compare ensemble score variance across all scenarios."""
    def score_var(results: List[Dict]) -> float:
        vals = [r["ensemble_signal"] for r in results if "ensemble_signal" in r]
        return float(np.var(vals)) if vals else 0.0

    base_var = score_var(baseline)
    abl_var = score_var(ablated)
    return "increase" if abl_var > base_var else "decrease"


def _compare_graham_A3(
    baseline: List[Dict], ablated: List[Dict], scenario_map: Dict,
) -> str:
    """Compare Graham's score on A3 scenarios before/after edge removal."""
    a3_scenarios = [sid for sid, (a, m) in scenario_map.items() if a == "A3_TOXIC"]

    def graham_mean(results: List[Dict]) -> float:
        scores = [
            r["expert_scores"].get("VAL_GRAHAM_001", 0.0)
            for r in results if r["scenario_id"] in a3_scenarios
        ]
        return np.mean(scores) if scores else 0.0

    base_val = graham_mean(baseline)
    abl_val = graham_mean(ablated)
    return "increase" if abl_val > base_val else "decrease"


def _compare_ensemble_extremity(
    baseline: List[Dict], ablated: List[Dict], scenario_map: Dict,
) -> str:
    """Check if flat ensemble makes extreme scenarios less extreme."""
    extreme_assets = ["A3_TOXIC", "A4_HYPE"]
    extreme_scenarios = [
        sid for sid, (a, m) in scenario_map.items() if a in extreme_assets
    ]

    def extremity(results: List[Dict]) -> float:
        vals = [
            abs(r["ensemble_signal"] - 0.5)
            for r in results
            if r["scenario_id"] in extreme_scenarios and "ensemble_signal" in r
        ]
        return np.mean(vals) if vals else 0.0

    base_ext = extremity(baseline)
    abl_ext = extremity(ablated)
    return "decrease" if abl_ext < base_ext else "increase"


# ═══════════════════════════════════════════════════
# Unified Computation
# ═══════════════════════════════════════════════════

def compute_all_metrics(
    main_results: List[Dict[str, Any]],
    ablation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all three metrics for all conditions."""
    report = {
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "ofs": {},
        "cdi": {},
        "taf": {},
    }

    for cond in ["C1", "C2", "C3", "C4"]:
        cond_results = [r for r in main_results if r.get("condition") == cond]
        if cond_results:
            report["ofs"][cond] = compute_ofs(main_results, condition_filter=cond)
            if cond != "C4":
                report["cdi"][cond] = compute_cdi(main_results, condition_filter=cond)

    baseline_c1 = [r for r in main_results if r.get("condition") == "C1"]
    if baseline_c1 and ablation_results:
        report["taf"] = compute_taf(baseline_c1, ablation_results)

    return report


def print_metrics_summary(report: Dict[str, Any]) -> None:
    """Print a human-readable summary of all metrics."""
    print("\n" + "=" * 60)
    print("  H-PIOS Experiment Metrics Summary")
    print("=" * 60)

    print("\n--- OFS (Ontology Faithfulness Score) ---")
    print(f"  {'Condition':<12} {'OFS':>8} {'Matches':>10} {'Total':>8}")
    print("  " + "-" * 42)
    for cond in ["C1", "C2", "C3", "C4"]:
        ofs_data = report.get("ofs", {}).get(cond, {})
        if ofs_data:
            print(
                f"  {cond:<12} {ofs_data.get('ofs_score', 0):.4f}"
                f"    {ofs_data.get('matches', 0):>5}/{ofs_data.get('total', 0):<5}"
            )

    print("\n--- CDI (Cross-Expert Differentiation Index) ---")
    print(f"  {'Condition':<12} {'CDI':>8} {'Mean |r|':>10} {'Experts':>10}")
    print("  " + "-" * 44)
    for cond in ["C1", "C2", "C3"]:
        cdi_data = report.get("cdi", {}).get(cond, {})
        if cdi_data:
            print(
                f"  {cond:<12} {cdi_data.get('cdi_score', 0):.4f}"
                f"    {cdi_data.get('mean_abs_correlation', 0):.4f}"
                f"    {cdi_data.get('n_experts', 0):>5}"
            )
    print("  C4           N/A (single score, no expert differentiation)")

    print("\n--- TAF (Targeted Ablation Fidelity) ---")
    taf_data = report.get("taf", {})
    if taf_data:
        print(f"  TAF Score: {taf_data.get('taf_score', 0):.4f} "
              f"({taf_data.get('correct', 0)}/{taf_data.get('total', 0)})")
        for d in taf_data.get("details", []):
            status = "O" if d.get("correct") else "X"
            print(f"  {status} {d['ablation']}: predicted={d.get('predicted_direction')}, "
                  f"actual={d.get('actual_direction')}")
    else:
        print("  No ablation data available")

    print("\n" + "=" * 60)


def save_metrics_report(
    report: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> None:
    """Save the full metrics report as JSON."""
    if output_path is None:
        results_dir = BASE_DIR / "EXP_RESULTS"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"metrics_report_{ts}.json"

    clean_report = _strip_details(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Metrics report saved: %s", output_path)


def _strip_details(report: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary version without per-entry details for readability."""
    clean = {"timestamp": report.get("timestamp")}

    clean["ofs_summary"] = {}
    for cond, data in report.get("ofs", {}).items():
        clean["ofs_summary"][cond] = {
            "ofs_score": data.get("ofs_score"),
            "matches": data.get("matches"),
            "total": data.get("total"),
            "per_expert_ofs": data.get("per_expert_ofs"),
        }

    clean["cdi_summary"] = {}
    for cond, data in report.get("cdi", {}).items():
        clean["cdi_summary"][cond] = {
            "cdi_score": data.get("cdi_score"),
            "n_experts": data.get("n_experts"),
            "n_scenarios": data.get("n_scenarios"),
            "mean_abs_correlation": data.get("mean_abs_correlation"),
            "pairwise_correlations": data.get("pairwise_correlations"),
        }

    clean["taf_summary"] = {
        "taf_score": report.get("taf", {}).get("taf_score"),
        "correct": report.get("taf", {}).get("correct"),
        "total": report.get("taf", {}).get("total"),
        "details": report.get("taf", {}).get("details"),
    }

    return clean


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main() -> None:
    """Load results from EXP_RESULTS and compute all metrics."""
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Compute H-PIOS experiment metrics")
    parser.add_argument("--results-dir", type=str, default=str(BASE_DIR / "EXP_RESULTS"))
    parser.add_argument("--main-file", type=str, default=None, help="Specific main results JSON")
    parser.add_argument("--ablation-file", type=str, default=None, help="Specific ablation results JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    results_dir = Path(args.results_dir)

    if args.main_file:
        main_path = Path(args.main_file)
    else:
        main_files = sorted(results_dir.glob("main_results_*.json"))
        if not main_files:
            logger.error("No main_results_*.json found in %s", results_dir)
            return
        main_path = main_files[-1]

    logger.info("Loading main results: %s", main_path)
    with open(main_path, "r", encoding="utf-8") as f:
        main_results = json.load(f)

    ablation_results = []
    if args.ablation_file:
        abl_path = Path(args.ablation_file)
    else:
        abl_files = sorted(results_dir.glob("ablation_results_*.json"))
        abl_path = abl_files[-1] if abl_files else None

    if abl_path and abl_path.exists():
        logger.info("Loading ablation results: %s", abl_path)
        with open(abl_path, "r", encoding="utf-8") as f:
            ablation_results = json.load(f)

    report = compute_all_metrics(main_results, ablation_results)

    print_metrics_summary(report)
    save_metrics_report(report)


if __name__ == "__main__":
    main()
