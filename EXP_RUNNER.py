"""Experiment runner for H-PIOS Executable Ontology study.

Implements automated execution of 4 conditions (C1-C4) + 5 ablations
across 24 scenarios from EXP_DATA_wdss.json.

Excluded from experiment (per EXP_PLAN.txt Section 2):
  - CORE_GRAPH_agent_flow.py (full 5-stage pipeline)
  - CORE_BRAIN_firmware.py (confound variable)
  - SPG (Stage 4.5, domain-specific guardrail)
  - Stage 5 (Portfolio Sizer)
  - Optimizer process (only its results used as initial conditions)

The runner calls GraphOrchestrator.resolve_signals() directly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from CORE_MODELS_models import (
    MasterEngineConfig,
    MarketDataPayload,
    MarketRegime,
    NodeExecutionResult,
    OrchestratorOutput,
)
from CORE_ENGINE_core import GraphOrchestrator
from EXP_ENGINES import TextPersonaEngine, VanillaLLMEngine

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
WDSS_PATH = BASE_DIR / "EXP_DATA_wdss.json"
OPTIMIZED_PATH = BASE_DIR / "DATA_WEIGHTS_optimized.json"
MASTER_PATHS = [
    BASE_DIR / "DATA_JSON_value_master.json",
    BASE_DIR / "DATA_JSON_growth_master.json",
    BASE_DIR / "DATA_JSON_macro_master.json",
    BASE_DIR / "DATA_JSON_risk_master.json",
]
RESULTS_DIR = BASE_DIR / "EXP_RESULTS"

PILOT_SCENARIOS = ["S03", "S09", "S13", "S20"]
CONDITIONS = ["C1", "C2", "C3", "C4"]
LLM_REPETITIONS = 3

ABLATIONS = {
    "no_graham": {
        "description": "Remove VAL_GRAHAM_001 node",
        "remove_nodes": ["VAL_GRAHAM_001"],
        "remove_edges": False,
        "flat_ensemble": False,
    },
    "no_taleb": {
        "description": "Remove RSK_TALEB_001 node",
        "remove_nodes": ["RSK_TALEB_001"],
        "remove_edges": False,
        "flat_ensemble": False,
    },
    "no_risk": {
        "description": "Remove all RSK_* nodes",
        "remove_nodes": ["RSK_TALEB_001", "RSK_SHANNON_001", "RSK_THORP_001"],
        "remove_edges": False,
        "flat_ensemble": False,
    },
    "no_edges": {
        "description": "Remove all Logical Edges",
        "remove_nodes": [],
        "remove_edges": True,
        "flat_ensemble": False,
    },
    "flat_ensemble": {
        "description": "Replace confidence-weighted ensemble with simple average",
        "remove_nodes": [],
        "remove_edges": False,
        "flat_ensemble": True,
    },
}


# ═══════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════

def load_wdss() -> Dict[str, Any]:
    with open(WDSS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_master_configs() -> List[MasterEngineConfig]:
    configs = []
    for path in MASTER_PATHS:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        configs.append(MasterEngineConfig(**data))
    return configs


def load_optimized_weights() -> Dict[str, Any]:
    with open(OPTIMIZED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_optimized_weights(
    configs: List[MasterEngineConfig],
    opt_weights: Dict[str, Any],
) -> List[MasterEngineConfig]:
    """Apply optimized win_rate and decay_factor to master configs."""
    node_weights = opt_weights.get("nodes", {})
    name_to_key = {
        "Benjamin Graham": "graham",
        "Warren Buffett": "buffett",
        "Charlie Munger": "munger",
        "Philip Fisher": "fisher",
        "Peter Lynch": "lynch",
        "George Soros": "soros",
        "Ray Dalio": "dalio",
        "Howard Marks": "marks",
        "Jim Simons": "simons",
        "Nassim Taleb": "taleb",
        "Claude Shannon": "shannon",
        "Edward Thorp": "thorp",
    }

    for config in configs:
        for node in config.Nodes:
            key = name_to_key.get(node.Master)
            if key and key in node_weights:
                w = node_weights[key]
                step3 = node.Intelligence_Structure.Step_3_Statistical_Correction
                if "historical_win_rate" in w:
                    step3.constants["historical_win_rate"] = w["historical_win_rate"]["optimized"]
                if "decay_factor" in w:
                    step3.constants["decay_factor"] = w["decay_factor"]["optimized"]

    return configs


def build_market_data_payload(
    wdss: Dict[str, Any],
    scenario_id: str,
) -> MarketDataPayload:
    """Build MarketDataPayload from wdss scenario (asset + macro metrics merged)."""
    scenario = wdss["scenarios"][scenario_id]
    asset_key = scenario["asset"]
    macro_key = scenario["macro"]

    asset_profile = wdss["asset_profiles"][asset_key]
    macro_regime_data = wdss["macro_regimes"][macro_key]

    metrics = dict(asset_profile["metrics"])
    macro_metrics = macro_regime_data.get("macro_metrics", {})
    metrics.update(macro_metrics)

    regime_str = macro_regime_data["market_regime"]
    try:
        regime = MarketRegime(regime_str)
    except ValueError:
        regime = None

    return MarketDataPayload(
        ticker=asset_profile.get("ticker", f"{asset_key}_{macro_key}"),
        metrics=metrics,
        current_regime=regime,
        regime_confidence=macro_regime_data.get("regime_confidence", 0.5),
    )


# ═══════════════════════════════════════════════════
# Config Manipulation (for ablation studies)
# ═══════════════════════════════════════════════════

def create_ablated_configs(
    base_configs: List[MasterEngineConfig],
    remove_nodes: List[str],
    remove_edges: bool,
) -> List[MasterEngineConfig]:
    """Create modified configs for ablation conditions."""
    ablated = []
    for config in base_configs:
        cfg_dict = config.model_dump()

        if remove_nodes:
            cfg_dict["Nodes"] = [
                n for n in cfg_dict["Nodes"]
                if n["Node_ID"] not in remove_nodes
            ]

        if remove_edges:
            cfg_dict["Logical_Edges"] = []
        else:
            if remove_nodes:
                cfg_dict["Logical_Edges"] = [
                    e for e in cfg_dict["Logical_Edges"]
                    if e["Source"] not in remove_nodes
                    and e["Target"] not in remove_nodes
                ]

        if cfg_dict["Nodes"]:
            ablated.append(MasterEngineConfig(**cfg_dict))

    return ablated


def compute_flat_ensemble(
    node_results: Dict[str, NodeExecutionResult],
) -> float:
    """Simple average of non-Risk domain scores (C3 flat ensemble variant)."""
    scores = [
        r.normalized_score
        for nid, r in node_results.items()
        if not nid.startswith("RSK_")
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_simple_ensemble_for_c2(
    expert_results: Dict[str, NodeExecutionResult],
) -> float:
    """Confidence-weighted ensemble for C2 text persona results.

    Uses the same logic as CORE_ENGINE Phase 4 but simplified
    (no SPG, no position sizing).
    """
    NEUTRAL_POINT = 0.55
    weighted_sum = 0.0
    total_confidence = 0.0

    for nid, result in expert_results.items():
        if nid.startswith("RSK_"):
            continue
        score = result.normalized_score
        dist = abs(score - NEUTRAL_POINT)
        if score >= NEUTRAL_POINT:
            relative_dist = dist / (1.0 - NEUTRAL_POINT + 1e-8)
        else:
            relative_dist = dist / (NEUTRAL_POINT + 1e-8)
        confidence = (relative_dist ** 2.8) + 0.05
        weighted_sum += score * confidence
        total_confidence += confidence

    if total_confidence <= 0:
        return 0.5
    ensemble = weighted_sum / total_confidence
    return max(0.0, min(1.0, ensemble))


# ═══════════════════════════════════════════════════
# Experiment Execution
# ═══════════════════════════════════════════════════

def run_c1(
    configs: List[MasterEngineConfig],
    market_data: MarketDataPayload,
    scenario_id: str,
) -> Dict[str, Any]:
    """C1: Full Ontology — deterministic formula-based scoring."""
    orchestrator = GraphOrchestrator(configs)
    output: OrchestratorOutput = orchestrator.resolve_signals(
        market_data=market_data,
        nlp_data=None,
        extra_context={"_variant_no_spg": True},
    )

    expert_scores = {
        nid: r.normalized_score
        for nid, r in output.node_results.items()
    }

    edge_effects = _extract_edge_effects(output.audit_log)

    return {
        "scenario_id": scenario_id,
        "condition": "C1",
        "repetition": 1,
        "expert_scores": expert_scores,
        "ensemble_signal": output.ensemble_signal,
        "final_position_size": output.final_position_size,
        "tension_score": output.tension_score,
        "override_active": output.override_active,
        "override_source": output.override_source,
        "spg_veto_active": output.spg_veto_active,
        "edge_effects": edge_effects,
        "confidence": output.confidence,
        "audit_log_length": len(output.audit_log),
    }


def run_c2(
    market_data: MarketDataPayload,
    scenario_id: str,
    repetition: int,
    text_persona_engine: TextPersonaEngine,
) -> Dict[str, Any]:
    """C2: Text Persona — LLM generates scores from natural language descriptions."""
    expert_results = text_persona_engine.execute_all_experts(
        market_data=market_data,
        current_regime=market_data.current_regime,
    )

    expert_scores = {
        nid: r.normalized_score for nid, r in expert_results.items()
    }

    ensemble_signal = compute_simple_ensemble_for_c2(expert_results)

    return {
        "scenario_id": scenario_id,
        "condition": "C2",
        "repetition": repetition,
        "expert_scores": expert_scores,
        "ensemble_signal": ensemble_signal,
        "final_position_size": 0.0,
        "tension_score": 0.0,
        "override_active": False,
        "override_source": None,
        "spg_veto_active": False,
        "edge_effects": {},
        "confidence": 1.0,
        "audit_log_length": 0,
    }


def run_c3(
    configs: List[MasterEngineConfig],
    market_data: MarketDataPayload,
    scenario_id: str,
) -> Dict[str, Any]:
    """C3: Flat Ontology — no edges, simple average ensemble."""
    flat_configs = create_ablated_configs(configs, remove_nodes=[], remove_edges=True)
    orchestrator = GraphOrchestrator(flat_configs)
    output: OrchestratorOutput = orchestrator.resolve_signals(
        market_data=market_data,
        nlp_data=None,
        extra_context={"_variant_no_spg": True},
    )

    expert_scores = {
        nid: r.normalized_score for nid, r in output.node_results.items()
    }

    flat_signal = compute_flat_ensemble(output.node_results)

    return {
        "scenario_id": scenario_id,
        "condition": "C3",
        "repetition": 1,
        "expert_scores": expert_scores,
        "ensemble_signal": flat_signal,
        "final_position_size": 0.0,
        "tension_score": output.tension_score,
        "override_active": False,
        "override_source": None,
        "spg_veto_active": False,
        "edge_effects": {},
        "confidence": output.confidence,
        "audit_log_length": len(output.audit_log),
    }


def run_c4(
    market_data: MarketDataPayload,
    scenario_id: str,
    repetition: int,
    vanilla_engine: VanillaLLMEngine,
) -> Dict[str, Any]:
    """C4: Vanilla LLM — single score without any expert framework."""
    details = vanilla_engine.execute_with_details(
        market_data=market_data,
        current_regime=market_data.current_regime,
    )

    return {
        "scenario_id": scenario_id,
        "condition": "C4",
        "repetition": repetition,
        "expert_scores": {},
        "ensemble_signal": details["score"],
        "final_position_size": 0.0,
        "tension_score": 0.0,
        "override_active": False,
        "override_source": None,
        "spg_veto_active": False,
        "edge_effects": {},
        "confidence": 1.0,
        "audit_log_length": 0,
        "llm_response": details.get("raw_response", ""),
    }


def run_ablation(
    base_configs: List[MasterEngineConfig],
    market_data: MarketDataPayload,
    scenario_id: str,
    ablation_name: str,
    ablation_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single ablation test."""
    ablated_configs = create_ablated_configs(
        base_configs,
        remove_nodes=ablation_spec["remove_nodes"],
        remove_edges=ablation_spec["remove_edges"],
    )

    orchestrator = GraphOrchestrator(ablated_configs)
    output: OrchestratorOutput = orchestrator.resolve_signals(
        market_data=market_data,
        nlp_data=None,
        extra_context={"_variant_no_spg": True},
    )

    expert_scores = {
        nid: r.normalized_score for nid, r in output.node_results.items()
    }

    if ablation_spec.get("flat_ensemble"):
        ensemble = compute_flat_ensemble(output.node_results)
    else:
        ensemble = output.ensemble_signal

    return {
        "scenario_id": scenario_id,
        "condition": f"C1_ablation_{ablation_name}",
        "repetition": 1,
        "ablation": ablation_name,
        "expert_scores": expert_scores,
        "ensemble_signal": ensemble,
        "final_position_size": output.final_position_size if not ablation_spec.get("flat_ensemble") else 0.0,
        "tension_score": output.tension_score,
        "override_active": output.override_active,
        "override_source": output.override_source,
        "spg_veto_active": output.spg_veto_active,
        "edge_effects": {},
        "confidence": output.confidence,
        "audit_log_length": len(output.audit_log),
    }


def _extract_edge_effects(audit_log: List[str]) -> Dict[str, Any]:
    """Extract edge effect information from audit log.

    ``GraphOrchestrator`` Phase 3 logs use Korean labels alongside arrows:
    ``시너지`` = synergy, ``억제`` = suppress, ``연속전이`` = continuous edge transfer.
    Phase 1 override logs may include the English substring ``Decay``.
    """
    effects = {}
    for line in audit_log:
        if "→" in line and ("시너지" in line or "억제" in line or "연속전이" in line or "Decay" in line):
            effects[line.strip()[:100]] = True
    return effects


# ═══════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════

def run_all_experiments(
    scenario_ids: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    run_ablations: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the full experiment matrix."""
    wdss = load_wdss()
    base_configs = load_master_configs()
    opt_weights = load_optimized_weights()
    configs = apply_optimized_weights(base_configs, opt_weights)

    if scenario_ids is None:
        scenario_ids = list(wdss["scenarios"].keys())
    if conditions is None:
        conditions = CONDITIONS

    text_persona_engine = None
    vanilla_engine = None
    if "C2" in conditions:
        text_persona_engine = TextPersonaEngine()
    if "C4" in conditions:
        vanilla_engine = VanillaLLMEngine()

    main_results: List[Dict] = []
    ablation_results: List[Dict] = []

    total_main = 0
    for cond in conditions:
        reps = LLM_REPETITIONS if cond in ("C2", "C4") else 1
        total_main += len(scenario_ids) * reps
    total_abl = len(ABLATIONS) * len(scenario_ids) if run_ablations else 0

    logger.info(
        "Experiment start: %d scenarios × %s conditions = %d main + %d ablation = %d total",
        len(scenario_ids), conditions, total_main, total_abl, total_main + total_abl,
    )

    counter = 0
    for scenario_id in scenario_ids:
        market_data = build_market_data_payload(wdss, scenario_id)
        scenario_label = wdss["scenarios"][scenario_id].get("label", scenario_id)

        for condition in conditions:
            reps = LLM_REPETITIONS if condition in ("C2", "C4") else 1

            for rep in range(1, reps + 1):
                counter += 1
                logger.info(
                    "[%d/%d] %s | %s | %s rep=%d",
                    counter, total_main + total_abl,
                    scenario_id, scenario_label, condition, rep,
                )

                try:
                    if condition == "C1":
                        result = run_c1(configs, market_data, scenario_id)
                    elif condition == "C2":
                        result = run_c2(market_data, scenario_id, rep, text_persona_engine)
                    elif condition == "C3":
                        result = run_c3(configs, market_data, scenario_id)
                    elif condition == "C4":
                        result = run_c4(market_data, scenario_id, rep, vanilla_engine)
                    else:
                        raise ValueError(f"Unknown condition: {condition}")

                    result["scenario_label"] = scenario_label
                    result["timestamp"] = datetime.now(timezone.utc).isoformat()
                    main_results.append(result)

                except Exception as e:
                    logger.error("FAILED: %s %s rep=%d — %s", scenario_id, condition, rep, e)
                    main_results.append({
                        "scenario_id": scenario_id,
                        "condition": condition,
                        "repetition": rep,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

    if run_ablations:
        for scenario_id in scenario_ids:
            market_data = build_market_data_payload(wdss, scenario_id)
            for abl_name, abl_spec in ABLATIONS.items():
                counter += 1
                logger.info(
                    "[%d/%d] %s | ablation: %s",
                    counter, total_main + total_abl, scenario_id, abl_name,
                )
                try:
                    result = run_ablation(
                        configs, market_data, scenario_id, abl_name, abl_spec,
                    )
                    result["timestamp"] = datetime.now(timezone.utc).isoformat()
                    ablation_results.append(result)
                except Exception as e:
                    logger.error("FAILED ablation: %s %s — %s", scenario_id, abl_name, e)
                    ablation_results.append({
                        "scenario_id": scenario_id,
                        "condition": f"C1_ablation_{abl_name}",
                        "ablation": abl_name,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

    return main_results, ablation_results


# ═══════════════════════════════════════════════════
# Result Persistence
# ═══════════════════════════════════════════════════

def save_results(
    main_results: List[Dict],
    ablation_results: List[Dict],
    output_dir: Optional[Path] = None,
) -> None:
    """Save results as JSON and CSV."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    main_json_path = output_dir / f"main_results_{ts}.json"
    with open(main_json_path, "w", encoding="utf-8") as f:
        json.dump(main_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Main results saved: %s (%d entries)", main_json_path, len(main_results))

    if ablation_results:
        abl_json_path = output_dir / f"ablation_results_{ts}.json"
        with open(abl_json_path, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Ablation results saved: %s (%d entries)", abl_json_path, len(ablation_results))

    csv_path = output_dir / f"main_results_{ts}.csv"
    _save_results_csv(main_results, csv_path)

    summary = _generate_summary(main_results, ablation_results)
    summary_path = output_dir / f"summary_{ts}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Summary saved: %s", summary_path)


def _save_results_csv(results: List[Dict], path: Path) -> None:
    """Flatten results to CSV for easy analysis."""
    if not results:
        return

    rows = []
    for r in results:
        if "error" in r:
            rows.append({
                "scenario_id": r.get("scenario_id"),
                "condition": r.get("condition"),
                "repetition": r.get("repetition"),
                "ensemble_signal": None,
                "error": r.get("error"),
            })
            continue

        base = {
            "scenario_id": r["scenario_id"],
            "condition": r["condition"],
            "repetition": r.get("repetition", 1),
            "ensemble_signal": r.get("ensemble_signal"),
            "tension_score": r.get("tension_score"),
            "override_active": r.get("override_active"),
            "confidence": r.get("confidence"),
        }

        for nid, score in r.get("expert_scores", {}).items():
            base[f"score_{nid}"] = score

        rows.append(base)

    if not rows:
        return

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _generate_summary(
    main_results: List[Dict],
    ablation_results: List[Dict],
) -> str:
    """Generate a human-readable summary of experiment results."""
    lines = [
        "=" * 70,
        "  H-PIOS Experiment Results Summary",
        f"  Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
        "",
    ]

    by_condition: Dict[str, List[Dict]] = {}
    for r in main_results:
        cond = r.get("condition", "?")
        by_condition.setdefault(cond, []).append(r)

    for cond in sorted(by_condition.keys()):
        results = by_condition[cond]
        errors = [r for r in results if "error" in r]
        success = [r for r in results if "error" not in r]

        ensembles = [r["ensemble_signal"] for r in success if "ensemble_signal" in r]

        lines.append(f"--- Condition {cond} ---")
        lines.append(f"  Total runs: {len(results)} (success: {len(success)}, errors: {len(errors)})")

        if ensembles:
            avg_ens = sum(ensembles) / len(ensembles)
            min_ens = min(ensembles)
            max_ens = max(ensembles)
            lines.append(f"  Ensemble signal: avg={avg_ens:.4f}, min={min_ens:.4f}, max={max_ens:.4f}")

        if success and success[0].get("expert_scores"):
            all_experts = set()
            for r in success:
                all_experts.update(r.get("expert_scores", {}).keys())
            for expert in sorted(all_experts):
                scores = [
                    r["expert_scores"][expert]
                    for r in success if expert in r.get("expert_scores", {})
                ]
                if scores:
                    avg = sum(scores) / len(scores)
                    lines.append(f"    {expert}: avg={avg:.4f} (n={len(scores)})")

        lines.append("")

    if ablation_results:
        lines.append("--- Ablation Results ---")
        by_abl: Dict[str, List[Dict]] = {}
        for r in ablation_results:
            abl = r.get("ablation", "?")
            by_abl.setdefault(abl, []).append(r)

        for abl_name in sorted(by_abl.keys()):
            results = by_abl[abl_name]
            success = [r for r in results if "error" not in r]
            ensembles = [r["ensemble_signal"] for r in success if "ensemble_signal" in r]
            if ensembles:
                avg = sum(ensembles) / len(ensembles)
                lines.append(f"  {abl_name}: avg_ensemble={avg:.4f} (n={len(ensembles)})")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H-PIOS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python EXP_RUNNER.py --pilot               # 4 scenarios × 4 conditions\n"
            "  python EXP_RUNNER.py --pilot --c1-only      # C1 only, no LLM calls\n"
            "  python EXP_RUNNER.py                        # Full 24 × 4 + ablations\n"
            "  python EXP_RUNNER.py --conditions C1 C3     # Deterministic only\n"
            "  python EXP_RUNNER.py --no-ablation          # Main experiments only\n"
        ),
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help=f"Run pilot mode with {len(PILOT_SCENARIOS)} scenarios only ({PILOT_SCENARIOS})",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None, choices=CONDITIONS,
        help="Which conditions to run (default: all)",
    )
    parser.add_argument(
        "--c1-only", action="store_true",
        help="Run C1 only (deterministic, no LLM calls)",
    )
    parser.add_argument(
        "--no-ablation", action="store_true",
        help="Skip ablation experiments",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Specific scenario IDs to run (e.g., S03 S09)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    scenario_ids = args.scenarios
    if args.pilot:
        scenario_ids = PILOT_SCENARIOS

    conditions = args.conditions
    if args.c1_only:
        conditions = ["C1"]

    run_abl = not args.no_ablation

    output_dir = Path(args.output_dir) if args.output_dir else None

    logger.info("=" * 50)
    logger.info("H-PIOS Experiment Runner v1.0")
    logger.info("Scenarios: %s", scenario_ids or "ALL (24)")
    logger.info("Conditions: %s", conditions or "ALL (C1-C4)")
    logger.info("Ablations: %s", "Yes" if run_abl else "No")
    logger.info("=" * 50)

    main_results, ablation_results = run_all_experiments(
        scenario_ids=scenario_ids,
        conditions=conditions,
        run_ablations=run_abl,
    )

    save_results(main_results, ablation_results, output_dir)

    logger.info("Experiment complete. %d main + %d ablation results.",
                len(main_results), len(ablation_results))


if __name__ == "__main__":
    main()
