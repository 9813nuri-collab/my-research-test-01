"""
Threshold-Independent Faithfulness AUC Analysis
================================================
Computes ROC-AUC and OFS-threshold sweep for C1, C1_fair, C3, C3_fair
(+ C2 as reference) to resolve the Faithfulness Paradox without
threshold dependency.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE = pathlib.Path(__file__).resolve().parent
REPO = BASE.parent

ASSET_TYPES = ["A1_QUALITY", "A2_BALANCED", "A3_TOXIC", "A4_HYPE"]
REGIMES = ["M1_BOOM", "M2_STAGFLATION", "M3_CRISIS", "M4_STEADY", "M5_MANIA", "M6_DEFLATION"]

ASSET_DEPENDENT = [
    "VAL_GRAHAM_001", "VAL_BUFFETT_001", "VAL_MUNGER_001",
    "GRO_FISHER_001", "GRO_LYNCH_001", "GRO_SOROS_001",
    "RSK_TALEB_001", "RSK_SHANNON_001", "RSK_THORP_001",
]
MACRO_DEPENDENT = ["MAC_DALIO_001", "MAC_MARKS_001", "MAC_SIMONS_001"]
ALL_EXPERTS = ASSET_DEPENDENT + MACRO_DEPENDENT

# S01-S04 = M1 x (A1,A2,A3,A4), S05-S08 = M2 x ..., etc.
def scenario_to_asset_regime(sid: str):
    idx = int(sid[1:]) - 1          # S01 -> 0
    regime_idx = idx // 4
    asset_idx = idx % 4
    return ASSET_TYPES[asset_idx], REGIMES[regime_idx]


def load_ground_truth():
    with open(REPO / "EXP_GROUND_TRUTH.json", encoding="utf-8") as f:
        gt = json.load(f)
    return gt


def build_labels(gt):
    """Return dict[(expert_id, scenario_id)] -> 0 or 1."""
    labels = {}
    for sid_num in range(1, 25):
        sid = f"S{sid_num:02d}"
        asset, regime = scenario_to_asset_regime(sid)
        for expert in ALL_EXPERTS:
            if expert in ASSET_DEPENDENT:
                expected = gt["asset_expectations"][expert][asset]["expected"]
            else:
                expected = gt["macro_expectations"][expert][regime]["expected"]
            labels[(expert, sid)] = 1 if expected == "HIGH" else 0
    return labels


def load_scores(condition_filter: str, source: str):
    """Load expert_scores for a given condition from a JSON results file.
    Returns dict[(expert_id, scenario_id)] -> float score.
    For C2 with multiple reps, average across reps.
    """
    with open(BASE / source, encoding="utf-8") as f:
        runs = json.load(f)

    # Group by (expert, scenario) for averaging across reps
    accum: dict[tuple, list[float]] = {}
    for run in runs:
        if run["condition"] != condition_filter:
            continue
        sid = run["scenario_id"]
        for expert, score in run["expert_scores"].items():
            key = (expert, sid)
            accum.setdefault(key, []).append(score)

    return {k: np.mean(v) for k, v in accum.items()}


def align_labels_scores(labels, scores):
    """Return aligned y_true, y_scores arrays for keys present in both."""
    keys = sorted(set(labels) & set(scores))
    y_true = np.array([labels[k] for k in keys])
    y_scores = np.array([scores[k] for k in keys])
    return y_true, y_scores


def compute_ofs_sweep(y_true, y_scores, thresholds):
    """OFS at each threshold tau."""
    ofs_values = []
    for tau in thresholds:
        predicted = (y_scores >= tau).astype(int)
        ofs = np.mean(predicted == y_true)
        ofs_values.append(ofs)
    return np.array(ofs_values)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
gt = load_ground_truth()
labels = build_labels(gt)

conditions = {
    "C1":      ("C1",      "main_results_FULL.json"),
    "C1_fair": ("C1_fair", "supplementary_fair_results.json"),
    "C3":      ("C3",      "main_results_FULL.json"),
    "C3_fair": ("C3_fair", "supplementary_fair_results.json"),
    "C2":      ("C2",      "main_results_FULL.json"),
}

STYLES = {
    "C1":      {"color": "#d62728", "ls": "-",  "lw": 2.0},
    "C1_fair": {"color": "#2ca02c", "ls": "-",  "lw": 2.5},
    "C3":      {"color": "#ff7f0e", "ls": "-",  "lw": 2.0},
    "C3_fair": {"color": "#1f77b4", "ls": "-",  "lw": 2.5},
    "C2":      {"color": "#9467bd", "ls": "--", "lw": 2.0},
}

DISPLAY = {
    "C1":      "C1 (Full Override)",
    "C1_fair": "C1_fair (No Override)",
    "C3":      "C3 (Flat + Override)",
    "C3_fair": "C3_fair (Flat, No Override)",
    "C2":      "C2 (LLM Persona)",
}

data = {}
for name, (cond, src) in conditions.items():
    scores = load_scores(cond, src)
    y_true, y_scores = align_labels_scores(labels, scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    data[name] = {
        "y_true": y_true, "y_scores": y_scores,
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
    }

# ──────────────────────────────────────────────
# Print AUC Table
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ROC-AUC Results (Threshold-Independent Faithfulness)")
print("=" * 55)
print(f"  {'Condition':<28s} {'AUC':>6s}  {'n':>4s}  {'HIGH':>4s}  {'LOW':>4s}")
print("-" * 55)
for name in ["C1", "C1_fair", "C3", "C3_fair", "C2"]:
    d = data[name]
    n_high = int(d["y_true"].sum())
    n_low = len(d["y_true"]) - n_high
    print(f"  {DISPLAY[name]:<28s} {d['auc']:.4f}  {len(d['y_true']):>4d}  {n_high:>4d}  {n_low:>4d}")
print("=" * 55)

# ──────────────────────────────────────────────
# Plot 1: ROC Curves
# ──────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 6.5))

for name in ["C1", "C1_fair", "C3", "C3_fair", "C2"]:
    d = data[name]
    s = STYLES[name]
    label = f"{DISPLAY[name]} (AUC = {d['auc']:.3f})"
    ax1.plot(d["fpr"], d["tpr"], color=s["color"], ls=s["ls"], lw=s["lw"], label=label)

ax1.plot([0, 1], [0, 1], "k:", lw=1.0, alpha=0.4, label="Random (AUC = 0.500)")
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate", fontsize=12)
ax1.set_title("ROC Curves: Threshold-Independent Faithfulness", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(BASE / "fig_roc_auc.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {BASE / 'fig_roc_auc.png'}")

# ──────────────────────────────────────────────
# Plot 2: OFS-Threshold Sweep
# ──────────────────────────────────────────────
tau_range = np.linspace(0.01, 0.99, 300)

fig2, ax2 = plt.subplots(figsize=(8, 5.5))

for name in ["C1", "C1_fair", "C3", "C3_fair", "C2"]:
    d = data[name]
    s = STYLES[name]
    ofs_vals = compute_ofs_sweep(d["y_true"], d["y_scores"], tau_range)
    ax2.plot(tau_range, ofs_vals, color=s["color"], ls=s["ls"], lw=s["lw"], label=DISPLAY[name])

ax2.axvline(x=0.5, color="gray", ls=":", lw=1.2, alpha=0.7)
ax2.axvline(x=0.633, color="gray", ls="--", lw=1.2, alpha=0.7)
ax2.text(0.505, 0.02, r"$\tau=0.5$", fontsize=9, color="gray", transform=ax2.get_xaxis_transform())
ax2.text(0.638, 0.02, r"$\tau=0.633$", fontsize=9, color="gray", transform=ax2.get_xaxis_transform())

ax2.set_xlabel(r"Decision Threshold $\tau$", fontsize=12)
ax2.set_ylabel("OFS (Ontology Faithfulness Score)", fontsize=12)
ax2.set_title(r"OFS vs. Threshold $\tau$: Faithfulness Paradox Reversal", fontsize=13, fontweight="bold")
ax2.legend(loc="best", fontsize=9, framealpha=0.9)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(BASE / "fig_ofs_threshold_sweep.png", dpi=300, bbox_inches="tight")
print(f"Saved: {BASE / 'fig_ofs_threshold_sweep.png'}")

print("\nDone.")
