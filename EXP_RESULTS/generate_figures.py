"""Generate publication-quality figures for the workshop paper."""

import json
import pathlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

HERE = pathlib.Path(__file__).resolve().parent


def fig_k_sensitivity():
    """Figure 2: Override Decay Coefficient (k) vs OFS / CDI — phase transition."""
    with open(HERE / "supp4_k_sensitivity_20260319_115717.json", encoding="utf-8") as f:
        data = json.load(f)

    ks, ofs_vals, cdi_vals, ens_vals = [], [], [], []
    for kv in data["k_values"]:
        entry = data["per_k"][f"k={kv}"]
        ks.append(kv)
        ofs_vals.append(entry["ofs_score"])
        cdi_vals.append(entry["cdi_score"])
        ens_vals.append(entry["ensemble_mean"])

    fig, ax1 = plt.subplots(figsize=(4.5, 2.8))

    color_ofs = "#2563eb"
    color_cdi = "#dc2626"

    ln1 = ax1.plot(ks, ofs_vals, "o-", color=color_ofs, markersize=5, linewidth=1.8,
                   label="OFS", zorder=3)
    ax1.set_xlabel("Override Decay Coefficient ($k$)")
    ax1.set_ylabel("OFS", color=color_ofs)
    ax1.tick_params(axis="y", labelcolor=color_ofs)
    ax1.set_ylim(0.4, 1.0)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(ks, cdi_vals, "s--", color=color_cdi, markersize=5, linewidth=1.8,
                   label="CDI", zorder=3)
    ax2.set_ylabel("CDI", color=color_cdi)
    ax2.tick_params(axis="y", labelcolor=color_cdi)
    ax2.set_ylim(0.0, 0.8)

    ax1.axvspan(0.5, 1.0, alpha=0.10, color="#facc15", zorder=0)
    mid_y = 0.95
    ax1.annotate("Phase\nTransition", xy=(0.75, mid_y), fontsize=7,
                 ha="center", va="top", style="italic", color="#92400e")

    ax1.axhline(y=0.82, color=color_ofs, linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.text(4.6, 0.83, "C2 OFS", fontsize=6.5, color=color_ofs, alpha=0.7)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", framealpha=0.9)

    ax1.set_xticks(ks)
    ax1.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(HERE / "fig_k_sensitivity.png")
    plt.close(fig)
    print("Saved fig_k_sensitivity.png")


def fig_taf_comparison():
    """Figure 3: Targeted Ablation Fidelity — C1 vs C2 comparison."""
    c1_ablations = [
        ("no_graham", True),
        ("no_taleb", True),
        ("no_risk", True),
        ("no_edges", False),
        ("flat_ensemble", True),
    ]
    c2_ablations = [
        ("no_graham", True),
        ("no_taleb", False),
        ("no_risk", False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True,
                             gridspec_kw={"width_ratios": [5, 3]})

    colors_correct = "#22c55e"
    colors_wrong = "#ef4444"

    for ax, ablations, title, taf_score in [
        (axes[0], c1_ablations, "Structured (C1)", "TAF = 80%"),
        (axes[1], c2_ablations, "LLM Persona (C2)", "TAF = 33%"),
    ]:
        labels = [a[0].replace("_", " ").title() for a in ablations]
        colors = [colors_correct if a[1] else colors_wrong for a in ablations]
        y_pos = np.arange(len(ablations))

        bars = ax.barh(y_pos, [1] * len(ablations), color=colors, height=0.6,
                       edgecolor="white", linewidth=0.5)

        for i, (name, correct) in enumerate(ablations):
            symbol = "O" if correct else "X"
            ax.text(0.5, i, symbol, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"{title}\n{taf_score}", fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_correct, label="Prediction Correct"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_wrong, label="Prediction Failed"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=7,
                   framealpha=0.9)

    fig.tight_layout()
    fig.savefig(HERE / "fig_taf_comparison.png")
    plt.close(fig)
    print("Saved fig_taf_comparison.png")


def fig_expert_ofs():
    """Figure 4: Per-Expert OFS Heatmap — 12 experts x 3 conditions."""
    with open(HERE / "supp1_adaptive_threshold_20260319_115717.json", encoding="utf-8") as f:
        data = json.load(f)

    expert_order = [
        "VAL_GRAHAM_001", "VAL_BUFFETT_001", "VAL_MUNGER_001",
        "GRO_FISHER_001", "GRO_LYNCH_001", "GRO_SOROS_001",
        "MAC_DALIO_001", "MAC_MARKS_001", "MAC_SIMONS_001",
        "RSK_TALEB_001", "RSK_SHANNON_001", "RSK_THORP_001",
    ]
    short_names = [e.split("_")[1].title() for e in expert_order]

    conditions = [
        ("C1$_{fair}$ @0.5", data["ofs_comparison"]["C1_fair"]["fixed_0.5"]["per_expert_ofs"]),
        ("C1$_{fair}$ @0.633", data["ofs_comparison"]["C1_fair"]["adaptive_0.633"]["per_expert_ofs"]),
        ("C2 @0.633", data["ofs_comparison"]["C2"]["adaptive_0.633"]["per_expert_ofs"]),
    ]

    matrix = np.zeros((len(expert_order), len(conditions)))
    for j, (_, ofs_dict) in enumerate(conditions):
        for i, eid in enumerate(expert_order):
            matrix[i, j] = ofs_dict.get(eid, 0.0)

    fig, ax = plt.subplots(figsize=(3.2, 4.0))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c[0] for c in conditions], fontsize=7, rotation=15, ha="right")
    ax.set_yticks(range(len(expert_order)))
    ax.set_yticklabels(short_names, fontsize=7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                    color=color, fontweight="bold" if val >= 0.9 else "normal")

    ax.axhline(2.5, color="white", linewidth=1.5)
    ax.axhline(5.5, color="white", linewidth=1.5)
    ax.axhline(8.5, color="white", linewidth=1.5)

    domain_labels = ["Value", "Growth", "Macro", "Risk"]
    domain_positions = [1, 4, 7, 10]
    for pos, label in zip(domain_positions, domain_labels):
        ax.text(-0.8, pos, label, ha="right", va="center", fontsize=7,
                fontstyle="italic", color="#6b7280")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("OFS", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Per-Expert Ontology Faithfulness", fontsize=9, fontweight="bold", pad=8)

    fig.tight_layout()
    fig.savefig(HERE / "fig_expert_ofs.png")
    plt.close(fig)
    print("Saved fig_expert_ofs.png")


def fig_main_metrics():
    """Figure 1 (supplementary): Main metrics bar chart — OFS, CDI across conditions."""
    conditions = ["C1", "C1$_{fair}$", "C2", "C3", "C3$_{fair}$"]
    ofs_vals = [0.632, 0.458, 0.818, 0.632, 0.514]
    cdi_vals = [0.167, 0.706, 0.723, 0.160, 0.683]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    bars1 = ax.bar(x - width / 2, ofs_vals, width, label="OFS",
                   color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, cdi_vals, width, label="CDI",
                   color="#dc2626", alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7,
                color="#2563eb", fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7,
                color="#dc2626", fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("OFS and CDI Across Experimental Conditions", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(HERE / "fig_main_metrics.png")
    plt.close(fig)
    print("Saved fig_main_metrics.png")


if __name__ == "__main__":
    fig_k_sensitivity()
    fig_taf_comparison()
    fig_expert_ofs()
    fig_main_metrics()
    print("\nAll figures generated successfully.")
