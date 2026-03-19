# The Faithfulness Paradox

**Threshold-Dependent Evaluation Reversals in Multi-Expert AI Systems**

This repository contains the source code, data, and experimental results for reproducing the findings presented in our paper. We propose three evaluation metrics (OFS, CDI, TAF) for multi-expert AI systems and discover a *Faithfulness Paradox* where evaluation conclusions reverse depending on threshold choice.

## Overview

We study how faithfulness should be evaluated in multi-expert AI systems where domain knowledge is encoded either as **executable ontologies** (structured formulas + interaction rules) or as **LLM persona descriptions** (natural language). Our key findings:

- **Faithfulness Paradox**: A formula engine that exactly executes expert rules appears *less* faithful (OFS = 0.46) than LLM personas (OFS = 0.82) under a fixed threshold (τ = 0.5), but this ranking **reverses** under an adaptive threshold (formula OFS = 0.82 vs. LLM OFS = 0.78).
- **Complementary Strengths**: LLM personas achieve higher intent alignment, while structured systems uniquely enable causal interpretability (TAF = 80% vs. 33%).
- **Phase Transition**: A single parameter (override decay coefficient *k*) separates an intent-preserving regime (OFS = 0.924) from an intent-distorting regime (OFS = 0.632).

## Repository Structure

### Core (experiment reproduction)

| File | Description |
|------|-------------|
| `CORE_ENGINE_core.py` | Formula engine, DAG orchestration, 6-phase signal resolution |
| `CORE_MODELS_models.py` | Pydantic v2 data models (`MasterNode`, edges, interactions) |
| `EXP_RUNNER.py` | Main experiment runner (C1–C4 conditions, 24 scenarios) |
| `EXP_SUPPLEMENT.py` | Supplementary experiments (fair conditions, k-sensitivity, C2 ablation) |
| `EXP_METRICS.py` | OFS, CDI, TAF metric implementations |
| `EXP_ENGINES.py` | LLM-based engines for C2 (Text Persona) and C4 (Vanilla LLM) |

### Data

| File | Description |
|------|-------------|
| `DATA_JSON_{value,growth,macro,risk}_master.json` | Expert ontology definitions (12 experts across 4 domains) |
| `DATA_WEIGHTS_optimized.json` | Calibrated expert weights |
| `EXP_DATA_wdss.json` | 24 synthetic market scenarios (4 assets × 6 macro regimes) |
| `EXP_GROUND_TRUTH.json` | OFS ground truth labels per expert per scenario |
| `EXP_TEXT_PERSONAS.json` | Natural language persona descriptions for C2 condition |

### Results (`EXP_RESULTS/`)

| File | Description |
|------|-------------|
| `ANALYSIS_REPORT.txt` | Comprehensive analysis of all 648 runs |
| `main_results_FULL.json` | Raw results for all main conditions |
| `metrics_report_*.json` | Computed OFS, CDI metrics |
| `ablation_results_*.json` | TAF ablation data |
| `supplementary_fair_results.json` | C1_fair / C3_fair results |
| `supp1_adaptive_threshold_*.json` | Adaptive threshold (τ = 0.633) analysis |
| `supp4_k_sensitivity_*.json` | Override decay coefficient sweep |
| `supp2_c2_taf_*.json`, `supp2_partial/` | C2 per-expert ablation results |
| `generate_figures.py` | Regenerate all paper figures |
| `fig_*.png` | Pre-generated figures |

### Full System (`full_system/`) — supplementary

The `full_system/` directory contains the complete production system implementation described in Section 3 of the paper. **These files are not required for reproducing experimental results** but are provided for completeness.

| File | Description |
|------|-------------|
| `CORE_GRAPH_agent_flow.py` | LangGraph multi-agent pipeline (4-stage intelligence, real-time data) |
| `CORE_BRAIN_firmware.py` | Cognitive firmware generator (synthesizes system prompts from ontologies) |
| `TOOL_OPT_optimizer.py` | Philosophy-inertia weight optimizer |

## Requirements & Setup

**Python 3.10+** is required.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```

The system uses **Gemini 2.0 Flash** (via `langchain-google-genai`) for LLM-based conditions (C2, C4). A valid API key is required only for re-running LLM conditions; all pre-computed results are included.

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pydantic | ≥2.0 | Data model validation |
| numpy | ≥1.24 | Numerical computation |
| pandas | ≥2.0 | Data manipulation |
| langgraph | ≥0.2 | Agent pipeline orchestration |
| langchain-google-genai | ≥2.0 | Gemini API integration |
| langchain-community | ≥0.3 | LangChain utilities |
| python-dotenv | ≥1.0 | Environment variable loading |

## Experimental Conditions

We evaluate six conditions across 24 scenarios (4 assets × 6 macro regimes), totaling **648 experimental runs**:

| Condition | Knowledge Repr. | Execution | Interactions | LLM Calls | Runs |
|-----------|----------------|-----------|--------------|-----------|------|
| **C1** (Full Ontology) | Structured JSON | Formula engine | Full DAG edges + override | 0 | 24 |
| **C1_fair** | Structured JSON | Formula engine | No override | 0 | 24 |
| **C2** (Text Persona) | Natural language | LLM (Gemini) | None | 12/scenario | 72 |
| **C3** (Flat Ontology) | Structured JSON | Formula engine | Flat (no edges) + override | 0 | 24 |
| **C3_fair** | Structured JSON | Formula engine | Flat, no override | 0 | 24 |
| **C4** (Vanilla LLM) | None | LLM (Gemini) | None | 12/scenario | 72 |

**Ablations** (120 runs): Five single-component removals from C1 to measure TAF.

## Evaluation Metrics

### Ontology Faithfulness Score (OFS)

Measures whether each expert's output aligns with the direction (bullish/bearish) expected from their investment philosophy:

```
OFS = (1/N) Σ 𝟙[sign(sᵢ − τ) = expectedᵢ]
```

where τ is the decision threshold, sᵢ is the expert's score, and expectedᵢ is the ground truth label.

### Cross-Expert Differentiation Index (CDI)

Quantifies whether experts maintain distinct outputs rather than collapsing to homogeneity:

```
CDI = 1 − (1/C(N,2)) Σᵢ<ⱼ |r(sᵢ, sⱼ)|
```

where r(sᵢ, sⱼ) is the Pearson correlation between expert score vectors.

### Targeted Ablation Fidelity (TAF)

Measures causal interpretability by testing whether pre-registered predictions about component removal effects hold:

```
TAF = correct_predictions / total_predictions
```

## Reproducing Experiments

### Main Experiments

```bash
# Run all conditions (C1–C4) across 24 scenarios
python EXP_RUNNER.py
```

Results are saved to `EXP_RESULTS/` with timestamped filenames.

### Supplementary Experiments

```bash
# Run fair conditions (C1_fair, C3_fair), k-sensitivity, C2 ablation
python EXP_SUPPLEMENT.py
```

### Generate Figures

```bash
# Regenerate all paper figures from result data
python EXP_RESULTS/generate_figures.py
```

Figures are saved as `EXP_RESULTS/fig_*.png`.

**Note**: C1, C3, C1_fair, and C3_fair conditions are fully deterministic (no LLM calls). C2 and C4 conditions use `temperature=0` for maximum reproducibility, but minor variations may occur across runs due to LLM non-determinism.

## Key Results

| Condition | OFS (τ=0.5) | OFS (τ=0.633) | CDI | TAF |
|-----------|-------------|---------------|------|-----|
| C1 (Full Ontology) | 0.632 | 0.924 | 0.167 | 4/5 (80%) |
| C1_fair (no override) | 0.458 | 0.819 | 0.706 | -- |
| C2 (Text Persona) | 0.818 | 0.778 | 0.723 | 1/3 (33%) |
| C3 (Flat Ontology) | 0.632 | -- | 0.160 | -- |
| C3_fair (no override) | 0.514 | -- | 0.683 | -- |
| C4 (Vanilla LLM) | N/A | N/A | N/A | -- |

The **Faithfulness Paradox**: At τ=0.5, C2 (LLM) > C1 (Ontology) in OFS. At τ=0.633 (adaptive), this reverses: C1 > C2.

## License

This repository is provided for academic review purposes.
