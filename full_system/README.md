# Full System Implementation (Supplementary)

These files implement the complete H-PIOS production system described in the paper's Section 3. **They are not required for reproducing the experimental results** — all experiments run from the root directory files.

| File | Description |
|------|-------------|
| `CORE_GRAPH_agent_flow.py` | LangGraph multi-agent pipeline with 4-stage intelligence process, real-time data fetching, and ensemble aggregation |
| `CORE_BRAIN_firmware.py` | Cognitive firmware generator that synthesizes system prompts from all ontology and configuration files |
| `TOOL_OPT_optimizer.py` | Philosophy-inertia optimizer for expert weight calibration using historical data |

These are provided for completeness to demonstrate that the ontology-based multi-expert system is fully implemented beyond the controlled experimental setup.
