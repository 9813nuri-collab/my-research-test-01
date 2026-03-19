"""Experimental engines for C2 (Text Persona) and C4 (Vanilla LLM) conditions.

Separated from CORE_ENGINE_core.py to avoid contaminating the core ontology engine.
Uses google-generativeai SDK directly for minimal dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from CORE_MODELS_models import (
    MarketDataPayload,
    MarketRegime,
    NodeExecutionResult,
)

load_dotenv()

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_DEFAULT_TEMPERATURE = 0.0
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0

_PERSONAS_PATH = Path(__file__).parent / "EXP_TEXT_PERSONAS.json"


def _load_personas() -> Dict[str, Any]:
    with open(_PERSONAS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_meta", None)
    return data


def _format_metrics_for_prompt(metrics: Dict[str, float]) -> str:
    lines = []
    for k, v in sorted(metrics.items()):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _call_gemini(prompt: str, temperature: float = _DEFAULT_TEMPERATURE) -> str:
    """Call Gemini API and return text response."""
    import google.generativeai as genai

    genai.configure(api_key=_GOOGLE_API_KEY)
    model = genai.GenerativeModel(_GEMINI_MODEL)

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=512,
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return response.text
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError(f"Gemini API call failed after {_MAX_RETRIES} retries: {e}")
    return ""


def _parse_score_from_response(response_text: str) -> float:
    """Extract a 0.0-1.0 score from LLM response text."""
    patterns = [
        r"(?:score|Score|SCORE)\s*[:=]\s*([01]\.?\d*)",
        r"\*\*([01]\.?\d*)\*\*",
        r"^([01]\.?\d*)\s*$",
        r"([01]\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.MULTILINE)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                continue

    numbers = re.findall(r"(\d+\.?\d*)", response_text)
    for n in numbers:
        val = float(n)
        if 0.0 <= val <= 1.0:
            return val

    return 0.5


class TextPersonaEngine:
    """C2 condition engine: LLM scores based on text persona descriptions.

    For each expert, loads their natural language persona from EXP_TEXT_PERSONAS.json,
    combines it with the financial data, and asks the LLM to produce a 0.0-1.0 score.
    Returns NodeExecutionResult for compatibility with GraphOrchestrator's ensemble.
    """

    def __init__(self, temperature: float = _DEFAULT_TEMPERATURE) -> None:
        self._personas = _load_personas()
        self._temperature = temperature

    def execute(
        self,
        expert_id: str,
        market_data: MarketDataPayload,
        current_regime: Optional[MarketRegime] = None,
    ) -> NodeExecutionResult:
        """Execute a single expert evaluation via LLM text persona."""
        persona_entry = self._personas.get(expert_id)
        if persona_entry is None:
            raise ValueError(f"No persona found for expert: {expert_id}")

        persona_text = persona_entry["persona_text"]
        master_name = persona_entry["master"]

        metrics_text = _format_metrics_for_prompt(market_data.metrics)
        regime_text = current_regime.value if current_regime else "Unknown"

        prompt = (
            f"{persona_text}\n\n"
            f"=== Financial Data ===\n"
            f"Ticker: {market_data.ticker or 'UNKNOWN'}\n"
            f"Current Market Regime: {regime_text} (confidence: {market_data.regime_confidence})\n"
            f"Metrics:\n{metrics_text}\n\n"
            f"=== Instructions ===\n"
            f"Based on the above data and your principles, provide your score.\n"
            f"You MUST output exactly one line in this format:\n"
            f"Score: X.XX\n"
            f"where X.XX is a number between 0.00 and 1.00.\n"
            f"Then briefly explain your reasoning in 2-3 sentences."
        )

        raw_response = _call_gemini(prompt, self._temperature)
        score = _parse_score_from_response(raw_response)

        return NodeExecutionResult(
            node_id=expert_id,
            master=master_name,
            raw_score=score,
            normalized_score=score,
            signal_type=f"TextPersona_{master_name}",
            active_state_flag="LLM_Generated",
            step1_output=score,
            step2_output=0.0,
            step3_output=score,
            used_win_rate=0.0,
            internal_monologue=raw_response[:500],
            metadata={
                "engine": "TextPersonaEngine",
                "condition": "C2",
                "temperature": self._temperature,
                "model": _GEMINI_MODEL,
            },
        )

    def execute_all_experts(
        self,
        market_data: MarketDataPayload,
        current_regime: Optional[MarketRegime] = None,
        exclude_experts: Optional[List[str]] = None,
    ) -> Dict[str, NodeExecutionResult]:
        """Execute experts and return results dict.

        Args:
            exclude_experts: Expert IDs to skip (for C2 ablation experiments).
        """
        expert_ids = [k for k in self._personas.keys() if not k.startswith("_")]
        if exclude_experts:
            expert_ids = [k for k in expert_ids if k not in exclude_experts]
        results = {}
        for eid in expert_ids:
            results[eid] = self.execute(eid, market_data, current_regime)
        return results


class VanillaLLMEngine:
    """C4 condition engine: Single LLM score without any expert framework.

    Provides only the raw financial data and asks for a general investment
    attractiveness score. No expert personas, no ontology structure.
    """

    def __init__(self, temperature: float = _DEFAULT_TEMPERATURE) -> None:
        self._temperature = temperature

    def execute(
        self,
        market_data: MarketDataPayload,
        current_regime: Optional[MarketRegime] = None,
    ) -> float:
        """Return a single 0.0-1.0 investment attractiveness score."""
        metrics_text = _format_metrics_for_prompt(market_data.metrics)
        regime_text = current_regime.value if current_regime else "Unknown"

        prompt = (
            f"You are a financial analyst. Evaluate the following stock's investment attractiveness.\n\n"
            f"=== Financial Data ===\n"
            f"Ticker: {market_data.ticker or 'UNKNOWN'}\n"
            f"Current Market Regime: {regime_text} (confidence: {market_data.regime_confidence})\n"
            f"Metrics:\n{metrics_text}\n\n"
            f"=== Instructions ===\n"
            f"Score this stock's investment attractiveness from 0.0 (complete avoid) to 1.0 (strong buy).\n"
            f"Consider both the individual stock metrics and the macro environment.\n"
            f"You MUST output exactly one line in this format:\n"
            f"Score: X.XX\n"
            f"where X.XX is a number between 0.00 and 1.00.\n"
            f"Then briefly explain your reasoning in 2-3 sentences."
        )

        raw_response = _call_gemini(prompt, self._temperature)
        score = _parse_score_from_response(raw_response)

        return score

    def execute_with_details(
        self,
        market_data: MarketDataPayload,
        current_regime: Optional[MarketRegime] = None,
    ) -> Dict[str, Any]:
        """Return score with full metadata for logging."""
        metrics_text = _format_metrics_for_prompt(market_data.metrics)
        regime_text = current_regime.value if current_regime else "Unknown"

        prompt = (
            f"You are a financial analyst. Evaluate the following stock's investment attractiveness.\n\n"
            f"=== Financial Data ===\n"
            f"Ticker: {market_data.ticker or 'UNKNOWN'}\n"
            f"Current Market Regime: {regime_text} (confidence: {market_data.regime_confidence})\n"
            f"Metrics:\n{metrics_text}\n\n"
            f"=== Instructions ===\n"
            f"Score this stock's investment attractiveness from 0.0 (complete avoid) to 1.0 (strong buy).\n"
            f"Consider both the individual stock metrics and the macro environment.\n"
            f"You MUST output exactly one line in this format:\n"
            f"Score: X.XX\n"
            f"where X.XX is a number between 0.00 and 1.00.\n"
            f"Then briefly explain your reasoning in 2-3 sentences."
        )

        raw_response = _call_gemini(prompt, self._temperature)
        score = _parse_score_from_response(raw_response)

        return {
            "score": score,
            "raw_response": raw_response[:500],
            "engine": "VanillaLLMEngine",
            "condition": "C4",
            "temperature": self._temperature,
            "model": _GEMINI_MODEL,
        }
