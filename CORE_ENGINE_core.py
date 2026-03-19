"""
H-PIOS v8.6 — Core Execution Engine (Balanced Intelligence Protocol)
======================================================================
12인 투자 거장의 마스터 JSON으로부터 동적 추론 로직을 구동하는
실전 헤지펀드 수준의 엔진 코어입니다.

[v8.6 Update Note]
------------------
1. **Pessimism Bias Systemic Fix**: 
   - Step 1 Quantitative Analysis의 시작 베이스 점수를 0.2로 설정함으로써, 
     데이터 부재가 곧바로 강력 매도 신호로 이어지는 편향을 제거함.
2. **Collective Intelligence (Weighted Ensemble)**:
   - 전문가의 확신도(Confidence, 0.18 중립점에서의 거리)에 비례한 지수 가중치 적용.
   - 침묵하는 다수의 소음보다 외치는 소수 전문가의 목소리를 우선함.
3. **Organic Tension Tuning**:
   - 시너지 증폭(Synergy Boost) 계수를 0.5에서 0.8로 상향하여 우량주 간 결합력 강화.
   - 리스크 감원폭 및 매크로 보정 계수의 유연성 확보.
"""

from __future__ import annotations

import ast
import math
import operator
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from CORE_MODELS_models import (
    EngineState,
    FinalOutput,
    LogicalEdge,
    MarketDataPayload,
    MarketRegime,
    MasterEngineConfig,
    MasterNode,
    NLPContextPayload,
    NodeExecutionResult,
    OrchestratorOutput,
    PerformanceState,
    QualitativeScenario,
    QuantMetric,
    RelationshipType,
    Step1QuantitativeAnalysis,
    Step2QualitativeContext,
    Step3StatisticalCorrection,
)

logger = logging.getLogger("H-PIOS.engine_core")


# ══════════════════════════════════════════════════════════
# 1. Safe Formula Evaluator — AST 기반 안전 수식 평가
# ══════════════════════════════════════════════════════════

class SafeFormulaEvaluator:
    """AST 기반 제한된 수식 평가기.

    보안 및 런타임 안정성을 위해 Python의 기본 ``eval()``을 사용하지 않고,
    ``ast`` 모듈로 파싱한 뒤 허용된 연산/함수만 화이트리스트로 실행합니다.

    허용 연산:
        - 산술: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``
        - 비교: ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``
        - 논리: ``and``, ``or``, ``not``
        - 단항: ``-``, ``+``
        - 내장 함수: ``MAX``, ``MIN``, ``ABS``, ``ROUND``, ``LOG``, ``EXP``, ``SQRT``
        - 삼항(IF-THEN-ELSE) 패턴도 전처리로 지원

    Usage::

        evaluator = SafeFormulaEvaluator()
        ctx = {"L_quant_graham": 0.6, "L_qual_graham": 0.5,
               "historical_win_rate": 0.62, "decay_factor": 0.85}
        result = evaluator.evaluate(
            "(L_quant_graham + L_qual_graham) * historical_win_rate * decay_factor",
            ctx
        )
    """

    # 허용된 이항 연산자 매핑
    _BINARY_OPS: Dict[type, Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    # 허용된 비교 연산자 매핑
    _COMPARE_OPS: Dict[type, Callable] = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    # 허용된 단항 연산자
    _UNARY_OPS: Dict[type, Callable] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }

    # 허용된 내장 함수 (대소문자 무시)
    _SAFE_FUNCTIONS: Dict[str, Callable] = {
        "MAX": max,
        "MIN": min,
        "ABS": abs,
        "ROUND": round,
        "LOG": math.log,
        "EXP": math.exp,
        "SQRT": math.sqrt,
        "max": max,
        "min": min,
        "abs": abs,
        "round": round,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
    }

    # AST 노드 최대 깊이 (무한 재귀 방어)
    _MAX_DEPTH: int = 50

    def __init__(self, extra_functions: Optional[Dict[str, Callable]] = None) -> None:
        """SafeFormulaEvaluator를 초기화합니다.

        Args:
            extra_functions: 추가 허용 함수 매핑 (선택).
                예: ``{"SIGMOID": lambda x: 1/(1+math.exp(-x))}``
        """
        self._functions = dict(self._SAFE_FUNCTIONS)
        if extra_functions:
            self._functions.update(extra_functions)

    # ── 전처리: JSON 수식의 IF-THEN-ELSE → Python 삼항 변환 ──

    @staticmethod
    def _preprocess_formula(formula: str) -> str:
        """JSON 수식의 비표준 패턴을 Python AST 호환 형태로 전처리.

        지원 변환:
            - ``IF <cond> THEN <val1> ELSE <val2>``
              → ``(<val1>) if (<cond>) else (<val2>)``
            - 변수명의 특수문자(``/``, ``-`` 등)를 ``_``로 치환하여
              Python AST 호환 식별자로 정규화

        Args:
            formula: 원본 수식 문자열

        Returns:
            Python AST 호환 수식 문자열
        """
        import re

        # IF ... THEN ... ELSE ... 패턴 변환 (중첩 미지원, 단일 레벨)
        pattern = r"IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)(?:\s*$|\s*\))"
        
        def _replace_if(match: re.Match) -> str:
            cond = match.group(1).strip()
            then_val = match.group(2).strip()
            else_val = match.group(3).strip()
            return f"(({then_val}) if ({cond}) else ({else_val}))"

        result = formula
        # 반복 치환 (중첩 대응)
        for _ in range(5):
            new_result = re.sub(pattern, _replace_if, result, flags=re.IGNORECASE)
            if new_result == result:
                break
            result = new_result

        # 변수명 특수문자 정규화: P/E_Ratio → P_E_Ratio 등
        # 알파벳/숫자/_ 로 구성된 식별자 사이의 / 를 _ 로 치환
        result = re.sub(r'(?<=[A-Za-z0-9_])/(?=[A-Za-z0-9_])', '_', result)
        # 식별자 내부의 - 도 _ 로 치환 (예: Debt-to-Equity → Debt_to_Equity)
        result = re.sub(r'(?<=[A-Za-z0-9_])-(?=[A-Za-z])', '_', result)

        return result

    # ── AST 노드 평가 (재귀적 트리 워크) ──

    def _eval_node(self, node: ast.AST, context: Dict[str, Any], depth: int = 0) -> Any:
        """AST 노드를 안전하게 재귀 평가합니다.

        Args:
            node: AST 노드
            context: 변수명 → 값 매핑
            depth: 현재 재귀 깊이

        Returns:
            평가 결과 (float, bool 등)

        Raises:
            RecursionError: 깊이 초과 시
            ValueError: 허용되지 않은 AST 노드 감지 시
        """
        if depth > self._MAX_DEPTH:
            raise RecursionError(
                f"수식 평가 깊이 초과 (최대 {self._MAX_DEPTH}). "
                "수식이 너무 복잡하거나 순환 참조가 있을 수 있습니다."
            )

        # ── 상수(Constant) ──
        if isinstance(node, ast.Constant):
            return node.value

        # ── 변수(Name) ──
        if isinstance(node, ast.Name):
            name = node.id
            # 내장 함수명이면 Callable 반환
            if name in self._functions:
                return self._functions[name]
            if name in context:
                return context[name]
            raise ValueError(
                f"알 수 없는 변수: '{name}'. "
                f"사용 가능한 변수: {list(context.keys())}"
            )

        # ── 이항 연산(BinOp) ──
        if isinstance(node, ast.BinOp):
            op_func = self._BINARY_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"허용되지 않은 이항 연산: {type(node.op).__name__}")
            left = self._eval_node(node.left, context, depth + 1)
            right = self._eval_node(node.right, context, depth + 1)
            return op_func(left, right)

        # ── 단항 연산(UnaryOp) ──
        if isinstance(node, ast.UnaryOp):
            op_func = self._UNARY_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"허용되지 않은 단항 연산: {type(node.op).__name__}")
            operand = self._eval_node(node.operand, context, depth + 1)
            return op_func(operand)

        # ── 비교(Compare) ──
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context, depth + 1)
            for op_node, comparator in zip(node.ops, node.comparators):
                op_func = self._COMPARE_OPS.get(type(op_node))
                if op_func is None:
                    raise ValueError(f"허용되지 않은 비교 연산: {type(op_node).__name__}")
                right = self._eval_node(comparator, context, depth + 1)
                if not op_func(left, right):
                    return False
                left = right
            return True

        # ── 논리 연산(BoolOp) ──
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(
                    self._eval_node(v, context, depth + 1) for v in node.values
                )
            if isinstance(node.op, ast.Or):
                return any(
                    self._eval_node(v, context, depth + 1) for v in node.values
                )
            raise ValueError(f"허용되지 않은 논리 연산: {type(node.op).__name__}")

        # ── 함수 호출(Call) ──
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func, context, depth + 1)
            if not callable(func):
                raise ValueError(f"호출 불가능한 객체: {func}")
            args = [self._eval_node(a, context, depth + 1) for a in node.args]
            return func(*args)

        # ── 삼항 표현식(IfExp) ──
        if isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, context, depth + 1)
            if test:
                return self._eval_node(node.body, context, depth + 1)
            return self._eval_node(node.orelse, context, depth + 1)

        # ── Expression 래퍼 ──
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, context, depth + 1)

        raise ValueError(
            f"허용되지 않은 AST 노드 유형: {type(node).__name__}. "
            "보안상 제한된 수식만 평가 가능합니다."
        )

    # ── 공개 API ──

    @staticmethod
    def _normalize_key(key: str) -> str:
        """변수명의 특수문자를 _로 정규화합니다.

        _preprocess_formula와 동일한 규칙으로 context 키를 정규화하여
        수식 내 변수명과 일치시킵니다.

        Args:
            key: 원본 변수명

        Returns:
            정규화된 변수명
        """
        import re
        result = re.sub(r'(?<=[A-Za-z0-9_])/(?=[A-Za-z0-9_])', '_', key)
        result = re.sub(r'(?<=[A-Za-z0-9_])-(?=[A-Za-z])', '_', result)
        return result

    def evaluate(self, formula: str, context: Dict[str, Any]) -> float:
        """수식 문자열을 안전하게 평가합니다.

        Args:
            formula: 수식 문자열 (JSON에서 로드)
            context: 변수명 → 값 매핑 (Step 1/2 출력, 상수 등)

        Returns:
            평가 결과 (float)

        Raises:
            ValueError: 수식 파싱/평가 실패 시
            RecursionError: 깊이 초과 시
        """
        preprocessed = self._preprocess_formula(formula)

        # context 키도 동일 규칙으로 정규화 (P/E_Ratio → P_E_Ratio)
        normalized_ctx: Dict[str, Any] = {}
        for k, v in context.items():
            normalized_ctx[self._normalize_key(k)] = v
            # 원본 키도 유지 (정규화 전 키로 참조되는 경우 대비)
            if k != self._normalize_key(k):
                normalized_ctx[k] = v

        try:
            tree = ast.parse(preprocessed, mode="eval")
        except SyntaxError as e:
            raise ValueError(
                f"수식 파싱 실패: '{formula}' → 전처리: '{preprocessed}'. "
                f"원인: {e}"
            ) from e

        result = self._eval_node(tree, normalized_ctx)

        # bool → float 변환 (True=1.0, False=0.0)
        if isinstance(result, bool):
            return 1.0 if result else 0.0
        return float(result)


# ══════════════════════════════════════════════════════════
# 2. Score Normalizer — 플러그인 가능한 정규화 전략
# ══════════════════════════════════════════════════════════

@runtime_checkable
class ScoreNormalizer(Protocol):
    """점수 정규화 프로토콜 (Pluggable Design).

    모든 정규화 전략은 이 프로토콜을 구현해야 합니다.
    0.0 ~ 1.0 사이의 정규화된 점수를 반환합니다.
    """

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """원시 점수를 0.0 ~ 1.0으로 정규화합니다.

        Args:
            raw_score: 정규화 전 원시 점수
            **kwargs: 전략별 추가 파라미터

        Returns:
            0.0 ~ 1.0 사이의 정규화 점수
        """
        ...


class SigmoidNormalizer:
    """Sigmoid 정규화기 (기본 전략).

    수식: f(x) = 1 / (1 + e^(-k * (x - x0)))

    Parameters:
        k: 기울기 계수 (클수록 급격한 전환). 기본 5.0
        x0: 중심점 (시그모이드의 0.5 지점). 기본 0.5
    """

    def __init__(self, k: float = 5.0, x0: float = 0.5) -> None:
        """SigmoidNormalizer를 초기화합니다.

        Args:
            k: 기울기 계수 (기본 5.0)
            x0: 중심점 (기본 0.5)
        """
        self.k = k
        # [Calibration] x0 set to 0.21 for academic zero-point (Baseline maps to ~0.55)
        self.x0 = 0.21

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Sigmoid 함수로 정규화합니다.

        Args:
            raw_score: 원시 점수
            **kwargs: 무시됨 (프로토콜 호환용)

        Returns:
            0.0 ~ 1.0 사이의 정규화 점수
        """
        try:
            return 1.0 / (1.0 + math.exp(-self.k * (raw_score - self.x0)))
        except OverflowError:
            # exp 오버플로우: 극단값 처리
            return 0.0 if raw_score < self.x0 else 1.0


class ZScoreNormalizer:
    """Z-Score 기반 정규화기.

    원시 점수를 이동 평균/표준편차 대비 표준화한 뒤,
    Sigmoid로 0~1 범위에 매핑합니다.

    Parameters:
        mean: 이동 평균 (기본 0.0)
        std: 이동 표준편차 (기본 1.0)
        sigmoid_k: 후단 Sigmoid 기울기 (기본 1.5)
    """

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, sigmoid_k: float = 1.5
    ) -> None:
        self.mean = mean
        self.std = std
        self._sigmoid = SigmoidNormalizer(k=sigmoid_k, x0=0.0)

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Z-Score → Sigmoid로 정규화합니다.

        Args:
            raw_score: 원시 점수
            **kwargs:
                mean (float): 동적 평균 (선택)
                std (float): 동적 표준편차 (선택)

        Returns:
            0.0 ~ 1.0 사이의 정규화 점수
        """
        mean = kwargs.get("mean", self.mean)
        std = kwargs.get("std", self.std)
        if std == 0:
            std = 1e-8  # 제로 분산 방어
        z = (raw_score - mean) / std
        return self._sigmoid.normalize(z)


class MinMaxNormalizer:
    """Min-Max 정규화기.

    이력 최솟값/최댓값 기반으로 선형 정규화합니다.

    Parameters:
        min_val: 이력 최솟값 (기본 0.0)
        max_val: 이력 최댓값 (기본 2.0)
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 2.0) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Min-Max 정규화를 수행합니다.

        Args:
            raw_score: 원시 점수
            **kwargs:
                min_val (float): 동적 최솟값 (선택)
                max_val (float): 동적 최댓값 (선택)

        Returns:
            0.0 ~ 1.0 사이의 정규화 점수 (클리핑 적용)
        """
        lo = kwargs.get("min_val", self.min_val)
        hi = kwargs.get("max_val", self.max_val)
        if hi == lo:
            return 0.5
        normalized = (raw_score - lo) / (hi - lo)
        return max(0.0, min(1.0, normalized))


# ══════════════════════════════════════════════════════════
# 3. AbstractMasterEngine — 개별 거장 노드 실행 엔진
# ══════════════════════════════════════════════════════════

class AbstractMasterEngine:
    """12인 투자 거장 개별 노드 실행 엔진.

    MasterNode JSON 구조를 기반으로 3-Step 추론 파이프라인을 실행하고,
    정규화된 최종 점수를 산출합니다.

    핵심 기능:
        - Step 1: 정량 분석 (logic_formula 동적 평가)
        - Step 2: 정성 평가 (NLP 키워드/코사인 유사도 기반)
        - Step 3: 통계 보정 (formula 동적 평가)
        - 정규화: 플러그인 가능한 Normalizer (기본 Sigmoid)
        - 승률 갱신: update_rolling_win_rate()로 동적 덮어쓰기

    Args:
        node: MasterNode Pydantic 모델
        normalizer: ScoreNormalizer 프로토콜 구현체 (기본 SigmoidNormalizer)
        evaluator: SafeFormulaEvaluator 인스턴스 (공유 가능)
    """

    def __init__(
        self,
        node: MasterNode,
        normalizer: Optional[ScoreNormalizer] = None,
        evaluator: Optional[SafeFormulaEvaluator] = None,
    ) -> None:
        self.node = node
        self.normalizer: ScoreNormalizer = normalizer or SigmoidNormalizer()
        self.evaluator = evaluator or SafeFormulaEvaluator()

        # JSON 초기값을 기본 승률로 저장
        self._default_win_rate: float = (
            node.Intelligence_Structure
            .Step_3_Statistical_Correction
            .historical_win_rate
        )
        # 동적 롤링 승률 (None이면 default 사용)
        self._rolling_win_rate: Optional[float] = None

        # 엔진 상태
        self.state = EngineState(node_id=node.Node_ID)

    # ── 프로퍼티 ──

    @property
    def node_id(self) -> str:
        """노드 고유 식별자."""
        return self.node.Node_ID

    @property
    def effective_win_rate(self) -> float:
        """현재 유효 승률.

        동적 롤링 승률이 설정되었으면 그 값을, 아니면 JSON 초기값을 반환합니다.
        """
        if self._rolling_win_rate is not None:
            return self._rolling_win_rate
        if self.state.rolling_win_rate is not None:
            return self.state.rolling_win_rate
        return self._default_win_rate

    # ── 승률 갱신 ──

    def update_rolling_win_rate(
        self, performance: PerformanceState
    ) -> float:
        """외부 백테스트/라이브 성과 데이터로 롤링 승률을 동적 갱신합니다.

        JSON의 historical_win_rate는 초기값(Default)으로만 취급되며,
        이 메서드가 호출되면 동적으로 덮어씁니다.

        Args:
            performance: 성과 추적 상태 (PerformanceState)

        Returns:
            갱신된 롤링 승률

        Raises:
            ValueError: 유효한 거래 이력이 없는 경우
        """
        calculated = performance.calculated_win_rate
        if calculated is None:
            raise ValueError(
                f"노드 {self.node_id}: 유효한 거래 이력이 없어 승률 갱신 불가. "
                f"total_trades={performance.total_trades}"
            )

        self._rolling_win_rate = calculated
        self.state.rolling_win_rate = calculated
        self.state.last_updated = datetime.utcnow()

        logger.info(
            "노드 %s 롤링 승률 갱신: %.4f → %.4f (trades=%d)",
            self.node_id, self._default_win_rate, calculated,
            performance.total_trades,
        )
        return calculated

    # ── Step 1: 정량 분석 ──

    def _execute_step1(
        self, market_data: MarketDataPayload
    ) -> float:
        """Step 1: 정량 분석을 실행합니다.

        각 QuantMetric의 logic_formula를 SafeFormulaEvaluator로 평가하고,
        결과를 합산합니다.

        Args:
            market_data: 재무/매크로 정량 데이터 페이로드

        Returns:
            Step 1 정량 점수 합계
        """
        step1 = self.node.Intelligence_Structure.Step_1_Quantitative_Analysis
        # [v8.6] Pessimism Bias Systemic Fix - 베이스 점수 하한 0.2 설정
        # (기존 additive 방식에서 max 방식으로 변경하여 Bullish Bias 방지)
        total_score = 0.0

        for metric_def in step1.metrics:
            # 원본 키 및 정규화 키 모두로 조회 (P/E_Ratio 등 대응)
            metric_value = market_data.metrics.get(metric_def.metric)
            if metric_value is None:
                normalized_key = self.evaluator._normalize_key(metric_def.metric)
                metric_value = market_data.metrics.get(normalized_key)

            if metric_value is None:
                logger.debug(
                    "노드 %s Step1: 지표 '%s' 데이터 없음 → 0점 처리",
                    self.node_id, metric_def.metric,
                )
                continue

            if metric_def.logic_formula:
                # logic_formula를 SafeEval로 동적 해석
                context = {metric_def.metric: metric_value}
                try:
                    score = self.evaluator.evaluate(metric_def.logic_formula, context)
                except (ValueError, RecursionError) as e:
                    logger.warning(
                        "노드 %s Step1: 수식 평가 실패 '%s' → 0점. 원인: %s",
                        self.node_id, metric_def.logic_formula, e,
                    )
                    score = 0.0
            else:
                # logic_formula 없으면 operator/threshold 기반 직접 비교
                score = self._evaluate_metric_direct(metric_def, metric_value)

            total_score += score
            
        return max(0.40, total_score)

    @staticmethod
    def _evaluate_metric_direct(metric_def: QuantMetric, value: float) -> float:
        """logic_formula 없을 때 operator/threshold로 직접 평가합니다.

        Args:
            metric_def: 정량 지표 정의
            value: 실제 지표 값

        Returns:
            조건 충족 시 weight, 미충족 시 0.0
        """
        op_map = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
        op_func = op_map.get(metric_def.operator)
        if op_func and op_func(value, metric_def.threshold):
            return metric_def.weight
        # Trend, Volatility 등 특수 연산자는 threshold 기반 단순 비교
        if metric_def.operator in ("Trend", "Volatility"):
            if value > metric_def.threshold:
                return metric_def.weight
        return 0.0

    # ── Step 2: 정성 평가 ──

    def _execute_step2(
        self, nlp_data: Optional[NLPContextPayload]
    ) -> Tuple[float, Optional[str]]:
        """Step 2: 정성 평가를 실행합니다.

        NLP 키워드 매칭 및 코사인 유사도 기반으로 시나리오 점수를 산출합니다.
        ``CRITICAL_INHIBIT`` 액션이 발동되면 특수 플래그를 반환합니다.

        Args:
            nlp_data: NLP 컨텍스트 페이로드 (None이면 0.0)

        Returns:
            (Step 2 정성 점수, 활성 특수 액션 또는 None)
        """
        if nlp_data is None:
            return 0.0, None

        step2 = self.node.Intelligence_Structure.Step_2_Qualitative_Context
        total_score = 0.0
        critical_action: Optional[str] = None

        for scenario in step2.scenarios:
            activation_score = self._evaluate_scenario(
                scenario, nlp_data, step2.nlp_confidence_threshold
            )

            if activation_score > 0.0 or (
                activation_score == 0.0
                and scenario.score_modifier < 0
                and self._check_keyword_match(scenario, nlp_data)
            ):
                # 시나리오 발동
                effective_modifier = scenario.score_modifier * activation_score
                total_score += effective_modifier

                if scenario.action == "CRITICAL_INHIBIT":
                    critical_action = "CRITICAL_INHIBIT"
                    total_score += scenario.score_modifier  # 강제 억제
                    logger.warning(
                        "노드 %s Step2: CRITICAL_INHIBIT 발동 — %s",
                        self.node_id, scenario.condition,
                    )

        return total_score, critical_action

    def _evaluate_scenario(
        self,
        scenario: QualitativeScenario,
        nlp_data: NLPContextPayload,
        confidence_threshold: float,
    ) -> float:
        """개별 시나리오의 발동 점수를 계산합니다.

        우선순위:
        1. semantic_similarity_scores에 해당 condition의 유사도가 있으면 사용
        2. 없으면 키워드 매칭 기반 휴리스틱

        Args:
            scenario: 정성 시나리오 정의
            nlp_data: NLP 데이터 페이로드
            confidence_threshold: NLP 신뢰도 하한

        Returns:
            발동 점수 (0.0 ~ 1.0, 미발동 시 0.0)
        """
        # NLP 모델 전역 신뢰도 체크
        if nlp_data.nlp_model_confidence < confidence_threshold:
            return 0.0

        # 1) 코사인 유사도 기반 평가 (Semantic NLP 뼈대)
        similarity = nlp_data.semantic_similarity_scores.get(scenario.condition)
        if similarity is not None:
            # 유사도가 confidence_threshold 이상이면 발동
            if similarity >= confidence_threshold:
                return similarity
            return 0.0

        # 2) 키워드 매칭 기반 폴백
        if self._check_keyword_match(scenario, nlp_data):
            return 1.0

        return 0.0

    @staticmethod
    def _check_keyword_match(
        scenario: QualitativeScenario,
        nlp_data: NLPContextPayload,
    ) -> bool:
        """키워드 매칭으로 시나리오 발동 여부를 판단합니다.

        detected_keywords에 시나리오 키워드가 존재하거나,
        raw_texts에 키워드가 포함되어 있으면 발동입니다.

        Args:
            scenario: 시나리오 정의
            nlp_data: NLP 데이터

        Returns:
            키워드 매칭 여부
        """
        if not scenario.keywords:
            return False

        # detected_keywords 매핑 체크
        for kw in scenario.keywords:
            if kw in nlp_data.detected_keywords:
                return True

        # raw_texts 전문 검색 폴백
        combined_text = " ".join(nlp_data.raw_texts).lower()
        for kw in scenario.keywords:
            if kw.lower() in combined_text:
                return True

        return False

    # ── Step 3: 통계 보정 ──

    def _execute_step3(
        self,
        step1_score: float,
        step2_score: float,
        current_regime: Optional[MarketRegime] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Step 3: 통계 보정을 실행합니다.

        Step 1 + Step 2 결과를 JSON의 formula로 결합하고,
        historical_win_rate (동적 갱신 가능), decay_factor로 보정합니다.

        Args:
            step1_score: Step 1 정량 점수
            step2_score: Step 2 정성 점수
            current_regime: 현재 시장 국면 (선택)
            extra_context: 추가 컨텍스트 변수 (예: 다른 노드 점수)

        Returns:
            보정된 최종 점수
        """
        step3 = self.node.Intelligence_Structure.Step_3_Statistical_Correction
        output_var_s1 = (
            self.node.Intelligence_Structure
            .Step_1_Quantitative_Analysis.output_variable
        )
        output_var_s2 = (
            self.node.Intelligence_Structure
            .Step_2_Qualitative_Context.output_variable
        )

        # 수식 컨텍스트 구성
        context: Dict[str, Any] = {
            output_var_s1: step1_score,
            output_var_s2: step2_score,
            "historical_win_rate": self.effective_win_rate,
            "decay_factor": step3.decay_factor,
        }

        # 국면별 성과 보정 (v8.6: N/A 및 타입 에러 방어 로직 추가)
        if current_regime:
            regime_key = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)
            if regime_key in step3.regime_performance:
                regime_perf = step3.regime_performance[regime_key]
                try:
                    if regime_perf != "N/A":
                        context["historical_win_rate"] = (
                            self.effective_win_rate * float(regime_perf)
                        )
                except (ValueError, TypeError):
                    pass

        # 추가 컨텍스트 (다른 노드 점수 등)
        if extra_context:
            context.update(extra_context)

        try:
            result = self.evaluator.evaluate(step3.formula, context)
        except (ValueError, RecursionError) as e:
            logger.error(
                "노드 %s Step3: 수식 평가 실패 '%s' → 0.0. 원인: %s",
                self.node_id, step3.formula, e,
            )
            result = 0.0

        return result

    # ── 상태 플래그 결정 ──

    def _determine_state_flag(self, normalized_score: float) -> Optional[str]:
        """정규화 점수 기반으로 활성 상태 플래그를 결정합니다.

        State_Flags 리스트의 인덱스를 점수 구간에 매핑합니다:
        - 높은 점수 → 첫 번째 플래그 (긍정적)
        - 낮은 점수 → 마지막 플래그 (경고)

        Args:
            normalized_score: 0.0 ~ 1.0 정규화 점수

        Returns:
            활성 상태 플래그 문자열 또는 None
        """
        flags = self.node.Final_Output.State_Flags
        if not flags:
            return None
        n = len(flags)
        # 점수를 n개 구간으로 분할
        idx = min(int(normalized_score * n), n - 1)
        # 높은 점수 → 첫 번째 플래그 (역순 매핑)
        return flags[n - 1 - idx]

    # ── 통합 실행 ──

    def execute(
        self,
        market_data: MarketDataPayload,
        nlp_data: Optional[NLPContextPayload] = None,
        current_regime: Optional[MarketRegime] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> NodeExecutionResult:
        """3-Step 파이프라인을 통합 실행하고 정규화된 결과를 반환합니다.

        Args:
            market_data: 정량 데이터 페이로드
            nlp_data: NLP 정성 데이터 페이로드 (선택)
            current_regime: 현재 시장 국면 (선택)
            extra_context: 추가 컨텍스트 변수 딕셔너리 (선택)

        Returns:
            NodeExecutionResult — 정규화 점수 포함
        """
        # Step 1: 정량 분석
        step1_score = self._execute_step1(market_data)

        # Step 2: 정성 평가
        step2_score, critical_action = self._execute_step2(nlp_data)

        # CRITICAL_INHIBIT 발동 시 즉시 0점
        if critical_action == "CRITICAL_INHIBIT":
            logger.warning(
                "노드 %s: CRITICAL_INHIBIT 발동 → 강제 0점 할당", self.node_id
            )
            result = NodeExecutionResult(
                node_id=self.node_id,
                master=self.node.Master,
                raw_score=0.0,
                normalized_score=0.0,
                signal_type=self.node.Final_Output.Signal_Type,
                active_state_flag="CRITICAL_INHIBIT",
                step1_output=step1_score,
                step2_output=step2_score,
                step3_output=0.0,
                used_win_rate=self.effective_win_rate,
                metadata={"critical_action": critical_action},
            )
            self._update_state(result)
            return result

        # Step 3: 통계 보정
        step3_score = self._execute_step3(
            step1_score, step2_score, current_regime, extra_context
        )

        # 정규화 (0.0 ~ 1.0)
        normalized = self.normalizer.normalize(step3_score)

        # 상태 플래그 결정
        state_flag = self._determine_state_flag(normalized)

        result = NodeExecutionResult(
            node_id=self.node_id,
            master=self.node.Master,
            raw_score=step3_score,
            normalized_score=normalized,
            signal_type=self.node.Final_Output.Signal_Type,
            active_state_flag=state_flag,
            step1_output=step1_score,
            step2_output=step2_score,
            step3_output=step3_score,
            used_win_rate=self.effective_win_rate,
        )

        self._update_state(result)
        return result

    def _update_state(self, result: NodeExecutionResult) -> None:
        """실행 결과로 엔진 상태를 갱신합니다.

        Args:
            result: 노드 실행 결과
        """
        self.state.last_score = result.normalized_score
        self.state.signal_history.append(result.normalized_score)
        # 최근 100개만 유지
        if len(self.state.signal_history) > 100:
            self.state.signal_history = self.state.signal_history[-100:]
        self.state.last_updated = datetime.utcnow()

        # 시그널 지속성 갱신
        if len(self.state.signal_history) >= 2:
            prev = self.state.signal_history[-2]
            curr = self.state.signal_history[-1]
            if (prev >= 0.5 and curr >= 0.5) or (prev < 0.5 and curr < 0.5):
                self.state.signal_persistence += 1
            else:
                self.state.signal_persistence = 0


# ══════════════════════════════════════════════════════════
# 4. GraphOrchestrator — 12인 거장 DAG 시그널 해소 시스템
# ══════════════════════════════════════════════════════════

class GraphOrchestrator:
    """12인 투자 거장 노드를 통합 관리하는 시냅스 시스템.

    핵심 기능:
        1. **위상 정렬 (Topological Sort)** — 노드 간 의존성(엣지)을 분석하여
           안전한 실행 순서를 보장합니다. 순환 참조 감지 시 예외 발생.

        2. **시그널 해소 (resolve_signals)** — 4단계 위계:
           (a) Priority Override: Taleb 블랙스완 등 치명적 리스크 → 강제 셧다운
           (b) Global Adjustment: Dalio 등 매크로 점수 → 연속 함수로 가중치 조절
           (c) Synergy/Suppress: 노드 간 상호작용 (멍거-버핏 등)
           (d) Position Sizing: Thorp Kelly 기준 최종 비중 체인링

        3. **연속적 가중치 조절** — Threshold 이분법 대신 Linear Interpolation /
           Exponential Decay 등 연속 함수로 부드러운 감쇠/증폭

    Args:
        configs: MasterEngineConfig 리스트 (4개 도메인 JSON)
        normalizer: 공유 ScoreNormalizer (기본 SigmoidNormalizer)
        evaluator: 공유 SafeFormulaEvaluator (기본 새 인스턴스)
    """

    # ── Priority Override 우선순위 (높을수록 우선) ──
    _OVERRIDE_PRIORITY: Dict[str, int] = {
        "Master_Override": 100,   # Taleb: 블랙스완, 유동성 동결
        "Override": 80,           # Munger: 사기/회계 위험
        "Global_Weight_Adjuster": 60,  # Dalio: 매크로 국면
        "Suppress": 40,           # 일반 억제
        "Synergize_With": 20,     # 시너지
    }

    # Taleb 임계값: 이 이상이면 마스터 오버라이드 발동
    _TALEB_RUIN_THRESHOLD: float = 0.75  # 정규화 기준

    def __init__(
        self,
        configs: List[MasterEngineConfig],
        normalizer: Optional[ScoreNormalizer] = None,
        evaluator: Optional[SafeFormulaEvaluator] = None,
    ) -> None:
        self._normalizer = normalizer or SigmoidNormalizer()
        self._evaluator = evaluator or SafeFormulaEvaluator()

        # 노드 엔진 레지스트리: node_id → AbstractMasterEngine
        self._engines: Dict[str, AbstractMasterEngine] = {}
        # 논리 엣지 전체 목록
        self._edges: List[LogicalEdge] = []
        # 위상 정렬 결과 캐시
        self._sorted_order: Optional[List[str]] = None

        # 글로벌 엔진 상태
        self.global_state = EngineState(node_id="GLOBAL")

        self._load_configs(configs)

    # ── 초기화 ──

    def _load_configs(self, configs: List[MasterEngineConfig]) -> None:
        """MasterEngineConfig 리스트로부터 엔진과 엣지를 로드합니다.

        Args:
            configs: 4개 도메인의 MasterEngineConfig
        """
        for config in configs:
            for node in config.Nodes:
                engine = AbstractMasterEngine(
                    node=node,
                    normalizer=self._normalizer,
                    evaluator=self._evaluator,
                )
                self._engines[node.Node_ID] = engine
                logger.info(
                    "노드 등록: %s (%s — %s)",
                    node.Node_ID, node.Master, config.Engine_Domain,
                )

            self._edges.extend(config.Logical_Edges)

        # 위상 정렬 수행
        self._sorted_order = self._topological_sort()
        logger.info("위상 정렬 완료: %s", self._sorted_order)

    # ── 위상 정렬 (DAG) ──

    def _topological_sort(self) -> List[str]:
        """노드 간 의존성을 분석하여 위상 정렬을 수행합니다.

        엣지의 Source → Target 방향으로 의존성 그래프를 구성하고,
        Kahn 알고리즘으로 안전한 실행 순서를 보장합니다.

        와일드카드 타겟(ALL_*)은 등록된 모든 노드로 확장합니다.

        Returns:
            노드 ID 실행 순서 리스트

        Raises:
            ValueError: 순환 참조 감지 시
        """
        all_node_ids = set(self._engines.keys())

        # 인접 리스트 및 진입 차수 구성
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = {nid: 0 for nid in all_node_ids}

        for edge in self._edges:
            source = edge.Source
            if source not in all_node_ids:
                continue

            # 타겟 확장 (와일드카드 처리)
            targets = self._resolve_targets(edge.Target, all_node_ids, source)

            for target in targets:
                if target in all_node_ids and target != source:
                    # Source가 먼저 실행되어야 Target에 영향을 줄 수 있음
                    if target not in adjacency[source]:
                        adjacency[source].add(target)
                        in_degree[target] += 1

        # Kahn 알고리즘
        queue: deque[str] = deque(
            nid for nid in all_node_ids if in_degree[nid] == 0
        )
        sorted_order: List[str] = []

        while queue:
            node_id = queue.popleft()
            sorted_order.append(node_id)
            for neighbor in adjacency.get(node_id, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(all_node_ids):
            missing = all_node_ids - set(sorted_order)
            raise ValueError(
                f"순환 참조 감지: 위상 정렬 불가. 미처리 노드: {missing}"
            )

        return sorted_order

    @staticmethod
    def _resolve_targets(
        target: str, all_node_ids: Set[str], source: str
    ) -> List[str]:
        """와일드카드 타겟을 실제 노드 ID 리스트로 확장합니다.

        Args:
            target: 타겟 문자열 (노드 ID 또는 'ALL_*')
            all_node_ids: 등록된 모든 노드 ID 집합
            source: 소스 노드 ID (자기 참조 방지)

        Returns:
            타겟 노드 ID 리스트
        """
        if target.startswith("ALL_"):
            if target == "ALL_OTHER_ENGINES":
                return [nid for nid in all_node_ids if nid != source]
            elif target == "ALL_GROWTH_AND_VALUE_NODES":
                return [
                    nid for nid in all_node_ids
                    if nid.startswith("GRO_") or nid.startswith("VAL_")
                ]
            else:
                # 기타 ALL_ 패턴: 소스 제외 전체
                return [nid for nid in all_node_ids if nid != source]
        return [target]

    # ── 공개 API: 시그널 해소 ──

    def resolve_signals(
        self,
        market_data: MarketDataPayload,
        nlp_data: Optional[NLPContextPayload] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorOutput:
        """12인 거장 시그널을 통합 해소하고 최종 포지션을 산출합니다.

        실행 위계:
            1. Priority Override — 치명적 시스템 리스크 감지 시 강제 셧다운
            2. Global Adjustment — 매크로 국면 기반 연속 가중치 조절
            3. Synergy/Suppress — 노드 간 상호작용 반영
            4. Position Sizing — Thorp Kelly 기준 최종 비중

        Args:
            market_data: 정량 데이터 페이로드
            nlp_data: NLP 정성 데이터 (선택)

        Returns:
            OrchestratorOutput — 노드별 결과, 앙상블 시그널, 최종 포지션
        """
        if self._sorted_order is None:
            self._sorted_order = self._topological_sort()

        current_regime = market_data.current_regime
        audit_log: List[str] = []
        node_results: Dict[str, NodeExecutionResult] = {}

        # ── Phase 0: 위상 정렬 순서대로 노드 실행 ──
        audit_log.append(f"[Phase 0] 위상 정렬 순서: {self._sorted_order}")

        for node_id in self._sorted_order:
            engine = self._engines[node_id]

            # 다른 노드 점수를 extra_context로 전달
            node_context: Dict[str, Any] = {}
            if extra_context:
                node_context.update(extra_context)

            for prev_id, prev_result in node_results.items():
                node_context[
                    self._engines[prev_id].node.Final_Output.Score_Variable
                ] = prev_result.normalized_score

            result = engine.execute(
                market_data=market_data,
                nlp_data=nlp_data,
                current_regime=current_regime,
                extra_context=node_context,
            )
            node_results[node_id] = result
            audit_log.append(
                f"  노드 {node_id} ({engine.node.Master}): "
                f"raw={result.raw_score:.4f}, norm={result.normalized_score:.4f}, "
                f"flag={result.active_state_flag}"
            )

        # ── Phase 0.5: Raw Tension Calculation (Sandbox Metric) ──
        # Calculate standard deviation of expert signals BEFORE any discounting or overrides
        expert_scores = [
            r.normalized_score for nid, r in node_results.items() 
            if nid.startswith(("VAL_", "GRO_", "MAC_"))
        ]
        raw_tension = 0.0
        if len(expert_scores) > 1:
            mean = sum(expert_scores) / len(expert_scores)
            var = sum((x - mean) ** 2 for x in expert_scores) / len(expert_scores)
            raw_tension = math.sqrt(var)
        
        audit_log.append(f"[Phase 0.5] Raw Philosophy Tension Score: {raw_tension:.4f}")

        # ── Phase 1: Priority Override / Master Influence — 리스크 감쇄 및 조정 ──
        # [v8.8] Soft-Shutdown: 하드 셧다운 대신 지수적 감쇄 적용
        # [v8.9.8] Variant Support: skip Phase 1 override entirely for controlled experiments
        skip_override = bool(extra_context.get("_variant_no_override", False)) if extra_context else False

        if skip_override:
            influence_active, influence_source, influence_factor = False, None, 1.0
            audit_log.append("[Phase 1] Override SKIPPED (_variant_no_override=True)")
        else:
            override_k = extra_context.get("_variant_override_k", None) if extra_context else None
            influence_active, influence_source, influence_factor = self._apply_master_influences(
                node_results, current_regime, audit_log, override_k=override_k
            )

        # [v8.9.6] Variant Support: HARD_OVERRIDE (Retro-design for ablation)
        is_hard_override = bool(extra_context.get("_variant_hard_override", False)) if extra_context else False

        if influence_active and (influence_factor < 0.15 or is_hard_override):
            if is_hard_override:
                # Traditional H-PIOS v8.5 behavior: Hard-Zero
                for nid in node_results:
                    if nid != influence_source:
                        node_results[nid] = node_results[nid].model_copy(
                            update={"normalized_score": 0.0}
                        )
                audit_log.append("[Phase 1] TRADITIONAL HARD-ZERO applied (Ablation Variant).")
                return OrchestratorOutput(
                    ticker=market_data.ticker,
                    node_results=node_results,
                    ensemble_signal=0.0,
                    final_position_size=0.0,
                    override_active=True,
                    override_source=influence_source,
                    active_regime=current_regime,
                    confidence=node_results.get("RSK_SHANNON_001").normalized_score if "RSK_SHANNON_001" in node_results else 1.0,
                    audit_log=audit_log,
                    tension_score=raw_tension
                )
            else:
                # [v8.9] Reduced Dictatorship: Soft-Shutdown with non-zero floor
                for nid in node_results:
                    if nid != influence_source:
                        new_val = max(0.01, node_results[nid].normalized_score * influence_factor)
                        node_results[nid] = node_results[nid].model_copy(
                            update={"normalized_score": new_val}
                        )
                audit_log.append(
                    f"[Phase 1] Extreme Risk Discount applied (Factor={influence_factor:.4f}) - Non-zero floor retained for deliberation."
                )
        elif influence_active:
            # Soft-Shutdown: 모든 노드 점수를 지수적으로 감쇄
            for nid in node_results:
                if nid != influence_source:
                    new_val = node_results[nid].normalized_score * influence_factor
                    node_results[nid] = node_results[nid].model_copy(
                        update={"normalized_score": new_val}
                    )
            audit_log.append(
                f"[Phase 1] Soft-Shutdown 적용 (Source={influence_source}, Factor={influence_factor:.4f})"
            )

        # ── Phase 2: Global Adjustment — 매크로 연속 가중치 조절 ──
        adjusted_scores = self._apply_global_adjustments(
            node_results, current_regime, audit_log
        )

        # ── Phase 3: Synergy/Suppress — 노드 간 상호작용 ──
        synergized_scores = self._apply_synergy_suppress(
            adjusted_scores, node_results, current_regime, audit_log
        )

        # ── Phase 4: 앙상블 & Position Sizing ──
        ensemble_signal, final_position = self._compute_ensemble_and_sizing(
            synergized_scores, node_results, audit_log
        )

        # ── Phase 5: Strategic Policy Governor (SPG) ──
        # [v8.5] Deterministic Guardrail for Hype/Toxic Asset Filtering
        # [v8.9.7] Variant Support: skip SPG entirely for controlled experiments
        skip_spg = bool(extra_context.get("_variant_no_spg", False)) if extra_context else False

        spg_veto = False
        spg_report: Dict[str, Any] = {}

        if skip_spg:
            audit_log.append("[Phase 5] SPG SKIPPED (_variant_no_spg=True)")
        else:
            spg = StrategicPolicyGovernor()
            spg_veto, spg_report = spg.evaluate(market_data, ensemble_signal, audit_log)

            if spg_veto:
                ensemble_signal = 0.0
                final_position = 0.0
                audit_log.append("[Phase 5] SPG VETO 발동 → 최종 시그널 및 비중 0.0 처리")
            elif spg_report.get("apply_floor"):
                floor_val = spg_report.get("strategic_floor", 0.0)
                ensemble_signal = max(ensemble_signal, floor_val)
                audit_log.append(f"[Phase 5] SPG Floor 적용 → Ensemble Signal {ensemble_signal:.4f}")

        # Shannon 신뢰도
        shannon_confidence = 1.0
        if "RSK_SHANNON_001" in node_results:
            shannon_confidence = node_results["RSK_SHANNON_001"].normalized_score

        # 글로벌 상태 갱신
        self.global_state.last_score = ensemble_signal
        self.global_state.previous_regime = current_regime
        self.global_state.last_updated = datetime.utcnow()

        # ── Phase 6: Final Return Preparation ──

        return OrchestratorOutput(
            ticker=market_data.ticker,
            node_results=node_results,
            ensemble_signal=ensemble_signal,
            final_position_size=final_position,
            override_active=False,
            spg_veto_active=spg_veto,
            spg_report=spg_report,
            active_regime=current_regime,
            confidence=shannon_confidence,
            tension_score=raw_tension,
            audit_log=audit_log,
        )

    # ── Phase 1: Priority Override ──

    def _apply_master_influences(
        self,
        results: Dict[str, NodeExecutionResult],
        regime: Optional[MarketRegime],
        audit_log: List[str],
        override_k: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """Master/Priority influence를 지수적 감쇄 함수로 처리합니다.

        하드코딩된 이분법적 셧다운 대신, 리스크 점수에 비례하여
        다른 노드들의 시그널을 지수적으로 감쇄(Exponential Decay)시킵니다.

        수식: factor = exp(-k * Ruin_Risk)
        - k=3.0 적용 시: Risk 0.5 -> factor 0.22, Risk 0.8 -> factor 0.09, Risk 1.0 -> factor 0.05

        Args:
            override_k: If provided, use this as the decay coefficient instead of default 3.5.

        Returns:
            (영향 활성 여부, 소스 노드 ID, 감쇄 인자)
        """
        k = override_k if override_k is not None else 3.5
        audit_log.append(f"[Phase 1] Master Influence (Soft-Shutdown) 평가 시작 (k={k})")

        # Taleb 직접 검사
        taleb_result = results.get("RSK_TALEB_001")
        if taleb_result:
            risk_score = taleb_result.normalized_score
            influence_factor = math.exp(-k * risk_score)
            
            if risk_score > 0.4: # 어느 정도 위험이 감지될 때부터 로그에 남김
                audit_log.append(
                    f"  ⚠ Taleb Master Influence: Risk={risk_score:.4f} → "
                    f"Decay Factor={influence_factor:.4f}"
                )
            
            # 위험이 매우 높으면(예: 1.2 이상) 실질적 셧다운으로 간주
            if risk_score > 1.5:
                return True, "RSK_TALEB_001", 0.0
                
            if risk_score > 0.1: # 유의미한 위험 존재 시
                return True, "RSK_TALEB_001", influence_factor

        return False, None, 1.0

    # ── Phase 2: Global Adjustment (연속 함수) ──

    def _apply_global_adjustments(
        self,
        results: Dict[str, NodeExecutionResult],
        regime: Optional[MarketRegime],
        audit_log: List[str],
    ) -> Dict[str, float]:
        """매크로 노드 점수를 기반으로 전역 가중치를 연속 함수로 조절합니다.

        하드코딩된 ``if > 0.8`` 이분법 대신, 점수에 비례하여
        가중치가 부드럽게 감쇠/증폭되는 연속 함수를 사용합니다.

        적용 함수:
            - growth_multiplier = linear_interp(dalio_score, [0.3, 0.8], [1.0, 0.5])
            - value_multiplier  = linear_interp(dalio_score, [0.3, 0.8], [1.0, 1.5])

        Args:
            results: 노드별 실행 결과
            regime: 현재 시장 국면
            audit_log: 감사 로그

        Returns:
            노드 ID → 조정된 점수 딕셔너리
        """
        audit_log.append("[Phase 2] Global Adjustment (연속 가중치) 적용 시작")
        adjusted: Dict[str, float] = {
            nid: r.normalized_score for nid, r in results.items()
        }

        for edge in self._edges:
            if edge.Relationship_Type != RelationshipType.GLOBAL_WEIGHT_ADJUSTER:
                continue

            source_result = results.get(edge.Source)
            if source_result is None:
                continue

            # 국면 조건 체크
            if edge.Condition_Regime and regime:
                if regime not in edge.Condition_Regime:
                    continue

            source_score = source_result.normalized_score
            targets = self._resolve_targets(
                edge.Target, set(self._engines.keys()), edge.Source
            )

            for target_id in targets:
                if target_id not in adjusted:
                    continue

                original = adjusted[target_id]

                # 연속 가중치 함수 적용
                multiplier = self._continuous_weight_function(
                    source_score, target_id, edge
                )
                adjusted[target_id] = max(0.0, min(1.0, original * multiplier))

                audit_log.append(
                    f"  {edge.Source}({source_score:.3f}) → {target_id}: "
                    f"multiplier={multiplier:.3f}, "
                    f"{original:.3f} → {adjusted[target_id]:.3f}"
                )

        return adjusted

    def _continuous_weight_function(
        self, source_score: float, target_id: str, edge: LogicalEdge
    ) -> float:
        """소스 점수를 기반으로 연속적 가중치 승수를 계산합니다.

        이분법적 ``if > threshold`` 대신 Linear Interpolation으로
        부드러운 감쇠/증폭을 구현합니다.

        성장(Growth) 노드 → 높은 매크로 리스크일수록 감쇠
        가치(Value) 노드 → 높은 매크로 리스크일수록 증폭

        Args:
            source_score: 소스 노드 정규화 점수 (0~1)
            target_id: 타겟 노드 ID
            edge: 논리 엣지

        Returns:
            가중치 승수 (0.3 ~ 2.0)
        """
        # 타겟이 성장 노드인지 가치 노드인지 판별
        if target_id.startswith("GRO_"):
            # Growth: Dalio 리스크 높을수록 감쇠
            # source_score 0.0→1.0, multiplier 1.0→0.3
            return self._linear_interpolation(
                source_score,
                x_range=(0.2, 0.9),
                y_range=(1.0, 0.3),
            )
        elif target_id.startswith("VAL_") and "Graham" in (
            self._engines.get(target_id, None) and
            self._engines[target_id].node.Master or ""
        ):
            # Graham Value: Dalio 리스크 높을수록 증폭 (역발상 기회)
            return self._linear_interpolation(
                source_score,
                x_range=(0.2, 0.9),
                y_range=(1.0, 1.8),
            )
        else:
            # 기타 노드: 약한 감쇠
            return self._exponential_decay(source_score, decay_rate=0.5)

    @staticmethod
    def _linear_interpolation(
        x: float,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (1.0, 0.5),
    ) -> float:
        """선형 보간으로 연속 가중치를 계산합니다.

        x가 x_range 범위 밖이면 클리핑합니다.

        Args:
            x: 입력값 (0~1)
            x_range: (x_min, x_max) 보간 범위
            y_range: (y_at_x_min, y_at_x_max) 출력 범위

        Returns:
            보간된 가중치
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        if x <= x_min:
            return y_min
        if x >= x_max:
            return y_max

        t = (x - x_min) / (x_max - x_min)
        return y_min + t * (y_max - y_min)

    @staticmethod
    def _exponential_decay(
        x: float, decay_rate: float = 0.5, base: float = 1.0
    ) -> float:
        """지수 감쇠로 연속 가중치를 계산합니다.

        수식: base * exp(-decay_rate * x)

        Args:
            x: 입력값 (0~1)
            decay_rate: 감쇠율
            base: 기저값

        Returns:
            감쇠된 가중치 (> 0)
        """
        return base * math.exp(-decay_rate * x)

    # ── Phase 3: Synergy / Suppress ──

    def _apply_synergy_suppress(
        self,
        adjusted_scores: Dict[str, float],
        results: Dict[str, NodeExecutionResult],
        regime: Optional[MarketRegime],
        audit_log: List[str],
    ) -> Dict[str, float]:
        """노드 간 시너지/억제 상호작용을 반영합니다.

        엣지의 Relationship_Type에 따라:
        - Synergize_With: 소스 + 타겟 모두 높을 때 증폭
        - Suppress: 소스가 높을 때 타겟 감쇠

        모든 상호작용은 연속 함수로 처리합니다.

        Args:
            adjusted_scores: Phase 2 이후 조정된 점수
            results: 원본 노드 결과 (상태 플래그 참조용)
            regime: 현재 시장 국면
            audit_log: 감사 로그

        Returns:
            시너지/억제 반영된 최종 점수 딕셔너리
        """
        audit_log.append("[Phase 3] Synergy/Suppress 상호작용 적용 시작")
        final_scores = dict(adjusted_scores)

        # 우선순위 기반 정렬 (높은 우선순위 먼저)
        interaction_edges = [
            e for e in self._edges
            if e.Relationship_Type in (
                RelationshipType.SYNERGIZE_WITH,
                RelationshipType.SUPPRESS,
                RelationshipType.CONTINUOUS_DISCOUNT,
                RelationshipType.CONTINUOUS_SYNERGY,
            )
        ]
        interaction_edges.sort(
            key=lambda e: self._OVERRIDE_PRIORITY.get(e.Relationship_Type, 0),
            reverse=True,
        )

        for edge in interaction_edges:
            source_score = final_scores.get(edge.Source)
            if source_score is None:
                continue

            # 국면 조건 체크
            if edge.Condition_Regime and regime:
                if regime not in edge.Condition_Regime:
                    continue

            targets = self._resolve_targets(
                edge.Target, set(self._engines.keys()), edge.Source
            )

            for target_id in targets:
                if target_id not in final_scores:
                    continue

                target_score = final_scores[target_id]

                if edge.Relationship_Type == RelationshipType.SYNERGIZE_WITH:
                    # 시너지: 소스 × 타겟의 기하평균에 비례하여 증폭
                    synergy_factor = self._compute_synergy_multiplier(
                        source_score, target_score
                    )
                    new_score = min(1.0, target_score * synergy_factor)
                    final_scores[target_id] = new_score
                    audit_log.append(
                        f"  시너지: {edge.Source}({source_score:.3f}) ↔ "
                        f"{target_id}({target_score:.3f}) → "
                        f"factor={synergy_factor:.3f}, new={new_score:.3f}"
                    )

                elif edge.Relationship_Type == RelationshipType.SUPPRESS:
                    # 억제: 소스 점수에 비례하여 타겟 감쇠
                    suppress_factor = self._compute_suppress_multiplier(
                        source_score, results.get(edge.Source)
                    )
                    new_score = max(0.0, target_score * suppress_factor)
                    final_scores[target_id] = new_score
                    audit_log.append(
                        f"  억제: {edge.Source}({source_score:.3f}) → "
                        f"{target_id}({target_score:.3f}) → "
                        f"factor={suppress_factor:.3f}, new={new_score:.3f}"
                    )

                elif edge.Relationship_Type in (RelationshipType.CONTINUOUS_DISCOUNT, RelationshipType.CONTINUOUS_SYNERGY):
                    # 연속적 가중치 전이: 소스 점수가 타겟에 직접 비례/반비례 영향
                    if edge.Relationship_Type == RelationshipType.CONTINUOUS_DISCOUNT:
                        # [Calibration] Sharper suppression: 0.05 -> 0.02
                        multiplier = source_score + 0.02
                    else:
                        # [Calibration] Stronger synergy: 0.5 -> 0.8 boost
                        multiplier = 1.0 + (source_score * 0.8)
                        
                    new_score = max(0.0, min(1.0, target_score * multiplier))
                    final_scores[target_id] = new_score
                    audit_log.append(
                        f"  연속전이: {edge.Source}({source_score:.3f}) → {target_id}: "
                        f"multiplier={multiplier:.3f}, new={new_score:.3f}"
                    )

        return final_scores

    @staticmethod
    def _compute_synergy_multiplier(
        source_score: float, target_score: float
    ) -> float:
        """시너지 승수를 계산합니다.

        두 노드 모두 높은 점수일 때 기하평균에 비례하여 증폭합니다.
        최대 1.5배까지 증폭 가능합니다.

        수식: 1.0 + 0.5 * sqrt(source * target)

        Args:
            source_score: 소스 노드 점수
            target_score: 타겟 노드 점수

        Returns:
            시너지 승수 (1.0 ~ 1.5)
        """
        geometric_mean = math.sqrt(max(0.0, source_score * target_score))
        # [v8.6] Synergy Boost Coefficient: 0.8 -> 0.95
        return 1.0 + 0.95 * geometric_mean

    @staticmethod
    def _compute_suppress_multiplier(
        source_score: float,
        source_result: Optional[NodeExecutionResult] = None,
    ) -> float:
        """억제 승수를 계산합니다.

        소스의 점수가 높을수록 타겟을 강하게 감쇠합니다.
        CRITICAL_INHIBIT 플래그가 있으면 완전 억제(0.0)합니다.

        수식: max(0.0, 1.0 - source_score)

        Args:
            source_score: 소스 노드 점수
            source_result: 소스 노드 실행 결과 (상태 플래그 참조)

        Returns:
            억제 승수 (0.0 ~ 1.0)
        """
        # CRITICAL_INHIBIT → 완전 억제
        if source_result and source_result.active_state_flag == "CRITICAL_INHIBIT":
            return 0.0

        # 연속 감쇠: 소스 점수 비례
        return max(0.0, 1.0 - source_score)

    # ── Phase 4: Ensemble & Position Sizing Pipeline ──

    def _compute_ensemble_and_sizing(
        self,
        final_scores: Dict[str, float],
        results: Dict[str, NodeExecutionResult],
        audit_log: List[str],
    ) -> Tuple[float, float]:
        """앙상블 시그널과 최종 포지션 사이즈를 산출합니다.

        앙상블:
            - Value, Growth, Macro 도메인 노드의 가중 평균
            - Thorp/Taleb/Shannon은 앙상블 대상에서 제외

        Position Sizing (Kelly Criterion):
            - 앙상블 시그널 × Shannon 신뢰도 → Thorp Kelly Fraction
            - Half-Kelly 방식으로 보수적 비중 산출

        Args:
            final_scores: Phase 3 이후 최종 점수
            results: 노드 실행 결과
            audit_log: 감사 로그

        Returns:
            (앙상블 시그널, 최종 포지션 사이즈)
        """
        audit_log.append("[Phase 4] 앙상블 & Position Sizing 파이프라인 시작")

        # ── 앙상블 (Risk 도메인 제외) ──
        ensemble_nodes = {
            nid: score for nid, score in final_scores.items()
            if not nid.startswith("RSK_")
        }

        if not ensemble_nodes:
            audit_log.append("  앙상블 대상 노드 없음 → 0.0")
            return 0.0, 0.0

        # [v8.6.2] Collective Intelligence: 전문가 확신도 기반 대칭 가중 앙상블
        # Calibration: Neutral Point shifted to 0.55 (A2_BALANCED Target)
        NEUTRAL_POINT = 0.55
        weighted_sum = 0.0
        total_confidence = 0.0
        
        for nid, score in ensemble_nodes.items():
            # [Refinement] 대칭 가중치: 하방 Expert(0.075)와 상방 Expert(0.925)가 
            # 중립점(0.18) 대비 동일한 발언권(Weight)을 갖도록 보정
            dist = abs(score - NEUTRAL_POINT)
            if score >= NEUTRAL_POINT:
                # 상방 가중치 스케일링 (0.18 ~ 1.0)
                relative_dist = dist / (1.0 - NEUTRAL_POINT + 1e-8)
            else:
                # 하방 가중치 스케일링 (0.0 ~ 0.18)
                relative_dist = dist / (NEUTRAL_POINT + 1e-8)
                
            # [Calibration] Exponent 2.8 + floor 0.05 for balanced transition
            confidence = (relative_dist ** 2.8) + 0.05
            weighted_sum += score * confidence
            total_confidence += confidence
            
        ensemble_signal = weighted_sum / total_confidence if total_confidence > 0 else 0.18
        ensemble_signal = max(0.0, min(1.0, ensemble_signal))
        
        audit_log.append(
            f"  [v8.6] symmetric_weighted_ensemble = {ensemble_signal:.4f} "
            f"(대상 {len(ensemble_nodes)}개 노드)"
        )

        # ── Shannon 신뢰도 필터 ──
        shannon_factor = 1.0
        shannon_result = results.get("RSK_SHANNON_001")
        if shannon_result:
            shannon_factor = shannon_result.normalized_score
            audit_log.append(f"  Shannon 신뢰도 필터 = {shannon_factor:.4f}")

        # ── Thorp Kelly Criterion Position Sizing ──
        final_position = self._kelly_position_sizing(
            ensemble_signal, shannon_factor, results, audit_log
        )

        return ensemble_signal, final_position

    def _kelly_position_sizing(
        self,
        ensemble_signal: float,
        shannon_factor: float,
        results: Dict[str, NodeExecutionResult],
        audit_log: List[str],
    ) -> float:
        """Kelly Criterion 기반 최종 포지션 비중을 산출합니다.

        수식: f* = (bp - q) / b
        - b: 손익비 (win/loss ratio)
        - p: 승률 (앙상블 시그널 기반)
        - q: 패율 (1 - p)

        Half-Kelly 적용으로 보수적 비중을 산출합니다.
        Shannon 신뢰도를 직접 승수로 적용합니다.

        Args:
            ensemble_signal: 앙상블 시그널 (0~1)
            shannon_factor: Shannon 신뢰도 (0.1~1.0)
            results: 노드 실행 결과
            audit_log: 감사 로그

        Returns:
            최종 포지션 비중 (0.0 ~ 1.0)
        """
        # Thorp 노드 결과에서 Edge Conviction Premium 반영
        thorp_result = results.get("RSK_THORP_001")
        edge_premium = 0.0
        if thorp_result:
            edge_premium = thorp_result.step2_output  # Edge_Conviction_Premium
            audit_log.append(
                f"  Thorp Edge Conviction Premium = {edge_premium:.4f}"
            )

        # 승률 추정: 앙상블 시그널을 기반으로
        estimated_win_rate = max(0.01, min(0.99, ensemble_signal))
        estimated_loss_rate = 1.0 - estimated_win_rate

        # 손익비 (기본 1.5:1, edge_premium으로 조정)
        win_loss_ratio = 1.5 * (1.0 + max(0.0, edge_premium))

        # Kelly Fraction: f* = (bp - q) / b
        kelly_fraction = (
            (win_loss_ratio * estimated_win_rate - estimated_loss_rate)
            / win_loss_ratio
        )

        # Half-Kelly (보수적)
        half_kelly = max(0.0, kelly_fraction * 0.5)

        # Shannon 신뢰도 적용
        final_size = half_kelly * shannon_factor

        # 상한 제한 (최대 100%)
        final_size = max(0.0, min(1.0, final_size))

        audit_log.append(
            f"  Kelly: p={estimated_win_rate:.3f}, b={win_loss_ratio:.3f}, "
            f"f*={kelly_fraction:.4f}, half_kelly={half_kelly:.4f}, "
            f"× shannon={shannon_factor:.3f} → final={final_size:.4f}"
        )

        return final_size

    # ── 유틸리티 ──

    def get_engine(self, node_id: str) -> Optional[AbstractMasterEngine]:
        """노드 ID로 엔진 인스턴스를 검색합니다.

        Args:
            node_id: 노드 식별자

        Returns:
            AbstractMasterEngine 인스턴스 또는 None
        """
        return self._engines.get(node_id)

    def update_node_win_rate(
        self, node_id: str, performance: PerformanceState
    ) -> float:
        """특정 노드의 롤링 승률을 외부 성과 데이터로 갱신합니다.

        Args:
            node_id: 대상 노드 ID
            performance: 성과 데이터

        Returns:
            갱신된 승률

        Raises:
            KeyError: 노드 ID가 존재하지 않을 때
        """
        engine = self._engines.get(node_id)
        if engine is None:
            raise KeyError(f"등록되지 않은 노드 ID: {node_id}")
        return engine.update_rolling_win_rate(performance)

    @property
    def registered_nodes(self) -> List[str]:
        """등록된 모든 노드 ID 목록."""
        return list(self._engines.keys())

    @property
    def execution_order(self) -> List[str]:
        """위상 정렬된 실행 순서."""
        return list(self._sorted_order) if self._sorted_order else []


# ══════════════════════════════════════════════════════════
# 5. Strategic Policy Governor (SPG) — 시스템 2 가드레일
# ══════════════════════════════════════════════════════════

class StrategicPolicyGovernor:
    """전략적 제어 및 거부권(Veto)을 행사하는 결정론적 가드레일 엔진.
    
    에이전트나 엔진의 '확신(Conviction)'과 별개로, 기본적 재무 건전성 및 
    상업적 상식에 근거하여 'Toxic' 주식을 필터링합니다.
    """
    
    def evaluate(
        self, 
        market_data: MarketDataPayload, 
        ensemble_signal: float, 
        audit_log: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """앙상블 시그널에 대해 최종 거부권을 행사합니다.
        
        Args:
            market_data: 재무 지표를 포함한 페이로드 (Debt, FCF, P/E 등)
            ensemble_signal: 엔진이 산출한 1차 시그널
            audit_log: 시스템 감사 로그
            
        Returns:
            (Veto 여부, 상세 판정 리포트)
        """
        audit_log.append("[Phase 5] Strategic Policy Governor (SPG) 판정 시작")
        
        metrics = market_data.metrics
        report: Dict[str, Any] = {
            "checks": {},
            "score": 0,
            "threshold": 3,
            "status": "PASS"
        }
        
        # 1. Debt Safety (부채 건전성)
        debt_to_equity = metrics.get("Debt_to_Equity", metrics.get("Debt/Equity", 0.0))
        debt_pass = debt_to_equity < 1.5
        report["checks"]["debt_safety"] = {"value": debt_to_equity, "pass": debt_pass}
        
        # 2. Cash Flow Health (현금흐름)
        fcf_yield = metrics.get("FCF_Yield", 0.0)
        cash_pass = fcf_yield > 0.01  # 최소 1% 이상의 FCF 수익률
        report["checks"]["cash_flow"] = {"fcf": fcf_yield, "pass": cash_pass}
        
        # 3. Valuation Sanity (밸류에이션 상한)
        pe_ratio = metrics.get("P/E_Ratio", 15.0)
        pe_pass = 0 < pe_ratio < 60
        report["checks"]["valuation"] = {"value": pe_ratio, "pass": pe_pass}
        
        # 4. Profitability (수익성)
        roic = metrics.get("ROIC_10yr_Avg", 0.0)
        roic_pass = roic > 0.08  # 최소 8% 이상의 ROIC
        report["checks"]["profitability"] = {"value": roic, "pass": roic_pass}
        
        # 5. Asset Quality (자산 가치)
        ncav_ratio = metrics.get("Price_to_NCAV", 1.0)
        ncav_pass = ncav_ratio < 15.0  # 극단적 거품 방지
        report["checks"]["asset_quality"] = {"value": ncav_ratio, "pass": ncav_pass}
        
        # 최종 점수 산출
        report["score"] = sum(1 for c in report["checks"].values() if c["pass"])
        
        # [NEW] Strategic Floor Logic: 우량주 비중 하방 방어
        # adj_floor = (strategy_base + contrarian_bonus) * regime_mult * quality_mult
        score = report["score"]
        strategy_base = 0.05
        # pe_ratio is already defined at line 1937
        contrarian_bonus = 0.02 if (pe_ratio < 10 and pe_ratio > 0) else 0.0
        quality_mult = 1.0 + (score / 5.0)  # 1.0 ~ 2.0
        
        # 국면 보정 (예시: 하락장에서는 우량주 비중 확대)
        regime_mult = 1.2 if market_data.current_regime == MarketRegime.BEAR_MARKET else 1.0
        
        strategic_floor = (strategy_base + contrarian_bonus) * regime_mult * quality_mult
        report["strategic_floor"] = strategic_floor
        
        # 거부권 행사 (Veto Logic)
        is_veto = False
        if ensemble_signal > 0.6 and score < 3:
            is_veto = True
            report["status"] = "VETO_BY_TOXIC_FUNDAMENTALS"
            audit_log.append(f"  ⚠ SPG VETO: Conviction={ensemble_signal:.3f}이나 Fundamentals({score}/5) 부실")
        elif ensemble_signal > 0.8 and score < 4:
            is_veto = True
            report["status"] = "VETO_BY_HIGH_HYPE_RISK"
            audit_log.append(f"  ⚠ SPG VETO: Extreme Hype({ensemble_signal:.3f}) 대비 재무 점수({score}/5) 부족")
        else:
            # Veto가 아닐 때, 품질이 극도로 좋으면(5/5) Floor 적용 검토
            if ensemble_signal < strategic_floor and score >= 5:
                report["apply_floor"] = True
                audit_log.append(f"  SPG Floor 적용 대상 탐지: {ensemble_signal:.3f} < Floor {strategic_floor:.3f}")
            else:
                report["apply_floor"] = False
            
            audit_log.append(f"  SPG PASS: Fundamental Score = {score}/5, Floor = {strategic_floor:.3f}")
            
        return is_veto, report
