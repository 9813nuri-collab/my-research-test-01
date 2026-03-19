"""
H-PIOS v8.5 — Brain Firmware Generator
========================================
8개 지식원천 파일(models.py, engine_core.py, optimizer.py,
DATA_WEIGHTS_optimized.json, 4개 DATA_JSON_*_master.json)을 하나의 통합 인지 커널로
증류(distill)하여, 에이전트의 System Prompt에 주입할 수 있는
'뇌 펌웨어' 텍스트를 생성합니다.

설계 원칙
---------
- 단순 RAG가 아닌, 지식의 '내재화' — 에이전트가 12인 거장의 사고방식을 체화
- 8개 파일의 유기적 관계를 보존 — 시냅스(Logical_Edge), 위계(4-Phase),
  학습 원칙(Philosophy Inertia)이 모두 포함
- Gemini 100만 토큰 컨텍스트 대비 <1% 사용으로 공간 제약 없음
- 동적 생성 — 소스 파일 변경 시 자동 반영
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("H-PIOS.brain_firmware")

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ══════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════

def _load_json(filename: str) -> dict:
    """프로젝트 루트에서 JSON 파일을 로드합니다."""
    path = _PROJECT_ROOT / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
# LAYER 0: 인지 문법 (models.py 기반)
# ══════════════════════════════════════════════════════════

def _build_layer0() -> str:
    """models.py에서 추출한 인지 문법 레이어."""

    # MarketRegime을 직접 import하여 동적으로 추출
    try:
        from CORE_MODELS_models import MarketRegime, RelationshipType
        regimes = [f"  - {r.value}" for r in MarketRegime]
        relationships = [f"  - {r.value}: " for r in RelationshipType]
    except ImportError:
        regimes = ["  (models.py import 실패 — MarketRegime 목록 생략)"]
        relationships = ["  (models.py import 실패 — RelationshipType 목록 생략)"]

    regime_list = "\n".join(regimes)
    rel_list = "\n".join(relationships)

    return f"""━━━ LAYER 0: 인지 문법 (Cognitive Grammar) ━━━
출처: models.py — 이 뇌가 세상을 인식하고 분류하는 모든 '개념'과 '규칙'

[0-1] 세계 인식 범주 (MarketRegime — 28개)
이 뇌가 인식할 수 있는 모든 시장 국면:
{regime_list}

[0-2] 사고 간 관계의 문법 (RelationshipType — 5가지)
12인 거장의 판단이 서로 연결되는 방식:
  - Override: 소스의 판단이 타겟의 시그널을 완전히 무시/대체
  - Synergize_With: 소스와 타겟이 모두 높을 때 기하평균으로 상호 증폭
  - Suppress: 소스가 높을 때 타겟을 연속 함수로 감쇠
  - Global_Weight_Adjuster: 소스가 전체 도메인의 가중치를 연속 곡선으로 조절
  - Master_Override: 최고 우선순위 — 전체 시스템을 셧다운 (Taleb 전용)

[0-3] 사고의 구조 (3-Step Intelligence Pipeline)
모든 거장 노드는 반드시 이 3단계를 거쳐 판단합니다:
  Step 1 (정량 분석): 수치 지표를 수식(logic_formula)으로 평가 → 조건 충족 시 가중치 합산
  Step 2 (정성 맥락): NLP 키워드/코사인유사도로 시나리오 발동 → score_modifier 적용
    ※ CRITICAL_INHIBIT: 치명적 위험 감지 시 즉시 0점 강제 (Munger 사기탐지, Simons 분식회계)
  Step 3 (통계 보정): Step1 + Step2를 formula로 결합 × historical_win_rate × decay_factor
    ※ 국면별 regime_performance로 승률 보정 (Bear_Market에서 Graham 승률 × 0.75 등)

[0-4] 감각 입력 포맷
  · MarketDataPayload: 정량 데이터 (ticker, timestamp, metrics 딕셔너리, current_regime, regime_confidence)
  · NLPContextPayload: 정성 데이터 (raw_texts, detected_keywords, semantic_similarity_scores, sentiment_scores)

[0-5] 상태 기억
  · EngineState: 각 노드의 롤링 변동성, 시그널 지속 카운터, 직전 국면, 롤링 승률, 최근 100개 시그널 이력
  · PerformanceState: 백테스트/라이브 거래 성과 (총 거래수, 승률, 샤프비율, 최대낙폭, 칼마비율)
  · 동적 승률 갱신: update_rolling_win_rate()로 JSON 초기값을 실시간 교체 가능"""


# ══════════════════════════════════════════════════════════
# LAYER 1: 전문 지식 영역 (4개 Master JSON 기반)
# ══════════════════════════════════════════════════════════

def _format_node(node: dict) -> str:
    """단일 마스터 노드를 텍스트로 포맷합니다."""
    intel = node["Intelligence_Structure"]

    # Step 1: 정량 지표
    metrics_lines = []
    for m in intel["Step_1_Quantitative_Analysis"]["metrics"]:
        formula_str = f' → {m["logic_formula"]}' if m.get("logic_formula") else ""
        unit_str = f' ({m["unit"]})' if m.get("unit") else ""
        metrics_lines.append(
            f'      · {m["metric"]} {m["operator"]} {m["threshold"]}{unit_str} '
            f'[w={m["weight"]}]{formula_str}'
        )
    metrics_text = "\n".join(metrics_lines)

    # Step 2: 정성 시나리오
    scenarios_lines = []
    for s in intel["Step_2_Qualitative_Context"]["scenarios"]:
        action_str = f' ⚠ {s["action"]}' if s.get("action") else ""
        kw_str = ", ".join(s.get("keywords", []))
        scenarios_lines.append(
            f'      · [{s["score_modifier"]:+.1f}]{action_str} {s["condition"]}\n'
            f'        키워드: [{kw_str}]\n'
            f'        맥락: {s["context"]}'
        )
    scenarios_text = "\n".join(scenarios_lines)

    # Step 3: 보정 수식
    step3 = intel["Step_3_Statistical_Correction"]
    constants = step3["constants"]
    regime_perf = constants.get("regime_performance", {})
    regime_str = ", ".join(f'{k}={v}' for k, v in regime_perf.items()) if regime_perf else "없음"

    # Final Output
    final = node["Final_Output"]
    flags_str = " | ".join(final["State_Flags"])

    return f"""
    [{node["Node_ID"]}] {node["Master"]} — {node["Core_Concept"]}
    "{node["Full_Description"]}"

    Step 1 정량 (→ {intel["Step_1_Quantitative_Analysis"]["output_variable"]}):
{metrics_text}

    Step 2 정성 (NLP ≥ {intel["Step_2_Qualitative_Context"]["nlp_confidence_threshold"]} → {intel["Step_2_Qualitative_Context"]["output_variable"]}):
{scenarios_text}

    Step 3 보정: {step3["formula"]}
      win_rate={constants.get("historical_win_rate", "N/A")}, decay={constants.get("decay_factor", "N/A")}
      국면 보정: {{{regime_str}}}

    출력: {final["Score_Variable"]} ({final["Signal_Type"]})
    상태: [{flags_str}]  범위: {final["Range"]}"""


def _format_edges(edges: list) -> str:
    """Logical Edges를 텍스트로 포맷합니다."""
    if not edges:
        return "    (없음)"
    lines = []
    for e in edges:
        regimes = ", ".join(e.get("Condition_Regime", []))
        lines.append(
            f'    [{e["Edge_ID"]}] {e["Source"]} →({e["Relationship_Type"]})→ {e["Target"]}\n'
            f'      국면: [{regimes}]\n'
            f'      로직: {e["Logic"]}'
        )
    return "\n".join(lines)


def _build_layer1() -> str:
    """4개 Master JSON에서 추출한 전문 지식 레이어."""

    json_files = [
        ("DATA_JSON_value_master.json",
         "🟦 가치 투자 (Value Investment)",
         "전두엽 배내측 — 보수적 의사결정, 안전마진, 실수 방지"),
        ("DATA_JSON_growth_master.json",
         "🟩 성장 투자 (Growth Investment)",
         "도파민 보상 회로 — 혁신 탐지, 성장 기대, 거품 경보"),
        ("DATA_JSON_macro_master.json",
         "🟨 매크로 투자 (Macro Investment)",
         "시상하부+편도체 — 경제 사이클 판별, 군중 심리, 패턴 인식"),
        ("DATA_JSON_risk_master.json",
         "🟥 리스크 & 사이징 (Risk & Sizing)",
         "뇌간+소뇌 — 생존 본능, 신호 필터링, 최적 자본 배분"),
    ]

    sections = []
    total_edges = 0

    for fname, title, brain_role in json_files:
        try:
            data = _load_json(fname)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            sections.append(f"\n  [{fname}] 로드 실패: {e}")
            continue

        nodes_text = "\n".join(_format_node(n) for n in data.get("Nodes", []))
        edges = data.get("Logical_Edges", [])
        edges_text = _format_edges(edges)
        total_edges += len(edges)

        sections.append(f"""
  ── {title} ──
  뇌 역할: {brain_role}
  도메인: {data.get("Engine_Domain", "N/A")}
{nodes_text}

  시냅스 (Logical Edges):
{edges_text}""")

    body = "\n".join(sections)

    return f"""━━━ LAYER 1: 전문 지식 영역 (12인 거장의 투자 철학) ━━━
출처: 4개 DATA_JSON_*_master.json — 각 거장의 지표, 시나리오, 보정 수식, 상호작용 규칙
{body}

  ═══ 시냅스 총계: {total_edges}개 Logical Edges ═══

  ★ 도메인 간 핵심 상호작용 요약:
  1. Taleb(RSK) → ALL_OTHER_ENGINES: Master_Override — 꼬리 위험 시 전체 셧다운
  2. Dalio(MAC) → ALL_GROWTH_AND_VALUE_NODES: Global_Weight_Adjuster — 긴축기 성장↓ 가치↑
  3. Shannon(RSK) → Thorp(RSK): Suppress — 노이즈 높으면 베팅 축소
  4. Munger(VAL) → Graham(VAL): Override — 정상/강세장에서 밸류트랩 방지
  5. Munger(VAL) → Buffett(VAL): Suppress — 사기 리스크 시 해자 점수 무력화
  6. Graham(VAL) + Buffett(VAL): Synergize — 시장 폭락 시 딥밸류 시너지 1.5x
  7. Fisher(GRO) + Lynch(GRO): Synergize — 혁신 사이클에서 성장 증폭 1.3x
  8. Soros(GRO) → Lynch(GRO): Suppress — 후기 강세장 버블 감지 시 70% 억제
  9. Marks(MAC) + Simons(MAC): Synergize — 패닉 매도 시 기술적 반등 극대화"""


# ══════════════════════════════════════════════════════════
# LAYER 2: 집행 신경계 (engine_core.py 기반)
# ══════════════════════════════════════════════════════════

def _build_layer2() -> str:
    """engine_core.py에서 추출한 집행 로직 레이어."""

    return """━━━ LAYER 2: 집행 신경계 (Signal Resolution Hierarchy) ━━━
출처: engine_core.py — 12인 거장의 사고를 하나의 판단으로 통합하는 실행 메커니즘

[2-1] SafeFormulaEvaluator — 시냅스 전달 메커니즘
  · AST 기반 안전 수식 평가기 (Python eval 대체)
  · IF-THEN-ELSE → Python 삼항 전처리, 변수명 정규화 (P/E_Ratio → P_E_Ratio)
  · 허용 연산: 산술(+,-,*,/,**,%), 비교(<,>,==), 논리(and,or,not)
  · 허용 함수: MAX, MIN, ABS, ROUND, LOG, EXP, SQRT
  · 최대 AST 깊이: 50 (무한 재귀 방어)

[2-2] ScoreNormalizer — 지각 정규화
  · SigmoidNormalizer (기본): f(x) = 1/(1+e^(-5*(x-0.5)))  — 중심 0.5, 급격도 5
  · ZScoreNormalizer: Z-Score → Sigmoid 2단계 (동적 평균/표준편차 지원)
  · MinMaxNormalizer: 이력 최소/최대 기반 선형 [0,1] 클리핑

[2-3] AbstractMasterEngine — 개별 뉴런 발화
  · 각 MasterNode에 대해 3-Step 파이프라인 실행:
    Step 1: logic_formula를 SafeEval로 평가 → 조건 충족 metric의 weight 합산
    Step 2: NLP 키워드/코사인유사도로 시나리오 발동 → score_modifier 적용
      ※ CRITICAL_INHIBIT 발동 → 즉시 0점 할당, Step 3 건너뜀
    Step 3: formula를 SafeEval로 평가 (Step1+Step2 결합 × win_rate × decay)
      ※ 국면별 regime_performance 승률 보정 자동 적용
  · 정규화: Sigmoid로 0.0~1.0 매핑
  · 상태 플래그: normalized_score를 State_Flags 구간에 매핑 (높은 점수 → 첫번째 플래그)
  · 상태 갱신: signal_history(최근 100개), signal_persistence(연속 동일 방향 카운터)
  · 동적 승률: PerformanceState 주입으로 JSON 초기 win_rate를 실시간 교체 가능

[2-4] GraphOrchestrator — 전두엽 통합 판단 (4단계 위계)

  Phase 0: 위상 정렬 (Topological Sort — Kahn 알고리즘)
    · 12개 노드의 안전한 실행 순서 보장 (Source가 Target보다 먼저 실행)
    · 와일드카드 확장: ALL_OTHER_ENGINES → 소스 제외 전체
    · ALL_GROWTH_AND_VALUE_NODES → GRO_* + VAL_* 노드
    · 순환 참조 감지 시 예외 발생
    · 실행 시 이전 노드 점수를 extra_context로 전달 (노드 간 수식 참조 지원)

  Phase 1: Priority Override (뇌간 반사 — 생존 최우선)
    우선순위 체계:
      Master_Override = 100 (Taleb 블랙스완)
      Override = 80 (Munger 사기/회계 위험)
      Global_Weight_Adjuster = 60 (Dalio 매크로)
      Suppress = 40 (일반 억제)
      Synergize_With = 20 (시너지)
    · Taleb 직접 검사: normalized_score > 0.75 → 즉시 전체 셧다운
    · 엣지 기반 검사: Override/Master_Override 소스 점수 > 0.7 → 최고 우선순위 발동
    · 발동 결과: 오버라이드 소스 제외 모든 시그널 = 0.0, position_size = 0.0

  Phase 2: Global Adjustment (호르몬 체계 — 연속 함수, 이분법 아님)
    · Global_Weight_Adjuster 엣지 처리 (주로 Dalio → 성장/가치 전체)
    · Growth 노드: linear_interp(dalio, [0.2,0.9], [1.0,0.3]) — 리스크↑ 성장↓
    · Graham Value: linear_interp(dalio, [0.2,0.9], [1.0,1.8]) — 리스크↑ 가치↑ (역발상)
    · 기타 노드: exponential_decay(source, rate=0.5)
    · 결과: [0.0, 1.0] 클리핑

  Phase 3: Synergy/Suppress (시냅스 가소성 — 연속 함수)
    · Synergize_With: multiplier = 1.0 + 0.5 × √(source × target) → 최대 1.5x 증폭
    · Suppress: multiplier = max(0.0, 1.0 - source_score) → 점수 비례 감쇠
    · CRITICAL_INHIBIT 플래그 감지 → multiplier = 0.0 (완전 억제)
    · 우선순위 기반 정렬 후 순차 적용

  Phase 4: Ensemble & Position Sizing (최종 운동 명령)
    앙상블:
      · Risk 도메인(RSK_*) 제외, Value+Growth+Macro 노드의 균등 가중 평균
      · 결과: ensemble_signal ∈ [0.0, 1.0]
    Shannon 신뢰도 필터:
      · Shannon_Confidence_Factor를 직접 곱셈 (노이즈 높으면 전체 축소)
    Kelly Criterion Position Sizing:
      · 승률 추정: p = clip(ensemble_signal, 0.01, 0.99)
      · 손익비: b = 1.5 × (1 + Thorp_Edge_Conviction_Premium)
      · Kelly: f* = (bp - q) / b
      · Half-Kelly: f = f* × 0.5 (보수적 적용)
      · 최종: final_size = half_kelly × shannon_factor → [0.0, 1.0]"""


# ══════════════════════════════════════════════════════════
# LAYER 3: 학습 메커니즘 (optimizer.py 기반)
# ══════════════════════════════════════════════════════════

def _build_layer3() -> str:
    """optimizer.py에서 추출한 학습 방법론 레이어."""

    return """━━━ LAYER 3: 학습 메커니즘 (Philosophy Inertia Optimization) ━━━
출처: optimizer.py — 과거 경험에서 배우되, 핵심 철학을 절대 포기하지 않는 학습 방법론

[3-1] 데이터 수집 (2019-01-01 ~ 현재)
  · 13개 티커: SP500, KOSPI, NASDAQ100, OIL, GOLD, USDKRW, DXY, TNX, IRX, VIX, HYG, IEF
  · 15개 파생 시계열: yield_spread, credit_spread_proxy, SP500_MA200_ratio,
    SP500_MA20_zscore, NDX_12m_ret, NDX_60d_ret, VIX_5d_roc, VIX_20d_roc,
    HYG_IEF_zscore, SP500_ret_252d, SP500_volatility_20d, SP500_r_squared_60d

[3-2] 역사적 국면 라벨링 (Event Regime Mapping)
  · Black_Swan: 2020-02-19 ~ 2020-04-09 (COVID 팬데믹 패닉)
  · Liquidity_Bubble: 2020-05 ~ 2021-11 (양적완화 유동성 랠리)
  · Deleveraging: 2022-01 ~ 2022-10 (연준 급격한 금리 인상)
  · Systemic_Risk: 2023-03 ~ 2023-04 (SVB 뱅크런 사태)
  · AI_Supercycle: 2023-05 ~ 2025-12 (생성형 AI 혁명 랠리)
  · Geopolitical_Stagflation: 2026-01 ~ 현재 (지정학적 긴장 + 스태그플레이션)

[3-3] Synthetic Fundamental Proxies (감각 교차 변환 — 핵심 혁신)
  엔진은 마이크로(개별 종목) 지표를 기대하지만, 데이터는 매크로(인덱스) 뿐인 패러독스를 해결:
  · Value: 200MA 이격도 + 신용 스프레드 + VIX공포 → Graham NCAV/PE, Buffett ROIC, Munger D/E·FCF
  · Growth: NASDAQ 모멘텀 + VIX 역수 → Fisher R&D/마진, Lynch PEG/EPS, Soros 모멘텀 백분위
  · Macro: Yield Spread + VIX + 신용 Z-Score → Dalio/Marks/Simons 직접 매핑
  · Risk: VIX 가속도 + 추세 R² → Taleb SKEW/OTM풋, Shannon R²/노이즈, Thorp 기대값/승률
  각 프록시는 해당 거장 JSON의 임계값(threshold) 범위에서 작동하도록 금융 수학적으로 교정됨

[3-4] Historical Simulator — 역사 재현
  · GraphOrchestrator를 1871 영업일 매일 실행 → "만약 이 뇌가 그때 있었다면?"
  · 시점별 기록: ensemble_signal, position_size, override_active, 노드별 점수/상태
  · 미래 수익률: T+5, T+20, T+60 호라이즌별 S&P500 전방 수익률 계산

[3-5] Attribution Tracker — 성찰과 자기 평가
  · 각 거장의 시그널이 실제 미래 수익과 얼마나 일치했는지 추적
  · 노드 유형별 최적 호라이즌:
    - Quant/Risk(Simons,Taleb,Shannon): T+5 (단기)
    - Growth/Macro(Fisher,Lynch,Soros,Marks,Dalio): T+20 (중기)
    - Value/Risk(Buffett,Graham,Munger,Thorp): T+60 (장기)
  · 승률 = (시그널 방향 == 수익률 방향인 횟수) / 전체 횟수
  · 기여도 = Σ (signal - 0.5) × forward_return

[3-6] Philosophy Inertia Optimizer — 가치 보존 학습 (핵심 알고리즘)
  손실 함수: Loss = Σ(αₕ × Errorₕ) + λ_eff × |current - original| / |original|

  호라이즌 가중치 (αₕ):
    · Value:  T+5=0.10, T+20=0.20, T+60=0.70 (장기 중심)
    · Growth: T+5=0.15, T+20=0.60, T+60=0.25 (중기 중심)
    · Macro:  T+5=0.20, T+20=0.55, T+60=0.25 (중기 중심)
    · Quant:  T+5=0.70, T+20=0.20, T+60=0.10 (단기 중심)
    · Risk:   T+5=0.40, T+20=0.35, T+60=0.25 (균형)

  핵심 제약 조건:
    · ±20% 바운드: 원본 파라미터의 ±20% 이내에서만 조정 (정체성 유지)
    · Philosophy Inertia λ=0.3: 원본에서 멀어질수록 페널티 증가 → 급격한 성격 변화 방지
    · Taleb Protection: RSK_TALEB_001, MAC_MARKS_001에 λ×10 = 3.0 관성
      → 위기 국면(Black_Swan, Systemic_Risk) 외에서는 사실상 파라미터 동결
      → "Taleb의 파산 방어 능력과 Marks의 역발상 감각은 평상시 낮은 승률 때문에
         약화되면 안 된다. 위기가 올 때 반드시 작동해야 한다."
    · Regime-Weighted Loss: Black_Swan=3.0x, Systemic_Risk=2.0x, Deleveraging=1.5x
      → 극단적 경험에서 더 깊이 학습
    · Learning Rate: 0.005 (매우 보수적)
    · Early Stopping: 20회 유예(min_epochs) 후 3회 연속 미개선 시 조기 종료 (과학습 방지)
    · Gradient Direction: 해당 호라이즌 승률 - 0.5 × 국면 가중 보정"""


# ══════════════════════════════════════════════════════════
# LAYER 4: 장기 기억 (DATA_WEIGHTS_optimized.json 기반)
# ══════════════════════════════════════════════════════════

def _build_layer4() -> str:
    """DATA_WEIGHTS_optimized.json에서 추출한 장기 기억 레이어."""

    try:
        weights = _load_json("DATA_WEIGHTS_optimized.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"━━━ LAYER 4: 장기 기억 ━━━\n(DATA_WEIGHTS_optimized.json 로드 실패: {e})"

    nodes = weights.get("nodes", {})
    attribution = weights.get("attribution", {})
    global_params = weights.get("global_parameters", {})

    # 노드별 파라미터 요약
    param_lines = []
    for name, params in nodes.items():
        wr = params.get("historical_win_rate", {})
        df = params.get("decay_factor", {})
        protection = wr.get("protection", "")
        prot_str = f" 🛡️[{protection}]" if protection else ""
        param_lines.append(
            f'    {name:>8s}: win_rate {wr.get("original",0):.2f} → {wr.get("optimized",0):.4f} '
            f'({wr.get("change_pct",0):+.2f}%) [{wr.get("status","?")}] | '
            f'decay {df.get("original",0):.2f} → {df.get("optimized",0):.4f} '
            f'({df.get("change_pct",0):+.2f}%) [{df.get("status","?")}]{prot_str}'
        )
    params_text = "\n".join(param_lines)

    # 기여도 요약 (best_fit_regime 중심)
    attr_lines = []
    for name, attr in attribution.items():
        primary_h = attr.get("primary_horizon", 20)
        wr_key = f"T{primary_h}_win_rate"
        wr_val = attr.get(wr_key, 0)
        best_regime = attr.get("best_fit_regime", "N/A")
        best_wr = attr.get("best_fit_win_rate", 0)
        ntype = attr.get("node_type", "?")
        attr_lines.append(
            f'    {name:>8s} [{ntype:>6s}] T+{primary_h:>2d} 승률={wr_val*100:5.1f}% | '
            f'최적 무대: {best_regime} ({best_wr*100:.1f}%)'
        )
    attr_text = "\n".join(attr_lines)

    # 국면 분포
    regime_dist = global_params.get("regime_distribution", {})
    total_days = sum(regime_dist.values()) if regime_dist else 1
    dist_lines = []
    for regime, count in sorted(regime_dist.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_days * 100
        dist_lines.append(f'    {regime:>30s}: {count:>4d}일 ({pct:5.1f}%)')
    dist_text = "\n".join(dist_lines)

    data_range = global_params.get("data_range", {})
    opt_config = global_params.get("optimizer_config", {})

    return f"""━━━ LAYER 4: 장기 기억 (Learned Parameters from History) ━━━
출처: DATA_WEIGHTS_optimized.json — {data_range.get("n_trading_days", 0)}영업일의 역사적 학습에서 굳어진 전문성

최적화 일자: {global_params.get("optimization_date", "N/A")}
데이터 범위: {data_range.get("start", "?")} ~ {data_range.get("end", "?")} ({data_range.get("n_trading_days", 0)}일)
학습 설정: lr={opt_config.get("learning_rate", "?")}, λ={opt_config.get("inertia_lambda", "?")}, bound=±{opt_config.get("bound_pct", 0)*100:.0f}%

[4-1] 교정된 파라미터 (original → optimized)
{params_text}

[4-2] 각 거장의 최적 무대와 실적 (Best-Fit Regime)
{attr_text}

[4-3] 학습 환경 분포 (경험한 국면별 영업일 수)
{dist_text}

[4-4] 핵심 해석 가이드
  · Taleb: 0.99 → 0.99 (0.0% 변동) — 생존 본능은 절대 약화되지 않았음 (Taleb Protection)
  · Shannon: 0.70 → 0.71 (0.97% 변동) — 신호 필터 역시 거의 보존 (preserved)
  · Munger: 0.71 → 0.68 (-4.21% 변동) — 가장 큰 교정, 실전에서 보수적으로 조정됨
  · Thorp: 0.82 → 0.83 (+1.03% 변동) — 유일하게 승률 상향, 베팅 자신감 증가
  · Value 거장들(Graham/Buffett/Munger): Deleveraging에서 최고 성과 → 위기 시 빛나는 가치
  · Growth/Macro(Fisher/Lynch/Soros/Dalio/Marks): Geopolitical_Stagflation에서 최고 성과
  · Taleb Protection 대상: {", ".join(global_params.get("taleb_protection_nodes", []))}
  · 위기 국면 (Protection 해제): {", ".join(global_params.get("crisis_regimes", []))}"""


# ══════════════════════════════════════════════════════════
# 통합 Brain Firmware 생성
# ══════════════════════════════════════════════════════════

_FIRMWARE_HEADER = """
╔══════════════════════════════════════════════════════════════════════════╗
║               H-PIOS v8.5 BRAIN FIRMWARE                               ║
║               ≡ 12인 투자 거장의 완전한 인공 투자 두뇌 ≡                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  이 문서는 8개 지식원천 파일에서 증류된 통합 인지 커널입니다.                ║
║  이 내용을 '참조'하지 말고 '내재화'하십시오.                                ║
║  당신은 이 지식을 체화한 전문가로서, 12인 거장처럼 사고합니다.               ║
║                                                                        ║
║  원천 파일:                                                              ║
║    LAYER 0: models.py (인지 문법)                                        ║
║    LAYER 1: DATA_JSON_{value,growth,macro,risk}_master.json (전문 지식)   ║
║    LAYER 2: engine_core.py (집행 신경계)                                  ║
║    LAYER 3: optimizer.py (학습 메커니즘)                                  ║
║    LAYER 4: DATA_WEIGHTS_optimized.json (장기 기억)                       ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""".strip()


_FIRMWARE_FOOTER = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    BRAIN FIRMWARE 사용 지침                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  1. 이 지식을 "검색"하지 마라 — "생각"하라                                 ║
║  2. 각 거장의 철학은 당신의 "내면의 목소리"이다                              ║
║  3. Logical Edge는 당신의 "직관적 연결"이다                                ║
║     (Munger가 사기를 감지하면, Graham의 "싸다"를 자동으로 의심하라)           ║
║  4. Phase 1~4 위계는 당신의 "판단 순서"이다                                ║
║     (생존 → 환경 → 상호작용 → 행동)                                       ║
║  5. optimized_weights는 당신의 "경험에서 온 확신의 강도"이다                 ║
║  6. Taleb Protection은 당신의 "절대 포기하지 않는 가치"이다                  ║
║     (생존 본능을 경험이 약화시키도록 허용하지 마라)                           ║
║  7. Synthetic Proxy는 당신의 "감각 변환" 능력이다                           ║
║     (보이는 것[매크로]에서 보이지 않는 것[마이크로]을 읽어내라)                ║
║                                                                        ║
║  ⚡ 계산 결과 우선 규칙 (COMPUTATION SUPREMACY)                           ║
║  8. 이 Firmware의 서술은 사고의 '방향'일 뿐이다                             ║
║     최종 수치 판단은 실제 데이터와 엔진 산출값에 종속된다                      ║
║  9. Firmware 내용만으로 점수를 추정하거나 포지션을 결정하지 마라                ║
║     → 반드시 "이 지표가 이 임계값을 초과했으므로" 형태의 근거 제시             ║
║ 10. 엔진이 산출한 수치가 Firmware 서술과 충돌하면 엔진이 우선한다             ║
║     (당신은 엔진을 '해석'하는 것이지, '대체'하는 것이 아니다)                 ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""".strip()


def generate_brain_firmware() -> str:
    """8개 지식원천 파일에서 완전한 Brain Firmware를 생성합니다.

    Returns:
        통합 Brain Firmware 텍스트 (System Prompt 주입용)
    """
    logger.info("Brain Firmware 생성 시작...")

    sections = [
        _FIRMWARE_HEADER,
        "",
        _build_layer0(),
        "",
        _build_layer1(),
        "",
        _build_layer2(),
        "",
        _build_layer3(),
        "",
        _build_layer4(),
        "",
        _FIRMWARE_FOOTER,
    ]

    firmware = "\n\n".join(sections)

    # 통계
    char_count = len(firmware)
    estimated_tokens = char_count // 4  # 대략 4자 = 1토큰
    line_count = firmware.count("\n") + 1

    logger.info(
        "Brain Firmware 생성 완료: %d 문자, %d 줄 ≈ %d 토큰 "
        "(Gemini 100만 토큰의 %.3f%%)",
        char_count, line_count, estimated_tokens,
        estimated_tokens / 1_000_000 * 100,
    )

    return firmware


# ── 캐시 ──
_FIRMWARE_CACHE: Optional[str] = None


def get_brain_firmware() -> str:
    """Brain Firmware를 캐시하여 반환합니다.

    최초 호출 시 생성하고, 이후 호출에서는 캐시된 값을 반환합니다.

    Returns:
        통합 Brain Firmware 텍스트
    """
    global _FIRMWARE_CACHE
    if _FIRMWARE_CACHE is None:
        _FIRMWARE_CACHE = generate_brain_firmware()
    return _FIRMWARE_CACHE


def clear_firmware_cache() -> None:
    """Brain Firmware 캐시를 초기화합니다.

    소스 파일 변경 후 재생성이 필요할 때 호출합니다.
    """
    global _FIRMWARE_CACHE
    _FIRMWARE_CACHE = None
    logger.info("Brain Firmware 캐시 초기화됨")


# ══════════════════════════════════════════════════════════
# CLI Entry Point — 단독 실행 시 Firmware 출력/저장
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

    firmware = generate_brain_firmware()

    # --save 플래그: 파일로 저장
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        output_path = _PROJECT_ROOT / "brain_firmware_output.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(firmware)
        logger.info("Brain Firmware 파일 저장: %s", output_path)
    else:
        print(firmware)

    # 통계 출력
    char_count = len(firmware)
    line_count = firmware.count("\n") + 1
    estimated_tokens = char_count // 4
    print(f"\n{'='*60}")
    print(f"  총 문자수: {char_count:,}")
    print(f"  총 줄수:   {line_count:,}")
    print(f"  추정 토큰: {estimated_tokens:,}")
    print(f"  Gemini 100만 토큰 대비: {estimated_tokens / 1_000_000 * 100:.3f}%")
    print(f"{'='*60}")
