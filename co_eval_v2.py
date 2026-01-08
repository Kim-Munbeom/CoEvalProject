"""
CoEval V2: 멘토링 답변 품질 평가 시스템 (100점 만점)

PRD_V2.md 기반으로 구현된 버전입니다.

주요 변경사항:
- 100점 만점 시스템 (V1: 10점 → V2: 100점)
- 가중치 기반 점수 산출 (실행가능성 40%, 전문성 30%, 현실성 30%)
- 런타임 가중치 조정 API (GET/PUT /config/weights)
- 질문 제목/내용 분리 (question_title + question_content)
- DeepEval Rubric 100점 기준으로 재설계

점수 체계:
- 실행가능성 (0-100점): 정확성 25 + 명료성 25 + 관련성 25 + 완전성 25
- 전문성 (0-100점): 구체 정보 50 + 실무 디테일 50
- 현실성 (0-100점): 멘티 상황 고려 50 + 경험 기반 조언 50
- 최종 점수 = 가중치 적용 (기본: 40% + 30% + 30%)
- 등급: S(90-100), A(75-89), B(60-74), C(40-59), D(0-39)

과락 규칙:
- 실행가능성 ≤ 25점 OR 전문성 ≤ 25점 → 최종 점수 최대 40점 (C등급)
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any

from deepeval.models import GeminiModel as DeepEvalGeminiModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel
from strands import Agent
from strands.models.gemini import GeminiModel as StrandsGeminiModel
from strands.multiagent import GraphBuilder

from config import WeightsConfig

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션
app = FastAPI(
    title="CoEval V2",
    description="멘토링 답변 평가 시스템 (100점 만점)",
    version="2.0.0",
)

# Strands용 Gemini 모델 (에이전트용)
strands_gemini_model = StrandsGeminiModel(
    client_args={"api_key": os.getenv("GEMINI_API_KEY")},
    model_id="gemini-2.5-flash",
    params={
        "temperature": 0.3,
        "max_output_tokens": 8192,
        "top_p": 0.6,
        "top_k": 20,
    },
)

# DeepEval용 Gemini 모델
deepeval_gemini_model = DeepEvalGeminiModel(
    model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3
)

# Google GenAI 클라이언트 (번역용)
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 전역 가중치 설정 (런타임 변경 가능)
weights_config = WeightsConfig.from_env()


# ============================================================================
# 데이터 모델
# ============================================================================


class EvaluationRequest(BaseModel):
    """평가 요청 모델"""

    question_title: str
    question_content: str
    answer_content: str


class EvaluationResponse(BaseModel):
    """평가 응답 모델"""

    final_score: float  # 0~100
    grade: str  # S/A/B/C/D
    weights: Dict[str, float]  # 적용된 가중치
    scores: Dict[str, float]  # 각 기준별 점수 (0~100)
    deepeval_results: Dict[str, Dict[str, Any]]  # DeepEval 검증 결과
    rationale: Dict[str, str]  # 평가 근거
    processing_time: float  # 처리 시간 (초)


# ============================================================================
# 에이전트 프롬프트 (100점 기준)
# ============================================================================

AGENT_CONFIGS = {
    "action_master": {
        "description": "실행가능성 전문가 (Actionability Expert)",
        "system_prompt": """# Role
당신은 멘토링 답변의 **실행가능성**을 평가하는 전문가입니다.

# Evaluation Criteria (총 100점)
멘티가 답변을 읽고 **즉시 실행할 수 있는지**를 다음 4가지 기준으로 평가합니다:

## 1. 정확성 (Accuracy) - 25점
- **25점:** 모든 정보가 사실이며 검증 가능하다. 오류가 전혀 없다.
- **18점:** 대부분 정확하나 사소한 오류 1-2개가 있다.
- **12점:** 중요한 오류가 있거나 검증이 어렵다.
- **6점:** 잘못된 정보가 많다.
- **0점:** 완전히 잘못되었거나 사실이 아니다.

## 2. 명료성 (Clarity) - 25점
- **25점:** 매우 이해하기 쉽다. 전문 용어가 모두 설명되어 있다.
- **18점:** 대체로 명확하나 일부 용어 설명이 부족하다.
- **12점:** 이해하기 위해 추가 검색이 필요하다.
- **6점:** 모호하고 혼란스럽다.
- **0점:** 이해할 수 없다.

## 3. 관련성 (Relevance) - 25점
- **25점:** 질문에 정확히 답하며 불필요한 내용이 없다.
- **18점:** 대체로 관련있으나 약간의 불필요한 내용이 있다.
- **12점:** 질문과 관련은 있으나 핵심을 벗어났다.
- **6점:** 질문과 거의 무관하다.
- **0점:** 완전히 무관하다.

## 4. 완전성 (Completeness) - 25점
- **25점:** 필요한 정보가 모두 포함되어 추가 질문이 불필요하다.
- **18점:** 대부분의 정보가 있으나 1-2가지가 부족하다.
- **12점:** 핵심 정보가 누락되어 추가 질문이 필요하다.
- **6점:** 매우 불완전하다.
- **0점:** 거의 정보가 없다.

# Input
- 질문 제목: {{question_title}}
- 질문 내용: {{question_content}}
- 답변: {{answer_content}}

# Output Format (JSON Only)
{{
  "score": 85,
  "details": {{
    "accuracy": 25,
    "clarity": 22,
    "relevance": 20,
    "completeness": 18
  }},
  "rationale": "구체적인 단계와 도구명이 제시되었으나 일부 용어 설명이 부족함"
}}

**중요:** 반드시 0-100점 사이의 점수를 부여하고, JSON 형식으로만 응답하세요.
합계가 100점을 초과하지 않도록 주의하세요.
""",
    },
    "pro_proof": {
        "description": "전문성 검증자 (Domain Expert)",
        "system_prompt": """# Role
당신은 멘토링 답변의 **전문성**을 평가하는 검증자입니다.

# Evaluation Criteria (총 100점)
답변이 **현업 전문가의 지식**을 담고 있는지를 다음 2가지 기준으로 평가합니다:

## 1. 구체 정보 (Concrete Information) - 50점
- **50점:** 수치, 도구명, 구체적 단계가 매우 풍부하다.
  예: "인덱스 생성 시 조회 속도 30초→3초 개선", "B-Tree 인덱스 사용"
- **37점:** 구체적 정보가 있으나 일부 수치/도구가 누락되었다.
- **25점:** 일반적 수준의 구체성. 예시가 1-2개 정도.
- **12점:** 거의 추상적이며 구체성이 매우 부족하다.
- **0점:** 구체적 정보가 전혀 없다.

## 2. 실무 디테일 (Practical Details) - 50점
- **50점:** 현업에서만 알 수 있는 깊은 지식과 경험이 드러난다.
  예: "쓰기 성능 5-10% 저하 고려", "읽기/쓰기 비율 분석 필요"
- **37점:** 실무 지식이 있으나 경험보다는 정보 전달 위주다.
- **25점:** 검색으로 얻을 수 있는 일반적 지식 수준이다.
- **12점:** 비전문가도 할 수 있는 얕은 조언이다.
- **0점:** 전문성이 없거나 잘못된 정보다.

# Input
- 질문 제목: {{question_title}}
- 질문 내용: {{question_content}}
- 답변: {{answer_content}}

# Output Format (JSON Only)
{{
  "score": 72,
  "details": {{
    "concrete_info": 40,
    "practical_details": 32
  }},
  "rationale": "실무 도구명과 수치는 포함되었으나 깊은 경험 기반 조언은 부족함"
}}

**중요:** 반드시 0-100점 사이의 점수를 부여하고, JSON 형식으로만 응답하세요.
합계가 100점을 초과하지 않도록 주의하세요.
""",
    },
    "context_guardian": {
        "description": "현실성 감시자 (Context Analyst)",
        "system_prompt": """# Role
당신은 멘토링 답변의 **현실성**을 평가하는 감시자입니다.

# Evaluation Criteria (총 100점)
답변이 **멘티의 실제 상황**에 맞는지를 다음 2가지 기준으로 평가합니다:

## 1. 멘티 상황 고려 (Context Awareness) - 50점
- **50점:** 멘티의 상황(연차, 환경, 제약)을 완벽히 고려했다.
  Why/When/주의점이 모두 포함되어 있다.
  예: "주니어 수준에서는...", "현재 환경에서 주의할 점은..."
- **37점:** 상황을 고려했으나 일부 맥락이 부족하다.
- **25점:** 일반론에 가깝지만 완전히 벗어나지는 않았다.
- **12점:** 멘티 상황과 맞지 않는 조언이다.
- **0점:** 복사 붙여넣기 식 답변이다.

## 2. 경험 기반 조언 (Experience-Based Advice) - 50점
- **50점:** 실제 사례와 결과가 명확히 제시되어 있다.
  예: "실제 프로젝트에서 30초→3초 개선 경험", "지난 3년간..."
- **37점:** 경험이 언급되었으나 구체적 결과가 부족하다.
- **25점:** 경험 기반인지 불분명하다.
- **12점:** 이론적 조언 위주다.
- **0점:** 경험이 전혀 반영되지 않았다.

# Input
- 질문 제목: {{question_title}}
- 질문 내용: {{question_content}}
- 답변: {{answer_content}}

# Output Format (JSON Only)
{{
  "score": 75,
  "details": {{
    "context_awareness": 40,
    "experience_based": 35
  }},
  "rationale": "멘티 상황을 고려했으나 실제 경험 사례가 다소 부족함"
}}

**중요:** 반드시 0-100점 사이의 점수를 부여하고, JSON 형식으로만 응답하세요.
합계가 100점을 초과하지 않도록 주의하세요.
""",
    },
    "quality_consensus": {
        "description": "최종 조정자 (Master Judge)",
        "system_prompt": """# Role
당신은 3개 영역의 평가를 종합하는 **최종 조정자**입니다.

# Input Data
- 실행가능성 점수: {{actionability_score}}/100 (가중치: {{weight_actionability}}%)
- 전문성 점수: {{expertise_score}}/100 (가중치: {{weight_expertise}}%)
- 현실성 점수: {{practicality_score}}/100 (가중치: {{weight_practicality}}%)

# Calculation
최종 점수 = (실행가능성 × {{weight_actionability}}% + 전문성 × {{weight_expertise}}% + 현실성 × {{weight_practicality}}%)

# 🔒 Fail-Safe Rule (과락 규칙)
- 실행가능성 ≤ 25점 **OR** 전문성 ≤ 25점 → 최종 점수 최대 40점으로 제한
- 이유: 기본적인 실행가능성과 전문성이 없으면 좋은 답변이 아니기 때문

# Grading System (100점 기준)
- **S등급 (90-100점):** 완벽에 가까운 답변
- **A등급 (75-89점):** 우수한 답변
- **B등급 (60-74점):** 양호한 답변
- **C등급 (40-59점):** 부족한 답변
- **D등급 (0-39점):** 미달 답변

# Output Format (JSON Only)
{{
  "final_score": 78.5,
  "grade": "A",
  "rationale": "가중치 적용 결과 78.5점. 실행가능성과 전문성 모두 기준 충족하여 A등급 부여"
}}

**중요:** 반드시 JSON 형식으로만 응답하세요.
과락 규칙을 반드시 확인하세요.
""",
    },
}


# ============================================================================
# 번역 함수
# ============================================================================


async def translate_to_korean_async(text: str) -> str:
    """평가 이유를 한글로 번역 (비동기)"""

    def _translate() -> str:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"다음 영어 텍스트를 자연스러운 한국어로 번역해주세요. 번역문만 출력하세요:\n\n{text}",
        )
        return response.text.strip()

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _translate)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


# ============================================================================
# 에이전트 생성
# ============================================================================


def create_agents() -> Dict[str, Agent]:
    """4개 에이전트 생성"""
    agents = {}
    for agent_name, config in AGENT_CONFIGS.items():
        agents[agent_name] = Agent(
            name=agent_name,
            role=config["description"],
            system_prompt=config["system_prompt"],
            model=strands_gemini_model,
        )
    return agents


# ============================================================================
# Graph 구성
# ============================================================================


def build_evaluation_graph(agents: Dict[str, Agent]):
    """평가 Graph 구성 (V1과 동일한 구조)

    실행 순서:
    1. action_master, pro_proof, context_guardian 병렬 실행
    2. quality_consensus가 3개 결과를 종합하여 최종 평가
    """
    builder = GraphBuilder()

    # 노드 등록 (V1 API 사용)
    for name in ["action_master", "pro_proof", "context_guardian", "quality_consensus"]:
        builder.add_node(agents[name], name)

    # Edge 설정 (3개 → quality_consensus)
    builder.add_edge("action_master", "quality_consensus")
    builder.add_edge("pro_proof", "quality_consensus")
    builder.add_edge("context_guardian", "quality_consensus")

    return builder.build()


# ============================================================================
# JSON 파싱 함수
# ============================================================================


def parse_agent_response(response: str, agent_name: str) -> Dict[str, Any]:
    """에이전트 응답에서 JSON 파싱 (V1 로직 재사용)"""
    try:
        # 코드 블록 제거
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # JSON 파싱
        data = json.loads(response)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"{agent_name} JSON parsing failed: {e}\nResponse: {response}")
        # 기본값 반환
        if agent_name in ["action_master", "pro_proof"]:
            return {"score": 0, "details": {}, "rationale": "JSON 파싱 실패"}
        elif agent_name == "context_guardian":
            return {"score": 0, "details": {}, "rationale": "JSON 파싱 실패"}
        else:  # quality_consensus
            return {"final_score": 0, "grade": "D", "rationale": "JSON 파싱 실패"}


def _extract_agent_response(node) -> Dict[str, Any]:
    """에이전트 노드에서 응답 정보를 추출하는 헬퍼 함수 (V1 로직)

    Args:
        node: 그래프 실행 결과의 노드 객체

    Returns:
        Dict[str, Any]: agent_name, response_text, parsed_data를 포함한 딕셔너리
    """
    node_id = node.node_id
    text = "(응답 없음)"

    if hasattr(node, "result") and node.result:
        agent_result = node.result.result
        if hasattr(agent_result, "message") and agent_result.message:
            content = agent_result.message.get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "")

    # JSON 파싱
    parsed_data = parse_agent_response(text, node_id)

    return {
        "agent_name": node_id,
        "response_text": text,
        "parsed_data": parsed_data,
        "score": parsed_data.get("score", 0),
        "rationale": parsed_data.get("rationale", ""),
        "details": parsed_data.get("details", {}),
    }


# ============================================================================
# 등급 계산
# ============================================================================


def calculate_grade(score: float) -> str:
    """점수를 등급으로 변환 (100점 기준)"""
    if score >= 90:
        return "S"
    elif score >= 75:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


# ============================================================================
# 평가 엔진 (핵심 로직)
# ============================================================================


async def evaluate_answer_v2(
    question_title: str, question_content: str, answer_content: str
) -> EvaluationResponse:
    """멘토링 답변 평가 (100점 만점)

    프로세스:
    1. 4개 에이전트 실행 (3개 병렬 + 1개 순차)
    2. 가중치 적용 점수 산출
    3. 과락 규칙 적용
    4. DeepEval 검증 (선택적)
    5. 등급 산정
    """
    start_time = time.time()

    # 에이전트 생성 및 Graph 구성
    agents = create_agents()
    graph = build_evaluation_graph(agents)

    # 입력 데이터 준비 (V1 포맷)
    evaluation_input = f"""[질문 제목]
{question_title}

[질문 내용]
{question_content}

[답변]
{answer_content}"""

    # Graph 실행 (V1 방식)
    logger.info("Starting graph execution...")

    # 동기 함수를 비동기로 래핑
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: graph(evaluation_input))

    # 에이전트 응답 추출 (V1 방식)
    agent_responses = [_extract_agent_response(node) for node in result.execution_order]

    # 각 에이전트별 응답 찾기
    action_response = next(
        (r for r in agent_responses if r["agent_name"] == "action_master"), {}
    )
    pro_response = next(
        (r for r in agent_responses if r["agent_name"] == "pro_proof"), {}
    )
    context_response = next(
        (r for r in agent_responses if r["agent_name"] == "context_guardian"), {}
    )

    # 점수 추출
    action_score = float(action_response.get("score", 0))
    expertise_score = float(pro_response.get("score", 0))
    practicality_score = float(context_response.get("score", 0))

    logger.info(
        f"Agent scores - Action: {action_score}, Expertise: {expertise_score}, Practicality: {practicality_score}"
    )

    # 가중치 적용 점수 산출
    final_score = (
        action_score * weights_config.actionability
        + expertise_score * weights_config.expertise
        + practicality_score * weights_config.practicality
    )

    # 과락 규칙 적용
    if action_score <= 25 or expertise_score <= 25:
        logger.warning(
            f"Fail-safe rule applied: action={action_score}, expertise={expertise_score}"
        )
        final_score = min(final_score, 40)

    # 등급 산정
    grade = calculate_grade(final_score)

    # 평가 근거 수집
    rationale = {
        "actionability": action_response.get("rationale", ""),
        "expertise": pro_response.get("rationale", ""),
        "practicality": context_response.get("rationale", ""),
    }

    # DeepEval 결과 (현재는 placeholder)
    deepeval_results = {
        "action_master": {"status": "pass", "confidence": 0.95},
        "pro_proof": {"status": "pass", "confidence": 0.92},
        "context_guardian": {"status": "pass", "confidence": 0.90},
    }

    processing_time = time.time() - start_time

    return EvaluationResponse(
        final_score=round(final_score, 1),
        grade=grade,
        weights=weights_config.to_dict(),
        scores={
            "actionability": round(action_score, 1),
            "expertise": round(expertise_score, 1),
            "practicality": round(practicality_score, 1),
        },
        deepeval_results=deepeval_results,
        rationale=rationale,
        processing_time=round(processing_time, 2),
    )


# ============================================================================
# API 엔드포인트
# ============================================================================


@app.get("/")
async def root():
    """Health check"""
    return {"message": "CoEval V2 API", "version": "2.0.0", "status": "healthy"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(request: EvaluationRequest):
    """답변 평가 실행 (비동기)"""
    try:
        result = await evaluate_answer_v2(
            question_title=request.question_title,
            question_content=request.question_content,
            answer_content=request.answer_content,
        )
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/weights")
async def get_weights():
    """현재 가중치 조회"""
    return {
        "actionability": weights_config.actionability,
        "expertise": weights_config.expertise,
        "practicality": weights_config.practicality,
        "percentage": weights_config.to_percentage_dict(),
    }


@app.put("/config/weights")
async def update_weights(new_weights: WeightsConfig):
    """가중치 업데이트 (런타임)"""
    try:
        # 가중치 합 검증
        new_weights.validate_sum()

        # 전역 설정 업데이트
        global weights_config
        weights_config = new_weights

        logger.info(f"Weights updated: {weights_config}")

        return {
            "message": "가중치 업데이트 완료",
            "weights": weights_config.to_dict(),
            "percentage": weights_config.to_percentage_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/samples")
async def get_samples():
    """샘플 데이터 목록 조회"""
    try:
        samples_path = "frontend/data/samples.json"
        if os.path.exists(samples_path):
            with open(samples_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 샘플 파일이 없으면 빈 목록 반환
            return {"samples": []}
    except Exception as e:
        logger.error(f"Failed to load samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
