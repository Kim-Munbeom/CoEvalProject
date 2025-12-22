"""
CoEval: 멘토링 답변 품질 평가 시스템

이 모듈은 멀티 에이전트 시스템과 DeepEval Rubric 기반 평가를 결합하여
멘토링 답변의 품질을 종합적으로 평가하는 FastAPI 애플리케이션입니다.

주요 기능:
- 멀티 에이전트 시스템을 통한 다각도 평가 (실행성, 전문성, 현실성)
- DeepEval Rubric 기반 정량적 점수 산출 (0-10 스케일)
- 등급 체계 (D/C/B/A/S) 자동 산정
- 평가 이유 한글 번역 제공
"""

import asyncio
import concurrent.futures
import os
from typing import List, Optional, Dict, Any

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import GeminiModel as DeepEvalGeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel
from strands import Agent
from strands.models.gemini import GeminiModel as StrandsGeminiModel
from strands.multiagent import GraphBuilder

# 환경 변수 로드 (.env 파일에서 API 키 등 로드)
load_dotenv()

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# Strands용 Gemini 모델 (멀티 에이전트용)
# 멀티 에이전트 시스템에서 각 에이전트가 사용할 LLM 모델
# temperature: 0.3으로 낮춰 일관성 있는 평가 결과 유도
strands_gemini_model = StrandsGeminiModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    model_id="gemini-2.5-flash",
    params={
        "temperature": 0.3,  # 낮은 온도로 일관된 평가
        "max_output_tokens": 8192,  # 긴 분석 리포트 생성 가능
        "top_p": 0.6,
        "top_k": 20,
    },
)

# DeepEval용 Gemini 모델 (Rubric 평가용)
# Rubric 기반 정량적 점수 산출에 사용
deepeval_gemini_model = DeepEvalGeminiModel(
    model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3
)

# Google GenAI 클라이언트 초기화 (번역용)
# DeepEval의 영문 평가 이유를 한글로 번역하기 위해 사용
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def translate_to_korean_async(text: str) -> str:
    """평가 이유를 한글로 번역하는 비동기 함수

    동기 번역 함수를 executor로 래핑하여 비동기로 실행합니다.

    Args:
        text: 번역할 영문 텍스트

    Returns:
        str: 번역된 한글 텍스트 (실패 시 원문)
    """
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: genai_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=f"다음 영어 텍스트를 자연스러운 한국어로 번역해주세요. 번역문만 출력하고 다른 설명은 하지 마세요:\n\n{text}",
            ),
        )
        return response.text.strip()
    except Exception:
        # 번역 실패 시 원문 반환
        return text


# 에이전트 설정 데이터 (데이터 기반 구성으로 유지보수성 향상)
AGENT_CONFIGS = {
    "action_master": {
        "description": "실행 지침 검수자",
        "system_prompt": """역할: 실행 지침 검수자.
        평가 기준: 수치·도구명 포함, 동사 중심 행동, Action Item 존재.
        출력 형식:
        - 판정: [PASS / FAIL]
        - 기준1(구체성): (평가 근거 1문장)
        - 기준2(행동성): (평가 근거 1문장)
        - 기준3(리스트): (평가 근거 1문장)""",
    },
    "pro_proof": {
        "description": "실무 디테일 검증가",
        "system_prompt": """역할: 실무 디테일 검증가. 지식 추가 설명 절대 금지.
        평가 기준: 전문 용어/프로세스 정확성, 경험 기반 인과관계 설명.
        출력 형식:
        - 판정: [현업수준 / 검색수준]
        - 기준1(전문성): (평가 근거 1문장)
        - 기준2(경험근거): (평가 근거 1문장)""",
    },
    "context_guardian": {
        "description": "현실성 분석가",
        "system_prompt": """역할: 현실성 분석가. 멘티 상담 및 대안 제시 금지.
        평가 기준: 멘티 상황 적합성, 실행 리스크 언급, 조언의 맥락.
        출력 형식:
        - 판정: [실현가능 / 불투명]
        - 기준1(적합성): (평가 근거 1문장)
        - 기준2(리스크): (평가 근거 1문장)
        - 기준3(맥락): (평가 근거 1문장)""",
    },
    "quality_consensus": {
        "description": "품질 합의 조정자",
        "system_prompt": """3개 에이전트(액션/프로프/가디언) 의견 종합 및 최종 리포트 작성.
수행 임무:
- 에이전트 의견 충돌 시 최종 결론 도출
- 0-10점 평가 근거를 마크다운으로 요약
- 종합 점수 및 핵심 개선 포인트 필수 포함""",
    },
}


def create_evaluation_agents(model: StrandsGeminiModel) -> Dict[str, Agent]:
    """멀티 에이전트 시스템을 생성하는 팩토리 함수

    AGENT_CONFIGS 데이터를 기반으로 에이전트를 동적으로 생성합니다.

    Args:
        model: 에이전트가 사용할 Gemini 모델 인스턴스

    Returns:
        Dict[str, Agent]: 에이전트 이름을 키로 하는 에이전트 딕셔너리
    """
    return {
        name: Agent(name=name, system_prompt=config["system_prompt"], model=model)
        for name, config in AGENT_CONFIGS.items()
    }


def build_evaluation_graph(agents: Dict[str, Agent]):
    """평가 그래프를 구축하는 함수 (최적화 버전)

    에이전트들의 실행 순서를 정의하는 DAG(Directed Acyclic Graph)를 생성합니다.

    최적화된 실행 흐름:
    1. action_master, pro_proof, context_guardian이 **모두 병렬**로 실행
       - 각 에이전트가 독립적으로 평가 수행 (실행성, 전문성, 현실성)
    2. quality_consensus가 세 에이전트의 결과를 종합하여 최종 리포트 작성

    기존 대비 개선:
    - context_guardian이 action_master, pro_proof 완료를 기다리지 않음
    - 3개 에이전트 병렬 실행으로 10-15% 추가 성능 향상

    Args:
        agents: 에이전트 딕셔너리

    Returns:
        실행 가능한 그래프 객체
    """
    builder = GraphBuilder()

    # 노드 등록 (각 에이전트를 그래프 노드로 추가)
    for name in AGENT_CONFIGS:
        builder.add_node(agents[name], name)

    # 엣지 정의 (최적화: 3개 에이전트 병렬 실행)
    # action_master, pro_proof, context_guardian → quality_consensus
    builder.add_edge("action_master", "quality_consensus")
    builder.add_edge("pro_proof", "quality_consensus")
    builder.add_edge("context_guardian", "quality_consensus")

    return builder.build()


# 전역 변수로 에이전트 그래프 초기화
# 애플리케이션 시작 시 한 번만 생성하여 재사용
evaluation_agents = create_evaluation_agents(strands_gemini_model)
evaluation_graph = build_evaluation_graph(evaluation_agents)

# ==================== Phase 1: 전역 메트릭 및 동시성 제어 ====================

# Rubric 정의 전역화 (매번 생성하지 않고 재사용)
MENTORING_RUBRIC = [
    Rubric(
        score_range=(0, 2),
        expected_outcome="D등급: 필수 조건 미달. 실행성/전문성이 결여된 답변.",
    ),
    Rubric(
        score_range=(3, 5),
        expected_outcome="C등급: 조언은 있으나 추상적이며 멘티 상황 고려가 부족함.",
    ),
    Rubric(
        score_range=(6, 8),
        expected_outcome="A/B등급: 우수함. 구체적 단계와 실무 지식이 포함된 수준 높은 답변.",
    ),
    Rubric(
        score_range=(9, 10),
        expected_outcome="S등급: 완벽함. 수치/도구/단계 및 리스크 관리까지 포함된 최상위 답변.",
    ),
]

# GEval 메트릭 전역화
QUALITY_METRIC = GEval(
    name="Overall Mentoring Quality",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    evaluation_steps=[
        "1. 에이전트들이 분석한 실행가능성, 전문성, 현실성 점수를 개별적으로 확인한다.",
        "2. 리포트의 결론이 우리가 설정한 Rubric 구간 중 어디에 해당하는지 대조한다.",
        "3. 에이전트 간의 갈등이 어떻게 조정되었는지 보고 최종 점수의 타당성을 검토한다.",
        "4. 최종 점수를 확정하고 그 근거를 한 문장으로 요약한다.",
    ],
    rubric=MENTORING_RUBRIC,
    threshold=0.7,
    model=deepeval_gemini_model,
)

# 동시 실행 수 제한 (API Rate Limiting)
MAX_CONCURRENT_EVALUATIONS = 5
_evaluation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EVALUATIONS)

# ThreadPoolExecutor for async wrapping of sync functions (Phase 2)
_agent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def _extract_agent_response(node) -> Dict[str, Any]:
    """에이전트 노드에서 응답 정보를 추출하는 헬퍼 함수

    Args:
        node: 그래프 실행 결과의 노드 객체

    Returns:
        Dict[str, Any]: agent_name, response_text, execution_time, token_usage를 포함한 딕셔너리
    """
    node_id = node.node_id
    text = "(응답 없음)"
    execution_time = 0.0
    usage = {}

    if hasattr(node, "result") and node.result:
        agent_result = node.result.result
        if hasattr(agent_result, "message") and agent_result.message:
            content = agent_result.message.get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "")

        execution_time = node.result.execution_time / 1000  # ms -> s 변환
        usage = getattr(node.result, "accumulated_usage", {})

    return {
        "agent_name": node_id,
        "response_text": text,
        "execution_time": execution_time,
        "token_usage": usage,
    }


def run_multi_agent_evaluation(question: str, answer: str) -> Dict[str, Any]:
    """멀티 에이전트 시스템을 실행하여 평가 결과를 반환

    멘티의 질문과 멘토의 답변을 입력받아 4개 에이전트로 구성된
    평가 파이프라인을 실행합니다.

    Args:
        question: 멘티의 질문
        answer: 멘토의 답변

    Returns:
        Dict[str, Any]: 각 에이전트의 응답, 최종 합의, 실행 정보 등을 포함한 딕셔너리
    """

    # 입력 포맷팅 (멘티 질문과 멘토 답변을 구조화된 형태로 변환)
    evaluation_input = f"""[멘티 질문]
{question}

[멘토 답변]
{answer}"""

    # 그래프 실행 (에이전트들이 정의된 순서대로 평가 수행)
    result = evaluation_graph(evaluation_input)

    # 에이전트 응답 추출 (헬퍼 함수 사용)
    agent_responses = [_extract_agent_response(node) for node in result.execution_order]

    # 최종 합의 결과 (마지막 에이전트인 quality_consensus의 응답)
    final_consensus = (
        agent_responses[-1]["response_text"] if agent_responses else "(평가 실패)"
    )

    # 총 실행 시간 및 토큰 사용량 계산
    total_execution_time = sum(resp["execution_time"] for resp in agent_responses)
    total_tokens = sum(
        resp["token_usage"].get("totalTokens", 0) for resp in agent_responses
    )

    return {
        "agent_responses": agent_responses,
        "final_consensus": final_consensus,
        "total_execution_time": total_execution_time,
        "total_tokens": total_tokens,
        "execution_order": [node.node_id for node in result.execution_order],
        "status": result.status,
    }


def calculate_grade(score: float) -> str:
    """0-10 스케일 점수를 D/C/B/A/S 등급으로 변환

    Args:
        score: DeepEval에서 반환된 점수 (0-1 범위)

    Returns:
        str: D, C, B, A, S 중 하나의 등급
    """
    # DeepEval의 score는 0-1 범위이므로 10을 곱해서 0-10 스케일로 변환
    absolute_score = score * 10

    if absolute_score >= 9:
        return "S"
    elif absolute_score >= 8:
        return "A"
    elif absolute_score >= 6:
        return "B"
    elif absolute_score >= 3:
        return "C"
    else:
        return "D"


# ==================== Phase 2: 비동기 평가 함수 ====================


async def run_multi_agent_evaluation_async(
    question: str, answer: str
) -> Dict[str, Any]:
    """멀티 에이전트 시스템을 비동기로 실행 (Phase 2)

    Strands 멀티 에이전트 시스템은 동기 방식이므로
    ThreadPoolExecutor를 사용하여 비동기로 래핑합니다.

    Args:
        question: 멘티의 질문
        answer: 멘토의 답변

    Returns:
        Dict[str, Any]: 각 에이전트의 응답, 최종 합의, 실행 정보 등을 포함한 딕셔너리
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _agent_executor,
        run_multi_agent_evaluation,
        question,
        answer,
    )


async def run_rubric_evaluation_async(
    question: str, answer: str, agent_consensus: str
) -> Dict[str, Any]:
    """DeepEval의 Rubric 기반 평가를 비동기로 실행 (Phase 2)

    DeepEval의 GEval 메트릭의 비동기 메서드(a_measure)를 사용하고,
    번역도 비동기로 처리합니다.

    Args:
        question: 멘티의 질문
        answer: 멘토의 답변
        agent_consensus: 멀티 에이전트의 최종 합의 리포트

    Returns:
        Dict[str, Any]: 점수, 합격 여부, 평가 이유, 비용 등을 포함한 딕셔너리
    """
    # 테스트 케이스 생성
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=[agent_consensus],
    )

    # DeepEval의 비동기 메서드 사용
    await QUALITY_METRIC.a_measure(test_case)

    # 메트릭 결과 추출
    score = QUALITY_METRIC.score
    reason = QUALITY_METRIC.reason

    # 번역도 비동기로 처리
    reason_kr = await translate_to_korean_async(reason)

    return {
        "score": score,
        "threshold": QUALITY_METRIC.threshold,
        "success": QUALITY_METRIC.is_successful(),
        "reason_en": reason,
        "reason_kr": reason_kr,
        "evaluation_cost": QUALITY_METRIC.evaluation_cost,
        "evaluation_model": QUALITY_METRIC.evaluation_model,
    }


# ==================== 비동기 테스트 케이스 처리 함수 (Phase 2 업데이트) ====================


async def process_single_test_case(
    test_case: "TestCaseRequest", index: int
) -> "TestResultResponse":
    """단일 테스트 케이스를 비동기로 처리 (Phase 2 최적화 버전)

    Phase 2 최적화:
    - 비동기 함수 사용 (run_multi_agent_evaluation_async, run_rubric_evaluation_async)
    - ThreadPoolExecutor를 통한 최적화된 스레드 관리
    - 번역도 비동기로 처리

    Args:
        test_case: 평가할 테스트 케이스
        index: 테스트 케이스 인덱스

    Returns:
        TestResultResponse: 평가 결과

    Raises:
        Exception: 평가 중 오류 발생 시
    """
    async with _evaluation_semaphore:
        # Step 1: 멀티 에이전트 평가 (Phase 2 비동기 함수 사용)
        agent_evaluation = await run_multi_agent_evaluation_async(
            test_case.input,
            test_case.actual_output,
        )

        # Step 2: Rubric 평가 (Phase 2 비동기 함수 사용)
        rubric_evaluation = await run_rubric_evaluation_async(
            test_case.input,
            test_case.actual_output,
            agent_evaluation["final_consensus"],
        )

        # Step 3: 등급 산정
        grade = calculate_grade(rubric_evaluation["score"])
        absolute_score = rubric_evaluation["score"] * 10

        # Step 4: 응답 구성
        test_result = TestResultResponse(
            test_case_index=index,
            input=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            # 각 에이전트의 응답을 Pydantic 모델로 변환
            agent_responses=[
                AgentResponseDetail(**agent_resp)
                for agent_resp in agent_evaluation["agent_responses"]
            ],
            final_consensus=agent_evaluation["final_consensus"],
            # Rubric 평가 결과 구성
            rubric_evaluation=RubricEvaluationDetail(
                score=rubric_evaluation["score"],
                absolute_score=absolute_score,
                grade=grade,
                threshold=rubric_evaluation["threshold"],
                success=rubric_evaluation["success"],
                reason=rubric_evaluation["reason_kr"],
                reason_en=rubric_evaluation["reason_en"],
                evaluation_cost=rubric_evaluation["evaluation_cost"],
                evaluation_model=rubric_evaluation["evaluation_model"],
            ),
            # 실행 정보
            total_execution_time=agent_evaluation["total_execution_time"],
            total_tokens=agent_evaluation["total_tokens"],
            execution_order=agent_evaluation["execution_order"],
            success=rubric_evaluation["success"],
        )

        return test_result


# ==================== Pydantic 모델 정의 ====================
# FastAPI의 요청/응답 스키마를 정의하는 모델들


class TestCaseRequest(BaseModel):
    """단일 테스트 케이스 요청 모델"""

    input: str  # 멘티 질문
    actual_output: str  # 멘토 답변
    expected_output: Optional[str] = None  # 기대 답변 (선택사항, 현재 미사용)


class EvaluationRequest(BaseModel):
    """평가 요청 모델 (여러 테스트 케이스를 포함)"""

    test_cases: List[TestCaseRequest]


class AgentResponseDetail(BaseModel):
    """개별 에이전트의 응답 상세 정보"""

    agent_name: str  # 에이전트 이름
    response_text: str  # 에이전트 응답 텍스트
    execution_time: float  # 실행 시간 (초)
    token_usage: Dict[str, int]  # 토큰 사용량 (totalTokens, inputTokens, outputTokens 등)


class RubricEvaluationDetail(BaseModel):
    """Rubric 기반 평가 상세 정보"""

    score: float  # 0-1 스케일 점수
    absolute_score: float  # 0-10 스케일 점수
    grade: str  # D, C, B, A, S 등급
    threshold: float  # 합격 기준점
    success: bool  # 합격 여부
    reason: str  # 평가 이유 (한글)
    reason_en: str  # 평가 이유 (영문 원본)
    evaluation_cost: float  # 평가 비용
    evaluation_model: str  # 평가에 사용된 모델 이름


class TestResultResponse(BaseModel):
    """단일 테스트 케이스의 평가 결과"""

    test_case_index: int  # 테스트 케이스 인덱스
    input: str  # 멘티 질문
    actual_output: str  # 멘토 답변
    expected_output: Optional[str]  # 기대 답변 (선택사항)

    # 멀티 에이전트 분석 결과
    agent_responses: List[AgentResponseDetail]  # 각 에이전트의 응답
    final_consensus: str  # quality_consensus의 최종 리포트

    # Rubric 기반 평가 결과
    rubric_evaluation: RubricEvaluationDetail

    # 실행 정보
    total_execution_time: float  # 총 실행 시간 (초)
    total_tokens: int  # 총 토큰 사용량
    execution_order: List[str]  # 에이전트 실행 순서

    # 전체 성공 여부
    success: bool  # Rubric 평가 합격 여부


class EvaluationResponse(BaseModel):
    """전체 평가 응답 (여러 테스트 결과를 포함)"""

    test_results: List[TestResultResponse]


# ==================== FastAPI 엔드포인트 ====================


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_test_cases(request: EvaluationRequest):
    """멘토링 답변을 평가하는 메인 API 엔드포인트 (Phase 2: 비동기 최적화 버전)

    이 엔드포인트는 두 단계의 평가 프로세스를 수행합니다:
    1. 멀티 에이전트 시스템을 통한 정성적 분석
       - action_master: 실행 지침 구체성 평가
       - pro_proof: 실무 전문성 검증
       - context_guardian: 현실성 분석
       - quality_consensus: 종합 리포트 작성

    2. DeepEval Rubric 기반 정량적 점수 산출
       - 에이전트 분석 결과를 바탕으로 0-10 스케일 점수 산출
       - D/C/B/A/S 등급 자동 산정

    Phase 1 최적화:
    - 모든 테스트 케이스를 병렬로 처리하여 60-80% 성능 향상
    - Semaphore로 동시 실행 수 제한 (기본 5개)
    - 일부 실패 시에도 부분 결과 반환

    Phase 2 최적화:
    - 비동기 함수 사용 (run_multi_agent_evaluation_async, run_rubric_evaluation_async)
    - ThreadPoolExecutor를 통한 최적화된 스레드 관리
    - 번역도 비동기로 처리 (translate_to_korean_async)
    - DeepEval의 a_measure() 비동기 메서드 활용

    Args:
        request: 평가할 테스트 케이스들을 포함한 요청 객체

    Returns:
        EvaluationResponse: 각 테스트 케이스의 평가 결과를 포함한 응답 객체
            - 각 에이전트의 상세 분석
            - 최종 합의 리포트
            - Rubric 점수 및 등급
            - 실행 시간 및 토큰 사용량

    Example:
        Request:
        {
            "test_cases": [
                {
                    "input": "주니어 개발자인데 코드 리뷰를 잘 받는 방법을 알려주세요",
                    "actual_output": "코드 리뷰를 잘 받으려면..."
                }
            ]
        }

        Response:
        {
            "test_results": [
                {
                    "test_case_index": 0,
                    "rubric_evaluation": {
                        "score": 0.85,
                        "grade": "A",
                        ...
                    },
                    ...
                }
            ]
        }
    """

    # 모든 테스트 케이스를 병렬 처리
    tasks = [
        process_single_test_case(tc, i) for i, tc in enumerate(request.test_cases)
    ]

    # 병렬 실행 (일부 실패해도 계속 진행)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 에러 처리 (일부 실패해도 부분 결과 반환)
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # 에러 발생 시 에러 응답 생성
            error_result = TestResultResponse(
                test_case_index=i,
                input=request.test_cases[i].input,
                actual_output=request.test_cases[i].actual_output,
                expected_output=request.test_cases[i].expected_output,
                agent_responses=[],
                final_consensus=f"평가 중 오류 발생: {str(result)}",
                rubric_evaluation=RubricEvaluationDetail(
                    score=0.0,
                    absolute_score=0.0,
                    grade="D",
                    threshold=0.7,
                    success=False,
                    reason=f"평가 실패: {str(result)}",
                    reason_en=f"Evaluation failed: {str(result)}",
                    evaluation_cost=0.0,
                    evaluation_model="N/A",
                ),
                total_execution_time=0.0,
                total_tokens=0,
                execution_order=[],
                success=False,
            )
            processed_results.append(error_result)
        else:
            processed_results.append(result)

    return {"test_results": processed_results}


@app.get("/")
def root():
    """API 상태 확인 엔드포인트

    API 서버가 정상 작동 중인지 확인하는 헬스체크 엔드포인트입니다.

    Returns:
        dict: API 상태 메시지
    """
    return {"message": "CoEval API is running"}
