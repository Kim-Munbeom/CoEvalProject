import os
from typing import List, Optional, Dict, Any

from deepeval.evaluate import evaluate
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

load_dotenv()

app = FastAPI()

# Strands용 Gemini 모델 (멀티 에이전트용)
strands_gemini_model = StrandsGeminiModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    model_id="gemini-2.5-flash",
    params={
        "temperature": 0.3,
        "max_output_tokens": 8192,
        "top_p": 0.6,
        "top_k": 20,
    },
)

# DeepEval용 Gemini 모델 (Rubric 평가용)
deepeval_gemini_model = DeepEvalGeminiModel(
    model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3
)

# Google GenAI 클라이언트 초기화 (번역용)
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def translate_to_korean(text: str) -> str:
    """평가 이유를 한글로 번역하는 함수"""
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"다음 영어 텍스트를 자연스러운 한국어로 번역해주세요. 번역문만 출력하고 다른 설명은 하지 마세요:\n\n{text}",
        )
        return response.text.strip()
    except Exception:
        # 번역 실패 시 원문 반환
        return text


def create_evaluation_agents(model: StrandsGeminiModel) -> Dict[str, Agent]:
    """멀티 에이전트 시스템을 생성하는 팩토리 함수"""

    action_master = Agent(
        name="action_master",
        system_prompt="""당신은 IT/직무 멘토링의 **'실행 지침 검수자'**입니다. 답변을 읽고 멘티가 **"내일 아침 출근해서 무엇을 해야 할지"**가 명확한지 평가하십시오.
[평가 루브릭]
1. 수치(%, 시간), 예시(Case), 도구명(Notion, Jira 등), 구체적 단계(Step 1, 2)가 포함되었는가?
2. '노력하세요', '열심히 하세요'와 같은 추상적 표현이 아닌 '동사' 중심의 행동 지침이 있는가?
3. 당신은 답변에서 구체적인 Action Item을 추출하여 리스트업해야 합니다.""",
        model=model,
    )

    pro_proof = Agent(
        name="pro_proof",
        system_prompt="""당신은 10년 차 이상의 **'실무 디테일 검증가'**입니다. 답변이 단순히 검색으로 알 수 있는 상식인지, 아니면 현업의 냄새가 나는 진짜 지식인지 판별하십시오.
[평가 루브릭]
1. 직무/업계 특유의 프로세스나 전문 용어가 정확하게 사용되었는가?
2. 실제 경험(Experience)이나 사례(Evidence)를 근거로 왜 그렇게 해야 하는지 이유를 설명하는가?""",
        model=model,
    )

    context_guardian = Agent(
        name="context_guardian",
        system_prompt="""당신은 멘티의 상황을 대변하는 **'현실성 분석가'**입니다. 조언이 아무리 훌륭해도 멘티의 현재 상황(연차, 환경)에서 실현 불가능한지를 찾아내십시오.
[평가 루브릭]
1. 질문에 담긴 멘티의 수준(주니어/시니어)과 처한 상황을 충분히 고려했는가?
2. 조언을 실행할 때 주의해야 할 리스크(Exception)나 조건(When)을 언급했는가?
3. 무조건적인 정답이 아니라 '왜(Why)' 이 상황에 이 조언이 최선인지 맥락을 설명하는가?""",
        model=model,
    )

    quality_consensus = Agent(
        name="quality_consensus",
        system_prompt="""당신은 앞선 세 에이전트(액션 마스터, 프로프, 가디언)의 의견을 종합하여 최종 멘토링 품질 리포트를 작성하는 조정자입니다.
[수행 임무]
- 세 에이전트 간에 의견이 충돌할 경우(예: 전문성은 높으나 실행이 너무 어려움), 이를 보완할 수 있는 최종 결론을 도출하십시오.
- 모든 분석 내용을 취합하여 DeepEval이 0~10점 사이의 점수를 매길 수 있도록 각 항목별 점수 근거를 마크다운 형식으로 요약하십시오.
- 최종 출력물에는 반드시 '종합 점수'와 '핵심 개선 포인트'를 포함해야 합니다.""",
        model=model,
    )

    return {
        "action_master": action_master,
        "pro_proof": pro_proof,
        "context_guardian": context_guardian,
        "quality_consensus": quality_consensus,
    }


def build_evaluation_graph(agents: Dict[str, Agent]):
    """평가 그래프를 구축하는 함수"""
    builder = GraphBuilder()

    # 노드 등록
    builder.add_node(agents["action_master"], "action_master")
    builder.add_node(agents["pro_proof"], "pro_proof")
    builder.add_node(agents["context_guardian"], "context_guardian")
    builder.add_node(agents["quality_consensus"], "quality_consensus")

    # 엣지 정의 (실행 순서)
    # 1차: action_master와 pro_proof 병렬 실행
    builder.add_edge("action_master", "context_guardian")
    builder.add_edge("pro_proof", "context_guardian")

    # 2차: context_guardian이 종합 후 quality_consensus로
    builder.add_edge("context_guardian", "quality_consensus")

    return builder.build()


# 전역 변수로 에이전트 그래프 초기화
evaluation_agents = create_evaluation_agents(strands_gemini_model)
evaluation_graph = build_evaluation_graph(evaluation_agents)


def run_multi_agent_evaluation(question: str, answer: str) -> Dict[str, Any]:
    """멀티 에이전트 시스템을 실행하여 평가 결과를 반환"""

    # 입력 포맷팅
    evaluation_input = f"""[멘티 질문]
{question}

[멘토 답변]
{answer}"""

    # 그래프 실행
    result = evaluation_graph(evaluation_input)

    # 에이전트 응답 추출
    agent_responses = []
    execution_info = {}

    for node in result.execution_order:
        node_id = node.node_id

        # 응답 텍스트 추출
        if hasattr(node, "result") and node.result:
            agent_result = node.result.result
            if hasattr(agent_result, "message") and agent_result.message:
                content = agent_result.message.get("content", [])
                if content and len(content) > 0:
                    text = content[0].get("text", "")
                else:
                    text = "(응답 없음)"
            else:
                text = "(응답 없음)"

            # 실행 정보 추출
            execution_time = node.result.execution_time / 1000  # ms -> s
            usage = (
                node.result.accumulated_usage
                if hasattr(node.result, "accumulated_usage")
                else {}
            )

            agent_responses.append(
                {
                    "agent_name": node_id,
                    "response_text": text,
                    "execution_time": execution_time,
                    "token_usage": usage,
                }
            )

            execution_info[node_id] = {
                "execution_time": execution_time,
                "usage": usage,
            }

    # 최종 합의 결과
    final_consensus = (
        agent_responses[-1]["response_text"] if agent_responses else "(평가 실패)"
    )

    # 총 실행 시간 및 토큰
    total_execution_time = sum(
        info["execution_time"] for info in execution_info.values()
    )
    total_tokens = sum(
        info["usage"].get("totalTokens", 0) for info in execution_info.values()
    )

    return {
        "agent_responses": agent_responses,
        "final_consensus": final_consensus,
        "total_execution_time": total_execution_time,
        "total_tokens": total_tokens,
        "execution_order": [node.node_id for node in result.execution_order],
        "status": result.status,
    }


def run_rubric_evaluation(
    question: str, answer: str, agent_consensus: str
) -> Dict[str, Any]:
    """DeepEval의 Rubric 기반 평가를 실행"""

    # Rubric 정의
    mentoring_rubric = [
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

    # GEval 메트릭 생성
    quality_metric = GEval(
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
        rubric=mentoring_rubric,
        threshold=0.7,  # 7점(0.7) 이상 시 합격
        model=deepeval_gemini_model,
    )

    # 테스트 케이스 생성
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=[agent_consensus],  # 에이전트 합의 결과를 컨텍스트로 사용
    )

    # 평가 실행
    results = evaluate(
        test_cases=[test_case],
        metrics=[quality_metric],
    )

    # 결과 추출
    metric_data = results.test_results[0].metrics_data[0]

    # 한글 번역
    reason_kr = translate_to_korean(metric_data.reason)

    return {
        "score": metric_data.score,
        "threshold": metric_data.threshold,
        "success": metric_data.success,
        "reason_en": metric_data.reason,
        "reason_kr": reason_kr,
        "evaluation_cost": metric_data.evaluation_cost,
        "evaluation_model": metric_data.evaluation_model,
    }


def calculate_grade(score: float) -> str:
    """0-10 스케일 점수를 D/C/B/A/S 등급으로 변환"""
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


# Pydantic 모델 정의
class TestCaseRequest(BaseModel):
    input: str
    actual_output: str
    expected_output: Optional[str] = None


class EvaluationRequest(BaseModel):
    test_cases: List[TestCaseRequest]


class AgentResponseDetail(BaseModel):
    agent_name: str
    response_text: str
    execution_time: float  # 초 단위
    token_usage: Dict[str, int]  # totalTokens, inputTokens, outputTokens 등


class RubricEvaluationDetail(BaseModel):
    score: float  # 0-1 스케일
    absolute_score: float  # 0-10 스케일
    grade: str  # D, C, B, A, S
    threshold: float
    success: bool
    reason: str  # 한글
    reason_en: str  # 영문 원본
    evaluation_cost: float
    evaluation_model: str


class TestResultResponse(BaseModel):
    test_case_index: int
    input: str
    actual_output: str
    expected_output: Optional[str]

    # 멀티 에이전트 분석 결과
    agent_responses: List[AgentResponseDetail]
    final_consensus: str  # quality_consensus의 최종 리포트

    # Rubric 기반 평가 결과
    rubric_evaluation: RubricEvaluationDetail

    # 실행 정보
    total_execution_time: float
    total_tokens: int
    execution_order: List[str]

    # 전체 성공 여부
    success: bool


class EvaluationResponse(BaseModel):
    test_results: List[TestResultResponse]


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_test_cases(request: EvaluationRequest):
    """멀티 에이전트 시스템으로 테스트 케이스를 평가하는 API"""

    response = {"test_results": []}

    for i, test_case in enumerate(request.test_cases):
        # Step 1: 멀티 에이전트 평가 실행
        agent_evaluation = run_multi_agent_evaluation(
            question=test_case.input, answer=test_case.actual_output
        )

        # Step 2: DeepEval Rubric 평가 실행
        rubric_evaluation = run_rubric_evaluation(
            question=test_case.input,
            answer=test_case.actual_output,
            agent_consensus=agent_evaluation["final_consensus"],
        )

        # Step 3: 등급 산정
        grade = calculate_grade(rubric_evaluation["score"])
        absolute_score = rubric_evaluation["score"] * 10

        # Step 4: 응답 구성
        test_result = TestResultResponse(
            test_case_index=i,
            input=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            agent_responses=[
                AgentResponseDetail(**agent_resp)
                for agent_resp in agent_evaluation["agent_responses"]
            ],
            final_consensus=agent_evaluation["final_consensus"],
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
            total_execution_time=agent_evaluation["total_execution_time"],
            total_tokens=agent_evaluation["total_tokens"],
            execution_order=agent_evaluation["execution_order"],
            success=rubric_evaluation["success"],
        )

        response["test_results"].append(test_result)

    return response


@app.get("/")
def root():
    """API 상태 확인"""
    return {"message": "CoEval API is running"}
