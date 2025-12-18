import logging
import os

from dotenv import load_dotenv
from strands import Agent
from strands.models.gemini import GeminiModel
from strands.multiagent import GraphBuilder

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval import evaluate
from deepeval.models import GeminiModel as DeepEvalGeminiModel
from google import genai

load_dotenv()

# logging.getLogger("strands.multiagent").setLevel(logging.DEBUG)
# logging.basicConfig(
#     format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()]
# )

gemini_model = GeminiModel(
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

action_master = Agent(
    name="action_master",
    system_prompt="""당신은 IT/직무 멘토링의 **'실행 지침 검수자'**입니다. 답변을 읽고 멘티가 **"내일 아침 출근해서 무엇을 해야 할지"**가 명확한지 평가하십시오.
[평가 루브릭]
1. 수치(%, 시간), 예시(Case), 도구명(Notion, Jira 등), 구체적 단계(Step 1, 2)가 포함되었는가?
2. '노력하세요', '열심히 하세요'와 같은 추상적 표현이 아닌 '동사' 중심의 행동 지침이 있는가?
3. 당신은 답변에서 구체적인 Action Item을 추출하여 리스트업해야 합니다.""",
    model=gemini_model,
)
pro_proof = Agent(
    name="pro_proof",
    system_prompt="""당신은 10년 차 이상의 **'실무 디테일 검증가'**입니다. 답변이 단순히 검색으로 알 수 있는 상식인지, 아니면 현업의 냄새가 나는 진짜 지식인지 판별하십시오.
[평가 루브릭]
1. 직무/업계 특유의 프로세스나 전문 용어가 정확하게 사용되었는가?
2. 실제 경험(Experience)이나 사례(Evidence)를 근거로 왜 그렇게 해야 하는지 이유를 설명하는가?""",
    model=gemini_model,
)
context_guardian = Agent(
    name="context_guardian",
    system_prompt="""당신은 멘티의 상황을 대변하는 **'현실성 분석가'**입니다. 조언이 아무리 훌륭해도 멘티의 현재 상황(연차, 환경)에서 실현 불가능한지를 찾아내십시오.
[평가 루브릭]
1. 질문에 담긴 멘티의 수준(주니어/시니어)과 처한 상황을 충분히 고려했는가?
2. 조언을 실행할 때 주의해야 할 리스크(Exception)나 조건(When)을 언급했는가?
3. 무조건적인 정답이 아니라 '왜(Why)' 이 상황에 이 조언이 최선인지 맥락을 설명하는가?""",
    model=gemini_model,
)
quality_consensus = Agent(
    name="quality_consensus",
    system_prompt="""당신은 앞선 세 에이전트(액션 마스터, 프로프, 가디언)의 의견을 종합하여 최종 멘토링 품질 리포트를 작성하는 조정자입니다.
[수행 임무]
- 세 에이전트 간에 의견이 충돌할 경우(예: 전문성은 높으나 실행이 너무 어려움), 이를 보완할 수 있는 최종 결론을 도출하십시오.
- 모든 분석 내용을 취합하여 DeepEval이 0~10점 사이의 점수를 매길 수 있도록 각 항목별 점수 근거를 마크다운 형식으로 요약하십시오.
- 최종 출력물에는 반드시 '종합 점수'와 '핵심 개선 포인트'를 포함해야 합니다.""",
    model=gemini_model,
)

# 그래프 구성
builder = GraphBuilder()

# 노드 등록
builder.add_node(action_master, "action_master")
builder.add_node(pro_proof, "pro_proof")
builder.add_node(context_guardian, "context_guardian")
builder.add_node(quality_consensus, "quality_consensus")

# 1차 실행가능성 및 전문성 평가
builder.add_edge("action_master", "context_guardian")
builder.add_edge("pro_proof", "context_guardian")

# 2차 현실성 평가
builder.add_edge("context_guardian", "quality_consensus")

graph = builder.build()

# 평가할 멘토링 대화
mentoring_question = "AI를 의료 분야에 어떻게 활용할 수 있을까요? 이 분야로 진출하려면 무엇을 준비해야 하나요?"
mentoring_answer = """AI를 의료에 활용하려면 데이터 보안이 중요합니다. 먼저 관련 논문을 찾아보시고 파이썬 공부를 열심히 하세요."""

# 질문과 답변을 결합하여 평가
evaluation_input = f"""[멘티 질문]
{mentoring_question}

[멘토 답변]
{mentoring_answer}"""

result = graph(evaluation_input)

# 각 에이전트의 응답을 변수에 저장
agent_responses = {}
execution_info = {}

for node in result.execution_order:
    node_id = node.node_id

    # NodeResult -> AgentResult -> message -> content -> text 경로로 접근
    if hasattr(node, "result") and node.result:
        agent_result = node.result.result
        if hasattr(agent_result, "message") and agent_result.message:
            content = agent_result.message.get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "")
                agent_responses[node_id] = text
            else:
                agent_responses[node_id] = "(응답 없음)"
        else:
            agent_responses[node_id] = "(응답 없음)"

        # 실행 정보 저장
        execution_info[node_id] = {
            "execution_time": node.result.execution_time / 1000,
            "usage": (
                node.result.accumulated_usage
                if hasattr(node.result, "accumulated_usage")
                else {}
            ),
        }

# 출력할 데이터를 변수화
evaluation_result = {
    "question": mentoring_question,
    "answer": mentoring_answer,
    "agent_responses": agent_responses,
    "execution_info": execution_info,
    "final_evaluation": agent_responses.get(
        "quality_consensus", "(최종 평가 결과 없음)"
    ),
    "total_execution_time": sum(
        info["execution_time"] for info in execution_info.values()
    ),
    "total_tokens": sum(
        info["usage"].get("totalTokens", 0) for info in execution_info.values()
    ),
    "execution_order": [node.node_id for node in result.execution_order],
    "status": result.status,
}

# 최종 품질 평가 결과만 출력
# print("\n" + "=" * 80)
# print("멘토링 품질 평가 결과")
# print("=" * 80)

# print("\n[평가 대상 멘토링]")
# print("-" * 80)
# print(f"질문: {evaluation_result['question']}")
# print(f"답변: {evaluation_result['answer']}")

# print("\n" + "=" * 80)
# print("최종 품질 평가")
# print("=" * 80)

# print(f"\n{evaluation_result['final_evaluation']}")

# # 전체 실행 정보 요약
# print("\n" + "=" * 80)
# print("실행 정보 요약")
# print("=" * 80)

# print(f"\n총 실행 시간: {evaluation_result['total_execution_time']:.2f}초")
# print(f"총 토큰 사용량: {evaluation_result['total_tokens']}")
# print(f"실행 순서: {' -> '.join(evaluation_result['execution_order'])}")

# print("\n" + "=" * 80)


deepeval_gemini_model = DeepEvalGeminiModel(
    model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3
)

# Google GenAI 클라이언트 초기화
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
    threshold=0.7,  # 7점(0.7) 이상 시 합격 처리
    model=deepeval_gemini_model,
)

test_case = LLMTestCase(
    input=evaluation_result["question"],
    actual_output=evaluation_result["answer"],
    context=[evaluation_result["final_evaluation"]],
)

results = evaluate(
    test_cases=[test_case],
    metrics=[quality_metric],
)

# DeepEval 결과를 변수화
metric_data = results.test_results[0].metrics_data[0]

deepeval_result = {
    "test_name": results.test_results[0].name,
    "success": results.test_results[0].success,
    "score": metric_data.score,
    "threshold": metric_data.threshold,
    "reason": metric_data.reason,
    "metric_name": metric_data.name,
    "evaluation_cost": metric_data.evaluation_cost,
    "evaluation_model": metric_data.evaluation_model,
}

# DeepEval의 토큰 사용량 확인 (verbose_logs에서 추출 시도)
# 참고: DeepEval은 기본적으로 상세한 토큰 사용량을 제공하지 않을 수 있음
# print("\n" + "=" * 80)
# print("DeepEval 메트릭 상세 정보 확인")
# print("=" * 80)
# print(f"\nMetricData 속성: {dir(metric_data)}")
# print(f"\nevaluation_cost: {metric_data.evaluation_cost}")
# print(f"evaluation_model: {metric_data.evaluation_model}")

# # 결과 출력
# print("\n" + "=" * 80)
# print("DeepEval 평가 결과")
# print("=" * 80)

# print(f"\n평가 지표: {deepeval_result['metric_name']}")
# print(f"평가 모델: {deepeval_result['evaluation_model']}")
# print(
#     f"스코어: {deepeval_result['score']:.2f} / 1.0 (통과 기준: {deepeval_result['threshold']})"
# )
# print(f"통과 여부: {'✅ 통과' if deepeval_result['success'] else '❌ 실패'}")
# print(f"평가 비용: ${deepeval_result['evaluation_cost']:.4f}")
# print(f"\n평가 이유:")
# print(f"{deepeval_result['reason']}")

# print("\n" + "=" * 80)
