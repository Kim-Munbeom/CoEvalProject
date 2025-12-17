import os
from typing import List, Optional

from deepeval.evaluate import evaluate
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

gemini_model = GeminiModel(
    model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3
)

# Google GenAI 클라이언트 초기화
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


# 평가 메트릭 정의
viability_metric = GEval(
    name="실행 가능성",
    evaluation_steps=[
        # 답변은 '바로 따라 할 수 있는 구체적인 행동 단계, 수치, 예시, 도구명'을 포함해야 하며, 조언이 명료하고 정확해야 합니다.
        "The answer should include 'specific action steps, figures, examples, tool names that can be followed immediately' and the advice should be clear and accurate.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8,
    model=gemini_model,
)

professionalism_metric = GEval(
    name="전문성",
    evaluation_steps=[
        # 답변은 '실무 경험, 직무 지식, 도구·지표·프로세스 등 전문적인 디테일'에 기반한 조언이어야 합니다.
        "Your answers should be based on 'practical experience, job knowledge, professional details such as tools, indicators, processes, etc.'",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8,
    model=gemini_model,
)

reality_metric = GEval(
    name="현실성",
    evaluation_steps=[
        # 답변은 '멘티의 현재 상황, 수준, 조건'을 고려해야 하며, 조언의 'Why, When, 주의점(리스크)' 등 현실적 맥락을 함께 제공해야 합니다.
        "The answer should take into account the 'current situation, level, and conditions of the mentee' and provide a realistic context such as 'Why, When, Attention (Risk)' of the advice.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8,
    model=gemini_model,
)


# Request 모델 정의
class TestCaseRequest(BaseModel):
    input: str
    actual_output: str
    expected_output: Optional[str] = None


class EvaluationRequest(BaseModel):
    test_cases: List[TestCaseRequest]


@app.post("/evaluate")
def evaluate_test_cases(request: EvaluationRequest):
    """테스트 케이스를 받아서 평가를 수행하는 API"""

    # LLMTestCase 객체로 변환
    test_cases = [
        LLMTestCase(
            input=tc.input,
            actual_output=tc.actual_output,
            expected_output=tc.expected_output,
        )
        for tc in request.test_cases
    ]

    # 평가 실행
    results = evaluate(
        test_cases=test_cases,
        metrics=[
            viability_metric,
            professionalism_metric,
            reality_metric,
        ],
    )

    # 결과를 JSON 형식으로 변환
    response = {"test_results": []}

    for i, test_case in enumerate(results.test_results):
        test_result = {
            "test_case_index": i,
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "success": test_case.success,
            "metrics": [],
        }

        for metric_data in test_case.metrics_data:
            # 평가 이유를 한글로 번역
            korean_reason = translate_to_korean(metric_data.reason)

            test_result["metrics"].append(
                {
                    "name": metric_data.name,
                    "score": metric_data.score,
                    "threshold": metric_data.threshold,
                    "success": metric_data.success,
                    "reason": korean_reason,
                    "reason_en": metric_data.reason,  # 원문도 함께 제공
                }
            )

        response["test_results"].append(test_result)

    return response


@app.get("/")
def root():
    """API 상태 확인"""
    return {"message": "CoEval API is running"}
