# CoEval - AI 답변 품질 평가 시스템

CoEval은 AI 멘토링 답변의 품질을 **실행 가능성**, **전문성**, **현실성** 세 가지 기준으로 평가하는 시스템입니다. DeepEval과 Google Gemini를 활용하여 답변의 품질을 정량적으로 측정하고 개선점을 제시합니다.

## 주요 기능

- **3가지 평가 기준**: 실행 가능성, 전문성, 현실성
- **자동 번역**: 평가 결과를 자동으로 한글로 번역
- **웹 UI**: Streamlit 기반의 사용자 친화적 인터페이스
- **REST API**: FastAPI 기반의 평가 API 엔드포인트
- **샘플 데이터**: 좋은 답변과 나쁜 답변 예시 제공

## 시스템 구성

- **백엔드 (co_eval.py)**: FastAPI 기반 REST API 서버
- **프론트엔드 (app.py)**: Streamlit 기반 웹 UI

## 평가 기준

### 1. 실행 가능성 (Viability)
- 바로 따라 할 수 있는 구체적인 행동 단계 포함
- 수치, 예시, 도구명 등 실용적 정보 제공
- 명료하고 정확한 조언

### 2. 전문성 (Professionalism)
- 실무 경험과 직무 지식 기반
- 전문적인 도구, 지표, 프로세스 등의 디테일 제공
- 실전에서 검증된 노하우

### 3. 현실성 (Reality)
- 멘티의 현재 상황, 수준, 조건 고려
- 조언의 Why, When, 주의점(리스크) 제공
- 실제 적용 가능한 현실적 맥락

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd CoEval
```

### 2. 의존성 설치

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다.

```bash
# uv가 설치되어 있지 않다면
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치
uv sync
```

또는 pip를 사용하는 경우:

```bash
pip install -r requirements.txt
# 또는
pip install deepeval fastapi[standard] google-genai httpx streamlit
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가합니다:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Google AI Studio에서 API 키를 발급받을 수 있습니다: https://aistudio.google.com/app/apikey

## 사용 방법

### 1. FastAPI 서버 실행 (co_eval.py)

터미널에서 다음 명령어를 실행합니다:

```bash
# uvicorn 직접 실행
uvicorn co_eval:app --reload

# 또는 포트를 지정하여 실행
uvicorn co_eval:app --host 0.0.0.0 --port 8000 --reload
```

서버가 실행되면 다음 URL에서 확인할 수 있습니다:
- API 엔드포인트: http://localhost:8000
- API 문서 (Swagger): http://localhost:8000/docs
- API 문서 (ReDoc): http://localhost:8000/redoc

### 2. Streamlit 앱 실행 (app.py)

새로운 터미널 창을 열고 다음 명령어를 실행합니다:

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며, http://localhost:8501 에서 앱에 접속할 수 있습니다.

## API 사용 예제

### 평가 API 호출

**엔드포인트**: `POST /evaluate`

**요청 예제**:

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "input": "주니어 백엔드 개발자가 실력을 빠르게 키우려면 어떻게 해야 하나요?",
        "actual_output": "매일 코딩 연습을 하고, 오픈소스 프로젝트에 기여하며, 기술 블로그를 읽고, 코드 리뷰를 받는 것이 좋습니다.",
        "expected_output": "구체적이고 실행 가능한 조언"
      }
    ]
  }'
```

**응답 예제**:

```json
{
  "test_results": [
    {
      "test_case_index": 0,
      "input": "주니어 백엔드 개발자가 실력을 빠르게 키우려면 어떻게 해야 하나요?",
      "actual_output": "매일 코딩 연습을 하고...",
      "expected_output": "구체적이고 실행 가능한 조언",
      "success": true,
      "metrics": [
        {
          "name": "실행 가능성",
          "score": 0.85,
          "threshold": 0.8,
          "success": true,
          "reason": "답변은 구체적인 행동 단계를 포함하고 있습니다...",
          "reason_en": "The answer includes specific action steps..."
        },
        {
          "name": "전문성",
          "score": 0.82,
          "threshold": 0.8,
          "success": true,
          "reason": "실무 경험에 기반한 조언을 제공합니다...",
          "reason_en": "Provides advice based on practical experience..."
        },
        {
          "name": "현실성",
          "score": 0.88,
          "threshold": 0.8,
          "success": true,
          "reason": "멘티의 수준을 고려한 현실적인 조언입니다...",
          "reason_en": "Realistic advice considering the mentee's level..."
        }
      ]
    }
  ]
}
```

### Python으로 API 호출

```python
import httpx

url = "http://localhost:8000/evaluate"
payload = {
    "test_cases": [
        {
            "input": "프론트엔드 개발자로 취업하려면 어떤 포트폴리오를 만들어야 하나요?",
            "actual_output": "React로 To-Do 앱을 만들고, GitHub에 올리면 됩니다.",
            "expected_output": None
        }
    ]
}

with httpx.Client() as client:
    response = client.post(url, json=payload, timeout=300.0)
    result = response.json()
    print(result)
```

## 프로젝트 구조

```
CoEval/
├── co_eval.py          # FastAPI 백엔드 서버
├── app.py              # Streamlit 웹 UI
├── sample_data.py      # 샘플 데이터 (좋은/나쁜 답변 예시)
├── pyproject.toml      # 프로젝트 설정 및 의존성
├── uv.lock            # uv 잠금 파일
├── .env               # 환경 변수 (API 키)
└── readme.md          # 프로젝트 문서
```

## 개발 환경

- Python: 3.14+
- 주요 라이브러리:
  - FastAPI: 웹 API 프레임워크
  - Streamlit: 웹 UI 프레임워크
  - DeepEval: LLM 평가 프레임워크
  - Google GenAI: Google Gemini API 클라이언트
  - httpx: HTTP 클라이언트

## 문제 해결

### API 서버 연결 오류
- FastAPI 서버가 실행 중인지 확인하세요 (`http://localhost:8000`)
- Streamlit 앱의 사이드바에서 API URL이 올바른지 확인하세요

### API 키 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인하세요
- `GEMINI_API_KEY`가 올바르게 설정되었는지 확인하세요
- API 키가 유효한지 Google AI Studio에서 확인하세요

### 평가 시간이 오래 걸림
- 평가 과정은 LLM을 사용하므로 몇 초에서 몇 분까지 걸릴 수 있습니다
- 네트워크 연결 상태를 확인하세요
- 타임아웃 설정을 늘려야 할 수 있습니다 (기본값: 300초)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 기여

이슈와 풀 리퀘스트는 언제나 환영합니다!