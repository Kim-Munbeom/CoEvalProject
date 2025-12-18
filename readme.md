# CoEval - AI 답변 품질 평가 시스템

CoEval은 **멀티 에이전트 시스템**을 활용하여 AI 멘토링 답변의 품질을 정량적으로 평가하는 시스템입니다. 4개의 전문 에이전트가 협력하여 답변을 분석하고, DeepEval의 Rubric 기반 평가를 통해 **0-10점 스케일**과 **등급(D/C/B/A/S)**을 제공합니다.

## 주요 기능

- **🤖 멀티 에이전트 평가**: 4개의 전문 에이전트가 다각도로 답변 분석
- **📊 등급 시스템**: 0-10점 스케일과 D/C/B/A/S 등급 제공
- **🔍 상세 분석**: 각 에이전트의 평가 근거와 최종 종합 리포트
- **🌐 자동 번역**: 평가 결과를 자동으로 한글로 번역
- **💻 웹 UI**: Streamlit 기반의 사용자 친화적 인터페이스
- **🚀 REST API**: FastAPI 기반의 평가 API 엔드포인트
- **📋 샘플 데이터**: 좋은 답변과 나쁜 답변 예시 제공

## 시스템 구성

- **백엔드 (co_eval.py)**: FastAPI 기반 REST API 서버
- **프론트엔드 (app.py)**: Streamlit 기반 웹 UI
- **평가 엔진**: Strands 멀티 에이전트 + DeepEval Rubric

## 멀티 에이전트 평가 시스템

### 평가 프로세스

```
입력 (질문 + 답변)
    ↓
[1단계: 병렬 평가]
├─ 🎯 Action Master (실행 지침 검수)
└─ 🔬 Pro Proof (실무 디테일 검증)
    ↓
[2단계: 종합 분석]
🌍 Context Guardian (현실성 분석)
    ↓
[3단계: 최종 조정]
📊 Quality Consensus (종합 리포트)
    ↓
[4단계: Rubric 평가]
DeepEval → 0-10점 스케일 + 등급
```

### 4개 에이전트 역할

#### 🎯 Action Master (실행 지침 검수자)
- **역할**: 멘티가 "내일 아침 출근해서 무엇을 해야 할지" 명확한지 평가
- **평가 항목**:
  - 수치(%, 시간), 예시, 도구명(Notion, Jira 등), 구체적 단계 포함 여부
  - 추상적 표현 대신 '동사' 중심의 행동 지침
  - 구체적인 Action Item 추출

#### 🔬 Pro Proof (실무 디테일 검증가)
- **역할**: 단순 상식이 아닌 실무 경험 기반 지식인지 판별
- **평가 항목**:
  - 직무/업계 특유의 프로세스나 전문 용어 정확성
  - 실제 경험(Experience)이나 사례(Evidence) 근거
  - 이유 기반 설명 포함 여부

#### 🌍 Context Guardian (현실성 분석가)
- **역할**: 멘티의 현재 상황에서 실현 가능한지 검증
- **평가 항목**:
  - 멘티의 수준(주니어/시니어)과 상황 고려
  - 리스크(Exception)나 조건(When) 언급
  - Why 기반 맥락 설명

#### 📊 Quality Consensus (최종 조정자)
- **역할**: 세 에이전트의 의견을 종합하여 최종 리포트 작성
- **수행 임무**:
  - 에이전트 간 충돌 해결 및 보완
  - 항목별 점수 근거를 마크다운 형식으로 요약
  - 종합 점수와 핵심 개선 포인트 제시

## 등급 체계

| 등급 | 점수 | 평가 기준 |
|------|------|----------|
| **S** | 9-10점 | 완벽함. 수치/도구/단계 및 리스크 관리까지 포함된 최상위 답변 |
| **A** | 8-9점 | 우수함. 구체적 단계와 실무 지식이 포함된 수준 높은 답변 |
| **B** | 6-8점 | 양호함. 구체적 단계와 실무 지식이 일부 포함됨 |
| **C** | 3-6점 | 부족함. 조언은 있으나 추상적이며 멘티 상황 고려가 부족함 |
| **D** | 0-3점 | 미달. 실행성/전문성이 결여된 답변 |

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
# uv를 사용하여 FastAPI 서버 실행
uv run fastapi run co_eval.py --host 0.0.0.0 --port 8000
```

서버가 실행되면 다음 URL에서 확인할 수 있습니다:
- API 엔드포인트: http://localhost:8000
- API 문서 (Swagger): http://localhost:8000/docs
- API 문서 (ReDoc): http://localhost:8000/redoc

### 2. Streamlit 앱 실행 (app.py)

새로운 터미널 창을 열고 다음 명령어를 실행합니다:

```bash
# uv를 사용하여 Streamlit 앱 실행
uv run streamlit run app.py
```

브라우저가 자동으로 열리며, http://localhost:8501 에서 앱에 접속할 수 있습니다.

### 💡 실행 팁

- **개발 모드**: 코드 변경 시 자동 재시작을 원한다면 `fastapi dev` 사용
  ```bash
  uv run fastapi dev co_eval.py --port 8000
  ```
- **프로덕션 모드**: 안정적인 배포를 위해서는 `fastapi run` 사용 (권장)
- **실행 시간**: 첫 평가는 약 20-35초 소요 (4개 에이전트 + Rubric 평가 + 번역)

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
      "actual_output": "매일 코딩 연습을 하고, 오픈소스 프로젝트에 기여하며...",
      "expected_output": "구체적이고 실행 가능한 조언",
      "success": true,

      "agent_responses": [
        {
          "agent_name": "action_master",
          "response_text": "**실행 지침 분석**\n\n구체적인 Action Items:\n- 매일 코딩 연습...",
          "execution_time": 3.2,
          "token_usage": {
            "totalTokens": 850,
            "inputTokens": 320,
            "outputTokens": 530
          }
        },
        {
          "agent_name": "pro_proof",
          "response_text": "**전문성 검증**\n\n실무 디테일 분석...",
          "execution_time": 3.5,
          "token_usage": {"totalTokens": 920}
        },
        {
          "agent_name": "context_guardian",
          "response_text": "**현실성 분석**\n\n멘티 수준 고려...",
          "execution_time": 4.1,
          "token_usage": {"totalTokens": 1050}
        },
        {
          "agent_name": "quality_consensus",
          "response_text": "**최종 종합 평가**\n\n## 종합 점수: 7.5/10 (A등급)...",
          "execution_time": 5.8,
          "token_usage": {"totalTokens": 1350}
        }
      ],

      "final_consensus": "**최종 종합 평가**\n\n## 종합 점수: 7.5/10 (A등급)\n\n### 강점\n1. 실행가능성: 8/10...",

      "rubric_evaluation": {
        "score": 0.75,
        "absolute_score": 7.5,
        "grade": "A",
        "threshold": 0.7,
        "success": true,
        "reason": "답변은 구체적인 행동 단계를 제시하고 실무 지식을 포함하고 있습니다...",
        "reason_en": "The answer provides specific action steps and includes practical knowledge...",
        "evaluation_cost": 0.0012,
        "evaluation_model": "gemini-2.5-flash"
      },

      "total_execution_time": 16.6,
      "total_tokens": 4170,
      "execution_order": ["action_master", "pro_proof", "context_guardian", "quality_consensus"]
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
- 패키지 관리자: uv
- 주요 라이브러리:
  - **FastAPI**: 웹 API 프레임워크
  - **Streamlit**: 웹 UI 프레임워크
  - **Strands Agents**: 멀티 에이전트 시스템 프레임워크
  - **DeepEval**: LLM 평가 및 Rubric 프레임워크
  - **Google GenAI**: Google Gemini API 클라이언트
  - **httpx**: HTTP 클라이언트

## 기술 스택

### 백엔드 (co_eval.py)
- **멀티 에이전트**: Strands (4개 에이전트 DAG 구조)
- **평가 엔진**: DeepEval GEval + Rubric
- **LLM**: Google Gemini 2.5 Flash
- **번역**: Google Gemini 2.5 Flash Lite
- **API**: FastAPI + Pydantic

### 프론트엔드 (app.py)
- **UI 프레임워크**: Streamlit
- **HTTP 클라이언트**: httpx
- **결과 시각화**: 등급 배지, 프로그레스 바, 메트릭 카드

## 성능 및 비용

### 실행 시간 (테스트 케이스당)
- 멀티 에이전트 평가: 10-20초
- Rubric 평가: 5-10초
- 한글 번역: 2-3초
- **총 예상 시간**: 20-35초

### 토큰 사용량 및 비용 (테스트 케이스당)
- Action Master: ~800 토큰 (~$0.0006)
- Pro Proof: ~800 토큰 (~$0.0006)
- Context Guardian: ~1,000 토큰 (~$0.0008)
- Quality Consensus: ~1,300 토큰 (~$0.001)
- Rubric 평가: ~1,700 토큰 (~$0.0012)
- 한글 번역: ~350 토큰 (~$0.0003)
- **총합**: ~6,000 토큰 (~$0.0045)

*가격은 Gemini 2.5 Flash 기준 (2025년 1월)*

## 문제 해결

### API 서버 연결 오류
- FastAPI 서버가 실행 중인지 확인하세요
  ```bash
  curl http://localhost:8000/
  ```
- Streamlit 앱의 사이드바에서 API URL이 `http://localhost:8000/evaluate`인지 확인
- 방화벽이나 보안 소프트웨어가 포트 8000을 차단하는지 확인

### API 키 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인
- `GEMINI_API_KEY`가 올바르게 설정되었는지 확인
  ```bash
  cat .env | grep GEMINI_API_KEY
  ```
- API 키가 유효한지 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 확인

### 평가 시간이 오래 걸림
- **정상 범위**: 테스트 케이스당 20-35초는 정상입니다 (4개 에이전트 순차 실행)
- 네트워크 연결 상태 확인
- 타임아웃 설정 확인 (기본값: 300초)
  - app.py에서 `timeout=300.0` 수정 가능

### 임포트 오류 또는 모듈 없음
```bash
# 의존성 재설치
uv sync

# 가상환경 확인
uv run python -c "import co_eval; print('✅ 임포트 성공')"
```

### 에이전트 실행 오류
- DeepEval 로그 확인: 평가 중 발생하는 상세 오류 메시지 참고
- Strands 버전 확인: `strands-agents>=1.20.0` 필요
- 그래프 사이클 경고는 무시 가능 (현재 DAG 구조는 사이클이 없음)

## 변경 이력

### v0.2.0 (2025-01-XX)
- ✨ 멀티 에이전트 평가 시스템 도입
- 📊 Rubric 기반 0-10점 스케일 + 등급 체계 (D/C/B/A/S)
- 🤖 4개 전문 에이전트: Action Master, Pro Proof, Context Guardian, Quality Consensus
- 🎨 UI 전면 개편 (에이전트별 분석 결과 표시)
- 📈 실행 정보 및 토큰 사용량 추적 기능

### v0.1.0 (초기 버전)
- 3개 독립 GEval 메트릭 평가
- 실행가능성, 전문성, 현실성 기준

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 기여

이슈와 풀 리퀘스트는 언제나 환영합니다!

## 관련 링크

- [Strands Agents 문서](https://github.com/anthropics/strands)
- [DeepEval 문서](https://docs.confident-ai.com/)
- [Google Gemini API](https://ai.google.dev/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Streamlit 문서](https://docs.streamlit.io/)