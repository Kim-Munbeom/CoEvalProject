import streamlit as st
import httpx
import json
from typing import Dict, Any

# 페이지 설정
st.set_page_config(
    page_title="CoEval V2 - 답변 평가 시스템 (100점 만점)",
    page_icon="📊",
    layout="wide"
)

# 타이틀
st.title("📊 CoEval V2 - 멘토링 답변 평가 시스템")
st.markdown(
    "멘토링 답변을 **3개 영역 전문 에이전트**가 평가하여 **0-100점 스케일**과 **등급(S/A/B/C/D)**을 제공합니다."
)
st.caption("🎯 실행가능성 | 🔬 전문성 | 🌍 현실성 → 📊 가중치 기반 종합 평가")

# API 엔드포인트 설정
API_BASE_URL = "http://localhost:8000"

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_base_url = st.text_input("API Base URL", value=API_BASE_URL)
    st.markdown("---")
    st.markdown("### 평가 시스템 V2")
    st.markdown("**4개 전문 에이전트 구성:**")
    st.markdown("- 🎯 **Action Master**: 실행가능성 전문가 (0-100점)")
    st.markdown("  - 정확성, 명료성, 관련성, 완전성 평가")
    st.markdown("- 🔬 **Pro Proof**: 전문성 검증자 (0-100점)")
    st.markdown("  - 구체 정보, 실무 디테일 평가")
    st.markdown("- 🌍 **Context Guardian**: 현실성 감시자 (0-100점)")
    st.markdown("  - 멘티 상황 고려, 경험 기반 조언 평가")
    st.markdown("- 📊 **Quality Consensus**: 최종 조정자")
    st.markdown("  - 3개 영역 가중치 적용 종합")
    st.markdown("---")
    st.markdown("### V2 주요 변경사항")
    st.markdown("- ✅ **100점 만점 시스템**")
    st.markdown("- ✅ **가중치 실시간 조정** (UI 슬라이더)")
    st.markdown("- ✅ **질문 제목/내용 분리**")
    st.markdown("- ✅ **DeepEval 검증 결과**")
    st.markdown("---")
    st.markdown("### 등급 체계 (100점 기준)")
    st.markdown("- **S등급 (90-100점)**: 완벽")
    st.markdown("- **A등급 (75-89점)**: 우수")
    st.markdown("- **B등급 (60-74점)**: 양호")
    st.markdown("- **C등급 (40-59점)**: 부족")
    st.markdown("- **D등급 (0-39점)**: 미달")

# 메인 컨텐츠

# ===== 샘플 데이터 선택 =====
st.header("📋 샘플 데이터 선택")

with st.expander("💡 샘플 데이터 불러오기", expanded=False):
    try:
        # GET /samples API 호출
        response = httpx.get(f"{api_base_url}/samples", timeout=10.0)
        if response.status_code == 200:
            samples_data = response.json()
            samples = samples_data.get("samples", [])

            if samples:
                # 샘플 선택 옵션 생성
                sample_options = {
                    f"{s['id']} - {s['question']['title']}": s
                    for s in samples
                }

                selected_sample_key = st.selectbox(
                    "샘플 선택",
                    options=["직접 입력"] + list(sample_options.keys()),
                    key="sample_selector"
                )

                if selected_sample_key != "직접 입력":
                    sample = sample_options[selected_sample_key]

                    # 샘플 미리보기
                    st.markdown("**📄 샘플 미리보기:**")
                    st.info(f"**질문 제목:** {sample['question']['title']}")
                    st.info(f"**질문 내용:** {sample['question']['content'][:100]}...")
                    st.info(f"**답변:** {sample['answer']['content'][:100]}...")

                    if st.button("📥 이 샘플 불러오기", use_container_width=True):
                        st.session_state.question_title = sample["question"]["title"]
                        st.session_state.question_content = sample["question"]["content"]
                        st.session_state.answer_content = sample["answer"]["content"]
                        st.success("✅ 샘플 데이터가 로드되었습니다!")
                        st.rerun()
            else:
                st.warning("샘플 데이터가 없습니다.")
        else:
            st.warning(f"샘플 데이터를 불러올 수 없습니다. (Status: {response.status_code})")
    except httpx.ConnectError:
        st.warning(f"⚠️ API 서버에 연결할 수 없습니다. 샘플 데이터를 불러오려면 서버를 실행하세요.")
    except Exception as e:
        st.warning(f"⚠️ 샘플 데이터 로드 중 오류: {str(e)}")

st.markdown("---")

# ===== 질문 및 답변 입력 =====
st.header("📝 평가할 질문 및 답변 입력")

col1, col2 = st.columns(2)

with col1:
    st.subheader("질문")
    question_title = st.text_input(
        "질문 제목",
        placeholder="예: SQL 쿼리 최적화 방법",
        value=st.session_state.get("question_title", ""),
        key="question_title_input"
    )
    question_content = st.text_area(
        "질문 내용",
        height=150,
        placeholder="대용량 데이터 조회 시 쿼리가 30초 이상 걸립니다. 어떻게 개선할 수 있나요?",
        value=st.session_state.get("question_content", ""),
        key="question_content_input"
    )

with col2:
    st.subheader("답변")
    answer_content = st.text_area(
        "답변 내용",
        height=220,
        placeholder="다음 3단계로 최적화하세요:\n\n1. 인덱스 추가\n- WHERE 절의 컬럼에 복합 인덱스 생성...",
        value=st.session_state.get("answer_content", ""),
        key="answer_content_input"
    )

st.markdown("---")

# ===== 가중치 조정 UI =====
st.header("⚖️ 평가 기준 가중치 설정")
st.caption("각 평가 항목의 중요도를 조정하세요 (합계 100%)")

col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

with col1:
    weight_action = st.slider(
        "🎯 실행가능성",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="정확성, 명료성, 관련성, 완전성 평가",
        key="weight_actionability"
    )

with col2:
    weight_expertise = st.slider(
        "🔬 전문성",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="구체 정보, 실무 디테일 평가",
        key="weight_expertise"
    )

with col3:
    weight_practicality = st.slider(
        "🌍 현실성",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="멘티 상황 고려, 경험 기반 조언 평가",
        key="weight_practicality"
    )

with col4:
    total_weight = weight_action + weight_expertise + weight_practicality
    if total_weight == 100:
        st.success(f"✅ {total_weight}%")
    else:
        st.error(f"❌ {total_weight}%")

# 가중치 합 검증
if total_weight != 100:
    st.warning("⚠️ 가중치 합계가 100%가 되도록 조정해주세요")

st.markdown("---")

# ===== 평가 실행 =====
st.header("🔍 평가 실행")

# 평가 버튼
eval_disabled = (
    not question_title or
    not question_content or
    not answer_content or
    total_weight != 100
)

if st.button("🔍 평가 시작", type="primary", use_container_width=True, disabled=eval_disabled):
    with st.spinner("평가 중... (최대 5분 소요)"):
        try:
            # 1. PUT /config/weights API 호출 (가중치 업데이트)
            weights_payload = {
                "actionability": weight_action / 100,
                "expertise": weight_expertise / 100,
                "practicality": weight_practicality / 100
            }

            weights_response = httpx.put(
                f"{api_base_url}/config/weights",
                json=weights_payload,
                timeout=10.0
            )
            weights_response.raise_for_status()

            # 2. POST /evaluate API 호출
            eval_payload = {
                "question_title": question_title,
                "question_content": question_content,
                "answer_content": answer_content
            }

            eval_response = httpx.post(
                f"{api_base_url}/evaluate",
                json=eval_payload,
                timeout=300.0  # 5분 타임아웃
            )
            eval_response.raise_for_status()

            result = eval_response.json()

            # ===== 결과 표시 =====
            st.success("✅ 평가 완료!")
            st.markdown("---")

            # 최종 점수 및 등급 표시
            st.header("🎯 최종 평가 결과")

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.metric("🎯 최종 점수", f"{result['final_score']:.1f}/100")

            with col2:
                grade = result['grade']
                grade_colors = {"S": "🟡", "A": "🟢", "B": "🔵", "C": "🟠", "D": "🔴"}
                st.metric("📊 등급", f"{grade_colors.get(grade, '⚪')} {grade}")

            with col3:
                st.metric("⏱️ 처리 시간", f"{result['processing_time']:.1f}초")

            # 적용된 가중치 표시
            st.caption(
                f"**적용된 가중치:** "
                f"실행가능성 {result['weights']['actionability']*100:.0f}% | "
                f"전문성 {result['weights']['expertise']*100:.0f}% | "
                f"현실성 {result['weights']['practicality']*100:.0f}%"
            )

            st.markdown("---")

            # 세부 점수
            st.subheader("📋 세부 점수")
            col1, col2, col3 = st.columns(3)

            with col1:
                action_score = result['scores']['actionability']
                st.metric("🎯 실행가능성", f"{action_score:.0f}/100")

            with col2:
                expertise_score = result['scores']['expertise']
                st.metric("🔬 전문성", f"{expertise_score:.0f}/100")

            with col3:
                practicality_score = result['scores']['practicality']
                st.metric("🌍 현실성", f"{practicality_score:.0f}/100")

            st.markdown("---")

            # 평가 근거
            st.subheader("💬 평가 근거")

            with st.expander("🎯 실행가능성 근거", expanded=True):
                st.markdown(result['rationale'].get('actionability', 'N/A'))

            with st.expander("🔬 전문성 근거", expanded=True):
                st.markdown(result['rationale'].get('expertise', 'N/A'))

            with st.expander("🌍 현실성 근거", expanded=True):
                st.markdown(result['rationale'].get('practicality', 'N/A'))

            st.markdown("---")

            # DeepEval 검증 결과
            if result.get('deepeval_results'):
                st.subheader("🔍 DeepEval 검증 결과")

                for agent_name, eval_result in result['deepeval_results'].items():
                    status = eval_result.get('status', 'unknown')
                    status_icon = "✅" if status == 'pass' else "❌"
                    confidence = eval_result.get('confidence', 0.0)
                    reason = eval_result.get('reason', 'N/A')

                    with st.expander(f"{status_icon} **{agent_name}**: {status.upper()} (신뢰도: {confidence:.2f})"):
                        st.write(reason)

            st.markdown("---")

            # JSON 결과 보기
            with st.expander("🔍 전체 JSON 결과 보기"):
                st.json(result)

        except httpx.ConnectError:
            st.error(
                f"❌ API 서버에 연결할 수 없습니다. {api_base_url}이 실행 중인지 확인하세요.\n\n"
                f"서버 실행: `uvicorn co_eval_v2:app --reload --port 8000`"
            )
        except httpx.HTTPStatusError as e:
            st.error(f"❌ API 요청 중 오류가 발생했습니다: {str(e)}\n\n응답: {e.response.text}")
        except Exception as e:
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")

# 입력 필드 안내
if eval_disabled:
    if not question_title or not question_content or not answer_content:
        st.info("💡 질문 제목, 질문 내용, 답변을 모두 입력해주세요.")
    if total_weight != 100:
        st.info("💡 가중치 합계를 100%로 조정해주세요.")

st.markdown("---")

# 푸터
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>CoEval V2 - AI 답변 품질 평가 시스템 (100점 만점) | Powered by DeepEval & Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)
