import streamlit as st
import httpx
import json
from sample_data import GOOD_EXAMPLES, BAD_EXAMPLES

# 페이지 설정
st.set_page_config(
    page_title="CoEval - 답변 평가 시스템", page_icon="📊", layout="wide"
)

# 타이틀
st.title("📊 CoEval - 멘토링 답변 평가 시스템")
st.markdown(
    "멘토링 답변을 **3개 영역 전문 에이전트**가 평가하여 **0-10점 스케일**과 **등급(D/C/B/A/S)**을 제공합니다."
)
st.caption("🎯 실행성 | 🔬 전문성 | 🌍 현실성 → 📊 종합 평가")

# API 엔드포인트 설정
API_URL = "http://localhost:8000/evaluate"

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_url = st.text_input("API URL", value=API_URL)
    st.markdown("---")
    st.markdown("### 평가 시스템")
    st.markdown("**4개 전문 에이전트 구성:**")
    st.markdown("- 🎯 **Action Master**: 실행성 전문가")
    st.markdown("  - 구체적 수치, 도구명, 단계별 지침 평가")
    st.markdown("- 🔬 **Pro Proof**: 전문성 검증자")
    st.markdown("  - 현업 지식 vs 검색 지식 판별")
    st.markdown("- 🌍 **Context Guardian**: 현실성 감시자")
    st.markdown("  - 멘티 상황별 실현 가능성 검토")
    st.markdown("- 📊 **Quality Consensus**: 최종 조정자")
    st.markdown("  - 3개 영역 점수 종합 및 조정")
    st.markdown("---")
    st.markdown("### 등급 체계")
    st.markdown("**3개 영역 종합 평가 (10점 만점):**")
    st.markdown("- **S등급 (9-10점)**: 완벽")
    st.markdown("  - 실행성·전문성·현실성 모두 우수")
    st.markdown("- **A등급 (7-8점)**: 우수")
    st.markdown("  - 수치/도구/단계 + 리스크 관리")
    st.markdown("- **B등급 (5-6점)**: 양호")
    st.markdown("  - 구체적 단계 + 실무 지식 포함")
    st.markdown("- **C등급 (3-4점)**: 부족")
    st.markdown("  - 추상적, 멘티 상황 고려 부족")
    st.markdown("- **D등급 (0-2점)**: 미달")
    st.markdown("  - 필수 조건 결여 (실행성/전문성)")

# 메인 컨텐츠
st.header("📝 평가할 답변 입력")

# 테스트 케이스 입력
col1, col2 = st.columns(2)

with col1:
    st.subheader("질문")
    question = st.text_area(
        "질문을 입력하세요",
        height=150,
        placeholder="예: 주니어 백엔드 개발자가 실력을 빠르게 키우려면 어떻게 해야 하나요?",
        key="question",
    )

with col2:
    st.subheader("답변")
    answer = st.text_area(
        "평가할 답변을 입력하세요",
        height=150,
        placeholder="평가할 답변을 입력하세요...",
        key="answer",
    )

# 평가 버튼
if st.button("🔍 평가 시작", type="primary", use_container_width=True):
    if not question or not answer:
        st.error("질문과 답변을 모두 입력해주세요.")
    else:
        with st.spinner("평가 중..."):
            # API 요청 준비
            payload = {
                "test_cases": [
                    {
                        "input": question,
                        "actual_output": answer,
                    }
                ]
            }

            try:
                # API 호출
                with httpx.Client() as client:
                    response = client.post(api_url, json=payload, timeout=300.0)
                    response.raise_for_status()

                    # 결과 파싱
                    result = response.json()

                # 결과 표시
                st.success("평가 완료!")
                st.markdown("---")

                if result.get("test_results"):
                    test_result = result["test_results"][0]
                    rubric = test_result["rubric_evaluation"]

                    # 최종 점수 및 등급 표시
                    st.header("🎯 최종 평가 결과")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        # 등급 색상 설정
                        grade_colors = {
                            "S": "#FFD700",  # 금색
                            "A": "#90EE90",  # 연두색
                            "B": "#87CEEB",  # 하늘색
                            "C": "#FFA500",  # 주황색
                            "D": "#FF6347"   # 빨간색
                        }
                        grade_color = grade_colors.get(rubric["grade"], "#808080")

                        st.markdown(
                            f"""
                            <div style="text-align: center; padding: 20px; background-color: {grade_color}; border-radius: 10px;">
                                <h1 style="color: white; margin: 0; font-size: 48px;">{rubric['grade']}</h1>
                                <p style="color: white; margin: 0; font-size: 14px;">등급</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.metric(
                            "총점 (10점 만점)",
                            f"{rubric['absolute_score']:.1f}",
                            delta=f"{rubric['grade']} 등급"
                        )

                    with col3:
                        st.metric(
                            "정규화 점수 (0-1)",
                            f"{rubric['score']:.2f}",
                            delta=f"합격 기준: {rubric['threshold']:.2f}"
                        )

                    with col4:
                        if rubric["success"]:
                            st.success("✅ 통과")
                        else:
                            st.error("❌ 미달")

                    # 평가 근거
                    st.markdown("---")
                    st.subheader("📝 평가 근거")
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 6px solid {grade_color};">
                            <p style="color: #1f1f1f; margin: 0; font-size: 16px; line-height: 1.8;">
                                {rubric['reason']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 에이전트별 상세 분석
                    st.markdown("---")
                    st.header("🤖 에이전트별 상세 분석")

                    agent_icons = {
                        "action_master": "🎯",
                        "pro_proof": "🔬",
                        "context_guardian": "🌍",
                        "quality_consensus": "📊"
                    }

                    agent_names = {
                        "action_master": "Action Master (실행성 전문가)",
                        "pro_proof": "Pro Proof (전문성 검증자)",
                        "context_guardian": "Context Guardian (현실성 감시자)",
                        "quality_consensus": "Quality Consensus (최종 조정자)"
                    }

                    for agent in test_result["agent_responses"]:
                        agent_id = agent["agent_name"]
                        icon = agent_icons.get(agent_id, "🤖")
                        name = agent_names.get(agent_id, agent_id)

                        with st.expander(
                            f"{icon} **{name}** (실행: {agent['execution_time']:.2f}초)",
                            expanded=(agent_id == "quality_consensus")
                        ):
                            st.markdown(agent["response_text"])

                            # 토큰 사용량 정보
                            if agent.get("token_usage"):
                                st.caption(
                                    f"📊 토큰 사용량: {agent['token_usage'].get('totalTokens', 0):,} "
                                    f"(입력: {agent['token_usage'].get('inputTokens', 0):,}, "
                                    f"출력: {agent['token_usage'].get('outputTokens', 0):,})"
                                )

                    # 실행 정보
                    st.markdown("---")
                    st.subheader("⚡ 실행 정보")

                    info_col1, info_col2, info_col3 = st.columns(3)

                    with info_col1:
                        st.metric("총 실행 시간", f"{test_result['total_execution_time']:.2f}초")

                    with info_col2:
                        st.metric("총 토큰 사용량", f"{test_result['total_tokens']:,}")

                    with info_col3:
                        st.metric("평가 비용", f"${rubric['evaluation_cost']:.4f}")

                    st.caption(f"실행 순서: {' → '.join(test_result['execution_order'])}")
                    st.caption(f"평가 모델: {rubric['evaluation_model']}")

                    # JSON 결과 보기
                    with st.expander("🔍 전체 JSON 결과 보기"):
                        st.json(result)

            except httpx.ConnectError:
                st.error(
                    f"❌ API 서버에 연결할 수 없습니다. {api_url}이 실행 중인지 확인하세요."
                )
            except httpx.HTTPStatusError as e:
                st.error(f"❌ API 요청 중 오류가 발생했습니다: {str(e)}")
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")

# 샘플 데이터 섹션
st.markdown("---")
st.header("📋 샘플 데이터")

# 샘플 목록 표시 상태 관리
if "show_samples" not in st.session_state:
    st.session_state.show_samples = None

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ 좋은 답변 예시 보기", use_container_width=True):
        st.session_state.show_samples = "good"

with col2:
    if st.button("❌ 나쁜 답변 예시 보기", use_container_width=True):
        st.session_state.show_samples = "bad"

# 선택된 샘플 목록 표시
if st.session_state.show_samples:
    st.markdown("---")
    samples = GOOD_EXAMPLES if st.session_state.show_samples == "good" else BAD_EXAMPLES
    sample_type = (
        "좋은 답변" if st.session_state.show_samples == "good" else "나쁜 답변"
    )

    st.subheader(f"📚 {sample_type} 예시 목록")

    for idx, example in enumerate(samples):
        with st.expander(
            f"예시 {idx + 1}: {example['question'][:50]}...", expanded=False
        ):
            st.markdown(f"**질문:**")
            st.info(example["question"])

            st.markdown(f"**답변:**")
            st.text_area(
                "",
                value=example["answer"],
                height=200,
                disabled=True,
                key=f"sample_answer_{idx}",
            )

            # 선택 버튼
            if st.button(
                f"📥 이 예시 불러오기",
                key=f"load_sample_{idx}",
                use_container_width=True,
            ):
                # 기존 위젯 키 삭제
                for key in ["question", "answer"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # 새 값 설정
                st.session_state.question = example["question"]
                st.session_state.answer = example["answer"]
                st.session_state.show_samples = None  # 목록 숨기기
                st.rerun()

    # 목록 닫기 버튼
    if st.button("✖️ 목록 닫기", use_container_width=True):
        st.session_state.show_samples = None
        st.rerun()

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>CoEval - AI 답변 품질 평가 시스템 | Powered by DeepEval & Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)
