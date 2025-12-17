import streamlit as st
import httpx
import json
from sample_data import GOOD_EXAMPLES, BAD_EXAMPLES

# 페이지 설정
st.set_page_config(
    page_title="CoEval - 답변 평가 시스템", page_icon="📊", layout="wide"
)

# 타이틀
st.title("📊 CoEval - 답변 평가 시스템")
st.markdown(
    "AI 답변의 품질을 **실행 가능성**, **전문성**, **현실성** 기준으로 평가합니다."
)

# API 엔드포인트 설정
API_URL = "http://localhost:8000/evaluate"

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_url = st.text_input("API URL", value=API_URL)
    st.markdown("---")
    st.markdown("### 평가 기준")
    st.markdown("**실행 가능성**: 구체적인 행동 단계, 수치, 예시, 도구명 포함")
    st.markdown("**전문성**: 실무 경험, 직무 지식, 전문적 디테일")
    st.markdown("**현실성**: 멘티 상황 고려, Why/When/리스크 제공")

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

expected_output = st.text_input(
    "기대 출력 (선택사항)",
    placeholder="예: 구체적이고 실행 가능한 조언",
    key="expected",
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
                        "expected_output": expected_output if expected_output else None,
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

                    # 전체 성공 여부
                    if test_result["success"]:
                        st.success("✅ 전체 평가 통과!")
                    else:
                        st.error("❌ 일부 평가 항목이 기준을 충족하지 못했습니다.")

                    st.markdown("---")

                    # 각 메트릭 결과 표시
                    st.header("📈 평가 결과 상세")

                    for idx, metric in enumerate(test_result["metrics"]):
                        with st.expander(
                            f"**{metric['name']}** - 점수: {metric['score']:.2f} / 기준: {metric['threshold']}",
                            expanded=True,
                        ):
                            col1, col2, col3 = st.columns([2, 1, 1])

                            with col1:
                                # 진행바
                                st.progress(metric["score"])

                            with col2:
                                st.metric("점수", f"{metric['score']:.2f}")

                            with col3:
                                if metric["success"]:
                                    st.success("통과 ✅")
                                else:
                                    st.error("미달 ❌")

                            # 평가 이유
                            st.markdown("**평가 근거:**")
                            st.markdown(
                                f"""
                                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50;">
                                    <p style="color: #1f1f1f; margin: 0; font-size: 16px; line-height: 1.6;">
                                        {metric['reason']}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

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

            st.markdown(f"**기대 출력:**")
            st.caption(example["expected"])

            # 선택 버튼
            if st.button(
                f"📥 이 예시 불러오기",
                key=f"load_sample_{idx}",
                use_container_width=True,
            ):
                # 기존 위젯 키 삭제
                for key in ["question", "answer", "expected"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # 새 값 설정
                st.session_state.question = example["question"]
                st.session_state.answer = example["answer"]
                st.session_state.expected = example["expected"]
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
