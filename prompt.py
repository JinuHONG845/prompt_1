import streamlit as st
import openai
from anthropic import Anthropic
import google.generativeai as genai
import plotly.graph_objects as go
import json

# 페이지 설정
st.set_page_config(page_title="LLM 모델 비교", layout="wide")

# API 키 설정
openai_api_key = st.secrets["OPENAI_API_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# 메인 화면
st.title("LLM 모델 비교 v1")

# 프롬프트 입력
user_prompt = st.text_area("프롬프트를 입력하세요:", 
    height=100,
    placeholder="질문을 입력해주세요...")

# 응답 생성 함수들
def get_gpt4_response(prompt):
    client = openai.OpenAI(api_key=openai_api_key)
    try:
        message_placeholder = st.empty()
        full_response = ""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"OpenAI 에러: {str(e)}")
        return None

def get_claude_response(prompt):
    client = Anthropic(api_key=anthropic_api_key)
    try:
        message_placeholder = st.empty()
        full_response = ""
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in message:
            if chunk.delta.text:
                full_response += chunk.delta.text
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Anthropic 에러: {str(e)}")
        return None

def get_gemini_response(prompt):
    try:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-pro')
        message_placeholder = st.empty()
        full_response = ""
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Gemini 에러: {str(e)}")
        return None

# 평가 기준 설명
evaluation_criteria = {
    "정확성": "제공된 정보의 사실적 정확성과 신뢰성",
    "완성도": "응답의 포괄성과 주제에 대한 충분한 설명",
    "명확성": "설명의 논리적 구조와 이해하기 쉬운 표현",
    "창의성": "독창적인 관점과 혁신적인 해결방안 제시",
    "유용성": "실제 적용 가능성과 실용적 가치"
}

# 레이더 차트 생성 함수
@st.cache_data(ttl=3600)
def create_radar_chart(evaluation_results):
    categories = list(evaluation_criteria.keys())
    fig = go.Figure()
    colors = {
        'GPT-4': 'rgb(0, 122, 255)',
        'Claude': 'rgb(128, 0, 128)',
        'Gemini': 'rgb(255, 64, 129)'
    }
    
    for model, scores in evaluation_results.items():
        fig.add_trace(go.Scatterpolar(
            r=[scores[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=model,
            line_color=colors.get(model, 'rgb(128, 128, 128)')
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="LLM 모델 성능 비교"
    )
    return fig

def evaluate_responses(responses):
    client = openai.OpenAI(api_key=openai_api_key)
    evaluation_prompt = f"""
    다음 AI 모델들의 응답을 5가지 기준(정확성, 완성도, 명확성, 창의성, 유용성)으로 1-10점으로 평가해 JSON 형식으로만 답변하세요.
    각 모델별로 다음 형식으로 응답해주세요:
    {{
        "GPT-4": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 8, "유용성": 7}},
        "Claude": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 8, "유용성": 7}},
        "Gemini": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 8, "유용성": 7}}
    }}

    평가할 응답:
    GPT-4: {responses.get("GPT-4", "응답 없음")}
    Claude: {responses.get("Claude", "응답 없음")}
    Gemini: {responses.get("Gemini", "응답 없음")}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"평가 중 오류 발생: {str(e)}")
        return None

if st.button("생성"):
    if user_prompt:
        col1, col2, col3 = st.columns(3)
        responses = {}

        with col1:
            st.subheader("GPT-4")
            responses["GPT-4"] = get_gpt4_response(user_prompt)

        with col2:
            st.subheader("Claude 3")
            responses["Claude"] = get_claude_response(user_prompt)

        with col3:
            st.subheader("Gemini Pro")
            responses["Gemini"] = get_gemini_response(user_prompt)

        # 평가 결과 표시
        if all(responses.values()):
            st.markdown("---")
            st.subheader("모델 평가")
            
            evaluation = evaluate_responses(responses)
            if evaluation:
                fig = create_radar_chart(evaluation)
                st.plotly_chart(fig)

                # 평가 기준 설명 (접이식)
                with st.expander("평가 기준 설명"):
                    for criterion, description in evaluation_criteria.items():
                        st.markdown(f"**{criterion}**: {description}")
    else:
        st.warning("프롬프트를 입력해주세요.")