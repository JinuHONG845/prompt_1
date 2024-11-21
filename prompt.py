import streamlit as st
import openai
from anthropic import Anthropic
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os
import plotly.graph_objects as go
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="LLM 모델 비교", layout="wide")
st.title("LLM 모델 비교 v1(241110)")

# 사이드바에 API 키 입력 필드 추가
with st.sidebar:
    st.header("API 키 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    google_api_key = st.text_input("Google API Key", type="password")

# 프롬프트 입력
user_prompt = st.text_area("프롬프트를 입력하세요:", height=100)

# 평가 함수 추가
def evaluate_responses(responses, openai_client):
    evaluation_prompt = """
    다음 AI 모델들의 응답을 5가지 기준으로 평가해주세요:
    1. 정확성 (Accuracy): 응답이 얼마나 사실에 기반하고 정확한가? (1-10점)
    2. 완성도 (Completeness): 질문에 대해 얼마나 포괄적으로 답변했는가? (1-10점)
    3. 명확성 (Clarity): 응답이 얼마나 명확하고 이해하기 쉬운가? (1-10점)
    4. 창의성 (Creativity): 응답이 얼마나 창의적이고 독창적인가? (1-10점)
    5. 유용성 (Usefulness): 응답이 실제로 얼마나 유용한가? (1-10점)

    각 모델의 응답:
    {responses}

    JSON 형식으로 평가 결과를 출력해주세요. 예시:
    {
        "GPT-4": {"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 6, "유용성": 8},
        "Claude": {"정확성": 7, "완성도": 8, "명확성": 7, "창의성": 7, "유용성": 7},
        "Gemini": {"정확성": 6, "완성도": 7, "명확성": 8, "창의성": 5, "유용성": 6}
    }
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt.format(responses=responses)}],
            temperature=0.7
        )
        return eval(response.choices[0].message.content)
    except Exception as e:
        st.error(f"평가 중 오류 발생: {str(e)}")
        return None

# 레이더 차트 생성 함수
def create_radar_chart(evaluation_results):
    categories = ['정확성', '완성도', '명확성', '창의성', '유용성']
    
    fig = go.Figure()
    colors = ['rgb(67, 67, 67)', 'rgb(115, 115, 115)', 'rgb(49, 130, 189)', 'rgb(189, 189, 189)']
    
    for i, (model, scores) in enumerate(evaluation_results.items()):
        fig.add_trace(go.Scatterpolar(
            r=[scores[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=model,
            line_color=colors[i]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="LLM 모델 성능 비교"
    )
    
    return fig

if st.button("생성"):
    if user_prompt:
        # 3개의 컬럼으로 변경
        col1, col2, col3 = st.columns(3)
        
        # 응답을 저장할 딕셔너리 초기화
        responses = {}
        
        with col1:
            st.subheader("GPT-4")
            if openai_api_key:
                try:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    client = openai.OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": user_prompt}],
                        stream=True
                    )
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    responses["GPT-4"] = full_response  # 여기서 응답 저장
                except Exception as e:
                    st.error(f"OpenAI 에러: {str(e)}")
            else:
                st.warning("OpenAI API 키를 입력해주세요.")

        with col2:
            st.subheader("Claude")
            if anthropic_api_key:
                try:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    client = Anthropic(api_key=anthropic_api_key)
                    message = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": user_prompt}],
                        stream=True
                    )
                    
                    for chunk in message:
                        if chunk.delta.text:
                            full_response += chunk.delta.text
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    responses["Claude"] = full_response  # 여기서 응답 저장
                except Exception as e:
                    st.error(f"Anthropic 에러: {str(e)}")
            else:
                st.warning("Anthropic API 키를 입력해주세요.")

        with col3:
            st.subheader("Gemini Pro")
            if google_api_key:
                try:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    genai.configure(api_key=google_api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(
                        contents=user_prompt,
                        generation_config={
                            "temperature": 0.9,
                            "top_p": 1,
                            "top_k": 1,
                            "max_output_tokens": 2048,
                        },
                        stream=True
                    )
                    
                    for chunk in response:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    responses["Gemini"] = full_response  # 여기서 응답 저장
                except Exception as e:
                    st.error(f"Google AI 에러: {str(e)}")
            else:
                st.warning("Google API 키를 입력해주세요.")

        # 구분선 추가
        st.markdown("---")
        st.subheader("모델 성능 평가")
        
        # GPT-4로 평가 수행
        if openai_api_key:
            evaluation_results = evaluate_responses(responses, openai.OpenAI(api_key=openai_api_key))
            
            if evaluation_results:
                # 레이더 차트 생성 및 표시
                fig = create_radar_chart(evaluation_results)
                st.plotly_chart(fig)
                
                # 상세 평가 결과 표시
                st.subheader("상세 평가 결과")
                for model, scores in evaluation_results.items():
                    st.write(f"**{model}**")
                    for criterion, score in scores.items():
                        st.write(f"- {criterion}: {score}/10")
        else:
            st.warning("평가를 위해 OpenAI API 키가 필요합니다.")
    else:
        st.warning("프롬프트를 입력해주세요.")