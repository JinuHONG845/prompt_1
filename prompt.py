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
    evaluation_prompt = f"""
다음 AI 모델들의 응답을 5가지 기준으로 평가해주세요. 
각 기준은 1-10점으로 평가하며, 반드시 아래 JSON 형식으로만 답변해주세요.

평가 기준:
1. 정확성 (Accuracy): 응답이 얼마나 사실에 기반하고 정확한가?
2. 완성도 (Completeness): 질문에 대해 얼마나 포괄적으로 답변했는가?
3. 명확성 (Clarity): 응답이 얼마나 명확하고 이해하기 쉬운가?
4. 창의성 (Creativity): 응답이 얼마나 창의적이고 독창적인가?
5. 유용성 (Usefulness): 응답이 실제로 얼마나 유용한가?

평가할 응답들:
GPT-4의 응답: {responses.get("GPT-4", "응답 없음")}
Claude의 응답: {responses.get("Claude", "응답 없음")}
Gemini의 응답: {responses.get("Gemini", "응답 없음")}

다음과 같은 JSON 형식으로만 답변해주세요:
{{
    "GPT-4": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 6, "유용성": 8}},
    "Claude": {{"정확성": 7, "완성도": 8, "명확성": 7, "창의성": 7, "유용성": 7}},
    "Gemini": {{"정확성": 6, "완성도": 7, "명확성": 8, "창의성": 5, "유용성": 6}}
}}
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates AI model responses and provides evaluations in valid JSON format only."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.7
        )
        
        # 응답에서 JSON 부분만 추출
        response_text = response.choices[0].message.content.strip()
        
        # JSON 형식이 아닌 텍스트 제거
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        response_text = response_text.strip()
        
        # 문자열을 딕셔너리로 변환
        import json
        evaluation_results = json.loads(response_text)
        
        return evaluation_results
    except Exception as e:
        st.error(f"평가 중 오류 발생: {str(e)}")
        return None

def evaluate_responses_gemini(responses, gemini_model):
    evaluation_prompt = f"""
다음 AI 모델들의 응답을 5가지 기준으로 평가해주세요. 
각 기준은 1-10점으로 평가하며, 반드시 아래 JSON 형식으로만 답변해주세요.

평가 기준:
1. 정확성 (Accuracy): 응답이 얼마나 사실에 기반하고 정확한가?
2. 완성도 (Completeness): 질문에 대해 얼마나 포괄적으로 답변했는가?
3. 명확성 (Clarity): 응답이 얼마나 명확하고 이해하기 쉬운가?
4. 창의성 (Creativity): 응답이 얼마나 창의적이고 독창적인가?
5. 유용성 (Usefulness): 응답이 실제로 얼마나 유용한가?

평가할 응답들:
GPT-4의 응답: {responses.get("GPT-4", "응답 없음")}
Claude의 응답: {responses.get("Claude", "응답 없음")}
Gemini의 응답: {responses.get("Gemini", "응답 없음")}

다음과 같은 JSON 형식으로만 답변해주세요:
{{
    "GPT-4": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 6, "유용성": 8}},
    "Claude": {{"정확성": 7, "완성도": 8, "명확성": 7, "창의성": 7, "유용성": 7}},
    "Gemini": {{"정확성": 6, "완성도": 7, "명확성": 8, "창의성": 5, "유용성": 6}}
}}
"""
    try:
        response = gemini_model.generate_content(evaluation_prompt)
        
        # 응답 텍스트 추출 방식 수정
        response_text = ""
        for part in response.parts:
            response_text += part.text
            
        # JSON 형식이 아닌 텍스트 제거
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        response_text = response_text.strip()
        
        # 문자열을 딕셔너리로 변환
        import json
        evaluation_results = json.loads(response_text)
        
        return evaluation_results
    except Exception as e:
        st.error(f"Gemini 평가 중 오류 발생: {str(e)}")
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

        # 모든 응답이 수집된 후 평가 진행
        if responses:
            st.markdown("---")
            
            # GPT-4 평가
            if openai_api_key:
                st.subheader("GPT-4의 성능 평가")
                evaluation_results_gpt = evaluate_responses(responses, openai.OpenAI(api_key=openai_api_key))
                if evaluation_results_gpt:
                    fig_gpt = create_radar_chart(evaluation_results_gpt)
                    st.plotly_chart(fig_gpt)
            
            # Gemini 평가
            if google_api_key:
                st.subheader("Gemini의 성능 평가")
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('gemini-pro')
                evaluation_results_gemini = evaluate_responses_gemini(responses, model)
                if evaluation_results_gemini:
                    fig_gemini = create_radar_chart(evaluation_results_gemini)
                    st.plotly_chart(fig_gemini)
            
            # 평가가 완료된 후에만 평가 기준 설명 표시
            if evaluation_results_gpt or evaluation_results_gemini:
                st.markdown("### 평가 기준 설명")
                st.markdown("""
                * **정확성 (Accuracy)**
                    * 응답이 사실에 기반하고 정확한 정보를 제공하는 정도
                    * 객관적 사실과의 일치성
                    * 정보의 신뢰성과 검증 가능성
                    * 오류나 잘못된 정보가 없는 정도

                * **완성도 (Completeness)**
                    * 주어진 질문에 대해 빠짐없이 포괄적으로 답변한 정도
                    * 질문의 모든 측면을 다루는 정도
                    * 답변의 깊이와 범위
                    * 추가 설명이나 예시의 충분성

                * **명확성 (Clarity)**
                    * 응답이 명확하고 이해하기 쉽게 작성된 정도
                    * 문장 구조의 명확성
                    * 전문 용어에 대한 적절한 설명
                    * 논리적 흐름과 구조화된 답변

                * **창의성 (Creativity)**
                    * 응답이 창의적이고 독창적인 관점을 제시하는 정도
                    * 새로운 시각이나 접근 방식 제시
                    * 다양한 관점의 통합
                    * 혁신적인 해결책이나 아이디어 제안

                * **유용성 (Usefulness)**
                    * 응답이 실제 사용자에게 실용적인 가치를 제공하는 정도
                    * 실제 적용 가능성
                    * 문제 해결에 도움이 되는 정도
                    * 실행 가능한 구체적 제안이나 해결책 제시
                """)
    else:
        st.warning("프롬프트를 입력해주세요.")