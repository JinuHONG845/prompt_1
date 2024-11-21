import streamlit as st
import openai
from anthropic import Anthropic
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

# API 키 가져오기
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# OpenAI 클라이언트 설정
client = OpenAI(api_key=openai_api_key)

# 테스트를 위한 디버그 출력 (개발 시에만 사용)
print(f"OpenAI API 키 확인: {openai_api_key[:10]}...") 

# OpenAI API 호출 예시
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
except Exception as e:
    print(f"OpenAI 에러 상세: {str(e)}")
    raise e

# 세션 상태에 저장
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = openai_api_key
if 'anthropic_api_key' not in st.session_state:
    st.session_state['anthropic_api_key'] = anthropic_api_key
if 'google_api_key' not in st.session_state:
    st.session_state['google_api_key'] = google_api_key

# API 키가 설정되었는지 확인 (디버깅용)
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# 페이지 설정
st.set_page_config(page_title="LLM 모델 비교", layout="wide")
st.title("LLM 모델 비교 v1(241111)")

# 프롬프트 입력
user_prompt = st.text_area("프롬프트를 입력하세요:", 
    height=100,
    placeholder="질문을 입력해주세요...",
    help="여러 줄의 텍스트를 입력할 수 있습니다.")

# 평가 함수 추가
def evaluate_responses(responses, openai_client):
    if not responses:
        st.error("평가할 응답이 없습니다.")
        return None
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
        if hasattr(response, 'candidates'):
            response_text = response.candidates[0].content.parts[0].text
        else:
            response_text = response.parts[0].text if hasattr(response, 'parts') else ""
            
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
        st.error(f"응답 구조: {type(response)}")
        st.error(f"응답 내용: {response_text if 'response_text' in locals() else '응답 없음'}")
        return None

# 상단에 캐싱 데코레이터 추가
@st.cache_data(ttl=3600)
def create_radar_chart(evaluation_results):
    categories = ['정확성', '완성도', '명확성', '창의성', '유용성']
    
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
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                ticktext=['0', '2', '4', '6', '8', '10'],
                tickvals=[0, 2, 4, 6, 8, 10]
            )),
        showlegend=True,
        title="LLM 모델 성능 비교",
        height=500  # 차트 크기 조정
    )
    
    return fig

if st.button("생성"):
    if user_prompt:
        # 3개의 컬럼으로 변경
        col1, col2, col3 = st.columns(3)
        
        # 응답을 저장할 딕셔너리 초기화
        responses = {}
        st.session_state['responses'] = responses  # 세션에 저장
        
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
            st.subheader("Claude 3.5")
            if anthropic_api_key:
                try:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    client = Anthropic(api_key=anthropic_api_key)
                    message = client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": user_prompt}],
                        stream=True
                    )
                    
                    for chunk in message:
                        if chunk.delta.text:
                            full_response += chunk.delta.text
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    responses["Claude"] = full_response
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
                
                # 3개의 컬럼 생성
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    * **정확성 (Accuracy)**
                        * 사실에 기반한 정확한 정보 제공
                        * 객관적 사실과의 일치도
                        * 정보의 신뢰성

                    &nbsp;

                    * **완성도 (Completeness)**
                        * 질문에 대한 포괄적 답변
                        * 모든 중요 측면 포함
                        * 충분한 설명과 예시
                    """)

                with col2:
                    st.markdown("""
                    * **명확성 (Clarity)**
                        * 이해하기 쉬운 설명
                        * 논리적인 구조
                        * 명확한 표현

                    &nbsp;

                    * **창의성 (Creativity)**
                        * 독창적인 관점
                        * 새로운 접근 방식
                        * 혁신적인 아이디어
                    """)

                with col3:
                    st.markdown("""
                    * **유용성 (Usefulness)**
                        * 실용적 가치
                        * 실제 적용 가능성
                        * 문제 해결 기여도
                    """)
    else:
        st.warning("프롬프트를 입력해주세요.")