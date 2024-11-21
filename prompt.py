import streamlit as st
import openai
from anthropic import Anthropic
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os
import requests  # Perplexity API를 위해 추가

# 페이지 설정
st.set_page_config(page_title="LLM 모델 비교", layout="wide")
st.title("LLM 모델 비교 v1(241110)")

# 사이드바에 API 키 입력 필드 추가
with st.sidebar:
    st.header("API 키 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    google_api_key = st.text_input("Google API Key", type="password")
    perplexity_api_key = st.text_input("Perplexity API Key", type="password")  # 추가

# 프롬프트 입력
user_prompt = st.text_area("프롬프트를 입력하세요:", height=100)

if st.button("생성"):
    if user_prompt:
        # 2x2 그리드로 변경
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("GPT-4")
            if openai_api_key:
                try:
                    # 스트리밍을 위한 빈 컨테이너
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
                except Exception as e:
                    st.error(f"OpenAI 에러: {str(e)}")
            else:
                st.warning("OpenAI API 키를 입력해주세요.")

        with row1_col2:
            st.subheader("Claude")
            if anthropic_api_key:
                try:
                    # 스트리밍을 위한 빈 컨테이너
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    client = Anthropic(api_key=anthropic_api_key)
                    message = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        messages=[
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        stream=True
                    )
                    
                    for chunk in message:
                        if chunk.delta.text:
                            full_response += chunk.delta.text
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Anthropic 에러: {str(e)}")
            else:
                st.warning("Anthropic API 키를 입력해주세요.")

        with row2_col1:
            st.subheader("Gemini Pro")
            if google_api_key:
                try:
                    # 스트리밍을 위한 빈 컨테이너
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
                except Exception as e:
                    st.error(f"Google AI 에러: {str(e)}")
            else:
                st.warning("Google API 키를 입력해주세요.")

        with row2_col2:
            st.subheader("Perplexity")
            if perplexity_api_key:
                try:
                    # 스트리밍을 위한 빈 컨테이너
                    message_placeholder = st.empty()
                    
                    headers = {
                        "Authorization": f"Bearer {perplexity_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "pplx-7b-online",  # or "pplx-70b-online"
                        "messages": [{"role": "user", "content": user_prompt}],
                        "stream": True
                    }
                    
                    response = requests.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=headers,
                        json=data,
                        stream=True
                    )
                    
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            if b"content" in line:
                                content = line.decode().split("content\":\"")[1].split("\"")[0]
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Perplexity 에러: {str(e)}")
            else:
                st.warning("Perplexity API 키를 입력해주세요.")
    else:
        st.warning("프롬프트를 입력해주세요.")