import streamlit as st
import openai
from anthropic import Anthropic
import google.generativeai as genai
import plotly.graph_objects as go
import json
from typing import Dict, Optional
import re

# 페이지 설정
st.set_page_config(page_title="LLM 모델 비교", layout="wide")

class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def validate_api_key(self) -> bool:
        raise NotImplementedError

    def generate_response(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def validate_api_key(self) -> bool:
        return self.api_key.startswith('sk-')

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            client = openai.OpenAI(api_key=self.api_key)
            message_placeholder = st.empty()
            full_response = ""
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
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
            st.error(f"GPT-4 에러: {str(e)}")
            return None

class ClaudeClient(LLMClient):
    def validate_api_key(self) -> bool:
        return bool(self.api_key)

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            client = Anthropic(api_key=self.api_key)
            message_placeholder = st.empty()
            full_response = ""
            
            # 스트리밍 없이 먼저 시도
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=4096,
                stream=False  # 스트리밍 비활성화
            )
            
            # 응답 텍스트 추출
            if response.content and len(response.content) > 0:
                full_response = response.content[0].text
                message_placeholder.markdown(full_response)
                return full_response
            else:
                st.error("Claude가 응답을 생성하지 못했습니다.")
                return None
            
        except Exception as e:
            st.error(f"Claude 에러 상세: {str(e)}")
            # 에러 타입도 출력
            st.error(f"에러 타입: {type(e).__name__}")
            return None

class GeminiClient(LLMClient):
    def validate_api_key(self) -> bool:
        return bool(self.api_key)  # Gemini API 키는 특별한 형식이 없음

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            message_placeholder = st.empty()
            full_response = ""
            
            response = model.generate_content(prompt, stream=True)
            
            for chunk in response:
                for part in chunk.parts:
                    if part.text:
                        full_response += part.text
                        message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            return full_response
        except Exception as e:
            st.error(f"Gemini 에러: {str(e)}")
            return None

class ModelEvaluator:
    CRITERIA = {
        "정확성": """• 제공된 정보의 사실적 정확성과 신뢰성을 평가
• 객관적 사실과 데이터에 기반한 응답의 정확도를 측정""",
        
        "완성도": """• 주제에 대한 포괄적인 설명과 필요한 모든 측면을 다루는 정도
• 누락된 정보 없이 질문의 모든 부분에 대한 충분한 답변 제공""",
        
        "명확성": """• 설명의 논리적 구조와 이해하기 쉬운 표현 방식
• 복잡한 개념을 명확하고 체계적으로 전달하는 능력""",
        
        "창의성": """• 새롭고 독창적인 관점과 해결방안 제시 능력
• 기존 아이디어를 혁신적으로 결합하고 발전시키는 정도""",
        
        "유용성": """• 실제 상황에서의 적용 가능성과 실용적 가치
• 사용자가 즉시 활용하고 실천할 수 있는 실용적인 정보 제공"""
    }

    def evaluate_with_model(self, responses: Dict[str, str], model_name: str, client: LLMClient) -> Optional[Dict]:
        try:
            evaluation_prompt = f"""
            다음 AI 모델들의 응답을 5가지 기준(정확성, 완성도, 명확성, 창의성, 유용성)으로 1-10점으로 평가해 JSON 형식으로만 답변하세요.
            반드시 아래 형식으로만 응답하세요:
            {{
                "ChatGPT 4O": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 8, "유용성": 7}},
                "Claude 3.5": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 8, "유용성": 7}},
                "Gemini Pro": {{"정확성": 8, "완성도": 7, "명확성": 9, "창의성": 8, "유용성": 7}}
            }}

            평가할 응답:
            ChatGPT 4O: {responses.get("GPT-4", "응답 없음")}
            Claude 3.5: {responses.get("Claude", "응답 없음")}
            Gemini Pro: {responses.get("Gemini", "응답 없음")}
            """
            
            if model_name == "GPT-4":
                response = client.generate_response(evaluation_prompt)
            elif model_name == "Claude":
                response = client.generate_response(evaluation_prompt)
            else:  # Gemini
                response = client.generate_response(evaluation_prompt)
            
            if response:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            return None
            
        except Exception as e:
            st.error(f"평가 중 오류 발생: {str(e)}")
            return None

def main():
    st.title("LLM 모델 비교 v2")
    
    # 사이드바에 API 키 입력 필드 추가
    with st.sidebar:
        st.header("API 키 설정")
        
        # API 키 입력 (빈 값으로 시작)
        openai_api_key = st.text_input(
            "OpenAI API 키",
            placeholder="sk-로 시작하는 키를 입력하세요",
            type="password"
        )
        
        anthropic_api_key = st.text_input(
            "Anthropic API 키",
            placeholder="sk-ant-로 시작하는 키를 입력하세요",
            type="password"
        )
        
        google_api_key = st.text_input(
            "Google API 키",
            placeholder="Google API 키를 입력하세요",
            type="password"
        )
        
        st.markdown("---")
        st.markdown("""
        ### API 키 발급 방법
        1. OpenAI API 키: [OpenAI 플랫폼](https://platform.openai.com/api-keys)
        2. Anthropic API 키: [Anthropic 콘솔](https://console.anthropic.com/)
        3. Google API 키: [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)
    
    # API 키 입력 확��
    if not openai_api_key or not anthropic_api_key or not google_api_key:
        st.warning("사이드바에서 모든 API 키를 입력해주세요.")
        st.stop()
    
    # API 클라이언트 초기화
    openai_client = OpenAIClient(openai_api_key)
    claude_client = ClaudeClient(anthropic_api_key)
    gemini_client = GeminiClient(google_api_key)
    
    # API 키 검증
    if not openai_client.validate_api_key():
        st.error("유효하지 않은 OpenAI API 키입니다.")
    if not claude_client.validate_api_key():
        st.error("유효하지 않은 Anthropic API 키입니다.")
    if not gemini_client.validate_api_key():
        st.error("유효하지 않은 Gemini API 키입니다.")
    
    # 프롬프트 입력
    user_prompt = st.text_area(
        "프롬프트를 입력하세요:",
        height=100,
        placeholder="분석하고 싶은 질문을 입력해주세요..."
    )
    
    if st.button("응답 생성"):
        if user_prompt:
            col1, col2, col3 = st.columns(3)
            responses = {}
            
            with col1:
                st.subheader("ChatGPT 4O")
                responses["GPT-4"] = openai_client.generate_response(user_prompt)
            
            with col2:
                st.subheader("Claude 3.5")
                responses["Claude"] = claude_client.generate_response(user_prompt)
            
            with col3:
                st.subheader("Gemini Pro")
                responses["Gemini"] = gemini_client.generate_response(user_prompt)
            
            # 각 모델별 평가 수행
            if all(responses.values()):
                st.markdown("---")
                st.subheader("모델별 평가 결과")
                
                evaluator = ModelEvaluator()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### ChatGPT 4O의 평가")
                    gpt_evaluation = evaluator.evaluate_with_model(responses, "GPT-4", openai_client)
                    if gpt_evaluation:
                        fig = evaluator.create_radar_chart(gpt_evaluation)
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### Claude 3.5의 평가")
                    claude_evaluation = evaluator.evaluate_with_model(responses, "Claude", claude_client)
                    if claude_evaluation:
                        fig = evaluator.create_radar_chart(claude_evaluation)
                        st.plotly_chart(fig, use_container_width=True)

                with col3:
                    st.markdown("### Gemini Pro의 평가")
                    gemini_evaluation = evaluator.evaluate_with_model(responses, "Gemini", gemini_client)
                    if gemini_evaluation:
                        fig = evaluator.create_radar_chart(gemini_evaluation)
                        st.plotly_chart(fig, use_container_width=True)
                
                # 평가 기준 설명 자동 표시
                st.markdown("---")
                st.markdown("### 평가 기준 설명")
                for criterion, description in ModelEvaluator.CRITERIA.items():
                    st.markdown(f"#### {criterion}")
                    st.markdown(description)
        else:
            st.warning("프롬프트를 입력해주세요.")

if __name__ == "__main__":
    main()