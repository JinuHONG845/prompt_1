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
        return self.api_key.startswith('sk-ant-')

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            client = Anthropic(api_key=self.api_key)
            message_placeholder = st.empty()
            full_response = ""
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'content') and chunk.content:
                    for content_block in chunk.content:
                        if content_block.text:
                            full_response += content_block.text
                            message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            return full_response
        except Exception as e:
            st.error(f"Claude 에러: {str(e)}")
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
        "정확성": "제공된 정보의 사실적 정확성과 신뢰성",
        "완성도": "응답의 포괄성과 주제에 대한 충분한 설명",
        "명확성": "설명의 논리적 구조와 이해하기 쉬운 표현",
        "창의성": "독창적인 관점과 혁신적인 해결방안 제시",
        "유용성": "실제 적용 가능성과 실용적 가치"
    }

    def __init__(self, gemini_client: GeminiClient):
        self.evaluator = gemini_client

    def evaluate(self, responses: Dict[str, str]) -> Optional[Dict]:
        try:
            evaluation_prompt = self._create_evaluation_prompt(responses)
            response = self.evaluator.generate_response(evaluation_prompt)
            
            if response:
                return self._parse_evaluation_response(response)
            return None
        except Exception as e:
            st.error(f"평가 중 오류 발생: {str(e)}")
            return None

    def _create_evaluation_prompt(self, responses: Dict[str, str]) -> str:
        return f"""
        다음 AI 모델들의 응답을 5가지 기준(정확성, 완성도, 명확성, 창의성, 유용성)으로 1-10점으로 평가해 JSON 형식으로만 답변하세요.
        반드시 아래 형식으로만 응답하세요:
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

    def _parse_evaluation_response(self, response: str) -> Optional[Dict]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return None
        except json.JSONDecodeError:
            return None

    def create_radar_chart(self, evaluation_results: Dict) -> go.Figure:
        categories = list(self.CRITERIA.keys())
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
    
    # API 키 입력 확인
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
        placeholder="질문을 입력해주세요..."
    )
    
    if st.button("생성"):
        if user_prompt:
            col1, col2, col3 = st.columns(3)
            responses = {}
            
            with col1:
                st.subheader("GPT-4")
                responses["GPT-4"] = openai_client.generate_response(user_prompt)
            
            with col2:
                st.subheader("Claude 3")
                responses["Claude"] = claude_client.generate_response(user_prompt)
            
            with col3:
                st.subheader("Gemini Pro")
                responses["Gemini"] = gemini_client.generate_response(user_prompt)
            
            # 평가 수행
            if all(responses.values()):
                st.markdown("---")
                st.subheader("모델 평가")
                
                evaluator = ModelEvaluator(gemini_client)
                evaluation = evaluator.evaluate(responses)
                
                if evaluation:
                    fig = evaluator.create_radar_chart(evaluation)
                    st.plotly_chart(fig)
                    
                    with st.expander("평가 기준 설명"):
                        for criterion, description in ModelEvaluator.CRITERIA.items():
                            st.markdown(f"**{criterion}**: {description}")
        else:
            st.warning("프롬프트를 입력해주세요.")

if __name__ == "__main__":
    main()