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

MODEL_COLORS = {
    'ChatGPT 4O': '#00A67E',
    'Claude 3.5': '#000000',
    'Gemini Pro': '#1A73E8'
}

MODEL_NAMES = {
    'GPT-4': 'ChatGPT 4O',
    'Claude': 'Claude 3.5',
    'Gemini': 'Gemini Pro'
}

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
            
            # 스트리밍 모드로 변경
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=4096,
                stream=True  # 스트리밍 활성화
            )
            
            # 스트리밍 응답 처리
            for chunk in response:
                if chunk.content and len(chunk.content) > 0:
                    content_text = chunk.content[0].text
                    full_response += content_text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            st.error(f"Claude 에러 상세: {str(e)}")
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
            다음 응답들을 5가지 기준으로 평가하여 1-10점으로 점수를 매겨주세요.
            오직 JSON 형식으로만 응답해주세요.

            [평가할 응답들]
            ChatGPT 4O의 응답:
            {responses.get("ChatGPT 4O", "응답 없음")}

            Claude 3.5의 응답:
            {responses.get("Claude 3.5", "응답 없음")}

            Gemini Pro의 응답:
            {responses.get("Gemini Pro", "응답 없음")}

            [평가 기준]
            - 정확성: 정보의 사실성과 신뢰성
            - 완성도: 응답의 포괄성과 충실성
            - 명확성: 설명의 논리성과 이해도
            - 창의성: 독창적 관점과 해결방안
            - 유용성: 실용적 가치와 적용성

            [응답 형식]
            {{
                "{model_name}": {{
                    "정확성": (1-10점),
                    "완성도": (1-10점),
                    "명확성": (1-10점),
                    "창의성": (1-10점),
                    "유용성": (1-10점)
                }}
            }}
            """
            
            response = client.generate_response(evaluation_prompt)
            if response:
                # JSON 형식 추출
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            return None
            
        except Exception as e:
            st.error(f"평가 중 오류 발생: {str(e)}")
            return None

    def create_radar_chart(self, evaluation_data: Dict) -> go.Figure:
        try:
            categories = ['정확성', '완성도', '명확성', '창의성', '유용성']
            fig = go.Figure()
            
            # 평가 데이터에서 모델 이름과 점수 추출
            model_name = list(evaluation_data.keys())[0]
            scores = evaluation_data[model_name]
            
            # 점수 리스트 생성
            values = [scores[cat] for cat in categories]
            
            # 레이더 차트 생성
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                name=model_name,
                fill='toself',
                line_color=MODEL_COLORS.get(model_name, '#000000')
            ))
            
            # 차트 레이아웃 설정
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=True,
                title=f"{model_name} 평가 결과",
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"차트 생성 중 오류: {str(e)}")
            return None

@st.cache_data(ttl=3600)  # 1시간 캐시
def evaluate_responses(responses: Dict[str, str], model_name: str, client: LLMClient) -> Optional[Dict]:
    evaluator = ModelEvaluator()
    return evaluator.evaluate_with_model(responses, model_name, client)

def main():
    st.title("LLM 모델 비교 v2")
    
    with st.sidebar:
        st.header("API 키 설정")
        with st.expander("API 키 입력", expanded=True):
            # API 키 입력 필드들...
        
        with st.expander("API 키 발급 방법"):
            st.markdown("""
            1. [OpenAI API 키](https://platform.openai.com/api-keys)
            2. [Anthropic API 키](https://console.anthropic.com/)
            3. [Google API 키](https://makersuite.google.com/app/apikey)
            """)
    
    # 메인 영역
    st.markdown("""
    ### 사용 방법
    1. 사이드바에서 각 모델의 API 키를 입력하세요
    2. 분석하고 싶은 질문을 입력하세요
    3. '응답 생성' 버튼을 클릭하세요
    """)

if __name__ == "__main__":
    main()