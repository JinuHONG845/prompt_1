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
            반드시 아래 JSON 형식으로만 응답해주세요. 다른 설명은 포함하지 마세요.

            [평가할 응답들]
            ChatGPT 4O의 응답:
            {responses.get("ChatGPT 4O", "응답 없음")}

            Claude 3.5의 응답:
            {responses.get("Claude 3.5", "응답 없음")}

            Gemini Pro의 응답:
            {responses.get("Gemini Pro", "응답 없음")}

            [평가 기준]
            - 정확성: 정보의 사실성과 신뢰성 (1-10점)
            - 완성도: 응답의 포괄성과 충실성 (1-10점)
            - 명확성: 설명의 논리성과 이해도 (1-10점)
            - 창의성: 독창적 관점과 해결방안 (1-10점)
            - 유용성: 실용적 가치와 적용성 (1-10점)

            [응답 형식]
            {{
                "{model_name}": {{
                    "정확성": 8,
                    "완성도": 7,
                    "명확성": 9,
                    "창의성": 6,
                    "유용성": 8
                }}
            }}
            """
            
            response = client.generate_response(evaluation_prompt)
            if response:
                try:
                    # JSON 문자열 추출 및 파싱
                    json_str = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', response)
                    if json_str:
                        evaluation_data = json.loads(json_str.group())
                        st.write("파싱된 평가 데이터:", evaluation_data)  # 디버깅용
                        return evaluation_data
                    else:
                        st.error("JSON 형식을 찾을 수 없습니다.")
                        return None
                except json.JSONDecodeError as e:
                    st.error(f"JSON 파싱 오류: {str(e)}")
                    return None
            return None
            
        except Exception as e:
            st.error(f"평가 중 오류 발생: {str(e)}")
            return None

    def create_radar_chart(self, evaluation_data: Dict) -> Optional[go.Figure]:
        try:
            st.write("입력 데이터:", evaluation_data)  # 디버깅용
            
            if not evaluation_data:
                st.error("평가 데이터가 비어있습니다.")
                return None
            
            categories = ['정확성', '완성도', '명확성', '창의성', '유용성']
            fig = go.Figure()
            
            # 첫 번째 모델의 데이터 추출
            model_name = list(evaluation_data.keys())[0]
            scores = evaluation_data[model_name]
            
            # 점수 리스트 생성
            values = [float(scores[cat]) for cat in categories]
            
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
            st.write("오류 발생 시 데이터:", evaluation_data)  # 디버깅용
            return None

@st.cache_data(ttl=3600)  # 1시간 캐시
def evaluate_responses(responses: Dict[str, str], model_name: str, client: LLMClient) -> Optional[Dict]:
    evaluator = ModelEvaluator()
    return evaluator.evaluate_with_model(responses, model_name, client)

def main():
    st.title("LLM 모델 비교 v2")
    
    with st.sidebar:
        st.header("API 키 설정")
        openai_api_key = st.text_input("OpenAI API 키", type="password", placeholder="sk-...")
        anthropic_api_key = st.text_input("Anthropic API 키", type="password", placeholder="sk-ant-...")
        google_api_key = st.text_input("Google API 키", type="password")
        
        with st.expander("API 키 발급 방법"):
            st.markdown("""
            1. [OpenAI API 키](https://platform.openai.com/api-keys)
            2. [Anthropic API 키](https://console.anthropic.com/)
            3. [Google API 키](https://makersuite.google.com/app/apikey)
            """)
    
    # API 키 검증
    if not all([openai_api_key, anthropic_api_key, google_api_key]):
        st.warning("모든 API 키를 입력해주세요.")
        st.stop()
    
    # 클라이언트 초기화
    openai_client = OpenAIClient(openai_api_key)
    claude_client = ClaudeClient(anthropic_api_key)
    gemini_client = GeminiClient(google_api_key)
    
    # 프롬프트 입력
    user_prompt = st.text_area("프롬프트를 입력하세요:", height=100)
    
    if st.button("응답 생성"):
        if user_prompt:
            col1, col2, col3 = st.columns(3)
            responses = {}
            
            # 각 모델의 응답 생성
            with col1:
                st.subheader("ChatGPT 4O")
                responses["ChatGPT 4O"] = openai_client.generate_response(user_prompt)
            
            with col2:
                st.subheader("Claude 3.5")
                responses["Claude 3.5"] = claude_client.generate_response(user_prompt)
            
            with col3:
                st.subheader("Gemini Pro")
                responses["Gemini Pro"] = gemini_client.generate_response(user_prompt)
            
            # 모델 평가 수행
            if all(responses.values()):
                st.markdown("---")
                st.subheader("모델별 평가 결과")
                
                evaluator = ModelEvaluator()
                
                # 각 모델의 평가 결과
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### ChatGPT 4O의 평가")
                    try:
                        gpt_evaluation = evaluator.evaluate_with_model(responses, "ChatGPT 4O", openai_client)
                        st.write("평가 데이터:", gpt_evaluation)  # 디버깅용
                        if gpt_evaluation and isinstance(gpt_evaluation, dict):
                            fig = evaluator.create_radar_chart(gpt_evaluation)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"ChatGPT 4O 평가 중 오류: {str(e)}")
                
                with col2:
                    st.markdown("### Claude 3.5의 평가")
                    try:
                        claude_evaluation = evaluator.evaluate_with_model(responses, "Claude 3.5", claude_client)
                        st.write("평가 데이터:", claude_evaluation)  # 디버깅용
                        if claude_evaluation and isinstance(claude_evaluation, dict):
                            fig = evaluator.create_radar_chart(claude_evaluation)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Claude 3.5 평가 중 오류: {str(e)}")
                
                with col3:
                    st.markdown("### Gemini Pro의 평가")
                    try:
                        gemini_evaluation = evaluator.evaluate_with_model(responses, "Gemini Pro", gemini_client)
                        st.write("평가 데이터:", gemini_evaluation)  # 디버깅용
                        if gemini_evaluation and isinstance(gemini_evaluation, dict):
                            fig = evaluator.create_radar_chart(gemini_evaluation)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Gemini Pro 평가 중 오류: {str(e)}")
                
                # 평가 기준 설명
                st.markdown("---")
                st.markdown("### 평가 기준 설명")
                for criterion, description in ModelEvaluator.CRITERIA.items():
                    st.markdown(f"#### {criterion}")
                    st.markdown(description)
        else:
            st.warning("프롬프트를 입력해주세요.")

if __name__ == "__main__":
    main()