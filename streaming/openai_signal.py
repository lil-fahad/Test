
import openai
import streamlit as st

# Load API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_summary(features):
    prompt = f"""You are a financial analyst. Summarize the current market signal:
    Features: {features}
    Return an actionable insight in one sentence."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=100
    )
    
    return response['choices'][0]['message']['content']
