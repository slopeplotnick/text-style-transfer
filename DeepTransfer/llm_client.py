import requests
import json
from typing import Optional


def clean_llm_output(text: str) -> str:
    """清理 LLM 输出，移除空行，将多行合并为单行"""
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    return ' '.join(non_empty_lines)


class LLMClient:
    """
    Simple client for OpenAI-compatible API.
    """
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            raw_content = result['choices'][0]['message']['content']
            return clean_llm_output(raw_content)
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return f"[Error generating text: {e}]"