# core/gemini_llm.py
import os
from google import genai
from google.genai import types

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_gemini_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 80,
) -> str:
    resp = _client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        ),
    )
    return (resp.text or "").strip()