# import requests

# OLLAMA_URL = "http://localhost:11434/api/chat"


# def call_ollama_model(
#     model: str,
#     system_prompt: str,
#     user_prompt: str,
#     temperature: float = 0.0,
#     num_predict: int = 256,
# ) -> str:
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "stream": False,
#         "options": {
#             "temperature": temperature,
#             "num_predict": num_predict,
#         },
#     }

#     try:
#         resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
#         resp.raise_for_status()
#     except requests.exceptions.RequestException as e:
#         raise RuntimeError(f"Ollama request failed: {e}")

#     try:
#         data = resp.json()
#     except ValueError:
#         raise RuntimeError(f"Invalid JSON response: {resp.text}")

#     # ✅ Safe extraction
#     if "message" in data and "content" in data["message"]:
#         return data["message"]["content"].strip()

#     # fallback (rare cases)
#     if "response" in data:
#         return data["response"].strip()

#     raise RuntimeError(f"Unexpected response format: {data}")

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_ollama_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    num_predict: int = 256,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()

    if isinstance(data.get("message"), dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content.strip()

    if isinstance(data.get("response"), str):
        return data["response"].strip()

    raise RuntimeError(f"Unexpected response format: {data}")