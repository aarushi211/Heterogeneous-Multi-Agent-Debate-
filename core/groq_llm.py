# import os
# import ollama


# def call_model(
#     model: str,
#     system_prompt: str,
#     user_prompt: str,
#     *,
#     num_predict: int | None = None,
#     temperature: float = 0.2,
# ) -> str:
#     options: dict = {}

#     env_cap = os.environ.get("OLLAMA_NUM_PREDICT", "").strip()
#     if env_cap.isdigit():
#         options["num_predict"] = int(env_cap)

#     if num_predict is not None:
#         options["num_predict"] = num_predict

#     options["temperature"] = temperature

#     combined_prompt = f"""
# SYSTEM:
# {system_prompt}

# USER:
# {user_prompt}
# """

#     try:
#         response = ollama.chat(
#             model=model,
#             messages=[{"role": "user", "content": combined_prompt}],
#             options=options,
#         )
#         return response["message"]["content"].strip()
#     except Exception as e:
#         return f"[ERROR] {str(e)}"

# core/groq_llm.py
import os
from groq import Groq

_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def call_groq_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content or ""