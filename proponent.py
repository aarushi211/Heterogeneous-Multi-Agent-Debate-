import json
from typing import Any, Optional

from core.ollama_llm import call_ollama_model  # adjust import path if needed

PROPONENT_MODEL = "gemma3:12b"

SYSTEM_PROMPT = """

You are a careful math solver.

Rules:
- Extract the given quantities first.
- Use only the information in the question.
- Do not write long explanations.
- Compute in at most 6 short steps.
- If someone drives away from home and later heads back toward home, track direction: motion toward home
  shrinks distance from home; motion away grows it. Do not add all segment lengths as if they were
  all in the same direction.
- For questions that ask how far someone is from home (or from a fixed place) at the end, compute net
  distance from that place after each leg — not the sum of every trip segment unless every segment
  moves farther from that place.
- After turning toward home: if D is miles from home before a leg and the leg covers L miles toward
  home, then afterward distance from home is D - L (subtraction), never D + L. Standstill adds 0
  toward home, so D is unchanged.
- For percent increases or decreases ("by 150%", "increased by 20%", etc.), name what the percentage
  is **of** (the base). The base is whatever the wording ties the change to (e.g. "value of the
  house" → the house's stated value), not automatically purchase price plus every later expense unless
  the problem explicitly says the percent applies to that combined total.
- Profit or net gain is usually final value minus **all** money spent; that total cost can differ from
  the **percentage base** used for a "value increased by P%" phrase—keep both straight.
- Identify what the question asks for: **total** for the whole group vs **per** person/animal/unit, time
  vs distance, etc. Your **Final answer** must match that target (e.g. total cups for the meal, not cups
  per chicken, unless the question asks per-chicken).
- If the problem states a uniform rate for each member (e.g. each chicken 3 cups/day), then
  (number of members × that rate) is the correct total daily amount unless the text says otherwise.
  Do not abandon that for a wrong quantity type when responding to a challenge.
- If a prober asks a why-question, answer that specific question directly before giving the math steps.
- Do not repeat the same answer verbatim.
- Return ONLY a JSON object with exactly these keys:
  - "steps": an array of short strings
  - "final_answer": a number

Do NOT include any text outside the JSON.

""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return valid JSON: {text}")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from model")

    return parsed


def ask_proponent(
    question: str,
    critique: str | None = None,
    previous_answer: dict[str, Any] | None = None,
    transcript: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    parts = [f"Original question:\n{question}"]

    if previous_answer is not None:
        parts.append(
            "Previous proponent answer:\n"
            + json.dumps(previous_answer, ensure_ascii=False, indent=2)
        )

    if critique:
      parts.append(
        "The prober asked:\n"
        f"{critique}\n\n"
        "Respond directly to that question, not just the original problem."
    )

    if transcript:
        parts.append(
            "Transcript so far:\n"
            + "\n".join(f"{item['role'].upper()}: {item['text']}" for item in transcript)
        )

    parts.append('Solve the original question and return JSON with keys "steps" and "final_answer".')
    user_prompt = "\n\n".join(parts)

    raw = call_ollama_model(
        PROPONENT_MODEL,
        SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        num_predict=256,
    )

    return _extract_json(raw)


# import os
# from core.groq_llm import call_groq_model

# PROPONENT_MODEL = os.environ.get("PROPONENT_MODEL", "llama-3.3-70b-versatile")

# SYSTEM_PROMPT = """
# You are a careful math solver.

# Rules:
# - Solve only the original question.
# - Use only the information in the question.
# - Extract the given quantities first.
# - Do not write long explanations.
# - Keep the solution to at most 6 short steps.
# - Stay on the same quantity the question asks for.
# - Do not switch from total to per-unit, or from distance to total distance, unless the question asks for that.
# - If the scenario involves traveling toward or away from home, track distance from home after each leg.
# - If the problem says a value changed by a percentage, identify the correct base before calculating the increase or decrease.
# - Do not introduce new facts, new assumptions, or a different interpretation of the question.
# - Output valid JSON only.
# - Do not wrap the JSON in markdown.

# Return this exact schema:
# {
#   "given": [...],
#   "steps": [...],
#   "result": <number>,
#   "final_answer": "<number>"
# }
# """

# def ask_proponent(question: str) -> str:
#     user_prompt = f"""
# Original question:
# {question}

# Solve the problem and return only JSON.
# """
#     return call_groq_model(
#         PROPONENT_MODEL,
#         SYSTEM_PROMPT,
#         user_prompt,
#         temperature=0.0,
#     )

# import json
# import os
# from typing import Any

# from core.groq_llm import call_groq_model

# PROPONENT_MODEL = os.environ.get("PROPONENT_MODEL", "deepseek-reasoner")

# SYSTEM_PROMPT = """
# Explain answer directly to the question asked by prober, don't repeat same answer for all questions.
# You are a careful math solver.

# Rules:
# - Extract the given quantities first.
# - Use only the information in the question.
# - Do not write long explanations.
# - Compute in at most 6 short steps.
# - If someone drives away from home and later heads back toward home, track direction: motion toward home
#   shrinks distance from home; motion away grows it. Do not add all segment lengths as if they were
#   all in the same direction.
# - For questions that ask how far someone is from home (or from a fixed place) at the end, compute net
#   distance from that place after each leg — not the sum of every trip segment unless every segment
#   moves farther from that place.
# - After turning toward home: if D is miles from home before a leg and the leg covers L miles toward
#   home, then afterward distance from home is D - L (subtraction), never D + L. Standstill adds 0
#   toward home, so D is unchanged.
# - For percent increases or decreases ("by 150%", "increased by 20%", etc.), name what the percentage
#   is **of** (the base). The base is whatever the wording ties the change to (e.g. "value of the
#   house" → the house's stated value), not automatically purchase price plus every later expense unless
#   the problem explicitly says the percent applies to that combined total.
# - Profit or net gain is usually final value minus **all** money spent; that total cost can differ from
#   the **percentage base** used for a "value increased by P%" phrase—keep both straight.
# - Identify what the question asks for: **total** for the whole group vs **per** person/animal/unit, time
#   vs distance, etc. Your **Final answer** must match that target (e.g. total cups for the meal, not cups
#   per chicken, unless the question asks per-chicken).
# - If the problem states a uniform rate for each member (e.g. each chicken 3 cups/day), then
#   (number of members × that rate) is the correct total daily amount unless the text says otherwise.
#   Do not abandon that for a wrong quantity type when responding to a challenge.
# - End with exactly one line:
# Final answer: <number>
# If a critique is provided:
# - identify the exact step being challenged
# - answer the critique directly in a separate field
# - only change the solution if the critique reveals a real error
# - do not repeat the entire answer verbatim unless necessary

# Return this exact schema:
# {
#   "given": [...],
#   "steps": [...],
#   "result": <number>,
#   "final_answer": "<number>"
# }
# """.strip()


# def _extract_json(text: str) -> dict[str, Any]:
#     text = text.strip()

#     try:
#         parsed = json.loads(text)
#         if isinstance(parsed, dict):
#             return parsed
#     except json.JSONDecodeError:
#         pass

#     start = text.find("{")
#     end = text.rfind("}")
#     if start == -1 or end == -1 or end <= start:
#         raise ValueError(f"Model did not return valid JSON: {text}")

#     parsed = json.loads(text[start : end + 1])
#     if not isinstance(parsed, dict):
#         raise ValueError("Expected JSON object from model")

#     return parsed


# def ask_proponent(
#     question: str,
#     critique: str | None = None,
#     previous_answer: dict[str, Any] | None = None,
#     transcript: list[dict[str, str]] | None = None,
# ) -> dict[str, Any]:
#     parts = [f"Original question:\n{question}"]

#     if previous_answer is not None:
#         parts.append(
#             "Previous proponent answer:\n"
#             + json.dumps(previous_answer, ensure_ascii=False, indent=2)
#         )

#     if critique:
#         parts.append(f"Prober critique:\n{critique}")

#     if transcript:
#         parts.append(
#             "Transcript so far:\n"
#             + "\n".join(f"{item['role'].upper()}: {item['text']}" for item in transcript)
#         )

#     parts.append("Solve the original question carefully and return only JSON.")
#     user_prompt = "\n\n".join(parts)

#     content = call_groq_model(
#         PROPONENT_MODEL,
#         SYSTEM_PROMPT,
#         user_prompt,
#         temperature=0.0,
#         top_p=1.0,
#     )
#     return _extract_json(content)