# from core.llm import call_model

# PROBER_MODEL = "qwen2.5:14b"

# SYSTEM_PROMPT = """
# You are a strict Socratic skeptic.

# Your task is to ask exactly ONE new question about the proponent's reasoning.

# Rules:
# - Ask exactly ONE question.
# - The question must start with "Why does" or "Why is".
# - Do not ask multiple questions.
# - Do not include explanations, commentary, or answers.
# - Do not question any numbers or facts explicitly stated in the original problem.
# - Do not ask why a count times a per-unit amount gives a total if that follows directly from the givens.
# - Do not ask why a copied relationship like "twice as many" or "4 times as many" is true if it is stated in the problem.
# - Only question a reasoning step, not a given fact.
# - If all steps are correct, question the least explicit reasoning step.
# - Do not repeat a previous probe.
# - Prefer a different reasoning target each turn if possible.
# - Keep it short and precise.
# - Output only the question.
# - End with exactly one question mark.
# """

# PROBE_TARGETS = [
#     "subtraction / removal",
#     "interpretation of the remainder",
#     "mapping to real-world meaning",
#     "final aggregation or multiplication",
#     "distance-from-home reasoning",
#     "percent-base reasoning",
#     "question interpretation",
# ]

# def _pick_probe_target(turn: int, used_probe_types: list[str] | None = None) -> str:
#     used_probe_types = used_probe_types or []
#     for target in PROBE_TARGETS:
#         if target not in used_probe_types:
#             return target
#     return PROBE_TARGETS[turn % len(PROBE_TARGETS)]

# def ask_prober(
#     question: str,
#     last_answer: str,
#     turn: int,
#     transcript: list[dict] | None = None,
#     previous_probes: list[str] | None = None,
#     used_probe_types: list[str] | None = None,
# ) -> str:
#     transcript = transcript or []
#     previous_probes = previous_probes or []
#     used_probe_types = used_probe_types or []

#     probes_text = "\n".join(f"- {p}" for p in previous_probes) if previous_probes else "None"
#     transcript_text = (
#         "\n".join(f"{m['role'].upper()}: {m['text']}" for m in transcript)
#         if transcript else "None"
#     )

#     target = _pick_probe_target(turn, used_probe_types)

#     user_prompt = f"""
# Original question:
# {question}

# Debate transcript:
# {transcript_text}

# Proponent's last answer:
# {last_answer}

# Previous probes:
# {probes_text}

# Reasoning target for this turn:
# {target}

# Task:
# Ask one NEW focused "Why?" question that probes the reasoning target above.
# This is probing turn {turn}.
# """

#     probe = call_model(
#         PROBER_MODEL,
#         SYSTEM_PROMPT,
#         user_prompt,
#         num_predict=64,
#     )

#     return probe.strip()
# from core.llm import call_model

# PROBER_MODEL = "qwen2.5:14b"

# SYSTEM_PROMPT = """
# You are a strict Socratic skeptic.

# Your task is to ask exactly one NEW question about the Proponent's reasoning.

# Rules:

# - Ask exactly ONE question.
# - The question must start with "Why does" or "Why is".
# - Do NOT ask multiple questions.
# - Do NOT include explanations or commentary.
# - Do NOT question any numbers or facts explicitly stated in the problem.
# - Do NOT ask why multiplying a **count** by a **per-unit amount given in the problem** yields a **total**
#   (e.g. why flock size × cups per chicken gives total daily feed); that follows from the givens. Probe a
#   different step instead, such as combining or subtracting partial totals.
# - Do NOT ask why one entity "has N times as many" or "twice as many" as another if that clause is copied from the problem; question a concrete calculation or aggregation step instead.
# - Do NOT invent alternative scenarios or "what if" cases.
# - Only question a reasoning step.
# - If all steps are correct, question the least explicitly justified reasoning step.
# - Do NOT repeat or paraphrase any previous probe.
# - Prefer a different reasoning target each turn if possible.
# - Keep the question short and precise.
# - Output ONLY the question.
# - End with exactly one question mark.
# - Prefer questions that reference the meaning of the operation, not just the numbers.
# - Don't hallucinate.

# - If a reasoning step is already correct and well-supported, do NOT challenge it again.
# - Move to a different step instead of re-questioning the same logic.

# Dont give like this below example.
# (eg1: 1st time:  "Why is Charleston's number of sheep calculated by multiplying Seattle's number by 4?"
# 2nd time: "Why does doubling Charleston's number of sheep give Toulouse's number?")

#  In this example you have questioned multiplication twice, which I don't want.
# If you have questioned multiplication during 1st time, then don't question it again second time also.Move to different methods like addition, subtraction or division.

# Reasoning types you can target:
# 1. Subtraction / removal (for example, 16 - 7)
# 2. Interpretation of the remainder (what "remaining" means)
# 3. Mapping to real-world meaning (for example, eggs available for sale)
# 4. Final calculation (for example, multiplication for earnings)

# If a reasoning type has already been used, choose a different type next time.
# Do NOT repeat the same reasoning category across turns.
# - If you already questioned one multiplication relationship or other methods, move to a different relationship or step (e.g., if aldready questiooned on multiplication , move to some other methods like addition ).

# Reasoning types (MUST choose a different one each turn):
# 1. First inference (e.g., Seattle → Charleston)
# 2. Second inference (e.g., Charleston → Toulouse)
# 3. Final aggregation (e.g., adding all values)
# 4. Interpretation of the question (what is being asked)

# If one type is already used, move to a different type.

# After questioning one inference, move to a different reasoning type next:
# 1. first inference
# 2. second inference
# 3. final addition / aggregation
# 4. question interpretation
# Do not repeat the same type twice.


# Good questions:
# - Why is subtracting 3 and 4 from 16 valid here?
# - Why does multiplying 9 by 2 give the total earnings?

# Bad questions:
# - Why 3 eggs?
# - Why 4 eggs?
# - Why not use a different method?
# """


# def ask_prober(
#     question: str,
#     last_answer: str,
#     turn: int,
#     transcript: list[dict] | None = None,
#     previous_probes: list[str] | None = None,
#     used_probe_types: list[str] | None = None,
# ) -> str:
#     transcript = transcript or []
#     previous_probes = previous_probes or []
#     used_probe_types = used_probe_types or []

#     probes_text = "\n".join(f"- {p}" for p in previous_probes) if previous_probes else "None"
#     types_text = ", ".join(used_probe_types) if used_probe_types else "None"
#     transcript_text = (
#         "\n".join(f"{m['role'].upper()}: {m['text']}" for m in transcript)
#         if transcript
#         else "None"
#     )

#     user_prompt = f"""
# Original question:
# {question}

# Debate transcript (context):
# {transcript_text}

# Proponent's last answer:
# {last_answer}

# Previous probes:
# {probes_text}

# Reasoning-operation categories already used in this debate:
# {types_text}
# If "multiplicative" appears above, you must NOT ask about multiplication, times, doubling, or "twice" again — target addition/aggregation, subtraction, division, or what the question is asking instead.

# Task:
# Ask one NEW focused "Why?" question.
# This is probing turn {turn}.
# """


#     return call_model(PROBER_MODEL, SYSTEM_PROMPT, user_prompt)

# import os
# from core.llm import call_model

# PROBER_MODEL = os.environ.get("PROBER_MODEL", "qwen/qwen3-32b")

# SYSTEM_PROMPT = """
# You are a strict Socratic skeptic.

# Your task is to ask exactly one NEW question about the Proponent's reasoning.

# Rules:

# - Ask exactly ONE question.
# - The question must start with "Why does" or "Why is".
# - Do NOT ask multiple questions.
# - Do NOT include explanations or commentary.
# - Do NOT question any numbers or facts explicitly stated in the problem.
# - Do NOT ask why multiplying a count by a per-unit amount given in the problem yields a total.
# - Do NOT ask why one entity has N times as many as another if that clause is copied from the problem.
# - Only question a reasoning step.
# - If all steps are correct, question the least explicitly justified reasoning step.
# - Do NOT repeat or paraphrase any previous probe.
# - Prefer a different reasoning target each turn if possible.
# - Keep the question short and precise.
# - Output ONLY the question.
# - End with exactly one question mark.
# """

# def ask_prober(
#     question: str,
#     last_answer: str,
#     turn: int,
#     transcript: list[dict] | None = None,
#     previous_probes: list[str] | None = None,
#     used_probe_types: list[str] | None = None,
# ) -> str:
#     transcript = transcript or []
#     previous_probes = previous_probes or []
#     used_probe_types = used_probe_types or []

#     probes_text = "\n".join(f"- {p}" for p in previous_probes) if previous_probes else "None"
#     types_text = ", ".join(used_probe_types) if used_probe_types else "None"
#     transcript_text = (
#         "\n".join(f"{m['role'].upper()}: {m['text']}" for m in transcript)
#         if transcript
#         else "None"
#     )

#     user_prompt = f"""
# Original question:
# {question}

# Debate transcript (context):
# {transcript_text}

# Proponent's last answer:
# {last_answer}

# Previous probes:
# {probes_text}

# Reasoning-operation categories already used in this debate:
# {types_text}

# Task:
# Ask one NEW focused "Why?" question.
# This is probing turn {turn}.
# """

#     return call_model(PROBER_MODEL, SYSTEM_PROMPT, user_prompt, temperature=0.0)

# import os
# from typing import Optional
# import time
# from google import genai
# from google.genai import types

# PROBER_MODEL = os.environ.get("PROBER_MODEL", "gemini-2.5-flash")

# SYSTEM_PROMPT = """
# You are a strict Socratic skeptic.

# Your task is to ask exactly one NEW question about the Proponent's reasoning.

# Rules:

# - Ask exactly ONE question.
# - The question must start with "Why does" or "Why is".
# - Do NOT ask multiple questions.
# - Do NOT include explanations or commentary.
# - Do NOT question any numbers or facts explicitly stated in the problem.
# - Do NOT ask why multiplying a count by a per-unit amount given in the problem yields a total.
# - Do NOT ask why one entity has N times as many as another if that clause is copied from the problem.
# - Only question a reasoning step.
# - If all steps are correct, question the least explicitly justified reasoning step.
# - Do NOT repeat or paraphrase any previous probe.
# - Prefer a different reasoning target each turn if possible.
# - Keep the question short and precise.
# - Output ONLY the question.
# - End with exactly one question mark.
# """.strip()

# _client = genai.Client()


# def ask_prober(
#     question: str,
#     last_answer: str,
#     turn: int,
#     transcript: list[dict] | None = None,
#     previous_probes: list[str] | None = None,
#     used_probe_types: list[str] | None = None,
# ) -> str:
#     transcript = transcript or []
#     previous_probes = previous_probes or []
#     used_probe_types = used_probe_types or []

#     probes_text = "\n".join(f"- {p}" for p in previous_probes) if previous_probes else "None"
#     types_text = ", ".join(used_probe_types) if used_probe_types else "None"
#     transcript_text = (
#         "\n".join(f"{m['role'].upper()}: {m['text']}" for m in transcript)
#         if transcript
#         else "None"
#     )

#     user_prompt = f"""
# Original question:
# {question}

# Debate transcript (context):
# {transcript_text}

# Proponent's last answer:
# {last_answer}

# Previous probes:
# {probes_text}

# Reasoning-operation categories already used in this debate:
# {types_text}

# Task:
# Ask one NEW focused "Why?" question.
# This is probing turn {turn}.
# """.strip()

#     def make_request():
#         return _client.models.generate_content(
#             model=PROBER_MODEL,
#             contents=user_prompt,
#             config=types.GenerateContentConfig(
#                 system_instruction=SYSTEM_PROMPT,
#                 temperature=0.0,
#                 max_output_tokens=80,
#             ),
#         )

#     # 🔁 retry logic
#     for attempt in range(5):
#         try:
#             response = make_request()
#             break
#         except Exception as e:
#             if "503" in str(e):
#                 wait = 2 ** attempt
#                 print(f"Gemini busy, retrying in {wait}s...")
#                 time.sleep(wait)
#             else:
#                 raise
#     else:
#         raise RuntimeError("Gemini failed after retries")

#     text = (response.text or "").strip()
#     return text
#     text = (response.text or "").strip()

#     # Defensive cleanup: keep only the first line and ensure it ends with '?'
#     if "\n" in text:
#         text = text.splitlines()[0].strip()
#     if not text.endswith("?"):
#         text += "?"

#     return text
import json
import os
from typing import Any

from core.ollama_llm import call_ollama_model

PROBER_MODEL = os.environ.get("PROBER_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """
You are a strict Socratic skeptic.

Your task is to ask exactly one NEW question about the Proponent's reasoning.

Rules:
Same Questions should not repeat.
- Ask exactly ONE question.
- The question must start with "Why does" or "Why is".
- Do NOT ask multiple questions.
- Do NOT include explanations or commentary.
- Do NOT question any numbers or facts explicitly stated in the problem.
- Do NOT ask why multiplying a **count** by a **per-unit amount given in the problem** yields a **total**
  (e.g. why flock size × cups per chicken gives total daily feed); that follows from the givens. Probe a
  different step instead, such as combining or subtracting partial totals.
- Do NOT ask why one entity "has N times as many" or "twice as many" as another if that clause is copied from the problem; question a concrete calculation or aggregation step instead.
- Do NOT invent alternative scenarios or "what if" cases.
- Only question a reasoning step.
- If all steps are correct, question the least explicitly justified reasoning step.
- Do NOT repeat or paraphrase any previous probe.
- Prefer a different reasoning target each turn if possible.
- Keep the question short and precise.
- Output ONLY the question.
- End with exactly one question mark.
- Prefer questions that reference the meaning of the operation, not just the numbers.
- Don't hallucinate.

- If a reasoning step is already correct and well-supported, do NOT challenge it again.
- Move to a different step instead of re-questioning the same logic.

Dont give like this below example.
(eg1: 1st time:  "Why is Charleston's number of sheep calculated by multiplying Seattle's number by 4?"
2nd time: "Why does doubling Charleston's number of sheep give Toulouse's number?")

 In this example you have questioned multiplication twice, which I don't want.
If you have questioned multiplication during 1st time, then don't question it again second time also.Move to different methods like addition, subtraction or division.

Reasoning types you can target:
1. Subtraction / removal (for example, 16 - 7)
2. Interpretation of the remainder (what "remaining" means)
3. Mapping to real-world meaning (for example, eggs available for sale)
4. Final calculation (for example, multiplication for earnings)

If a reasoning type has already been used, choose a different type next time.
Do NOT repeat the same reasoning category across turns.
- If you already questioned one multiplication relationship or other methods, move to a different relationship or step (e.g., if aldready questiooned on multiplication , move to some other methods like addition ).

Reasoning types (MUST choose a different one each turn):
1. First inference (e.g., Seattle → Charleston)
2. Second inference (e.g., Charleston → Toulouse)
3. Final aggregation (e.g., adding all values)
4. Interpretation of the question (what is being asked)

If one type is already used, move to a different type.

After questioning one inference, move to a different reasoning type next:
1. first inference
2. second inference
3. final addition / aggregation
4. question interpretation
Do not repeat the same type twice.


Good questions:
- Why is subtracting 3 and 4 from 16 valid here?
- Why does multiplying 9 by 2 give the total earnings?

Bad questions:
- Why 3 eggs?
- Why 4 eggs?
- Why not use a different method?
""".strip()


def ask_prober(
    question: str,
    proponent_output: dict[str, Any],
    transcript: list[dict[str, str]] | None = None,
) -> str:
    transcript_text = ""
    if transcript:
        transcript_text = "\n".join(
            f"{item['role'].upper()}: {item['text']}" for item in transcript
        )

    user_prompt = f"""
Original question:
{question}

Proponent output:
{json.dumps(proponent_output, ensure_ascii=False, indent=2)}

Transcript so far:
{transcript_text if transcript_text else "(none)"}

Ask one concise challenge question that targets a specific step.
""".strip()

    return call_ollama_model(
        model=PROBER_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )