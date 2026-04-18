# # judge.py
# import json
# import os
# import re
# from decimal import Decimal, InvalidOperation
# from pathlib import Path
# from typing import Any

# from groq import Groq


# # ----------------------------
# # Config
# # ----------------------------
# INPUT_PATH = Path("outputs/debate_results.json")
# OUTPUT_PATH = Path("outputs/judged_results_new.json")

# # Strict structured outputs are supported on these models.
# # Groq docs list strict:true support for openai/gpt-oss-20b and openai/gpt-oss-120b.
# JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")

# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# # ----------------------------
# # Helpers
# # ----------------------------
# def normalize_numeric(text: Any) -> str:
#     """
#     Normalize common numeric answer formats:
#     - remove commas
#     - strip whitespace
#     - keep the first numeric-looking token if present
#     """
#     s = str(text).strip().replace(",", "")
#     m = re.search(r"-?\d+(?:\.\d+)?", s)
#     return m.group(0) if m else s


# def numeric_equal(a: Any, b: Any) -> bool:
#     """
#     Compare two answers numerically when possible.
#     Falls back to normalized string comparison.
#     """
#     na = normalize_numeric(a)
#     nb = normalize_numeric(b)

#     try:
#         return Decimal(na) == Decimal(nb)
#     except (InvalidOperation, ValueError):
#         return na == nb


# def safe_json_loads(text: str) -> dict[str, Any]:
#     try:
#         obj = json.loads(text)
#         if isinstance(obj, dict):
#             return obj
#     except json.JSONDecodeError:
#         pass
#     return {}


# # ----------------------------
# # Judge schema + prompt
# # ----------------------------
# JUDGE_SYSTEM_PROMPT = """
# You are a strict reasoning judge for math debate transcripts.

# Evaluate the model's transcript against the question and gold answer.

# You must score:
# - answer_correct: whether the final answer matches the gold answer
# - reasoning_score: overall reasoning quality from 1 to 5
# - probe_response_score: how well the proponent responds to the prober from 1 to 5
# - consistency_score: whether the transcript stays consistent from 1 to 5
# - - overall_judgment: "pass" if:
#     - the final answer is correct AND
#     - reasoning_score >= 2

# - otherwise "fail"
# - notes: a short explanation

# Use these guidelines:
# - 5 = excellent
# - 4 = mostly correct
# - 3 = mixed
# - 2 = weak
# - 1 = invalid

# Be strict.
# """


# JUDGE_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "answer_correct": {"type": "boolean"},
#         "reasoning_score": {"type": "integer", "minimum": 1, "maximum": 5},
#         "probe_response_score": {"type": "integer", "minimum": 1, "maximum": 5},
#         "consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
#         "overall_judgment": {"type": "string", "enum": ["pass", "fail"]},
#         "notes": {"type": "string"},
#     },
#     "required": [
#         "answer_correct",
#         "reasoning_score",
#         "probe_response_score",
#         "consistency_score",
#         "overall_judgment",
#         "notes",
#     ],
#     "additionalProperties": False,
# }

# def coerce_sample(obj: Any) -> dict[str, Any] | None:
#     if isinstance(obj, dict):
#         return obj

#     if isinstance(obj, str):
#         try:
#             parsed = json.loads(obj)
#         except json.JSONDecodeError:
#             return None
#         if isinstance(parsed, dict):
#             return parsed
#         if isinstance(parsed, str):
#             try:
#                 parsed2 = json.loads(parsed)
#                 if isinstance(parsed2, dict):
#                     return parsed2
#             except json.JSONDecodeError:
#                 return None

#     return None


# def load_samples(path: Path) -> list[dict[str, Any]]:
#     raw = path.read_text(encoding="utf-8").strip()
#     if not raw:
#         return []

#     try:
#         data = json.loads(raw)
#         if isinstance(data, list):
#             out: list[dict[str, Any]] = []
#             for item in data:
#                 sample = coerce_sample(item)
#                 if sample is not None:
#                     out.append(sample)
#             return out

#         sample = coerce_sample(data)
#         return [sample] if sample is not None else []
#     except json.JSONDecodeError:
#         pass

#     samples: list[dict[str, Any]] = []
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 item = json.loads(line)
#             except json.JSONDecodeError as e:
#                 print(f"Skipping invalid JSON on line {line_no}: {e}")
#                 continue

#             sample = coerce_sample(item)
#             if sample is None:
#                 print(f"Skipping non-object on line {line_no}: {type(item).__name__}")
#                 continue

#             samples.append(sample)

#     return samples
# def judge_sample(sample: Any) -> dict[str, Any]:
#     sample = coerce_sample(sample) or {}

#     debate = sample.get("debate", {})
#     if not isinstance(debate, dict):
#         debate = {}

#     payload = {
#         "question": sample.get("question", ""),
#         "gold_answer": debate.get("final_answer", ""),
#         "final_answer": debate.get("final_answer", ""),
#         "initial_answer": debate.get("initial_answer", ""),
#         "final_proponent_text": debate.get("final_proponent_text", ""),
#         "transcript": debate.get("transcript", []),
#         "previous_probes": debate.get("previous_probes", []),
#         "used_probe_types": debate.get("used_probe_types", []),
#     }

#     response = client.chat.completions.create(
#         model=JUDGE_MODEL,
#         messages=[
#             {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
#             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
#         ],
#         temperature=0,
#         top_p=1,
#         response_format={
#             "type": "json_schema",
#             "json_schema": {
#                 "name": "debate_judge",
#                 "strict": True,
#                 "schema": JUDGE_SCHEMA,
#             },
#         },
#     )

#     raw_text = response.choices[0].message.content or "{}"
#     judge_obj = safe_json_loads(raw_text)

#     if "answer_correct" not in judge_obj:
#         judge_obj["answer_correct"] = numeric_equal(
#             debate.get("final_answer"), debate.get("gold_answer")
#         )

#     if "reasoning_score" not in judge_obj:
#         judge_obj["reasoning_score"] = 1 if not judge_obj["answer_correct"] else 3
#     if "probe_response_score" not in judge_obj:
#         judge_obj["probe_response_score"] = 1 if not judge_obj["answer_correct"] else 3
#     if "consistency_score" not in judge_obj:
#         judge_obj["consistency_score"] = 1 if not judge_obj["answer_correct"] else 3
#     if "overall_judgment" not in judge_obj:
#         judge_obj["overall_judgment"] = "pass" if judge_obj["answer_correct"] else "fail"
#     if "notes" not in judge_obj:
#         judge_obj["notes"] = "Fallback verdict generated because the structured response was incomplete."

#     sample["judge"] = judge_obj
#     debate = sample.get("debate", {})
#     if not isinstance(debate, dict):
#         debate = {}

#     sample["judge_match"] = numeric_equal(
#     debate.get("final_answer"),
#     judge_obj.get("answer_correct")  # not right for comparison
# )
#     sample["judge_answer_correct"] = bool(judge_obj.get("answer_correct"))
#     sample["judge_reasoning_score"] = int(judge_obj.get("reasoning_score", 0))
#     sample["judge_probe_response_score"] = int(judge_obj.get("probe_response_score", 0))
#     sample["judge_consistency_score"] = int(judge_obj.get("consistency_score", 0))
#     sample["judge_overall_judgment"] = judge_obj.get("overall_judgment", "fail")
#     return sample

# def main() -> None:
#     if not os.environ.get("GROQ_API_KEY"):
#         raise RuntimeError("GROQ_API_KEY is not set.")

#     if not INPUT_PATH.exists():
#         raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

#     OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

#     samples = load_samples(INPUT_PATH)

#     with OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
#         for idx, sample in enumerate(samples, start=1):
#             judged = judge_sample(sample)
#             f_out.write(json.dumps(judged, ensure_ascii=False) + "\n")

#             print(
#                 f"Saved id={judged.get('id', idx)} | "
#                 f"answer_correct={judged['judge_answer_correct']} | "
#                 f"reasoning_score={judged['judge_reasoning_score']} | "
#                 f"overall={judged['judge_overall_judgment']}"
#             )

#     print(f"Done. Judged results saved to {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()

# if __name__ == "__main__":
#     main()

# # judge.py
# import json
# import os
# import re
# from decimal import Decimal, InvalidOperation
# from pathlib import Path
# from typing import Any

# from groq import Groq


# # ----------------------------
# # Config
# # ----------------------------
# INPUT_PATH = Path("outputs/debate_results.json")
# OUTPUT_PATH = Path("outputs/judged_results_new.json")

# # Groq strict structured outputs support
# JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")

# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# # ----------------------------
# # Helpers
# # ----------------------------
# def normalize_numeric(text: Any) -> str:
#     """
#     Normalize common numeric answer formats:
#     - remove commas
#     - strip whitespace
#     - keep the first numeric-looking token if present
#     """
#     s = str(text).strip().replace(",", "")
#     m = re.search(r"-?\d+(?:\.\d+)?", s)
#     return m.group(0) if m else s


# def numeric_equal(a: Any, b: Any) -> bool:
#     """
#     Compare two answers numerically when possible.
#     Falls back to normalized string comparison.
#     """
#     na = normalize_numeric(a)
#     nb = normalize_numeric(b)

#     try:
#         return Decimal(na) == Decimal(nb)
#     except (InvalidOperation, ValueError):
#         return na == nb


# def safe_json_loads(text: str) -> dict[str, Any]:
#     try:
#         obj = json.loads(text)
#         if isinstance(obj, dict):
#             return obj
#     except json.JSONDecodeError:
#         pass
#     return {}


# def coerce_sample(obj: Any) -> dict[str, Any] | None:
#     """
#     Convert a loaded JSON value into a dict sample.
#     Handles:
#     - dict sample
#     - stringified JSON dict
#     - nested stringified JSON
#     """
#     if isinstance(obj, dict):
#         return obj

#     if isinstance(obj, str):
#         try:
#             parsed = json.loads(obj)
#         except json.JSONDecodeError:
#             return None
#         if isinstance(parsed, dict):
#             return parsed
#         if isinstance(parsed, str):
#             try:
#                 parsed2 = json.loads(parsed)
#                 if isinstance(parsed2, dict):
#                     return parsed2
#             except json.JSONDecodeError:
#                 return None

#     return None


# def load_samples(path: Path) -> list[dict[str, Any]]:
#     """
#     Load samples from either:
#     - JSON array
#     - JSON object
#     - JSONL (one JSON object per line)
#     - stringified JSON variants
#     """
#     raw = path.read_text(encoding="utf-8").strip()
#     if not raw:
#         return []

#     # First try to read as full JSON
#     try:
#         data = json.loads(raw)
#         if isinstance(data, list):
#             out: list[dict[str, Any]] = []
#             for item in data:
#                 sample = coerce_sample(item)
#                 if sample is not None:
#                     out.append(sample)
#             return out

#         sample = coerce_sample(data)
#         return [sample] if sample is not None else []
#     except json.JSONDecodeError:
#         pass

#     # Fallback to JSONL
#     samples: list[dict[str, Any]] = []
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 item = json.loads(line)
#             except json.JSONDecodeError as e:
#                 print(f"Skipping invalid JSON on line {line_no}: {e}")
#                 continue

#             sample = coerce_sample(item)
#             if sample is None:
#                 print(f"Skipping non-object on line {line_no}: {type(item).__name__}")
#                 continue

#             samples.append(sample)

#     return samples


# # ----------------------------
# # Judge schema + prompt
# # ----------------------------
# JUDGE_SYSTEM_PROMPT = """
# You are a strict reasoning judge for math debate transcripts.

# Evaluate the model's transcript against the question and gold answer.

# You must score:
# - answer_correct: whether the final answer matches the gold answer
# - reasoning_score: overall reasoning quality from 1 to 5
# - probe_response_score: how well the proponent responds to the prober from 1 to 5
# - consistency_score: whether the transcript stays consistent from 1 to 5
# - overall_judgment: "pass" if reasoning is solid, otherwise "fail"
# - notes: a short explanation

# Use these guidelines:
# - 5 = excellent
# - 4 = mostly correct
# - 3 = mixed
# - 2 = weak
# - 1 = invalid

# Be strict.
# """.strip()


# JUDGE_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "answer_correct": {"type": "boolean"},
#         "reasoning_score": {"type": "integer", "minimum": 1, "maximum": 5},
#         "probe_response_score": {"type": "integer", "minimum": 1, "maximum": 5},
#         "consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
#         "overall_judgment": {"type": "string", "enum": ["pass", "fail"]},
#         "notes": {"type": "string"},
#     },
#     "required": [
#         "answer_correct",
#         "reasoning_score",
#         "probe_response_score",
#         "consistency_score",
#         "overall_judgment",
#         "notes",
#     ],
#     "additionalProperties": False,
# }


# # ----------------------------
# # Judge logic
# # ----------------------------
# def judge_sample(sample: dict[str, Any]) -> dict[str, Any]:
#     """
#     Judge one debate sample and attach a structured reasoning verdict.
#     """
#     if not isinstance(sample, dict):
#         raise TypeError(f"judge_sample expected dict, got {type(sample).__name__}")

#     payload = {
#         "question": sample.get("question", ""),
#         "gold_answer": sample.get("gold_answer", ""),
#         "final_answer": sample.get("final_answer", ""),
#         "initial_answer": sample.get("initial_answer", ""),
#         "final_proponent_text": sample.get("final_proponent_text", ""),
#         "transcript": sample.get("transcript", []),
#         "previous_probes": sample.get("previous_probes", []),
#         "used_probe_types": sample.get("used_probe_types", []),
#     }

#     response = client.chat.completions.create(
#         model=JUDGE_MODEL,
#         messages=[
#             {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
#             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
#         ],
#         temperature=0,
#         top_p=1,
#         response_format={
#             "type": "json_schema",
#             "json_schema": {
#                 "name": "debate_judge",
#                 "strict": True,
#                 "schema": JUDGE_SCHEMA,
#             },
#         },
#     )

#     raw_text = response.choices[0].message.content or "{}"
#     judge_obj = safe_json_loads(raw_text)

#     # Fallbacks if the structured response is incomplete
#     if "answer_correct" not in judge_obj:
#         judge_obj["answer_correct"] = numeric_equal(
#             sample.get("final_answer"), sample.get("gold_answer")
#         )
#     if "reasoning_score" not in judge_obj:
#         judge_obj["reasoning_score"] = 1 if not judge_obj["answer_correct"] else 3
#     if "probe_response_score" not in judge_obj:
#         judge_obj["probe_response_score"] = 1 if not judge_obj["answer_correct"] else 3
#     if "consistency_score" not in judge_obj:
#         judge_obj["consistency_score"] = 1 if not judge_obj["answer_correct"] else 3
#     if "overall_judgment" not in judge_obj:
#         judge_obj["overall_judgment"] = "pass" if judge_obj["answer_correct"] else "fail"
#     if "notes" not in judge_obj:
#         judge_obj["notes"] = (
#             "Fallback verdict generated because the structured response was incomplete."
#         )

#     sample["judge"] = judge_obj
#     sample["judge_match"] = numeric_equal(sample.get("final_answer"), sample.get("gold_answer"))
#     sample["judge_answer_correct"] = bool(judge_obj.get("answer_correct"))
#     sample["judge_reasoning_score"] = int(judge_obj.get("reasoning_score", 0))
#     sample["judge_probe_response_score"] = int(judge_obj.get("probe_response_score", 0))
#     sample["judge_consistency_score"] = int(judge_obj.get("consistency_score", 0))
#     sample["judge_overall_judgment"] = judge_obj.get("overall_judgment", "fail")

#     return sample


# def main() -> None:
#     if not os.environ.get("GROQ_API_KEY"):
#         raise RuntimeError("GROQ_API_KEY is not set.")

#     if not INPUT_PATH.exists():
#         raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

#     OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

#     samples = load_samples(INPUT_PATH)
#     if not samples:
#         raise RuntimeError(f"No valid samples found in {INPUT_PATH}")

#     with OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
#         for idx, sample in enumerate(samples, start=1):
#             try:
#                 judged = judge_sample(sample)
#             except Exception as e:
#                 print(f"Skipping sample {idx}: {e}")
#                 continue

#             f_out.write(json.dumps(judged, ensure_ascii=False) + "\n")

#             print(
#                 f"Saved id={judged.get('id', idx)} | "
#                 f"answer_correct={judged['judge_answer_correct']} | "
#                 f"reasoning_score={judged['judge_reasoning_score']} | "
#                 f"overall={judged['judge_overall_judgment']}"
#             )

#     print(f"Done. Judged results saved to {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()
# judge.py
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from groq import Groq


# ----------------------------
# Config
# ----------------------------
INPUT_PATH = Path("outputs/debate_results.json")
OUTPUT_PATH = Path("outputs/judged_results_new.json")

# Strict structured outputs are supported on these models.
JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ----------------------------
# Helpers
# ----------------------------
def normalize_numeric(text: Any) -> str:
    """
    Normalize common numeric answer formats:
    - remove commas
    - strip whitespace
    - keep the first numeric-looking token if present
    """
    s = str(text).strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def numeric_equal(a: Any, b: Any) -> bool:
    """
    Compare two answers numerically when possible.
    Falls back to normalized string comparison.
    """
    na = normalize_numeric(a)
    nb = normalize_numeric(b)

    try:
        return Decimal(na) == Decimal(nb)
    except (InvalidOperation, ValueError):
        return na == nb


def safe_json_loads(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {}


def coerce_sample(obj: Any) -> dict[str, Any] | None:
    """
    Convert a loaded JSON value into a dict sample.
    Handles:
    - dict sample
    - stringified JSON dict
    - nested stringified JSON
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
                if isinstance(parsed2, dict):
                    return parsed2
            except json.JSONDecodeError:
                return None

    return None


def load_samples(path: Path) -> list[dict[str, Any]]:
    """
    Load samples from either:
    - JSON array
    - JSON object
    - JSONL (one JSON object per line)
    - stringified JSON variants
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # Try whole-file JSON first.
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out: list[dict[str, Any]] = []
            for item in data:
                sample = coerce_sample(item)
                if sample is not None:
                    out.append(sample)
            return out

        sample = coerce_sample(data)
        return [sample] if sample is not None else []
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL.
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_no}: {e}")
                continue

            sample = coerce_sample(item)
            if sample is None:
                print(f"Skipping non-object on line {line_no}: {type(item).__name__}")
                continue

            samples.append(sample)

    return samples


def extract_debate(sample: dict[str, Any]) -> dict[str, Any]:
    debate = sample.get("debate", {})
    if not isinstance(debate, dict):
        return {}
    return debate


def extract_final_answer(sample: dict[str, Any]) -> Any:
    """
    Prefer the top-level debate.final_answer.
    If blank, try to parse final_proponent_text as JSON and pull a final-answer field.
    """
    debate = extract_debate(sample)

    final_answer = debate.get("final_answer", "")
    if final_answer not in ("", None):
        return final_answer

    final_proponent_text = debate.get("final_proponent_text", "")
    if isinstance(final_proponent_text, str) and final_proponent_text.strip():
        parsed = safe_json_loads(final_proponent_text)
        for key in ("Final answer", "final answer", "Final Answer", "answer", "Answer"):
            if key in parsed:
                return parsed[key]

        # Last resort: pull the last numeric token from the text.
        m = re.findall(r"-?\d+(?:\.\d+)?", final_proponent_text.replace(",", ""))
        if m:
            return m[-1]

    return ""


def extract_gold_answer(sample: dict[str, Any]) -> Any:
    """
    Use an explicit gold answer if present.
    If not present, try common alternate locations.
    """
    if "gold_answer" in sample and sample["gold_answer"] not in ("", None):
        return sample["gold_answer"]

    debate = extract_debate(sample)
    for key in ("gold_answer", "answer", "correct_answer"):
        val = debate.get(key)
        if val not in ("", None):
            return val

    return ""


# ----------------------------
# Judge schema + prompt
# ----------------------------
JUDGE_SYSTEM_PROMPT = """
You are a strict reasoning judge for math debate transcripts.

Evaluate the model's transcript against the question and gold answer.

You must score:
- answer_correct: whether the final answer matches the gold answer
- reasoning_score: overall reasoning quality from 1 to 5
- probe_response_score: how well the proponent responds to the prober from 1 to 5
- consistency_score: whether the transcript stays consistent from 1 to 5
- overall_judgment: "pass" if the final answer is correct and reasoning_score >= 2, otherwise "fail"
- notes: a short explanation

Use these guidelines:
- 5 = excellent
- 4 = mostly correct
- 3 = mixed
- 2 = weak
- 1 = invalid

Be strict.
""".strip()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_correct": {"type": "boolean"},
        "reasoning_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "probe_response_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall_judgment": {"type": "string", "enum": ["pass", "fail"]},
        "notes": {"type": "string"},
    },
    "required": [
        "answer_correct",
        "reasoning_score",
        "probe_response_score",
        "consistency_score",
        "overall_judgment",
        "notes",
    ],
    "additionalProperties": False,
}


def judge_sample(sample: Any) -> dict[str, Any]:
    sample = coerce_sample(sample) or {}

    debate = extract_debate(sample)
    final_answer = extract_final_answer(sample)
    gold_answer = extract_gold_answer(sample)

    payload = {
        "question": sample.get("question", ""),
        "gold_answer": gold_answer,
        "final_answer": final_answer,
        "initial_answer": debate.get("initial_answer", ""),
        "final_proponent_text": debate.get("final_proponent_text", ""),
        "transcript": debate.get("transcript", []),
        "previous_probes": debate.get("previous_probes", []),
        "used_probe_types": debate.get("used_probe_types", []),
    }

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
        top_p=1,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "debate_judge",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            },
        },
    )

    raw_text = response.choices[0].message.content or "{}"
    judge_obj = safe_json_loads(raw_text)

    expected_correct = False
    if gold_answer not in ("", None) and final_answer not in ("", None):
        expected_correct = numeric_equal(final_answer, gold_answer)

    if "answer_correct" not in judge_obj:
        judge_obj["answer_correct"] = expected_correct

    if "reasoning_score" not in judge_obj:
        judge_obj["reasoning_score"] = 1 if not judge_obj["answer_correct"] else 3
    if "probe_response_score" not in judge_obj:
        judge_obj["probe_response_score"] = 1 if not judge_obj["answer_correct"] else 3
    if "consistency_score" not in judge_obj:
        judge_obj["consistency_score"] = 1 if not judge_obj["answer_correct"] else 3
    if "overall_judgment" not in judge_obj:
        judge_obj["overall_judgment"] = "pass" if (judge_obj["answer_correct"] and int(judge_obj["reasoning_score"]) >= 2) else "fail"
    if "notes" not in judge_obj:
        judge_obj["notes"] = "Fallback verdict generated because the structured response was incomplete."

    sample["judge"] = judge_obj
    sample["judge_match"] = bool(judge_obj.get("answer_correct")) == expected_correct if gold_answer not in ("", None) and final_answer not in ("", None) else bool(judge_obj.get("answer_correct"))
    sample["judge_answer_correct"] = bool(judge_obj.get("answer_correct"))
    sample["judge_reasoning_score"] = int(judge_obj.get("reasoning_score", 0))
    sample["judge_probe_response_score"] = int(judge_obj.get("probe_response_score", 0))
    sample["judge_consistency_score"] = int(judge_obj.get("consistency_score", 0))
    sample["judge_overall_judgment"] = judge_obj.get("overall_judgment", "fail")
    return sample


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(INPUT_PATH)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(samples, start=1):
            judged = judge_sample(sample)
            f_out.write(json.dumps(judged, ensure_ascii=False) + "\n")

            print(
                f"Saved id={judged.get('id', idx)} | "
                f"answer_correct={judged['judge_answer_correct']} | "
                f"reasoning_score={judged['judge_reasoning_score']} | "
                f"overall={judged['judge_overall_judgment']}"
            )

    print(f"Done. Judged results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()