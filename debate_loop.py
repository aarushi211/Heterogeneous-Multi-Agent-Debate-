# import re
# from typing import Any

# from proponent import ask_proponent
# from prober import ask_prober


# def extract_final_answer(text: str) -> str:
#     """
#     Extract the final numeric answer from model output.
#     Prefers the last number found in the text.
#     """
#     matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
#     if not matches:
#         return ""
#     return matches[-1].replace(",", "")


# def infer_probe_type(probe: str) -> str:
#     """
#     Best-effort classification so the prober can avoid repeating the same kind of challenge.
#     """
#     p = probe.lower()

#     if any(k in p for k in ("multiply", "times", "twice", "double", "product")):
#         return "multiplicative"
#     if any(k in p for k in ("add", "sum", "total", "combine", "aggregation", "together")):
#         return "aggregation"
#     if any(k in p for k in ("subtract", "minus", "left", "remain", "remaining", "difference", "remove")):
#         return "subtraction"
#     if any(k in p for k in ("asked", "asking", "what is being asked", "interpret")):
#         return "interpretation"

#     return "other"


# def run_debate(
#     question: str,
#     max_turns: int = 3,
# ) -> dict[str, Any]:
#     """
#     Run one Proponent–Prober debate.

#     Flow:
#     1. Proponent answers the original question.
#     2. Prober asks one Socratic question.
#     3. Proponent answers the challenge.
#     4. Repeat for max_turns.
#     """
#     transcript: list[dict[str, str]] = []
#     previous_probes: list[str] = []
#     used_probe_types: list[str] = []

#     proponent_answer = ask_proponent(question).strip()
#     transcript.append({"role": "proponent", "text": proponent_answer})

#     turn_records: list[dict[str, Any]] = []

#     for turn in range(1, max_turns + 1):
#         probe = ask_prober(
#             question=question,
#             last_answer=proponent_answer,
#             turn=turn,
#             transcript=transcript,
#             previous_probes=previous_probes,
#             used_probe_types=used_probe_types,
#         ).strip()

#         probe_type = infer_probe_type(probe)

#         previous_probes.append(probe)
#         if probe_type not in used_probe_types:
#             used_probe_types.append(probe_type)

#         transcript.append({"role": "prober", "text": probe})

#         challenged_answer = ask_proponent(question, challenge=probe).strip()
#         transcript.append({"role": "proponent", "text": challenged_answer})

#         turn_records.append(
#             {
#                 "turn": turn,
#                 "probe_type": probe_type,
#                 "probe": probe,
#                 "proponent_answer": challenged_answer,
#             }
#         )

#         proponent_answer = challenged_answer

#     final_answer = extract_final_answer(proponent_answer)

#     return {
#         "question": question,
#         "initial_answer": transcript[0]["text"] if transcript else "",
#         "final_proponent_text": proponent_answer,
#         "final_answer": final_answer,
#         "transcript": transcript,
#         "previous_probes": previous_probes,
#         "used_probe_types": used_probe_types,
#         "turns": turn_records,
#         "num_turns": max_turns,
#         "correct_format": bool(final_answer),
#     }

# # debate_loop.py
# import json
# from typing import Any

# from proponent import ask_proponent, ask_proponent_revision
# from prober import ask_prober


# def extract_final_answer(text: str) -> str:
#     try:
#         data = json.loads(text)
#         return str(data.get("final_answer", "")).strip()
#     except Exception:
#         return ""


# def run_debate(question: str, max_turns: int = 3) -> dict[str, Any]:
#     transcript: list[dict[str, str]] = []
#     previous_probes: list[str] = []
#     used_probe_types: list[str] = []

#     proponent_answer = ask_proponent(question).strip()
#     transcript.append({"role": "proponent", "text": proponent_answer})

#     turn_records: list[dict[str, Any]] = []

#     for turn in range(1, max_turns + 1):
#         probe = ask_prober(
#             question=question,
#             last_answer=proponent_answer,
#             turn=turn,
#             transcript=transcript,
#             previous_probes=previous_probes,
#             used_probe_types=used_probe_types,
#         ).strip()

#         probe_type = infer_probe_type(probe)
#         previous_probes.append(probe)
#         if probe_type not in used_probe_types:
#             used_probe_types.append(probe_type)

#         transcript.append({"role": "prober", "text": probe})

#         challenged_answer = ask_proponent_revision(
#             question=question,
#             prior_answer=proponent_answer,
#             probe=probe,
#         ).strip()
#         transcript.append({"role": "proponent", "text": challenged_answer})

#         turn_records.append(
#             {
#                 "turn": turn,
#                 "probe_type": probe_type,
#                 "probe": probe,
#                 "proponent_answer": challenged_answer,
#             }
#         )

#         proponent_answer = challenged_answer

#     final_answer = extract_final_answer(proponent_answer)

#     return {
#         "question": question,
#         "initial_answer": transcript[0]["text"] if transcript else "",
#         "final_proponent_text": proponent_answer,
#         "final_answer": final_answer,
#         "transcript": transcript,
#         "previous_probes": previous_probes,
#         "used_probe_types": used_probe_types,
#         "turns": turn_records,
#         "num_turns": max_turns,
#         "correct_format": bool(final_answer),
#     }
# import re
# from typing import Any

# from proponent import ask_proponent
# from prober import ask_prober


# def extract_final_answer(text: str) -> str:
#     """
#     Extract the final numeric answer from model output.
#     Prefers the last number found in the text.
#     """
#     matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
#     if not matches:
#         return ""
#     return matches[-1].replace(",", "")


# def run_debate(
#     question: str,
#     max_turns: int = 3,
# ) -> dict[str, Any]:
#     """
#     Run one Proponent–Prober debate.

#     Flow:
#     1. Proponent answers the original question.
#     2. Prober asks one Socratic question.
#     3. Proponent answers the challenge.
#     4. Repeat for max_turns.
#     """
#     transcript: list[dict[str, str]] = []
#     previous_probes: list[str] = []

#     proponent_answer = ask_proponent(question).strip()
#     transcript.append({"role": "proponent", "text": proponent_answer})

#     turn_records: list[dict[str, Any]] = []

#     for turn in range(1, max_turns + 1):
#         probe = ask_prober(
#             question=question,
#             last_answer=proponent_answer,
#             turn=turn,
#             transcript=transcript,
#             previous_probes=previous_probes,
#             used_probe_types=None,
#         ).strip()

#         previous_probes.append(probe)
#         transcript.append({"role": "prober", "text": probe})

#         challenged_answer = ask_proponent(question).strip()
#         transcript.append({"role": "proponent", "text": challenged_answer})

#         turn_records.append(
#             {
#                 "turn": turn,
#                 "probe": probe,
#                 "proponent_answer": challenged_answer,
#             }
#         )

#         proponent_answer = challenged_answer

#     final_answer = extract_final_answer(proponent_answer)

#     return {
#         "question": question,
#         "initial_answer": transcript[0]["text"] if transcript else "",
#         "final_proponent_text": proponent_answer,
#         "final_answer": final_answer,
#         "transcript": transcript,
#         "previous_probes": previous_probes,
#         "turns": turn_records,
#         "num_turns": max_turns,
#         "correct_format": bool(final_answer),
#     }

import json
from typing import Any

from proponent import ask_proponent
from prober import ask_prober

def run_debate(question: str, gold_answer: Any | None = None, max_turns: int = 3) -> dict[str, Any]:
    transcript: list[dict[str, str]] = []
    rounds: list[dict[str, Any]] = []

    current_json = ask_proponent(question)
    current_answer = str(current_json.get("final_answer", ""))

    transcript.append({
        "role": "proponent",
        "text": json.dumps(current_json, ensure_ascii=False),
    })

    for turn in range(1, max_turns + 1):
        probe = ask_prober(
            question=question,
            proponent_output=current_json,
            transcript=transcript,
        ).strip()

        transcript.append({"role": "prober", "text": probe})

        revised_json = ask_proponent(
            question=question,
            critique=probe,
            previous_answer=current_json,
            transcript=transcript,
        )

        current_answer = str(revised_json.get("final_answer", ""))

        transcript.append({
            "role": "proponent",
            "text": json.dumps(revised_json, ensure_ascii=False),
        })

        rounds.append({
            "turn": turn,
            "probe": probe,
            "proponent_answer": revised_json,
        })

        current_json = revised_json

    return {
        "question": question,
        "gold_answer": gold_answer,
        "final_answer": current_answer,
        "final_proponent_text": json.dumps(current_json, ensure_ascii=False),
        "transcript": transcript,
        "rounds": rounds,
        "num_turns": max_turns,
    }