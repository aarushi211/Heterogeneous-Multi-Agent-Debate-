# import json
# import re
# from decimal import Decimal, InvalidOperation
# from pathlib import Path

# from datasets import load_dataset

# from debate_loop import run_debate  # change this if your file is named debate_loop.py


# def extract_answer(answer_text: str) -> str | None:
#     match = re.search(r"####\s*(.*)", answer_text)
#     return match.group(1).strip() if match else None


# def normalize_numeric(text: str | None):
#     if text is None:
#         return None
#     cleaned = str(text).strip().replace(",", "")
#     match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
#     if not match:
#         return cleaned
#     try:
#         return Decimal(match.group(0))
#     except InvalidOperation:
#         return cleaned


# def main():
#     dataset = load_dataset("gsm8k", "main")
#     test_data = dataset["test"]

#     out_path = Path("outputs/results.jsonl")
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     samples = test_data.select(range(min(10, len(test_data))))

#     with out_path.open("w", encoding="utf-8") as f:
#         for idx, sample in enumerate(samples):
#             question = sample["question"]
#             gold_raw = extract_answer(sample["answer"])
#             gold_norm = normalize_numeric(gold_raw)

#             result = run_debate(question=question, max_turns=3)
#             pred_norm = normalize_numeric(result["final_answer"])

#             result["id"] = idx
#             result["gold_answer"] = gold_raw
#             result["is_correct"] = pred_norm == gold_norm

#             f.write(json.dumps(result, ensure_ascii=False) + "\n")
#             print(f"Saved sample {idx}: correct={result['is_correct']}")

#     print(f"Done. Results saved to {out_path}")


# if __name__ == "__main__":
#     main()

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from datasets import load_dataset

from debate_loop import run_debate


def extract_answer(answer_text: str) -> str | None:
    match = re.search(r"####\s*(.*)", answer_text)
    return match.group(1).strip() if match else None


def normalize_numeric(text: str | None):
    if text is None:
        return None

    cleaned = str(text).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return cleaned

    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return cleaned


def main():
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    out_path = Path("outputs/results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = test_data.select(range(min(10, len(test_data))))

    with out_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            question = sample["question"]
            gold_raw = extract_answer(sample["answer"])
            gold_norm = normalize_numeric(gold_raw)

            result = run_debate(question=question, max_turns=3)
            pred_norm = normalize_numeric(result["final_answer"])

            result["id"] = idx
            result["gold_answer"] = gold_raw
            result["is_correct"] = pred_norm == gold_norm

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Saved sample {idx}: correct={result['is_correct']}")

    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()