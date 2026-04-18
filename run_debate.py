import json
from pathlib import Path

from debate_loop import run_debate


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    input_path = Path("gsm8k_sample.jsonl")
    output_path = Path("outputs/debate_results.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(input_path)
    results = []

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        print(f"Running sample {idx}...")

        try:
            debate = run_debate(question=question, max_turns=3)
            status = "ok"
            error = None
        except Exception as e:
            debate = {}
            status = "error"
            error = str(e)

        record = {
            "id": sample.get("id", idx),
            "question": question,
            "debate": debate if status == "ok" else {"error": error},
            "status": status,
        }
        results.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()