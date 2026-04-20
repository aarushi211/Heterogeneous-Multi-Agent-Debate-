import json
from pathlib import Path

from proponent import ask_proponent


def main():
    input_path = Path("gsm8k_sample.jsonl")
    output_path = Path("outputs/proponent.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ FIX: Load JSONL properly
    dataset = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    results = []

    for idx, sample in enumerate(dataset):
        question = sample["question"]

        try:
            proponent_output = ask_proponent(question)
            status = "ok"
        except Exception as e:
            proponent_output = {
                "given": [],
                "steps": [],
                "result": None,
                "final_answer": "",
                "error": str(e),
            }
            status = "error"

        record = {
            "id": sample.get("id", idx),
            "question": question,
            "proponent_output": proponent_output,
            "status": status,
        }

        results.append(record)
        print(f"Processed {idx}: {status}")

    # Save as JSON (array)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()