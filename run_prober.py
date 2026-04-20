import json
from pathlib import Path

from prober import ask_prober


def main():
    input_path = Path("outputs/proponent.json")
    output_path = Path("outputs/prober.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        proponent_output = sample["proponent_output"]

        try:
            probe = ask_prober(
                question=question,
                proponent_output=proponent_output,
                transcript=None,
            )
            status = "ok"
        except Exception as e:
            probe = ""
            status = "error"
            error_msg = str(e)

        record = {
            "id": sample.get("id", idx),
            "question": question,
            "proponent_output": proponent_output,
            "probe": probe,
            "status": status,
        }

        if status == "error":
            record["error"] = error_msg

        results.append(record)
        print(f"Processed {idx}: {status}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved prober outputs to {output_path}")


if __name__ == "__main__":
    main()