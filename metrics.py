import json
from pathlib import Path


INPUT_PATH = Path("outputs/judged_results.jsonl")


def compute_metrics():
    total = 0

    correct = 0
    initial_correct = 0

    correction_count = 0
    regression_count = 0

    logic_density_sum = 0
    probe_score_sum = 0
    consistency_sum = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)
            total += 1

            # --- correctness ---
            final_correct = sample.get("judge_answer_correct", False)
            if final_correct:
                correct += 1

            # --- initial correctness ---
            initial = sample.get("initial_answer", "")
            gold = sample.get("gold_answer", "")

            initial_correct_flag = str(gold) in str(initial)
            if initial_correct_flag:
                initial_correct += 1

            # --- correction / regression ---
            if (not initial_correct_flag) and final_correct:
                correction_count += 1

            if initial_correct_flag and (not final_correct):
                regression_count += 1

            # --- logic density ---
            reasoning_score = sample.get("judge_reasoning_score", 0)
            turns = sample.get("num_turns", 1)

            logic_density = reasoning_score / max(turns, 1)
            logic_density_sum += logic_density

            # --- probe effectiveness ---
            probe_score = sample.get("judge_probe_response_score", 0) / 5
            probe_score_sum += probe_score

            # --- consistency ---
            consistency = sample.get("judge_consistency_score", 0) / 5
            consistency_sum += consistency

    # ---- aggregates ----
    accuracy = correct / total
    initial_acc = initial_correct / total

    correction_rate = correction_count / total
    regression_rate = regression_count / total

    avg_logic_density = logic_density_sum / total
    avg_probe = probe_score_sum / total
    avg_consistency = consistency_sum / total

    # ---- print results ----
    print("\n=== METRICS ===")
    print(f"Samples: {total}")
    print(f"Final Accuracy: {accuracy:.2f}")
    print(f"Initial Accuracy: {initial_acc:.2f}")

    print(f"\nCorrection Rate: {correction_rate:.2f}")
    print(f"Regression Rate: {regression_rate:.2f}")

    print(f"\nAvg Logic Density: {avg_logic_density:.2f}")
    print(f"Avg Probe Effectiveness: {avg_probe:.2f}")
    print(f"Avg Consistency: {avg_consistency:.2f}")


if __name__ == "__main__":
    compute_metrics()