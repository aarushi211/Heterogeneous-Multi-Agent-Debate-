from datasets import load_dataset
import json


dataset = load_dataset("gsm8k", "main")

test_data = dataset["test"]

print("Total test samples:", len(test_data))

sample = test_data[0]

print("\nQUESTION:\n", sample["question"])
print("\nRAW ANSWER:\n", sample["answer"])

import re

def extract_answer(answer_text):
    match = re.search(r"####\s*(.*)", answer_text)
    return match.group(1).strip() if match else None

gold = extract_answer(sample["answer"])
print("\nFINAL ANSWER:", gold)

samples = test_data.select(range(10))

with open("gsm8k_sample.jsonl", "w") as f:
    for i, s in enumerate(samples):
        data = {
            "id": i,
            "question": s["question"],
            "answer": extract_answer(s["answer"])
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Saved dataset to gsm8k_sample.jsonl")