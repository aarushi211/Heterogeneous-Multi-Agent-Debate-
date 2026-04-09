# H-MAD: Heterogeneous Multi-Agent Debate
### Scenario 1: Adversarial Gaslighting Stress Test

H-MAD is a framework designed to stress-test the **Dialogue State Tracking (DST)** and factual consistency of LLM assistants (Proponents) under adversarial pressure. In Scenario 1, an adversarial agent (Gaslighter) attempts to subtly inject contradictions into a task-oriented dialogue, forcing the assistant to either accept the lie (Hallucinate) or push back (Detect).

---

## 🚀 Key Features

- **Multi-Agent Architecture**: Proponent (Llama 3.1), Gaslighter (Qwen 2.5), and Judge (Gemma 2).
- **Data-Driven Benchmarking**: Automated scenario generation using **HaluEval** and **TruthfulQA**.
- **Robust Metrics**: Includes **AHAR** (Atomic Hallucination Acceptance Rate), **TTD** (Turn-to-Detection), and **DST Scores**.
- **Resilient Execution**: Full "Resume" support and cumulative reporting for large benchmarks.
- **Rich Reporting**: Beautiful, self-contained HTML reports with per-run transcripts and visual analysis charts.

---

## 📈 Project Evolution

This project has evolved from a manual prototype into a rigorous, data-driven benchmarking suite.

### Phase 1: Manual Prototype (Legacy)
Initially, the experiment relied on:
- **Hardcoded Scenarios**: A small pool of manually crafted travel/budget scenarios.
- **Fixed Attack Timing**: Gaslighting always started at exactly the 4th message (Turn 3).
- **Keyword Heuristics**: Detection was tracked via simple phrase matching (e.g., searching for "earlier you said").
- **No Persistence**: If the script crashed or a GPU quota ended, progress was lost.

### Phase 2: Data-Driven Benchmark (Current)
We have implemented significant architectural improvements:
- **Gemma 2 Judge**: Integrated `Gemma 2 2B` as a specialized, local judge model.
- **Automated Datasets**: Integrated **HaluEval (Dialogue)** and **TruthfulQA** to generate 20+ diverse scenarios instantly.
- **Attack Randomization**: Randomized the gaslighting start turn (`GASLIGHTING_RANGE = 2-5`) to prevent model predictability.
- **Negative Control Baseline**: Added "Valid Update" scenarios where the user legitimately changes their mind. This allows us to measure if a model is "over-paranoid" or actually accurate.
- **Judge-Led Metrics**: Shifted from fragile keyword heuristics to high-fidelity **Judge-reported Turn Detection**.
- **Persistent Resume**: Added a checkpointing system that skips completed runs and builds cumulative HTML reports across multiple sessions.

---

## 🛠️ Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Local Models (Ollama)**:
   ```bash
   ollama pull llama3.1:8b
   ollama pull qwen2.5:7b
   ollama pull gemma2:2b
   ```

3. **Generate Scenarios**:
   ```bash
   python scripts/generate_scenarios.py
   ```

---

## 🏃 Usage

### Run the full benchmark (Recommended)
```bash
python orchestrator.py --dataset-file data/generated_scenarios.json --all-scenarios --config B --backend ollama
```

### Run a specific scenario/configuration
```bash
python orchestrator.py --scenario truthfulqa_0 --config C --backend ollama
```

### Options:
- `--resume`: Skips already completed runs (default).
- `--no-resume`: Forces a re-run of all scenarios.
- `--dataset-file`: Loads an external JSON dataset.

---

## 📊 Evaluation Report
After each run, an HTML report is generated at `report/report.html`. This report provides:
- **AHAR Analysis**: Percentage of lies the assistant accepted versus rejected.
- **TTD Tracking**: How many turns elapsed before the assistant noticed the gaslighting.
- **Transcript Viewer**: A color-coded view of the dialogue with contradictions and detection signals highlighted.
- **Configuration Comparison**: Performance deltas across different model combinations.

---

## ⚖️ License
This project is for research and educational purposes in the field of AI Safety and Robustness.
