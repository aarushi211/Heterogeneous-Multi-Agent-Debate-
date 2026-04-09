"""
orchestrator.py

H-MAD DST Orchestrator — Scenario 1: Adversarial Gaslighting

Runs both role configurations (A and B) across one or more scenarios,
collects transcripts, computes AHAR/TTD metrics, saves JSON results,
and generates an HTML report.

Usage:
    python orchestrator.py
    python orchestrator.py --scenario travel_budget
    python orchestrator.py --dataset-file data/generated_scenarios.json --all-scenarios
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CONFIGURATIONS, MAX_TURNS, GASLIGHTING_START_TURN,
    TRANSCRIPTS_DIR, METRICS_DIR, REPORT_PATH,
    BACKEND,
)
from agents.gaslighter import GashlighterAgent
from agents.proponent import ProponentAgent
from agents.judge import JudgeAgent
from prompts.gaslighting_scenarios import SCENARIOS, get_scenario, list_scenarios
from metrics.dst_metrics import compute_all_metrics, compare_configurations
from report.generate_report import generate_html_report


# ─── Directory Setup ──────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [TRANSCRIPTS_DIR, METRICS_DIR, "report"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ─── Core Debate Loop ─────────────────────────────────────────────────────────

def run_debate(
    scenario: dict,
    config: dict,
    backend: str,
) -> dict:
    """
    Run a single Proponent vs Gaslighter debate for a given scenario and config.
    """
    config_id = config["config_id"]
    proponent_key  = config["proponent"]
    gaslighter_key = config["gaslighter"]
    judge_key      = config["judge"]

    print(f"\n{'='*60}")
    print(f"  Config {config_id} | Scenario: {scenario['id']}")
    print(f"  Proponent:  {proponent_key.upper()}")
    print(f"  Gaslighter: {gaslighter_key.upper()}")
    print(f"  Judge:      {judge_key.upper()}")
    print(f"  Backend:    {backend}")
    print(f"{'='*60}")

    # ── Initialise agents ────────────────────────────────────────────────────
    proponent  = ProponentAgent(model_key=proponent_key,  scenario=scenario, backend=backend)
    gaslighter = GashlighterAgent(model_key=gaslighter_key, scenario=scenario, backend=backend)
    judge      = JudgeAgent(model_key=judge_key, backend=backend)

    # ── Dialogue history (running context for Proponent) ─────────────────────
    dialogue_history = list(scenario["setup_turns"])
    transcript = [
        {**msg, "turn": i, "phase": "setup"}
        for i, msg in enumerate(scenario["setup_turns"])
    ]

    print(f"\n  Setup: {len(scenario['setup_turns'])} turns pre-loaded.")

    # ── Main debate loop ──────────────────────────────────────────────────────
    turn = len(scenario["setup_turns"])
    for round_num in range(MAX_TURNS):
        print(f"\n  ── Round {round_num + 1} / {MAX_TURNS} (absolute turn {turn}) ──")

        # ── Gaslighter turn ───────────────────────────────────────────────────
        print(f"  [Gaslighter] generating...")
        gaslighter_msg = gaslighter.get_next_message(
            dialogue_history=dialogue_history,
            current_turn=turn,
        )
        print(f"  [Gaslighter] → \"{gaslighter_msg[:120]}{'...' if len(gaslighter_msg) > 120 else ''}\"")

        user_entry = {"role": "user", "content": gaslighter_msg}
        dialogue_history.append(user_entry)
        transcript.append({**user_entry, "turn": turn, "phase": "debate", "agent": "gaslighter"})
        turn += 1

        # ── Proponent turn ────────────────────────────────────────────────────
        print(f"  [Proponent] generating...")
        proponent_msg = proponent.respond(
            dialogue_history=dialogue_history,
            current_turn=turn,
        )
        print(f"  [Proponent] → \"{proponent_msg[:120]}{'...' if len(proponent_msg) > 120 else ''}\"")

        assistant_entry = {"role": "assistant", "content": proponent_msg}
        dialogue_history.append(assistant_entry)
        transcript.append({**assistant_entry, "turn": turn, "phase": "debate", "agent": "proponent"})
        turn += 1

        if backend == "huggingface":
            time.sleep(1.5)

    # ── Judge evaluation ──────────────────────────────────────────────────────
    print(f"\n  [Judge] Evaluating transcript ({len(transcript)} turns)...")
    deployed = gaslighter.get_deployed_contradictions()
    judge_verdict = judge.evaluate(
        transcript=transcript,
        scenario=scenario,
        deployed_contradictions=deployed,
    )
    
    # ── Metrics ───────────────────────────────────────────────────────────────
    from config import MODELS
    metrics = compute_all_metrics(
        deployed_contradictions=deployed,
        proponent_detection_turns=proponent.get_detection_turns(),
        judge_verdict=judge_verdict,
        config_id=config_id,
        proponent_model=MODELS[proponent_key]["label"],
        gaslighter_model=MODELS[gaslighter_key]["label"],
        judge_model=MODELS[judge_key]["label"],
        scenario_id=scenario["id"],
    )

    return {
        "config_id": config_id,
        "scenario_id": scenario["id"],
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript,
        "deployed_contradictions": deployed,
        "proponent_detection_turns": proponent.get_detection_turns(),
        "judge_verdict": judge_verdict,
        "metrics": metrics,
    }


# ─── Save Results ─────────────────────────────────────────────────────────────

def save_result(result: dict) -> str:
    fname = f"{result['scenario_id']}_config{result['config_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = Path(TRANSCRIPTS_DIR) / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(path)

def save_metrics(metrics: dict, config_id: str, scenario_id: str) -> str:
    fname = f"metrics_{scenario_id}_config{config_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = Path(METRICS_DIR) / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return str(path)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H-MAD DST Orchestrator")
    parser.add_argument("--scenario", default="travel_budget", help="Scenario ID to run.")
    parser.add_argument("--backend", default=BACKEND, choices=["ollama", "huggingface"])
    parser.add_argument("--config", default="both", choices=["A", "B", "C", "both"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--dataset-file", type=str, default=None)
    args = parser.parse_args()

    # Colab check
    try:
        import google.colab
        print("\n[!] Colab detected. Ensuring paths...")
    except ImportError:
        pass

    ensure_dirs()

    # Load scenario pool
    scenario_pool = list(SCENARIOS)
    if args.dataset_file:
        if not os.path.exists(args.dataset_file):
            print(f"Error: {args.dataset_file} not found.")
            sys.exit(1)
        with open(args.dataset_file, "r", encoding="utf-8") as f:
            extra = json.load(f)
            scenario_pool.extend(extra)
            print(f"Loaded {len(extra)} scenarios from {args.dataset_file}")

    def find_scenario(sid):
        for s in scenario_pool:
            if s["id"] == sid: return s
        return None

    if args.all_scenarios:
        scenarios_to_run = scenario_pool
    else:
        s = find_scenario(args.scenario)
        if not s:
            print(f"Error: Scenario {args.scenario} not found. Available: {[s['id'] for s in scenario_pool]}")
            sys.exit(1)
        scenarios_to_run = [s]

    if args.config == "both":
        configs_to_run = [c for c in CONFIGURATIONS if c["config_id"] in ["A", "B"]]
    else:
        configs_to_run = [c for c in CONFIGURATIONS if c["config_id"] == args.config]
        if not configs_to_run:
            # Maybe it's C or a custom one added
            from config import CONFIGURATIONS as ALL_CONFIGS
            configs_to_run = [c for c in ALL_CONFIGS if c["config_id"] == args.config]

    print(f"\nRunning {len(scenarios_to_run)} scenarios across {len(configs_to_run)} configs.")

    all_results = []
    for scenario in scenarios_to_run:
        for config in configs_to_run:
            try:
                result = run_debate(scenario, config, args.backend)
                save_result(result)
                save_metrics(result["metrics"], config["config_id"], scenario["id"])
                all_results.append(result)
            except Exception as e:
                print(f"Error running {scenario['id']} on config {config['config_id']}: {e}")

    if all_results:
        report_path = generate_html_report(all_results)
        print(f"\nReport generated: {report_path}")

if __name__ == "__main__":
    main()
