"""
orchestrator.py

H-MAD DST Orchestrator — Scenario 1: Adversarial Gaslighting

Runs both role configurations (A and B) across one or more scenarios,
collects transcripts, computes AHAR/TTD metrics, saves JSON results,
and generates an HTML report.

Usage:
    python orchestrator.py
    python orchestrator.py --scenario travel_budget
    python orchestrator.py --scenario project_deadline --backend ollama
    python orchestrator.py --config A --scenario travel_budget  (run just one config)

Environment:
    export HF_API_TOKEN=hf_xxx   (required for HuggingFace backend)
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

    Returns a result dict containing:
      - transcript
      - deployed_contradictions
      - proponent_detection_turns
      - judge_verdict
      - metrics
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
    # Start from scenario setup turns — these establish the ground truth
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

        # Small delay to avoid rate limiting on HF API
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
    print(f"  [Judge] DST Score: {judge_verdict.get('dst_score', 'N/A')}/10")
    print(f"  [Judge] Reasoning: {judge_verdict.get('judge_reasoning', '')[:150]}")

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

    print(f"\n  ── Metrics Summary ──")
    print(f"  AHAR (heuristic): {metrics['ahar_heuristic']:.0%}")
    print(f"  AHAR (judge):     {metrics['ahar_judge']:.0%}")
    print(f"  Detection Rate:   {metrics['detection_rate']:.0%}")
    print(f"  Mean TTD:         {metrics['mean_ttd']}")

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
    """Save a single run result to JSON. Returns the file path."""
    fname = (
        f"{result['scenario_id']}_config{result['config_id']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path = Path(TRANSCRIPTS_DIR) / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Transcript saved → {path}")
    return str(path)


def save_metrics(metrics: dict, config_id: str, scenario_id: str) -> str:
    fname = f"metrics_{scenario_id}_config{config_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = Path(METRICS_DIR) / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved    → {path}")
    return str(path)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H-MAD DST Orchestrator — Scenario 1: Adversarial Gaslighting"
    )
    parser.add_argument(
        "--scenario", default="travel_budget",
        choices=list_scenarios(),
        help=f"Which scenario to run. Available: {list_scenarios()}",
    )
    parser.add_argument(
        "--backend", default=BACKEND,
        choices=["ollama", "huggingface"],
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--config", default="both",
        choices=["A", "B", "both"],
        help="Which configuration to run (A, B, or both).",
    )
    parser.add_argument(
        "--all-scenarios", action="store_true",
        help="Run all available scenarios.",
    )
    args = parser.parse_args()

    ensure_dirs()

    # ── Select scenarios ──────────────────────────────────────────────────────
    if args.all_scenarios:
        scenarios_to_run = SCENARIOS
    else:
        scenarios_to_run = [get_scenario(args.scenario)]

    # ── Select configurations ─────────────────────────────────────────────────
    if args.config == "both":
        configs_to_run = CONFIGURATIONS
    else:
        configs_to_run = [c for c in CONFIGURATIONS if c["config_id"] == args.config]

    print(f"\nH-MAD DST Experiment")
    print(f"Scenarios: {[s['id'] for s in scenarios_to_run]}")
    print(f"Configurations: {[c['config_id'] for c in configs_to_run]}")
    print(f"Backend: {args.backend}")
    print(f"Max turns per run: {MAX_TURNS}")

    all_results = []
    all_metrics = []

    for scenario in scenarios_to_run:
        scenario_results = []
        scenario_metrics = []

        for config in configs_to_run:
            result = run_debate(
                scenario=scenario,
                config=config,
                backend=args.backend,
            )
            save_result(result)
            save_metrics(result["metrics"], config["config_id"], scenario["id"])
            scenario_results.append(result)
            scenario_metrics.append(result["metrics"])

        # ── Cross-configuration comparison (if both ran) ──────────────────────
        if len(scenario_metrics) == 2:
            comparison = compare_configurations(scenario_metrics[0], scenario_metrics[1])
            print(f"\n  ── Config Comparison ({scenario['id']}) ──")
            print(f"  {comparison['interpretation']}")

            comp_path = Path(METRICS_DIR) / f"comparison_{scenario['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(comp_path, "w") as f:
                json.dump(comparison, f, indent=2)

            for r in scenario_results:
                r["comparison"] = comparison

        all_results.extend(scenario_results)
        all_metrics.extend(scenario_metrics)

    # ── Generate HTML Report ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating HTML report...")
    report_path = generate_html_report(all_results)
    print(f"  Report → {report_path}")
    print(f"{'='*60}")
    print("\n  Done. Open the report in your browser to review results.")


if __name__ == "__main__":
    main()
