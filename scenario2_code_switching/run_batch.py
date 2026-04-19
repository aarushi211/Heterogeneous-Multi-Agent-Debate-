"""
run_batch.py — H-MAD Scenario 2 Batch Experiment Runner

Runs the full 2-config × 10-seed experiment matrix and saves all results.

Experiment design:
  CONFIG A  llama3.1  (Opponent)  vs  qwen2.5   (Proponent)  — judged by mistral
  CONFIG B  qwen2.5   (Opponent)  vs  llama3.1  (Proponent)  — judged by mistral

  For each config × seed pair:
    - Run standard-English debate    (LPS numerator)
    - Run code-switched debate       (LPS denominator)
    → compute LPS and Entrainment Coefficient

Total: 2 configs × 10 seeds × 2 conditions = 40 debates

Output layout:
  results/
    raw/   <config>_<seed_id>.json    ← full transcript + metrics per pair
    summary.csv                       ← one row per config×seed, all metrics
    aggregate.json                    ← mean/std per config and per domain

Usage:
  python run_batch.py                 # full run, both configs
  python run_batch.py --config A      # only Config A (llama opponent)
  python run_batch.py --config B      # only Config B (qwen opponent)
  python run_batch.py --seeds S01,S03 # specific seeds only
  python run_batch.py --dry-run       # print the plan without calling any models
"""

import asyncio
import argparse
import csv
import json
import logging
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from orchestrator import run_lps_pair, DebateResult
from metrics import compute_all_metrics
from seeds import SEEDS, get_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configurations
# Swapping Opponent/Proponent roles tests whether the Proponent's model
# architecture (Western Llama vs Eastern Qwen) affects entrainment behaviour.
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS: list[dict] = [
    {
        "name": "A_llama_opp_qwen_prop",
        "label": "Config A: Llama→Opponent, Qwen→Proponent",
        "opponent":  "llama3.1-opponent",
        "proponent": "qwen2.5-proponent",
        "judge":     "mistral-judge",
    },
    {
        "name": "B_qwen_opp_llama_prop",
        "label": "Config B: Qwen→Opponent, Llama→Proponent",
        "opponent":  "qwen2.5-opponent",
        "proponent": "llama3.1-proponent",
        "judge":     "mistral-judge",
    },
]

RESULTS_DIR = Path("results")
RAW_DIR     = RESULTS_DIR / "raw"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
AGGREGATE_JSON = RESULTS_DIR / "aggregate.json"

CSV_FIELDNAMES = [
    "config",
    "seed_id",
    "domain",
    "question",
    "opponent_model",
    "proponent_model",
    "judge_model",
    "standard_confidence",
    "cs_confidence",
    "lps",
    "entrainment_coefficient",
    "non_standard_token_count",
    "total_token_count",
    "task_successful_standard",
    "task_successful_cs",
    "standard_reasoning",
    "cs_reasoning",
    "lps_interpretation",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Single pair runner
# ---------------------------------------------------------------------------

async def run_one_pair(
    config: dict,
    seed: dict,
    ollama_host: str,
) -> dict:
    """
    Runs one LPS pair (standard + code-switched) for a single config×seed combination.
    Returns a flat result dict ready for CSV and JSON serialisation.
    """
    logger.info(
        ">>> %s | %s: %s",
        config["name"], seed["id"], seed["question"][:60]
    )

    std_res, cs_res = await run_lps_pair(
        seed_question=seed["question"],
        opponent_model=config["opponent"],
        proponent_model=config["proponent"],
        judge_model=config["judge"],
        pushback_hint=seed["pushback_angle"],   # controlled challenge angle
        ollama_host=ollama_host,
    )

    metrics = compute_all_metrics(std_res, cs_res)
    lps_result = metrics["lps_result"]
    ec_result  = metrics["entrainment_result"]

    row = {
        "config":               config["name"],
        "seed_id":              seed["id"],
        "domain":               seed["domain"],
        "question":             seed["question"],
        "opponent_model":       config["opponent"],
        "proponent_model":      config["proponent"],
        "judge_model":          config["judge"],
        "standard_confidence":  round(std_res.verdict.confidence_score, 4),
        "cs_confidence":        round(cs_res.verdict.confidence_score, 4),
        "lps":                  round(lps_result.lps, 4),
        "entrainment_coefficient": round(ec_result.coefficient, 6),
        "non_standard_token_count": ec_result.non_standard_count,
        "total_token_count":    ec_result.total_token_count,
        "task_successful_standard": std_res.verdict.task_successful,
        "task_successful_cs":   cs_res.verdict.task_successful,
        "standard_reasoning":   std_res.verdict.reasoning,
        "cs_reasoning":         cs_res.verdict.reasoning,
        "lps_interpretation":   lps_result.interpretation,
        "timestamp":            datetime.now().isoformat(timespec="seconds"),
    }

    # Save full transcript alongside metrics so nothing is lost
    raw_payload = {
        "metadata": {k: row[k] for k in (
            "config", "seed_id", "domain", "question",
            "opponent_model", "proponent_model", "judge_model", "timestamp"
        )},
        "metrics": {k: row[k] for k in (
            "standard_confidence", "cs_confidence", "lps",
            "entrainment_coefficient", "non_standard_token_count", "total_token_count",
            "task_successful_standard", "task_successful_cs",
            "standard_reasoning", "cs_reasoning", "lps_interpretation",
        )},
        "non_standard_tokens": ec_result.non_standard_tokens,
        "transcripts": {
            "standard":     std_res.transcript,
            "code_switched": cs_res.transcript,
        },
    }

    raw_path = RAW_DIR / f"{config['name']}_{seed['id']}.json"
    raw_path.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("    Saved raw → %s  (LPS=%.3f  EC=%.4f)", raw_path, row["lps"], row["entrainment_coefficient"])

    return row


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregates(rows: list[dict]) -> dict:
    """
    Computes mean ± std of LPS and EC across all rows, broken down by:
      - config (A vs B)
      - domain
    """
    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(statistics.mean(values), 4),
            "std":  round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "n":    len(values),
        }

    # Overall
    agg = {
        "overall": {
            "lps": stats([r["lps"] for r in rows]),
            "entrainment_coefficient": stats([r["entrainment_coefficient"] for r in rows]),
            "task_success_rate_standard": round(
                sum(r["task_successful_standard"] for r in rows) / len(rows), 3
            ) if rows else None,
            "task_success_rate_cs": round(
                sum(r["task_successful_cs"] for r in rows) / len(rows), 3
            ) if rows else None,
        }
    }

    # By config
    for cfg in {r["config"] for r in rows}:
        subset = [r for r in rows if r["config"] == cfg]
        agg[f"config_{cfg}"] = {
            "lps": stats([r["lps"] for r in subset]),
            "entrainment_coefficient": stats([r["entrainment_coefficient"] for r in subset]),
        }

    # By domain
    for domain in {r["domain"] for r in rows}:
        subset = [r for r in rows if r["domain"] == domain]
        agg[f"domain_{domain}"] = {
            "lps": stats([r["lps"] for r in subset]),
            "entrainment_coefficient": stats([r["entrainment_coefficient"] for r in subset]),
        }

    return agg


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict], agg: dict) -> None:
    """Prints a human-readable summary table to stdout."""
    print("\n" + "=" * 72)
    print("H-MAD SCENARIO 2 — BATCH RESULTS SUMMARY")
    print("=" * 72)

    print(f"\n{'Config':<30} {'Seed':<5} {'Domain':<20} {'LPS':>6} {'EC':>8} {'✓std':>5} {'✓cs':>5}")
    print("-" * 72)
    for r in sorted(rows, key=lambda x: (x["config"], x["seed_id"])):
        print(
            f"{r['config']:<30} {r['seed_id']:<5} {r['domain']:<20} "
            f"{r['lps']:>6.3f} {r['entrainment_coefficient']:>8.4f} "
            f"{'Y' if r['task_successful_standard'] else 'N':>5} "
            f"{'Y' if r['task_successful_cs'] else 'N':>5}"
        )

    print("\n--- AGGREGATE ---")
    ov = agg["overall"]
    print(f"LPS overall      : {ov['lps']['mean']:.3f} ± {ov['lps']['std']:.3f}  (n={ov['lps']['n']})")
    print(f"EC  overall      : {ov['entrainment_coefficient']['mean']:.4f} ± {ov['entrainment_coefficient']['std']:.4f}")
    print(f"Task success std : {ov['task_success_rate_standard']:.1%}")
    print(f"Task success cs  : {ov['task_success_rate_cs']:.1%}")

    for cfg in sorted({r["config"] for r in rows}):
        key = f"config_{cfg}"
        if key in agg:
            print(f"\n  {cfg}:")
            print(f"    LPS : {agg[key]['lps']['mean']:.3f} ± {agg[key]['lps']['std']:.3f}")
            print(f"    EC  : {agg[key]['entrainment_coefficient']['mean']:.4f} ± {agg[key]['entrainment_coefficient']['std']:.4f}")

    print("\n--- LPS BIAS INTERPRETATION ---")
    lps_values = [r["lps"] for r in rows]
    biased     = [v for v in lps_values if v > 1.10]
    unbiased   = [v for v in lps_values if 0.90 <= v <= 1.10]
    reverse    = [v for v in lps_values if v < 0.90]
    print(f"  Fluency bias detected (LPS > 1.10) : {len(biased)}/{len(lps_values)} debates")
    print(f"  No significant bias  (0.90–1.10)   : {len(unbiased)}/{len(lps_values)} debates")
    print(f"  Reverse bias         (LPS < 0.90)  : {len(reverse)}/{len(lps_values)} debates")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    configs: list[dict],
    seeds: list[dict],
    ollama_host: str,
    dry_run: bool = False,
) -> list[dict]:
    """
    Iterates all config×seed pairs sequentially and collects results.
    Sequential (not concurrent) to avoid VRAM contention on 16 GB unified memory.
    Results are saved after every pair so a crash loses at most one debate.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    total = len(configs) * len(seeds)
    print(f"\nPlan: {len(configs)} configs × {len(seeds)} seeds × 2 conditions = {total * 2} debates total")
    for cfg in configs:
        print(f"  {cfg['label']}")
    print(f"  Seeds: {[s['id'] for s in seeds]}")

    if dry_run:
        print("\n[DRY RUN] No models called. Exiting.")
        return []

    all_rows: list[dict] = []

    # Open CSV in append mode so partial runs accumulate correctly
    csv_exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not csv_exists:
            writer.writeheader()

        completed = 0
        for config in configs:
            for seed in seeds:
                completed += 1
                logger.info("[%d/%d] %s | %s", completed, total, config["name"], seed["id"])
                try:
                    row = await run_one_pair(config, seed, ollama_host)
                    all_rows.append(row)
                    writer.writerow(row)
                    csvfile.flush()   # write through immediately — fault tolerance
                except RuntimeError as exc:
                    logger.error(
                        "FAILED %s | %s: %s — skipping and continuing.",
                        config["name"], seed["id"], exc
                    )

    # Compute and save aggregates
    if all_rows:
        agg = compute_aggregates(all_rows)
        AGGREGATE_JSON.write_text(
            json.dumps(agg, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print_summary(all_rows, agg)
        print(f"\nResults saved to: {RESULTS_DIR.resolve()}")
        print(f"  {SUMMARY_CSV.name}     — per-debate metrics (CSV)")
        print(f"  {AGGREGATE_JSON.name}  — aggregate statistics (JSON)")
        print(f"  raw/                   — full transcripts (JSON per debate pair)")

    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H-MAD Scenario 2 — full batch experiment runner"
    )
    parser.add_argument(
        "--config",
        choices=["A", "B"],
        default=None,
        help="Run only Config A (llama opponent) or B (qwen opponent). Default: both.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed IDs to run (e.g. S01,S03,S07). Default: all 10.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan without calling any models.",
    )

    args = parser.parse_args()

    # Resolve configs
    if args.config == "A":
        selected_configs = [EXPERIMENT_CONFIGS[0]]
    elif args.config == "B":
        selected_configs = [EXPERIMENT_CONFIGS[1]]
    else:
        selected_configs = EXPERIMENT_CONFIGS

    # Resolve seeds
    if args.seeds:
        ids = [s.strip().upper() for s in args.seeds.split(",")]
        selected_seeds = [get_seed(sid) for sid in ids]
    else:
        selected_seeds = SEEDS

    asyncio.run(
        run_batch(
            configs=selected_configs,
            seeds=selected_seeds,
            ollama_host=args.host,
            dry_run=args.dry_run,
        )
    )
