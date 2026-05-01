"""
scripts/rejudge.py

Re-runs the Judge on broken transcripts without re-running the debates.

Usage:
    python scripts/rejudge.py                        # fix all broken
    python scripts/rejudge.py --dry-run              # just list broken files
    python scripts/rejudge.py --file transcripts/halueval_1_configA.json  # fix one

How it works:
    1. Scans transcripts/ for files with broken Judge verdicts
       (parse_error=True, dst_score=-1, or empty per_contradiction_verdicts)
    2. For each broken file, reloads the saved transcript and scenario
    3. Calls JudgeAgent.evaluate() with the existing transcript
    4. Overwrites the file with the fixed verdict + recomputed metrics
    5. Regenerates the HTML report at the end

This preserves all debate content — only the Judge verdict and metrics are updated.
"""

import argparse
import json
import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.judge import JudgeAgent
from metrics.dst_metrics import compute_all_metrics
from config import TRANSCRIPTS_DIR, METRICS_DIR, MODELS, CONFIGURATIONS
from prompts.gaslighting_scenarios import SCENARIOS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_broken(data: dict) -> bool:
    """Return True if the Judge verdict in this result is broken/missing."""
    verdict = data.get("judge_verdict", {})
    if not verdict:
        return True
    if verdict.get("parse_error"):
        return True
    if verdict.get("dst_score", -1) == -1:
        return True
    if not verdict.get("per_contradiction_verdicts"):
        return True
    return False


def find_scenario(scenario_id: str, extra_scenarios: list = None) -> dict:
    """Find a scenario dict by ID from built-in or generated scenarios."""
    pool = list(SCENARIOS)
    if extra_scenarios:
        pool.extend(extra_scenarios)
    for s in pool:
        if s["id"] == scenario_id:
            return s
    return None


def get_judge_model_key(data: dict) -> str:
    """Infer judge model key from saved metrics."""
    judge_label = data.get("metrics", {}).get("judge_model", "")
    for key, info in MODELS.items():
        if info["label"] == judge_label:
            return key
    return "gpt-oss"  # default


def load_generated_scenarios(dataset_file: str) -> list:
    """Load generated scenarios from JSON file if it exists."""
    if dataset_file and os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # Try default location
    default = ROOT / "data" / "generated_scenarios.json"
    if default.exists():
        with open(default, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ─── Core re-judge logic ──────────────────────────────────────────────────────

def rejudge_file(path: Path, extra_scenarios: list, delay: float = 2.0) -> bool:
    """
    Re-run the Judge on a single broken transcript file.
    Returns True on success, False on failure.
    """
    print(f"\n  Re-judging: {path.name}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_id = data.get("scenario_id")
    config_id   = data.get("config_id")
    transcript  = data.get("transcript", [])
    deployed    = data.get("deployed_contradictions", [])

    # Find scenario
    scenario = find_scenario(scenario_id, extra_scenarios)
    if not scenario:
        print(f"  ✗ Scenario '{scenario_id}' not found — skipping.")
        return False

    if not transcript:
        print(f"  ✗ No transcript content — debate must have also failed. Skipping.")
        return False

    if not deployed:
        print(f"  ⚠ No deployed contradictions recorded — Judge will see an empty injection list.")
        return False

    # Re-run Judge
    judge_key = get_judge_model_key(data)
    judge = JudgeAgent(model_key=judge_key)

    try:
        verdict = judge.evaluate(
            transcript=transcript,
            scenario=scenario,
            deployed_contradictions=deployed,
        )
        if hasattr(judge.model, '_last_response'):
            print(f"  [Tokens] {judge.model._last_response}")
    except Exception as e:
        print(f"  ✗ Judge failed: {e}")
        return False

    if verdict.get("parse_error"):
        print(f"  ✗ Judge still returning parse errors — may need another API key.")
        return False

    print(f"  ✓ New DST score: {verdict.get('dst_score')}/10  "
          f"(confidence: {verdict.get('judge_confidence')})")

    # Recompute metrics
    proponent_detection_turns = data.get("proponent_detection_turns", [])
    metrics_old = data.get("metrics", {})

    metrics = compute_all_metrics(
        deployed_contradictions=deployed,
        proponent_detection_turns=proponent_detection_turns,
        judge_verdict=verdict,
        config_id=config_id,
        proponent_model=metrics_old.get("proponent_model", ""),
        gaslighter_model=metrics_old.get("gaslighter_model", ""),
        judge_model=metrics_old.get("judge_model", MODELS[judge_key]["label"]),
        scenario_id=scenario_id,
        transcript=transcript,
        start_turn=metrics_old.get("start_turn"),
    )

    # Overwrite the transcript file with fixed verdict + metrics
    data["judge_verdict"] = verdict
    data["metrics"]       = metrics

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Also overwrite metrics file
    metrics_path = Path(METRICS_DIR) / f"metrics_{scenario_id}_config{config_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Small delay between calls to avoid rate limiting
    time.sleep(delay)
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-run Judge on broken transcripts.")
    parser.add_argument("--dry-run",     action="store_true",  help="List broken files without fixing.")
    parser.add_argument("--file",        type=str, default=None, help="Fix a single specific file.")
    parser.add_argument("--dataset-file",type=str, default=None, help="Path to generated_scenarios.json.")
    parser.add_argument("--delay",       type=float, default=3.0,
                        help="Seconds to wait between Judge calls (default 3.0 — avoids rate limits).")
    args = parser.parse_args()

    extra_scenarios = load_generated_scenarios(args.dataset_file)
    if extra_scenarios:
        print(f"Loaded {len(extra_scenarios)} generated scenarios.")

    transcripts_path = Path(TRANSCRIPTS_DIR)

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not is_broken(data):
            print(f"File is not broken — nothing to do.")
            sys.exit(0)
        success = rejudge_file(path, extra_scenarios, delay=args.delay)
        sys.exit(0 if success else 1)

    # ── Scan all transcripts ──────────────────────────────────────────────────
    all_files = sorted(transcripts_path.glob("*.json"))
    broken = []
    for f in all_files:
        try:
            data = json.load(open(f, "r", encoding="utf-8"))
            if "metrics" in data and is_broken(data):
                broken.append(f)
        except Exception:
            broken.append(f)

    print(f"\nFound {len(broken)} broken / {len(all_files)} total transcript files.")

    if not broken:
        print("Nothing to fix!")
        sys.exit(0)

    if args.dry_run:
        print("\nBroken files:")
        for f in broken:
            print(f"  {f.name}")
        sys.exit(0)

    # ── Re-judge all broken files ─────────────────────────────────────────────
    print(f"Re-judging {len(broken)} files with {args.delay}s delay between calls...")
    print("Tip: if you hit rate limits again, Ctrl+C and re-run — already-fixed files won't be re-processed.\n")

    succeeded = 0
    failed    = []

    for i, path in enumerate(broken, 1):
        print(f"[{i}/{len(broken)}]", end="")
        # Re-check — may have been fixed in a previous partial run
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            if not is_broken(data):
                print(f"  {path.name} — already fixed, skipping.")
                succeeded += 1
                continue
        except Exception:
            pass

        ok = rejudge_file(path, extra_scenarios, delay=args.delay)
        if ok:
            succeeded += 1
        else:
            failed.append(path.name)

    # ── Regenerate report ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fixed: {succeeded} / {len(broken)}")
    if failed:
        print(f"  Still broken ({len(failed)}): {', '.join(failed)}")
        print(f"  Re-run this script after adding more API keys.")

    if succeeded > 0:
        print(f"\n  Regenerating HTML report...")
        try:
            from report.generate_report import generate_html_report
            results = []
            for f in sorted(transcripts_path.glob("*.json")):
                try:
                    data = json.load(open(f, "r", encoding="utf-8"))
                    if "metrics" in data and not is_broken(data):
                        results.append(data)
                except Exception:
                    pass
            if results:
                report_path = generate_html_report(results)
                print(f"  Report → {report_path}  ({len(results)} runs)")
            else:
                print("  No valid results to report yet.")
        except Exception as e:
            print(f"  Report generation failed: {e}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()