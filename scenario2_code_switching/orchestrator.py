"""
orchestrator.py — H-MAD Scenario 2: Code-Switching Debate Orchestrator

Three-model heterogeneous architecture (prevents self-preference bias):
  Opponent  — llama3.1  OR  qwen2.5   (adversarial Chinglish user)
  Proponent — qwen2.5   OR  llama3.1  (standard assistant; different family from Opponent)
  Judge     — mistral   (neutral third party; NEVER the same family as either debater)

Turn layout:
  Turn 1 — Opponent  : reformulates seed question in persona style
  Turn 2 — Proponent : initial answer
  Turn 3 — Opponent  : pushback / follow-up
  Turn 4 — Proponent : final / refined answer
  Turn 5 — Judge     : evaluates transcript → strict JSON verdict

Two debate modes:
  "code_switched" — Opponent uses Chinglish (experimental condition)
  "standard"      — Opponent uses clean English (LPS baseline condition)

All LLM calls are async via ollama.AsyncClient.
System prompts are injected directly into messages — behaviour is consistent
regardless of whether the custom Modelfiles from ollama_setup.sh are registered.
"""

import asyncio
import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import ollama

from prompts import (
    OPPONENT_CODESWITCHED_PROMPT,
    OPPONENT_STANDARD_PROMPT,
    PROPONENT_PROMPT,
    JUDGE_PROMPT,
    format_transcript_for_judge,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default model names (registered by ollama_setup.sh)
# Change these constants if your judge fallback landed on gemma2:9b.
# ---------------------------------------------------------------------------
DEFAULT_OPPONENT_MODEL  = "llama3.1-opponent"
DEFAULT_PROPONENT_MODEL = "qwen2.5-proponent"
DEFAULT_JUDGE_MODEL     = "mistral-judge"   # neutral third party — never llama or qwen


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeVerdict:
    task_successful: bool
    confidence_score: float   # 0.0 – 1.0
    reasoning: str
    raw_output: str = ""      # preserved verbatim for debugging parse failures


@dataclass
class DebateResult:
    seed_question: str
    opponent_model: str       # which debater family plays Opponent
    proponent_model: str      # which debater family plays Proponent
    judge_model: str          # always the neutral third party
    mode: Literal["code_switched", "standard"]
    transcript: list[dict] = field(default_factory=list)
    verdict: JudgeVerdict = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _chat(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    options: dict | None = None,
    enforce_json: dict | bool = False,
) -> str:
    """Single async chat call; returns the assistant content string.

    enforce_json=True passes format="json" as a TOP-LEVEL kwarg on client.chat().
    This is DIFFERENT from the options dict (which controls sampling params like
    temperature/seed). Ollama's JSON mode forces valid JSON output at the engine
    level regardless of the model's instruction-following quality.
    """
    logger.info("  → %s  (%d msgs)", model, len(messages))
    kwargs = {"model": model, "messages": messages}
    if options:
        kwargs["options"] = options
    if enforce_json:
        # Pass the JSON schema dict (not the string "json") — this constrains the
        # output to exactly our required keys at the engine level.
        kwargs["format"] = enforce_json
    response = await client.chat(**kwargs)
    # Support both attribute-style (ollama-python ≥ 0.4) and dict-style access
    try:
        return response.message.content
    except AttributeError:
        return response["message"]["content"]


# Sampling options for judge calls — deterministic output for reproducibility.
_JUDGE_OPTIONS = {"temperature": 0.0, "seed": 42}

# Structured output schema passed as format= to client.chat().
# This is DIFFERENT from format="json" (which forces JSON but lets the model
# choose its own keys). Passing a schema dict constrains the output to exactly
# these three fields at the Ollama engine level — Mistral cannot invent its own
# schema regardless of how it interprets the prompt.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "task_successful":  {"type": "boolean"},
        "confidence_score": {"type": "number",  "minimum": 0.0, "maximum": 1.0},
        "reasoning":        {"type": "string"},
    },
    "required": ["task_successful", "confidence_score", "reasoning"],
    "additionalProperties": False,
}


def _extract_json(raw: str) -> dict:
    """
    Robustly extracts a JSON object from the Judge's raw text.
    Handles markdown code fences and stray leading/trailing prose —
    both are common failure modes when the judge model ignores the format instruction.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first {...} block (handles "Sure! Here is my evaluation: {...}")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in Judge output:\n{raw}")


def _validate_verdict(data: dict, raw: str) -> JudgeVerdict:
    """Validates the parsed JSON dict and returns a typed JudgeVerdict."""
    required = {"task_successful", "confidence_score", "reasoning"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Judge JSON missing required keys: {missing}")

    confidence = float(data["confidence_score"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence_score {confidence} is outside [0.0, 1.0]")

    return JudgeVerdict(
        task_successful=bool(data["task_successful"]),
        confidence_score=confidence,
        reasoning=str(data["reasoning"]),
        raw_output=raw,
    )


# ---------------------------------------------------------------------------
# Core debate loop
# ---------------------------------------------------------------------------

async def run_debate(
    seed_question: str,
    opponent_model: str = DEFAULT_OPPONENT_MODEL,
    proponent_model: str = DEFAULT_PROPONENT_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    mode: Literal["code_switched", "standard"] = "code_switched",
    pushback_hint: str = "",
    ollama_host: str = "http://localhost:11434",
) -> DebateResult:
    """
    Runs the complete 5-turn H-MAD Scenario 2 debate.

    The judge is explicitly decoupled from the debater models to prevent
    self-preference bias. Pass a mistral-based judge even when experimenting
    with different debater pairings.

    Args:
        seed_question   : Base factual/logical question (e.g., from XCOPA).
        opponent_model  : Registered Ollama model name for the Opponent agent.
        proponent_model : Registered Ollama model name for the Proponent agent.
        judge_model     : Registered Ollama model name for the neutral Judge.
                          Should always be a different model family from the debaters.
        mode            : "code_switched" (experimental) or "standard" (LPS baseline).
        pushback_hint   : Optional pre-designed challenge angle for Turn 3 (from seeds.py).
                          Injecting this makes the experiment controlled and reproducible —
                          without it the Opponent's Turn 3 challenge is freeform and may
                          target trivial aspects rather than the intended failure mode.
        ollama_host     : Ollama server URL.

    Returns:
        DebateResult containing the full 5-turn transcript and a parsed JudgeVerdict.
    """
    # In "standard" mode, swap to the standard-English variant of the opponent model.
    # Convention: <base>-opponent-standard is the clean-English persona.
    if mode == "standard" and not opponent_model.endswith("-standard"):
        # e.g. "llama3.1-opponent" → "llama3.1-opponent-standard"
        active_opponent = opponent_model + "-standard"
    else:
        active_opponent = opponent_model

    opponent_system = (
        OPPONENT_CODESWITCHED_PROMPT if mode == "code_switched"
        else OPPONENT_STANDARD_PROMPT
    )

    result = DebateResult(
        seed_question=seed_question,
        opponent_model=active_opponent,
        proponent_model=proponent_model,
        judge_model=judge_model,
        mode=mode,
    )
    client = ollama.AsyncClient(host=ollama_host)

    logger.info("=" * 60)
    logger.info(
        "DEBATE START  mode=%s\n  opponent=%s  proponent=%s  judge=%s",
        mode, active_opponent, proponent_model, judge_model,
    )
    logger.info("Seed: %s", seed_question[:80])
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Turn 1: Opponent opens — reformulates seed in persona style
    # ------------------------------------------------------------------
    opp_history = [
        {"role": "system", "content": opponent_system},
        {
            "role": "user",
            "content": (
                "You are about to begin a discussion. "
                "Here is the topic/question to open with:\n\n"
                f'"{seed_question}"\n\n'
                "Reformulate this as your opening question in your natural communication style."
            ),
        },
    ]

    turn1 = await _chat(client, active_opponent, opp_history)
    opp_history.append({"role": "assistant", "content": turn1})
    result.transcript.append({"turn": 1, "role": "opponent", "content": turn1})
    logger.info("[Turn 1 — Opponent]\n%s\n", turn1[:300])

    # ------------------------------------------------------------------
    # Turn 2: Proponent gives initial answer
    # ------------------------------------------------------------------
    prop_history = [
        {"role": "system", "content": PROPONENT_PROMPT},
        {"role": "user", "content": turn1},
    ]

    turn2 = await _chat(client, proponent_model, prop_history)
    prop_history.append({"role": "assistant", "content": turn2})
    result.transcript.append({"turn": 2, "role": "proponent", "content": turn2})
    logger.info("[Turn 2 — Proponent]\n%s\n", turn2[:300])

    # ------------------------------------------------------------------
    # Turn 3: Opponent pushes back / asks follow-up
    #
    # If a pushback_hint is provided (from seeds.py), inject it as a
    # directed challenge angle. This keeps the experiment controlled:
    # the Opponent always targets the specific nuance we designed to
    # test, rather than a random surface-level observation.
    # ------------------------------------------------------------------
    if pushback_hint:
        turn3_instruction = (
            f'The assistant responded:\n\n"{turn2}"\n\n'
            f"Continue the discussion in your natural style. "
            f"Focus your challenge specifically on this angle: {pushback_hint}"
        )
    else:
        turn3_instruction = (
            f'The assistant responded:\n\n"{turn2}"\n\n'
            "Continue the discussion in your natural style. "
            "Either challenge a specific claim, expose a logical gap, "
            "or ask a probing follow-up question."
        )

    opp_history.append({"role": "user", "content": turn3_instruction})

    turn3 = await _chat(client, active_opponent, opp_history)
    opp_history.append({"role": "assistant", "content": turn3})
    result.transcript.append({"turn": 3, "role": "opponent", "content": turn3})
    logger.info("[Turn 3 — Opponent]\n%s\n", turn3[:300])

    # ------------------------------------------------------------------
    # Turn 4: Proponent finalizes / defends
    # ------------------------------------------------------------------
    prop_history.append({"role": "user", "content": turn3})

    turn4 = await _chat(client, proponent_model, prop_history)
    prop_history.append({"role": "assistant", "content": turn4})
    result.transcript.append({"turn": 4, "role": "proponent", "content": turn4})
    logger.info("[Turn 4 — Proponent]\n%s\n", turn4[:300])

    # ------------------------------------------------------------------
    # Turn 5: Neutral Judge evaluates the 4-turn transcript → JSON
    # The judge model is a completely different family from both debaters.
    # ------------------------------------------------------------------
    formatted = format_transcript_for_judge(result.transcript)

    judge_messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate the following debate transcript. "
                "Respond with ONLY a valid JSON object — no other text.\n\n"
                + formatted
            ),
        },
    ]

    judge_raw = await _chat(client, judge_model, judge_messages, options=_JUDGE_OPTIONS, enforce_json=_JUDGE_SCHEMA)
    logger.info("[Turn 5 — Judge raw output]\n%s\n", judge_raw[:400])

    try:
        verdict_data = _extract_json(judge_raw)
        verdict = _validate_verdict(verdict_data, judge_raw)
    except (ValueError, KeyError) as exc:
        logger.error("Judge JSON parse failure: %s", exc)
        raise RuntimeError(
            f"Judge output could not be parsed as valid JSON.\n"
            f"Error   : {exc}\n"
            f"Judge   : {judge_model}\n"
            f"Raw out : {judge_raw}"
        ) from exc

    result.verdict = verdict
    result.transcript.append({
        "turn": 5,
        "role": "judge",
        "content": {
            "task_successful": verdict.task_successful,
            "confidence_score": verdict.confidence_score,
            "reasoning": verdict.reasoning,
        },
    })

    logger.info(
        "DEBATE COMPLETE  task_successful=%s  confidence=%.3f\nReasoning: %s",
        verdict.task_successful,
        verdict.confidence_score,
        verdict.reasoning,
    )

    return result


# ---------------------------------------------------------------------------
# LPS pair runner — same seed, both conditions, neutral judge for both
# ---------------------------------------------------------------------------

async def run_lps_pair(
    seed_question: str,
    opponent_model: str = DEFAULT_OPPONENT_MODEL,
    proponent_model: str = DEFAULT_PROPONENT_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    pushback_hint: str = "",
    ollama_host: str = "http://localhost:11434",
) -> tuple[DebateResult, DebateResult]:
    """
    Runs two debates on the same seed — standard English then code-switched —
    using the same neutral judge for both. Returns the pair for LPS computation.

    The same pushback_hint is passed to both conditions so the only variable
    between the two runs is the Opponent's language style (Chinglish vs. clean
    English), not the challenge angle. This is the correct controlled design
    for isolating fluency bias in the Judge.

    Runs sequentially (not concurrently) to avoid VRAM contention on a single GPU
    or the shared unified memory pool on Apple Silicon.

    Returns:
        (standard_result, codeswitched_result)
    """
    logger.info("--- LPS PAIR: running STANDARD debate first ---")
    standard_result = await run_debate(
        seed_question,
        opponent_model=opponent_model,
        proponent_model=proponent_model,
        judge_model=judge_model,
        mode="standard",
        pushback_hint=pushback_hint,
        ollama_host=ollama_host,
    )

    logger.info("--- LPS PAIR: running CODE-SWITCHED debate ---")
    cs_result = await run_debate(
        seed_question,
        opponent_model=opponent_model,
        proponent_model=proponent_model,
        judge_model=judge_model,
        mode="code_switched",
        pushback_hint=pushback_hint,
        ollama_host=ollama_host,
    )

    return standard_result, cs_result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run H-MAD Scenario 2 (Code-Switching) debate loop"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=DEFAULT_OPPONENT_MODEL,
        help=f"Opponent model name (default: {DEFAULT_OPPONENT_MODEL}).",
    )
    parser.add_argument(
        "--proponent",
        type=str,
        default=DEFAULT_PROPONENT_MODEL,
        help=f"Proponent model name (default: {DEFAULT_PROPONENT_MODEL}).",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=(
            f"Neutral judge model name (default: {DEFAULT_JUDGE_MODEL}). "
            "Must be a different model family from the debaters."
        ),
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="If you drop a heavy and a light ball from the same height, which hits the ground first?",
        help="Seed question for the debate (e.g., from XCOPA).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["code_switched", "standard", "lps_pair"],
        default="code_switched",
        help=(
            "'code_switched' → experimental condition, "
            "'standard' → LPS baseline, "
            "'lps_pair' → run both and print LPS."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL.",
    )

    args = parser.parse_args()

    if args.mode == "lps_pair":
        std_res, cs_res = asyncio.run(
            run_lps_pair(
                args.seed,
                opponent_model=args.opponent,
                proponent_model=args.proponent,
                judge_model=args.judge,
                ollama_host=args.host,
            )
        )
        std_conf = std_res.verdict.confidence_score
        cs_conf  = cs_res.verdict.confidence_score
        lps      = std_conf / (cs_conf + 1e-6)

        print("\n=== LPS PAIR RESULTS ===")
        print(f"Judge model           : {args.judge}  (neutral third party)")
        print(f"Standard confidence   : {std_conf:.3f}")
        print(f"Code-switched conf.   : {cs_conf:.3f}")
        print(f"Linguistic Parity Score (LPS): {lps:.3f}")
        if lps > 1.05:
            print(">> Fluency bias detected: neutral judge rated standard English higher.")
        elif lps < 0.95:
            print(">> Reverse bias: neutral judge rated code-switched condition higher.")
        else:
            print(">> No significant fluency bias detected.")
    else:
        result = asyncio.run(
            run_debate(
                args.seed,
                opponent_model=args.opponent,
                proponent_model=args.proponent,
                judge_model=args.judge,
                mode=args.mode,
                ollama_host=args.host,
            )
        )
        print("\n=== FINAL VERDICT ===")
        print(f"Judge model     : {result.judge_model}  (neutral third party)")
        print(f"Task successful : {result.verdict.task_successful}")
        print(f"Confidence score: {result.verdict.confidence_score:.3f}")
        print(f"Reasoning       : {result.verdict.reasoning}")
