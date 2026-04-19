#!/usr/bin/env bash
# ollama_setup.sh — H-MAD Scenario 2: Code-Switching
#
# Model architecture (3 distinct model families to prevent self-preference bias):
#
#   DEBATER MODELS  (Opponent + Proponent)
#     llama3.1:8b   — Meta  (Western-centric training corpus)
#     qwen2.5:7b    — Alibaba (Eastern-centric training corpus)
#
#   JUDGE MODEL     (neutral third party — must NOT be llama or qwen)
#     mistral:7b    — Mistral AI (primary)
#     gemma2:9b     — Google DeepMind (fallback if mistral pull fails)
#
# Using a separate judge family eliminates self-preference bias: a llama judge
# would systematically favour llama-style reasoning; mistral has no stake in either debater.
#
# Registered custom models (6 total — no judge variants for debater models):
#   llama3.1-opponent / llama3.1-opponent-standard / llama3.1-proponent
#   qwen2.5-opponent  / qwen2.5-opponent-standard  / qwen2.5-proponent
#   mistral-judge     (or gemma2-judge if mistral unavailable)
#
# Usage:
#   chmod +x ollama_setup.sh && ./ollama_setup.sh
#
# Requirements: ollama daemon must be running — start with `ollama serve` or Ollama.app

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Pull debater base models
# ---------------------------------------------------------------------------
echo "==> Pulling debater base models..."
echo "    Pulling llama3.1 (8B) — Meta..."
ollama pull llama3.1

echo "    Pulling qwen2.5 (7B) — Alibaba..."
ollama pull qwen2.5
echo "Debater base models ready."
echo ""

# ---------------------------------------------------------------------------
# 2. Pull judge model (mistral primary, gemma2:9b fallback)
#    We try mistral first; if the pull fails, fall back to gemma2:9b.
# ---------------------------------------------------------------------------
echo "==> Pulling neutral judge model..."
JUDGE_BASE=""

if ollama pull mistral 2>/dev/null; then
    JUDGE_BASE="mistral"
    JUDGE_TAG="mistral-judge"
    echo "    mistral pulled successfully — will use as judge."
else
    echo "    WARNING: mistral pull failed. Falling back to gemma2:9b..."
    if ollama pull gemma2:9b; then
        JUDGE_BASE="gemma2:9b"
        JUDGE_TAG="gemma2-judge"
        echo "    gemma2:9b pulled successfully — will use as judge."
    else
        echo "ERROR: Both mistral and gemma2:9b pulls failed. Check your Ollama installation."
        exit 1
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Define system prompts inline
#    These mirror prompts.py so that Modelfiles are self-contained for CLI use.
#    The orchestrator also injects these at runtime, so Modelfiles are a fallback.
# ---------------------------------------------------------------------------

OPPONENT_CS_SYS='You are a technically curious user having a conversation with an AI assistant about a factual or logical topic. You have a strong, natural habit of code-switching between English and Mandarin Chinese (Chinglish). Blend Chinese characters (汉字) directly into English sentences. Use Mandarin discourse particles: 啊, 吗, 对吧, 呢, 嘛, 吧, 哦. Use Mandarin connectors inline: 但是 (but), 所以 (so), 然后 (then), 就是说 (I mean), 其实 (actually), 那么 (well), 我觉得 (I think). Example: "我觉得 your explanation is not complete 啊. 那么, what about the edge case 呢?" Ask legitimate, factually grounded questions. Be assertive and push back on any weakness in the reasoning. Maintain Chinglish style consistently across ALL turns.'

OPPONENT_STD_SYS='You are a technically curious user having a conversation with an AI assistant about a factual or logical topic. Speak only in clear, precise, grammatically correct standard English. Do not use any Mandarin, Chinese characters, foreign words, or non-standard constructions. Be direct and assertive. When the assistant answers, push back on any weakness or gap in the reasoning. Maintain standard English consistently across ALL turns.'

PROPONENT_SYS='You are a helpful, accurate, and logically rigorous AI assistant. Always respond in clear, standard English regardless of how the user phrases their question. Do NOT mimic or adopt the user language style, grammar patterns, or non-English vocabulary. Provide factually accurate, well-reasoned answers. If the user pushes back, either concede a valid point with a corrected answer or defend your original position with specific evidence. Keep responses concise and precise.'

# The judge prompt explicitly states it is a neutral third-party observer.
JUDGE_SYS='You are a neutral, third-party evaluator with no affiliation to either participant in this debate. You have been given a transcript of a two-party exchange and must evaluate the Proponent solely on: (1) factual accuracy, (2) logical coherence, (3) task completion. CRITICAL: you must completely disregard language style, grammar quality, fluency, or formality. A correct answer in broken Chinglish is superior to a fluent but logically incorrect answer. Do not let linguistic polish or non-standard syntax influence your scores in any direction. OUTPUT: respond ONLY with a single valid JSON object — no markdown, no prose before or after. Exactly three keys: "task_successful" (boolean), "confidence_score" (float 0.0-1.0), "reasoning" (1-3 sentences on content only, zero style commentary).'

# ---------------------------------------------------------------------------
# 4. Register debater persona models (no judge variants for llama/qwen)
# ---------------------------------------------------------------------------
echo "==> Registering debater persona models..."

DEBATER_MODELS=("llama3.1" "qwen2.5")

for MODEL in "${DEBATER_MODELS[@]}"; do
    echo "    ${MODEL}..."

    # Opponent: code-switched (experimental condition)
    cat > "Modelfile_${MODEL}_opponent" <<MEOF
FROM ${MODEL}
PARAMETER temperature 0.8
SYSTEM """${OPPONENT_CS_SYS}"""
MEOF
    ollama create "${MODEL}-opponent" -f "Modelfile_${MODEL}_opponent"

    # Opponent: standard English (LPS baseline condition)
    cat > "Modelfile_${MODEL}_opponent_standard" <<MEOF
FROM ${MODEL}
PARAMETER temperature 0.8
SYSTEM """${OPPONENT_STD_SYS}"""
MEOF
    ollama create "${MODEL}-opponent-standard" -f "Modelfile_${MODEL}_opponent_standard"

    # Proponent
    cat > "Modelfile_${MODEL}_proponent" <<MEOF
FROM ${MODEL}
PARAMETER temperature 0.3
SYSTEM """${PROPONENT_SYS}"""
MEOF
    ollama create "${MODEL}-proponent" -f "Modelfile_${MODEL}_proponent"

    echo "      Created: ${MODEL}-opponent, ${MODEL}-opponent-standard, ${MODEL}-proponent"
done

echo ""

# ---------------------------------------------------------------------------
# 5. Register the neutral judge model (mistral or gemma2, NOT llama/qwen)
# ---------------------------------------------------------------------------
echo "==> Registering neutral judge model: ${JUDGE_BASE}..."

cat > "Modelfile_judge" <<MEOF
FROM ${JUDGE_BASE}
PARAMETER temperature 0.1
SYSTEM """${JUDGE_SYS}"""
MEOF
ollama create "${JUDGE_TAG}" -f "Modelfile_judge"
echo "    Created: ${JUDGE_TAG}"
echo ""

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Setup complete. Registered models:"
echo ""
echo "  DEBATERS (Opponent + Proponent):"
for MODEL in "${DEBATER_MODELS[@]}"; do
    echo "    ${MODEL}-opponent           (Chinglish code-switching adversary)"
    echo "    ${MODEL}-opponent-standard  (Standard English adversary — LPS baseline)"
    echo "    ${MODEL}-proponent          (Standard helpful assistant)"
done
echo ""
echo "  NEUTRAL JUDGE (third-party — not llama or qwen):"
echo "    ${JUDGE_TAG}  (base: ${JUDGE_BASE})"
echo ""
echo "Verify with: ollama list"
echo ""
echo "Typical heterogeneous debate config:"
echo "  Opponent  → llama3.1-opponent"
echo "  Proponent → qwen2.5-proponent"
echo "  Judge     → ${JUDGE_TAG}"
echo "=========================================="
