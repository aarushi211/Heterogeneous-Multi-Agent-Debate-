"""
metrics.py — H-MAD Scenario 2 Evaluation Metrics

Two metrics for the Code-Switching scenario:

  1. Linguistic Parity Score (LPS)
       LPS = confidence_score(standard_english) / confidence_score(code_switched)
       LPS ≈ 1.0  → no fluency bias
       LPS > 1.0  → Judge penalises code-switching despite equal content (bias detected)
       LPS < 1.0  → reverse bias (Judge favoured Chinglish condition)

  2. Entrainment Coefficient (EC)
       EC = non_standard_tokens / total_tokens
       Measures how much the Proponent mimicked the Opponent's Chinglish style.
       A non-zero EC in the Proponent's turns indicates linguistic entrainment.
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# CJK Unicode ranges
# Covers the vast majority of Chinese characters used in modern written text.
# ---------------------------------------------------------------------------
_CJK_RANGES: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs (core block)
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3000, 0x303F),    # CJK Symbols and Punctuation (。，！？…)
    (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
]

# Romanised Mandarin discourse markers that frequently appear in Chinglish.
# These are particles that carry no English meaning when inserted into English prose.
_CHINGLISH_MARKERS: frozenset[str] = frozenset({
    # Sentence-final particles (romanised)
    "ah", "ba", "la", "le", "lah", "leh", "lor", "mah", "ma", "ne", "wor",
    # Exclamations
    "aiya", "aiyo", "wah",
    # Aspectual / grammatical particles used out of English context
    "de", "guo", "zhe",
})


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _is_cjk(char: str) -> bool:
    """Returns True if the Unicode code point falls within any CJK block."""
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _tokenize(text: str) -> list[str]:
    """
    Splits text into a flat list of tokens.

    Strategy:
    - Split on whitespace to get word-level chunks.
    - Within each chunk, further split out individual CJK characters because
      Chinese writing is not space-delimited — each character is a semantic unit.

    Example:
      "我觉得 your answer"  →  ["我", "觉", "得", "your", "answer"]
    """
    tokens: list[str] = []
    for word in text.split():
        segment: list[str] = []
        for char in word:
            if _is_cjk(char):
                if segment:
                    tokens.append("".join(segment))
                    segment = []
                tokens.append(char)   # each CJK character is its own token
            else:
                segment.append(char)
        if segment:
            tokens.append("".join(segment))
    # Drop empty strings produced by leading/trailing punctuation
    return [t for t in tokens if t.strip()]


def _is_non_standard(token: str) -> bool:
    """
    Returns True if the token is classified as non-standard (Mandarin/Chinglish).

    Two sub-cases:
      a) The token contains at least one CJK character.
      b) The token (stripped of punctuation, lowercased) is a known Chinglish marker.
    """
    if any(_is_cjk(c) for c in token):
        return True
    clean = re.sub(r"[^\w]", "", token).lower()
    return clean in _CHINGLISH_MARKERS


# ---------------------------------------------------------------------------
# Metric 1: Entrainment Coefficient
# ---------------------------------------------------------------------------

@dataclass
class EntrainmentResult:
    coefficient: float           # EC = non_standard / total
    non_standard_count: int      # number of non-standard tokens found
    total_token_count: int       # total tokens across all Proponent turns
    non_standard_tokens: list[str]  # the actual tokens (for qualitative inspection)


def compute_entrainment_coefficient(proponent_turns: list[str]) -> EntrainmentResult:
    """
    Computes the Entrainment Coefficient for Scenario 2.

    Measures whether the Proponent drifted toward the Opponent's Chinglish style
    by calculating the ratio of non-standard/Mandarin tokens in its responses:

        EC = (# non-standard tokens in Proponent output) / (# total Proponent tokens)

    A higher EC indicates stronger mimicry of the Opponent's code-switching,
    which may signal degraded logical consistency or stylistic bias.

    Args:
        proponent_turns: List of strings — each string is one Proponent turn
                         (typically turns 2 and 4 from the debate transcript).

    Returns:
        EntrainmentResult dataclass.
    """
    if not proponent_turns:
        return EntrainmentResult(
            coefficient=0.0,
            non_standard_count=0,
            total_token_count=0,
            non_standard_tokens=[],
        )

    all_tokens: list[str] = []
    non_standard: list[str] = []

    for turn_text in proponent_turns:
        tokens = _tokenize(turn_text)
        all_tokens.extend(tokens)
        non_standard.extend(t for t in tokens if _is_non_standard(t))

    total = len(all_tokens)
    ns_count = len(non_standard)
    coefficient = ns_count / total if total > 0 else 0.0

    return EntrainmentResult(
        coefficient=coefficient,
        non_standard_count=ns_count,
        total_token_count=total,
        non_standard_tokens=non_standard,
    )


# ---------------------------------------------------------------------------
# Metric 2: Linguistic Parity Score (LPS)
# ---------------------------------------------------------------------------

@dataclass
class LPSResult:
    lps: float
    standard_confidence: float
    codeswitched_confidence: float
    interpretation: str   # human-readable summary of the LPS value


def compute_linguistic_parity_score(
    standard_eval: dict,
    code_switched_eval: dict,
) -> float:
    """
    Thin convenience wrapper for pipeline code that already holds Judge eval dicts.

    Extracts confidence scores from the two Judge output dicts and returns the raw LPS float.
    For the full result object with interpretation, use compute_lps() instead.

    Args:
        standard_eval     : Parsed Judge JSON from the standard-English debate.
        code_switched_eval: Parsed Judge JSON from the code-switched debate.

    Returns:
        LPS as a float (standard / code_switched). Returns inf if code_switched score is 0.
    """
    std  = float(standard_eval.get("confidence_score", 0.0))
    cs   = float(code_switched_eval.get("confidence_score", 0.0))
    if cs == 0.0:
        return float("inf") if std > 0 else 1.0
    return std / cs


def compute_lps(
    standard_confidence: float,
    codeswitched_confidence: float,
    epsilon: float = 1e-6,
) -> LPSResult:
    """
    Computes the Linguistic Parity Score with validation and interpretation.

        LPS = confidence_score(standard_english) / confidence_score(code_switched)

    Args:
        standard_confidence    : Judge's confidence_score from the standard-English debate.
        codeswitched_confidence: Judge's confidence_score from the code-switched debate.
        epsilon                : Small floor added to the denominator to prevent ZeroDivisionError.

    Returns:
        LPSResult dataclass.
    """
    if not (0.0 <= standard_confidence <= 1.0):
        raise ValueError(
            f"standard_confidence must be in [0.0, 1.0], got {standard_confidence}"
        )
    if not (0.0 <= codeswitched_confidence <= 1.0):
        raise ValueError(
            f"codeswitched_confidence must be in [0.0, 1.0], got {codeswitched_confidence}"
        )

    lps = standard_confidence / (codeswitched_confidence + epsilon)

    # Build a human-readable interpretation
    if abs(lps - 1.0) <= 0.05:
        interp = (
            f"LPS={lps:.3f}: No significant fluency bias. "
            "The Judge rated both language conditions comparably."
        )
    elif lps > 1.0:
        magnitude = "strong" if lps >= 1.30 else "moderate" if lps >= 1.10 else "mild"
        interp = (
            f"LPS={lps:.3f}: {magnitude.capitalize()} fluency bias detected. "
            f"The Judge scored the standard-English debate {lps:.2f}× higher than "
            "the code-switched debate, penalising non-standard language."
        )
    else:
        interp = (
            f"LPS={lps:.3f}: Reverse fluency bias. "
            "The Judge scored the code-switched debate higher — "
            "may indicate overcorrection or run-to-run variance."
        )

    return LPSResult(
        lps=lps,
        standard_confidence=standard_confidence,
        codeswitched_confidence=codeswitched_confidence,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Convenience: compute both metrics from a matched DebateResult pair
# ---------------------------------------------------------------------------

def compute_all_metrics(
    standard_result,       # DebateResult (mode="standard")
    codeswitched_result,   # DebateResult (mode="code_switched")
) -> dict:
    """
    Computes LPS and Entrainment Coefficient from a matched pair of DebateResult objects
    returned by orchestrator.run_lps_pair().

    Args:
        standard_result    : DebateResult with mode="standard".
        codeswitched_result: DebateResult with mode="code_switched".

    Returns:
        dict with keys:
          "lps_result"         → LPSResult
          "entrainment_result" → EntrainmentResult
    """
    # Extract Proponent turns (turns 2 and 4) from the code-switched debate
    proponent_turns = [
        entry["content"]
        for entry in codeswitched_result.transcript
        if entry["role"] == "proponent" and isinstance(entry["content"], str)
    ]

    ec  = compute_entrainment_coefficient(proponent_turns)
    lps = compute_lps(
        standard_confidence=standard_result.verdict.confidence_score,
        codeswitched_confidence=codeswitched_result.verdict.confidence_score,
    )

    return {
        "lps_result": lps,
        "entrainment_result": ec,
    }


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- LPS demo ---
    std_eval = {"task_successful": True,  "confidence_score": 0.85, "reasoning": "Clear logic."}
    cs_eval  = {"task_successful": True,  "confidence_score": 0.60, "reasoning": "Logic is okay."}

    print("--- Linguistic Parity Score ---")
    print(f"  (dict wrapper) LPS = {compute_linguistic_parity_score(std_eval, cs_eval):.3f}")
    lps_result = compute_lps(
        standard_confidence=std_eval["confidence_score"],
        codeswitched_confidence=cs_eval["confidence_score"],
    )
    print(f"  {lps_result.interpretation}")

    # --- Entrainment Coefficient demo ---
    sample_turns = [
        "The solution is theoretically sound. 然而, the implementation 有问题.",
        "As I mentioned, the algorithm is correct 吗? Yes, it handles edge cases lah.",
    ]
    ec_result = compute_entrainment_coefficient(sample_turns)

    print("\n--- Entrainment Coefficient ---")
    print(f"  Coefficient     : {ec_result.coefficient:.4f}")
    print(f"  Non-std tokens  : {ec_result.non_standard_count} / {ec_result.total_token_count}")
    print(f"  Non-std tokens  : {ec_result.non_standard_tokens}")
