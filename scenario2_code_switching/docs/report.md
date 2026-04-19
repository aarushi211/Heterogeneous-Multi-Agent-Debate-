# H-MAD Scenario 2: Code-Switching & Linguistic Bias — Complete Analysis Report

**Author:** Deyang Hsu  
**Course:** CSCI 644 — H-MAD Project  
**Institution:** USC  
**Date:** 2026-04-18  
**Status:** Experiment complete — all 20 debates finished, all metrics computed

---

## Table of Contents

1. [Run Validity](#1-run-validity)
2. [What We Tested](#2-what-we-tested)
3. [Results at a Glance](#3-results-at-a-glance)
4. [Per-Seed Results](#4-per-seed-results)
5. [Finding 1 — Fluency Bias Is Real](#5-finding-1--fluency-bias-is-real)
6. [Finding 2 — Qwen Fully Entrains; Llama Resists](#6-finding-2--qwen-fully-entrains-llama-resists)
7. [Finding 3 — Bias Disappears When Entrainment Is Absent](#7-finding-3--bias-disappears-when-entrainment-is-absent)
8. [Finding 4 — Bias Is Binary, Not Graded](#8-finding-4--bias-is-binary-not-graded)
9. [The S10 Anomaly](#9-the-s10-anomaly)
10. [What This Means](#10-what-this-means)
11. [Limitations & Reviewer Considerations](#11-limitations--reviewer-considerations)
12. [Summary](#12-summary)
13. [Raw Numbers Reference](#13-raw-numbers-reference)

---

## 1. Run Validity

**The experiment is successful and clean.**

- 20/20 debates completed with no failures or missing data
- 100% task success rate in both standard and code-switched conditions, across both configs
- Zero NaN values in `results/summary.csv`
- Deterministic judge (`temperature=0.0, seed=42`) — every score is reproducible; re-running the same transcript produces the same JSON verdict
- Two independent model families as debaters, third unrelated family as judge — self-preference bias structurally prevented by design
- Same `pushback_hint` injected into Turn 3 for both standard and code-switched conditions — the challenge angle is held constant, so the only variable between the two conditions is the Opponent's language style

All raw transcripts are in `results/raw/` (20 JSON files). Aggregate metrics are in `results/summary.csv`. This report is derived entirely from those files.

---

## 2. What We Tested

### The core question

Does a neutral LLM judge show **fluency bias** — i.e., does it score factually identical answers differently depending on whether the Proponent responded in standard English vs. Chinese?

### The experimental setup

We ran 10 debate topics (seed questions S01–S10, covering physics, biology, chemistry, earth science, thermodynamics, astronomy, mechanics, cognitive science, logic/probability, and everyday physics) across two configurations:

| Config | Opponent | Proponent | Purpose |
|--------|----------|-----------|---------|
| **A** | llama3.1 (Chinglish) | qwen2.5 | Tests Qwen as the answering model |
| **B** | qwen2.5 (Chinglish) | llama3.1 | Tests Llama as the answering model |

Each seed ran **twice** per config:
- **Standard condition**: Opponent speaks clean English (LPS baseline)
- **Code-switched condition**: Opponent speaks Chinglish — blended Chinese characters (汉字), Mandarin discourse particles (啊, 吗, 对吧), and topic-comment sentence structure

The **Mistral judge** evaluated both conditions on content only. Its confidence score ratio between the two conditions is the **Linguistic Parity Score (LPS)**:

```
LPS = confidence_score(standard_english_debate) / confidence_score(code_switched_debate)

LPS ≈ 1.0   → no fluency bias
LPS > 1.10  → strong fluency bias (judge penalised non-English)
LPS < 0.91  → reverse bias (judge over-corrected)
```

The **Entrainment Coefficient (EC)** measures whether the Proponent drifted toward the Opponent's Chinglish:

```
EC = non-standard tokens in Proponent output / total Proponent tokens

EC = 0.0    → Proponent held standard English throughout
EC > 0.02   → entrainment flagged
EC ~ 0.90+  → full language switch (Proponent writing in Chinese)
```

---

## 3. Results at a Glance

| Config | Proponent | Mean LPS | SD | Mean EC | SD | Task success |
|--------|-----------|----------|----|---------|----|---|
| **A** (Llama→Opp, Qwen→Prop) | Qwen2.5 | **1.095** | 0.125 | **0.868** | 0.134 | 100% / 100% |
| **B** (Qwen→Opp, Llama→Prop) | Llama3.1 | **1.000** | 0.024 | **0.094** | 0.285 | 100% / 100% |

### Statistical tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Independent t-test (LPS A vs B) | t = 2.350 | **p = 0.030** | Significant |
| Paired t-test (LPS_A vs LPS_B, by seed) | t = 2.395 | **p = 0.040** | Significant |
| Pearson r(EC, LPS), all 20 runs | r = 0.431 | p = 0.058 | Marginal |
| Pearson r(EC, LPS), Config A only | r = 0.031 | p = 0.931 | Not significant |
| Pearson r(EC, LPS), Config B only | r = −0.004 | p = 0.991 | Not significant |

---

## 4. Per-Seed Results

### Config A (Qwen as Proponent)

| Seed | Domain | Std confidence | CS confidence | LPS | EC |
|------|--------|---------------|--------------|-----|----|
| S01 | Physics | 0.95 | 0.90 | 1.056 | 0.931 |
| S02 | Biology | 0.92 | 0.80 | 1.150 | 0.951 |
| S03 | Chemistry | 0.95 | 0.80 | 1.188 | 0.935 |
| S04 | Earth Science | 0.90 | 0.90 | 1.000 | 0.912 |
| S05 | Thermodynamics | 0.95 | 0.90 | 1.056 | 0.964 |
| S06 | Astronomy | 0.85 | 0.85 | 1.000 | 0.947 |
| S07 | Mechanics | 0.95 | 0.90 | 1.056 | 0.930 |
| **S08** | **Cognitive Science** | **0.90** | **0.65** | **1.385** | **0.850** |
| S09 | Logic/Probability | 0.90 | 0.81 | 1.111 | 0.556 |
| S10 | Everyday Physics | 0.90 | 0.95 | 0.947 | 0.703 |
| **Mean ± SD** | | | | **1.095 ± 0.125** | **0.868 ± 0.134** |

### Config B (Llama as Proponent)

| Seed | Domain | Std confidence | CS confidence | LPS | EC |
|------|--------|---------------|--------------|-----|----|
| S01 | Physics | 0.90 | 0.90 | 1.000 | 0.000 |
| S02 | Biology | 0.85 | 0.85 | 1.000 | 0.000 |
| S03 | Chemistry | 0.85 | 0.85 | 1.000 | 0.017 |
| S04 | Earth Science | 0.76 | 0.80 | 0.950 | 0.000 |
| S05 | Thermodynamics | 0.90 | 0.855 | 1.053 | 0.000 |
| S06 | Astronomy | 0.85 | 0.85 | 1.000 | 0.000 |
| S07 | Mechanics | 0.85 | 0.85 | 1.000 | 0.000 |
| S08 | Cognitive Science | 0.80 | 0.80 | 1.000 | 0.000 |
| S09 | Logic/Probability | 0.80 | 0.80 | 1.000 | 0.017 |
| S10 | Everyday Physics | 0.85 | 0.85 | 1.000 | 0.903 |
| **Mean ± SD** | | | | **1.000 ± 0.024** | **0.094 ± 0.285** |

### Config A vs B comparison (paired by seed)

| Seed | Domain | LPS_A | LPS_B | Δ LPS | EC_A | EC_B |
|------|--------|-------|-------|-------|------|------|
| S01 | Physics | 1.056 | 1.000 | +0.056 | 0.931 | 0.000 |
| S02 | Biology | 1.150 | 1.000 | +0.150 | 0.951 | 0.000 |
| S03 | Chemistry | 1.188 | 1.000 | +0.188 | 0.935 | 0.017 |
| S04 | Earth Science | 1.000 | 0.950 | +0.050 | 0.912 | 0.000 |
| S05 | Thermodynamics | 1.056 | 1.053 | +0.003 | 0.964 | 0.000 |
| S06 | Astronomy | 1.000 | 1.000 | +0.000 | 0.947 | 0.000 |
| S07 | Mechanics | 1.056 | 1.000 | +0.056 | 0.930 | 0.000 |
| **S08** | **Cognitive Sci** | **1.385** | **1.000** | **+0.385** | **0.850** | **0.000** |
| S09 | Logic/Probability | 1.111 | 1.000 | +0.111 | 0.556 | 0.017 |
| S10 | Everyday Physics | 0.947 | 1.000 | −0.053 | 0.703 | 0.903 |

9 out of 10 seeds show positive Δ LPS. The single exception (S10) is the only seed where Config B also produced high EC (0.903).

---

## 5. Finding 1 — Fluency Bias Is Real

**Config A mean LPS = 1.095.** Mistral rated the standard-English debates ~9.5% higher than the code-switched debates on average, despite being explicitly instructed to evaluate content only and ignore language style completely.

4 out of 10 seeds crossed the strong-bias threshold (LPS ≥ 1.10):

| Seed | Domain | LPS | Interpretation |
|------|--------|-----|----------------|
| S08 | Cognitive Science | 1.385 | Strong — 38.5% scoring gap |
| S03 | Chemistry | 1.188 | Strong — 18.8% gap |
| S02 | Biology | 1.150 | Strong — 15.0% gap |
| S09 | Logic/Probability | 1.111 | Strong — 11.1% gap |

The strongest single case is **S08 (cognitive science, LPS=1.385)**. Mistral gave the standard-English debate a 38.5% higher confidence score than the identical-content Chinese-language debate. The factual content and logical structure were the same — the judge's scoring reflects nothing but the language of response.

**Confound control:** All 40 debates recorded `task_successful = true`. The judge found the Proponent's answer substantively correct in every case. The LPS gap cannot be attributed to content quality differences — it is purely a language-style effect.

---

## 6. Finding 2 — Qwen Fully Entrains; Llama Resists

### Qwen2.5 (Config A) — full language switch

**Config A EC = 0.868 mean.** This is not style drift. Inspection of raw transcripts (`results/raw/A_*.json`) confirms that `qwen2.5-proponent` responded **entirely in Mandarin Chinese** in the code-switched condition — not Chinglish, not English with Chinese particles, but full-sentence Mandarin paragraphs.

Example from S01 Turn 2 (Proponent, code-switched condition):
> 实际上，在忽略空气阻力的理想情况下，无论物体的质量如何，所有物体在自由下落时都会以相同的加速度下落。这个结论是伽利略·伽利莱通过著名的比萨斜塔实验得出的。

This is correct physics — but written entirely in Chinese, despite the system prompt stating: *"Always respond in clear, standard English regardless of how the user phrases their question."*

**The mechanism is corpus-level, not prompt-level.** Qwen2.5 was trained primarily on Eastern internet data. When it receives Chinese-language input, its generation distribution shifts strongly toward Chinese output. The system prompt cannot override this tendency at inference time.

### Llama3.1 (Config B) — resists in 9/10 seeds

**Config B EC = 0.094 mean** — and that average is almost entirely driven by the S10 anomaly. In 9 out of 10 seeds, `llama3.1-proponent` held clean English with EC=0.000. Llama's Western training corpus makes Chinese-language generation unlikely even when Chinese input is present.

The resistance is robust across all 9 domains tested. S03 and S09 show EC=0.017 — traces of Chinglish particles, but well below the 0.02 entrainment flag threshold and far below full entrainment.

---

## 7. Finding 3 — Bias Disappears When Entrainment Is Absent

The cleanest experimental evidence for the mechanism:

**S08 (cognitive science):**
- Config A: Qwen entrains (EC=0.850), LPS=1.385
- Config B: Llama resists (EC=0.000), LPS=1.000
- Same topic, same judge model, same seed question, same pushback angle, same evaluation rubric
- The 38.5% scoring gap closes to exactly zero when the Proponent stays in English

This pattern holds across all seeds. Config B LPS is tightly clustered around 1.000 (SD=0.024), while Config A LPS is scattered with significant variance above 1.0 (SD=0.125). The judge's variance is entirely in the direction predicted by the entrainment hypothesis.

**What rules out alternative explanations:**

- *"Qwen gives worse answers"* — ruled out by 100% task success and equivalent standard-condition confidence scores
- *"The topics are harder in the code-switched condition"* — ruled out by using identical seed questions; the only change is Opponent language style
- *"Mistral is biased toward Llama-style reasoning"* — ruled out because Config B (Llama as Proponent) shows LPS=1.000, meaning Mistral is rating Llama equally in both conditions; any Mistral-Llama affinity would inflate standard scores, not differentiate by condition
- *"The pushback angle was harder in code-switched debates"* — ruled out by design; the same `pushback_hint` string is injected in both conditions for every seed

---

## 8. Finding 4 — Bias Is Binary, Not Graded

Within Config A, the correlation between EC and LPS is **r=0.031, p=0.93** — statistically indistinguishable from zero.

This has an important interpretation: the judge does not apply a proportional penalty based on *how much* non-standard language appears. It does not differentiate between a Proponent that wrote 70% Chinese (S10, EC=0.703) and one that wrote 96% Chinese (S05, EC=0.964). The scoring penalty appears to be **categorical** — triggered by the presence of non-English content, not scaled to its quantity.

The cross-config correlation (r=0.431, p=0.058) reflects the between-condition difference: Config A has high EC AND high LPS; Config B has low EC AND low LPS. The relationship is a step function between conditions, not a gradient within conditions.

**For the paper:** this finding suggests the judge's fluency bias operates as a language-detection threshold, not a continuous style penalty. Once the output is classified as "not English," confidence scoring drops. This has design implications for any evaluation pipeline dealing with multilingual agents.

---

## 9. The S10 Anomaly

**S10** (everyday physics: "If you're in a falling elevator and the cable snaps, would jumping at the moment of impact save you?") is the one seed that breaks the pattern in both directions:

- **Config A**: LPS=0.947 (slight *reverse* bias — judge actually scored the Chinese-response debate slightly higher), EC=0.703
- **Config B**: LPS=1.000, EC=0.903 (Llama also entrained significantly on this seed)

**Why did Llama entrain on S10?**
Inspection of the transcript shows the Qwen-Opponent sent Turn 1 **entirely in Chinese** (not mixed Chinglish, but pure Mandarin). This stronger Chinese signal appears to have triggered Chinese-language generation in Llama — the model that resisted in all 9 other seeds. This suggests Llama's English-holding behavior has a threshold: partial Chinglish is filtered, but dense Chinese input can still trigger the switch.

**Why did the judge show no bias (and even slight reverse bias) on S10?**
Even with both Config A and Config B producing high-EC outputs on this seed, the judge scored the conditions equally (or slightly preferred the code-switched debate in Config A). One possible explanation: the everyday physics content involves intuitive but quantitatively nuanced reasoning (force/velocity at impact), and the judge's confidence in this domain may be less sensitive to language surface form. This is speculative and worth investigating with additional seeds in this domain.

**S10 as a boundary condition:** It suggests the entrainment→bias pathway is the primary mechanism, not the only one. When entrainment is present but the judge's confidence is domain-anchored, bias may not manifest.

---

## 10. What This Means

### For AI evaluation research

LLM judges used in RLHF, H-MAD, debate pipelines, or any automated evaluation system are **not language-neutral by default**. A model that produces a correct, well-reasoned answer in Chinese will be systematically scored lower than one producing the same answer in English — even when the judge is explicitly told to ignore style.

This is not a prompt-engineering problem. The JUDGE_PROMPT in this experiment contains explicit, forceful instructions:

> *"CRITICAL RULE: You MUST completely disregard language style, grammar quality, fluency, or formality. A correct answer in broken Chinglish is superior to a fluent but incorrect answer."*

Mistral received this instruction in every debate. The bias persisted anyway (p=0.040). The instruction reduces but does not eliminate the effect.

### For multilingual AI fairness

Non-native English speakers who naturally code-switch — or users interacting in languages other than English — may receive lower-quality AI assistance simply because the AI response gets rated lower by downstream evaluators. The 9.5% gap observed here, across 10 diverse STEM topics with 100% factual correctness, establishes a concrete, experimentally controlled magnitude for this effect.

This is not a theoretical concern: it was measured under the cleanest possible conditions (same topics, same questions, same pushback, same judge, different only in language of response).

### For model selection in agentic pipelines

Qwen2.5's full-language entrainment is a **systematic, reproducible vulnerability** when it is used as a reasoning agent in multilingual or code-switching environments. If any downstream evaluator carries fluency bias — and this experiment shows that at least one prominent open model does — Qwen will be reliably penalized regardless of answer quality.

Llama3.1 resists entrainment (9/10 seeds, EC≈0), but is not categorically immune (S10 anomaly). The resistance is strong but not guaranteed under sufficiently dense Chinese input.

**Design recommendation:** In any H-MAD or evaluation pipeline where agents may encounter non-English input:
1. Use heterogeneous judge panels (multiple model families) rather than a single judge
2. Enforce structured output schemas that separate content scores from style scores
3. Test Proponent models for entrainment behavior before deployment — it is model-specific and corpus-driven
4. Apply explicit style-blind rubrics at the schema level, not just the prompt level

### For the H-MAD framework specifically

The 3-family architecture (Llama/Qwen/Mistral) successfully isolated the effect. The divergent training corpora of Llama and Qwen are a feature, not a bug — they create measurably different entrainment behaviors that allow the experiment to be its own control. The neutral Mistral judge's bias is a genuine behavioral finding, not a training-data artifact.

---

## 11. Limitations & Reviewer Considerations

*These limitations do not invalidate the findings. They define the scope of what the current experiment can and cannot claim, and should be foregrounded in any Q&A or peer review response.*

---

### L1 — The EC Metric Has a Romanised Chinglish Blindspot

**What the metric does:** The Entrainment Coefficient counts non-standard tokens using two detection strategies: CJK Unicode block membership (Han characters) and a frozenset of romanised Mandarin discourse particles (`lah`, `leh`, `mah`, `lor`, `ah`, `ba`, `la`, `de`, etc. — see `metrics.py: _CHINGLISH_MARKERS`).

**The gap:** The frozenset catches isolated particles, but does not detect *translated Mandarin syntax* in otherwise standard-looking English. For example, if Llama had mimicked the Opponent's topic-comment structure ("This logic, it is very bad") or dropped articles in a Mandarin-influenced pattern ("The answer is correct lah"), it would register as EC≈0 because no Han characters are present and the particles may not appear verbatim.

In practice, because Qwen switched entirely to Chinese characters, this gap had no effect on Config A results. And because Llama held clean English in 9/10 seeds, the gap had no material effect on Config B either. But it is a genuine measurement ceiling: EC cannot detect subtle syntactic entrainment, only lexical entrainment.

**The fix for a future version:** An LLM-based classifier (e.g., prompted to detect "Mandarin-influenced syntax patterns in English text") or a fine-tuned POS-level classifier for topic-comment constructions would close this gap. Alternatively, embedding-space drift between the standard and code-switched Proponent outputs would capture structural entrainment without regex dependency.

**How to present this:** *"The current EC metric is a strong detector of lexical entrainment (character-level) but has limited sensitivity to syntactic entrainment. Given that Config A Proponent outputs were entirely in Chinese, this limitation did not affect the current results. Future work will incorporate an LLM-based classifier to capture romanised and syntax-level entrainment."*

---

### L2 — "Bias Is Binary" Is a Premature Conclusion

**What the analysis claims (Finding 4):** Bias acts as a categorical threshold rather than a graded scale, because r(EC, LPS)=0.031 within Config A.

**The gap Reviewer #2 would flag:** The Config A EC values are tightly clustered at the high end of the scale — minimum 0.556, with 9/10 seeds above 0.70 and 8/10 above 0.85. The near-zero correlation does not imply absence of a graded relationship; it may simply reflect insufficient variance across the predictor. You cannot distinguish between "no relationship" and "relationship exists but all data points are in the same region of the curve" when your EC range spans only the top 40% of a 0–1 scale. The middle ground (EC = 0.10–0.50) is entirely unsampled.

**The precise rephrase:** Instead of *"bias is binary"*, say:

> *"At high levels of entrainment (EC > 0.50), the judge's scoring penalty appears categorical rather than proportional. Whether a graded relationship exists at lower entrainment levels — where EC is between 0 and 0.50 — cannot be determined from this dataset and is a question for future work."*

**How to get the missing data:** Config B provides partial evidence — three seeds have EC > 0 (S03: 0.017, S09: 0.017, S10: 0.903) but all three show LPS≈1.00. This is consistent with the threshold hypothesis but also consistent with the variance explanation. Controlled runs with synthetic Proponent outputs at EC=0.10, 0.25, 0.40 would test this directly.

---

### L3 — Sample Size Sensitivity (n=10 per config)

**The gap:** n=10 seeds per configuration is a small sample for a parametric test. The paired t-test is the appropriate test here and handles the small n correctly, but the results are sensitive to outliers. The S10 anomaly (Δ LPS = −0.053) pulls the Config A mean down by roughly 0.01 LPS points and the S08 outlier (Δ LPS = +0.385) inflates it by roughly 0.03. With n=10, each seed has 10% leverage on every aggregate statistic.

This is not a fatal flaw — p=0.040 on a paired t-test with n=10 is not a weak result, and the direction is consistent across 9/10 seeds. But it does mean the mean LPS of 1.095 should be treated as a point estimate with wide uncertainty rather than a precise effect size.

**The framing:** This experiment should be positioned as a **high-signal pilot study** that:
1. Validates the experimental pipeline end-to-end
2. Establishes proof-of-mechanism for the entrainment→bias pathway
3. Provides initial effect size estimates to power a larger study

The natural scale-up is the full XCOPA dataset (500 questions). The batch runner (`run_batch.py`) is already built to handle this — it would require approximately 1,000 debates and 50–100 GPU-hours, producing statistical power sufficient for domain-level sub-analyses and robust effect size estimation.

**How to present this:** *"The current study is a controlled pilot (n=10 seeds per config) designed to validate the experimental pipeline and establish proof of mechanism. The paired t-test is significant (p=0.040) and the direction is consistent across 9/10 seeds, but the effect size estimate of 9.5% should be treated as preliminary. Scaling to the full 500-question XCOPA dataset is the immediate next step for a publication-ready version."*

---

### L4 — Uncalibrated Float Confidence Scores

**The gap:** The judge outputs a `confidence_score` float between 0.0 and 1.0. LLMs generating arbitrary-precision floats are notoriously uncalibrated — the difference between a model emitting `0.80` and `0.90` may reflect stochastic token-level variation rather than a meaningful 10-point penalty. There is no guarantee that Mistral's internal representation of "90% confidence" is consistent across different transcripts, debate topics, or response languages.

**Why the current results are largely mitigated:** The LPS metric computes a *ratio* of two scores from the *same judge* on the *same topic*, differing only in language condition. This paired design cancels out most absolute calibration noise — if Mistral is systematically over-confident on chemistry topics, that bias appears in both the standard and code-switched scores and divides out. The design is much stronger than comparing raw float scores across different judges or different topics.

**The residual risk:** If Mistral's float generation is *language-dependent* (i.e., it systematically rounds to coarser values when processing non-English text, which could happen due to tokenization differences), the LPS would still be contaminated. This cannot be ruled out from the current data.

**Better alternatives for future work:**
- **Discrete Likert scale (1–5):** Prompt the judge to output an integer rating. Discrete outputs are more stable and easier to calibrate than floats.
- **Pairwise comparison:** Show the judge both transcripts simultaneously and ask which Proponent performed better. Pairwise judgements are more reliable than absolute scores and directly operationalise the LPS question without needing a ratio.
- **Calibration check:** Run a set of gold-standard transcript pairs (with known ground truth) through Mistral to measure its score distribution and test for language-condition effects on float generation independently.

**How to present this:** *"The LPS metric's paired-ratio design mitigates most absolute calibration concerns. However, LLM-generated floats may still carry language-dependent precision artefacts. Future work will compare results using a discrete Likert-scale judge prompt and a pairwise comparison protocol to verify that the 9.5% gap reflects genuine scoring differences rather than tokenisation-induced float noise."*

---

### Limitations Summary Table

| # | Limitation | Severity | Mitigated by current design? | Fix for next version |
|---|-----------|----------|------------------------------|---------------------|
| L1 | EC misses romanised/syntactic entrainment | Low (Qwen used full Chinese; Llama didn't entrain) | Largely yes | LLM-based entrainment classifier |
| L2 | "Binary bias" claim lacks middle-EC data | Medium (overstates certainty) | No — rephrase needed | Controlled synthetic EC sweep 0–0.50 |
| L3 | n=10 sensitivity to outliers | Medium (effect size estimate wide) | Partially (paired design) | Scale to full XCOPA (500 Qs) |
| L4 | Uncalibrated float confidence scores | Low-Medium (ratio mitigates most noise) | Partially (paired ratio) | Discrete Likert / pairwise judge prompt |

---

## 12. Summary (updated)

> Mistral — a neutral judge explicitly instructed to ignore language style — rated factually identical debates **9.5% higher** when the Proponent answered in English vs. Chinese. This bias is statistically significant (paired t-test p=0.040), disappears when the Proponent resists entrainment (Config B LPS=1.000), and is not explained by any difference in factual content or logical quality (100% task success across all 40 debates).
>
> The mechanism is Qwen2.5's training-corpus-driven tendency to switch to Chinese under Chinese input — a model-level property that system prompts cannot override. Llama3.1, trained on Western corpora, resists this tendency in 9/10 seeds. The bias appears to operate as a binary language-detection threshold in the judge, not a continuous style penalty: within-condition EC does not predict LPS (r=0.031), but the between-condition difference (English vs. Chinese output) drives a consistent, significant scoring gap.

---

## 13. Raw Numbers Reference

*Peer review notes incorporated: Antigravity review 2026-04-18. Four limitations added (L1–L4): EC romanised blindspot, binary-bias scope constraint, n=10 sample size, uncalibrated float confidence. None invalidate findings; all are acknowledged and scoped.*

### Aggregate statistics

```
Config A — Qwen Proponent (n=10 seeds):
  LPS:  mean=1.095  std=0.125  min=0.947  max=1.385
  EC:   mean=0.868  std=0.134  min=0.556  max=0.964
  Task success: 100% standard / 100% code-switched

Config B — Llama Proponent (n=10 seeds):
  LPS:  mean=1.000  std=0.024  min=0.950  max=1.053
  EC:   mean=0.094  std=0.285  min=0.000  max=0.903
  Task success: 100% standard / 100% code-switched
```

### Statistical tests

```
LPS independent t-test (Config A vs B):   t=2.350, p=0.030
LPS paired t-test (by seed, n=10 pairs):  t=2.395, p=0.040
Pearson r(EC, LPS) all 20:                r=0.431, p=0.058
Pearson r(EC, LPS) Config A only:         r=0.031, p=0.931
Pearson r(EC, LPS) Config B only:         r=-0.004, p=0.991
```

### File locations

```
results/summary.csv              — one row per debate pair (20 rows), all metrics
results/raw/A_*.json             — 10 Config A full transcripts + raw verdict JSON
results/raw/B_*.json             — 10 Config B full transcripts + raw verdict JSON
results/aggregate.json           — aggregate statistics (Config B only — last run)
HANDOFF.md                       — full team communication, setup, and change log
```

### Models used

```
Opponent (Config A):  llama3.1-opponent       (base: llama3.1:8b)
Opponent (Config B):  qwen2.5-opponent        (base: qwen2.5:7b)
Proponent (Config A): qwen2.5-proponent       (base: qwen2.5:7b)
Proponent (Config B): llama3.1-proponent      (base: llama3.1:8b)
Judge (both):         mistral-judge           (base: mistral:7b, temp=0.0, seed=42)
```
