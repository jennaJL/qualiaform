# Results and analysis

All numbers come from saved runs in `results/`. Nothing was re-tuned after the
test scores were seen. Machine-readable summaries are in
`reports/portfolio/summary.json` and `results/local_comparison.json`.

## 1. Corpus extraction (UD Chinese GSD Simplified)

`src/qualiaform_part2/extract.py` reads the three CoNLL-U files and keeps
each common noun that is an argument (`obj`, `nsubj`, `obl`, `iobj`) of a
verb. It then decides the noun phrase's surface form:

- **BN**: no numeral, determiner or classifier anywhere in the NP subtree.
- **YI_CL_N**: exactly `一` + one classifier + noun, in that order, with no
  determiner.
- Anything else is excluded, and the reason is counted.

| Split | Nouns seen | Not a verbal argument | Other determination | Nested determination | BN | YI_CL_N |
|---|---:|---:|---:|---:|---:|---:|
| train | 22,925 | 13,599 | 1,391 | 670 | 7,060 | 205 |
| dev | 3,134 | 1,930 | 163 | 66 | 950 | 25 |
| test | 2,760 | 1,621 | 170 | 56 | 887 | 26 |

That gives 9,153 candidates, with 4,025 distinct nouns in 3,868 sentences and
no sentence shared across splits. The whole NP subtree is replaced by
`[TARGET_NP]` in the model input. This blocks the answer from leaking, but it
also removes modifiers that could be informative.

**Label caveat.** These are *observed forms*, not judgements about
indefiniteness. Many corpus bare nouns are definite, generic, or part of
compounds and fixed expressions; 2,196 candidates contain a proper name. YI
is rare (2.8%). Because the BN class mixes these readings, the results in
sections 2 and 3 describe surface-form predictability, not indefinite form
choice, and are not a test of the qualia hypothesis.

**Reproducibility.** `scripts/verify_freeze.py` re-extracts the snapshot from
the raw files and requires byte-identical output. The refactored extractor
passes this check.

## 2. Supervised baselines (frozen encoder + PyTorch heads)

Setup: a pinned multilingual MiniLM encodes `noun + verb + relation + masked
context` into 384-dimensional vectors. The encoder is frozen. On the full
train split (7,265 items) I trained a linear head and a one-hidden-layer MLP
head, three seeds each, using AdamW (lr 1e-3), square-root class weights,
gradient clipping, batch size 256 and 20 epochs (final epoch kept). The decision threshold for each model was tuned on
dev. The served model was chosen on dev using the primary seed only. The
baseline is character 1–3-gram TF-IDF with class-balanced logistic
regression.

Test set: 913 items (887 BN, 26 YI).

| Model | Balanced acc. | Macro-F1 | YI recall |
|---|---:|---:|---:|
| majority BN | 0.500 | 0.493 | 0.000 |
| char TF-IDF + logreg | 0.514 | 0.517 | 0.038 |
| linear head (3 seeds) | 0.497–0.499 | 0.491–0.493 | 0.000 |
| MLP seed 1729 (**selected on dev**) | 0.542 | 0.538 | 0.115 |
| MLP seed 1730 | 0.496 | 0.491 | 0.000 |
| MLP seed 1731 | 0.540 | 0.534 | 0.115 |

- The selected model finds 3 of the 26 YI items.
- Its sentence-level bootstrap 95% interval is macro-F1 [0.486, 0.598] and
  balanced accuracy [0.484, 0.616]. Both include the majority baseline.
- Mean MLP macro-F1 across the three seeds is 0.521 (SD 0.022). The seed
  matters about as much as the model type.
- Slices: on the 836 test items whose verb–noun pair never occurs in train,
  macro-F1 is 0.543. Only one YI item has a pair that was seen in train, so
  the seen/unseen comparison says little.

**Cross-platform check.** After the refactor, I re-ran training on Linux
(torch 2.14). The original run used macOS with torch 2.9. On the same
platform, the old and new code give byte-identical weights, predictions and
metrics. Across platforms, the linear heads and two of the three MLP seeds
are identical. The selected MLP and the TF-IDF baseline move by about 0.01
macro-F1 (0.538 → 0.551), which is one or two test items. The reported
numbers stay the original ones. See `reports/portfolio/cross_platform_rerun.json`.

## 3. Small local LLM (Qwen2.5-0.5B-Instruct, CPU)

Conditions: **direct** (predict the form) vs. **structured** (predict
typicality and qualia, then the form), each zero-shot or with four train
examples. Output was constrained to valid JSON enums by a token trie. The
model's own logits choose among the allowed tokens.

| Condition | Balanced acc. | Macro-F1 | Predictions |
|---|---:|---:|---|
| direct, zero-shot | 0.500 | 0.493 | 913 × BN |
| direct, 4-shot | 0.499 | 0.493 | 912 BN, 1 YI |
| structured, zero-shot | 0.500 | 0.493 | 913 × BN; qualia always NONE |
| structured, 4-shot | 0.500 | 0.493 | 913 × BN; qualia always NONE |
| LoRA (256 balanced train examples, 64 steps) | 0.517 | 0.128 | 813 YI, 100 BN |

The 0.5B model ignores the input and always gives the same answer. LoRA on a
class-balanced sample flips that bias the other way: YI recall 0.92, but YI
precision is about 3%. Validation loss fell from 0.61 to 0.07, which shows
that a falling loss mostly reflects learning the output format, not the task.

**Takeaways.** A model of this size gives no evidence either way about the
qualia hypothesis. The semantic few-shot labels in this run were provisional
LLM-assisted labels, not expert-checked gold, so Part 3 uses the real gold
set and a 7B model.

## 4. Gold-set baselines without an LLM (Part 3 reference points)

The 155 gold items with Chinese text (31 BN / 124 YI-CL-N), on Part 1's
grouped folds:

| System | Balanced acc. | BN recall | YI recall |
|---|---:|---:|---:|
| majority (YI-CL-N) | 0.500 | 0.00 | 1.00 |
| logreg: context features only | 0.657 | 0.39 | 0.93 |
| logreg: context + **gold** qualia | 0.710 | 0.71 | 0.71 |
| rule: BN iff gold qualia ∈ {TELIC, AGENTIVE, HAVE} | 0.645 | 0.77 | 0.52 |

The logistic protocol reproduces Part 1 exactly: 0.6512 and 0.7147 on all
156 items, checked by a test.

BN rate by gold qualia label: TELIC 6/25 (0.24), AGENTIVE 10/33 (0.30), HAVE
8/26 (0.31), NONE 7/71 (0.10).

Part 3 asks how much of the gap between "context only" and "context + gold
qualia" an LLM annotator can recover, and whether an LLM that annotates first
beats one that predicts the form directly. Design: `GOLD_LLM_EXPERIMENT.md`;
results: section 5.

## 5. Part 3: LLM annotation on the gold set

Model: `qwen2.5:7b-instruct` via Ollama (7.6B parameters, Q4_K_M
quantisation, digest `845dbda0…`), run locally in November 2026. Items: the
155 gold items with Chinese text (31 BN / 124 YI-CL-N), on the same grouped
folds as Part 1. Few-shot demonstrations come only from the other four folds.
No condition produced an invalid output. Full tables:
[`reports/gold_llm/qwen2.5_7b-instruct/report.md`](../../reports/gold_llm/qwen2.5_7b-instruct/report.md)
and `summary.json` in the same folder.

**In short.** The 7B model reproduces my qualia annotation only moderately
(κ ≈ 0.44), and its labels do not carry the predictive signal that the gold
labels carry. Asking the model to annotate before predicting did not beat
asking it directly. Explicit linguistic structure helped only when it was the
expert's structure, and even that gain is not statistically clear on 155
items. Most of these results are negative; I report them in full.

### 5.1 Can the LLM reproduce the expert annotation?

| Condition | Accuracy | Cohen's κ | Macro-F1 | Typicality κ |
|---|---:|---:|---:|---:|
| annotate, zero-shot | 0.600 | 0.436 | 0.585 | 0.556 |
| annotate, few-shot | 0.606 | 0.439 | 0.591 | 0.465 |
| structured, zero-shot | 0.542 | 0.375 | 0.534 | 0.524 |
| structured, few-shot | 0.613 | **0.454** | 0.615 | 0.516 |

- Agreement is moderate in every condition (κ 0.38–0.45). Few-shot
  demonstrations help the structured prompt (κ 0.375 → 0.454) but barely
  change the plain annotation prompt (0.436 → 0.439).
- **HAVE** is the easiest relation (F1 0.63–0.76) and **TELIC** the hardest
  (F1 0.41–0.49). The model over-uses TELIC: with structured few-shot it
  assigns TELIC to 49 items, of which only 18 are gold TELIC (precision
  0.37).
- The main confusions are gold **NONE → TELIC** (12–20 of 71 items, depending
  on the condition) and **AGENTIVE → TELIC** (8–15 of 33). The first is the
  distinction between using an object for its purpose and merely handling it,
  which the design notes flagged as the most likely difficulty.

### 5.2 Form prediction: direct vs. linguistically informed

Balanced accuracy, 95% item-level bootstrap intervals.

| System | Balanced acc. | 95% CI | BN recall | YI recall |
|---|---:|---|---:|---:|
| majority (always YI-CL-N) | 0.500 | – | 0.00 | 1.00 |
| LLM direct, zero-shot | 0.504 | [0.427, 0.574] | 0.84 | 0.17 |
| LLM direct, few-shot | 0.560 | [0.461, 0.657] | 0.68 | 0.44 |
| LLM direct with reasoning, zero-shot | 0.520 | [0.428, 0.614] | 0.65 | 0.40 |
| LLM structured (annotate, then predict), zero-shot | 0.468 | [0.395, 0.545] | 0.16 | 0.77 |
| LLM structured (annotate, then predict), few-shot | 0.500 | [0.414, 0.593] | 0.29 | 0.71 |
| rule: BN iff LLM qualia ∈ {TELIC, AGENTIVE, HAVE} | 0.472–0.552 | – | 0.55–0.74 | 0.32–0.46 |
| logreg: context only | 0.657 | [0.566, 0.748] | 0.39 | 0.93 |
| logreg: context + LLM qualia (4 conditions) | 0.633–0.645 | – | 0.48–0.55 | 0.72–0.81 |
| rule: BN iff **gold** qualia ∈ {TELIC, AGENTIVE, HAVE} | 0.645 | [0.556, 0.729] | 0.77 | 0.52 |
| logreg: context + **gold** qualia (oracle) | **0.710** | [0.616, 0.799] | 0.71 | 0.71 |

Paired differences (bootstrap over items):

| Comparison | Δ balanced acc. | 95% CI | Resamples with Δ > 0 |
|---|---:|---|---:|
| LLM structured few-shot − LLM direct few-shot | −0.060 | [−0.195, +0.077] | 19% |
| LLM structured zero-shot − LLM direct zero-shot | −0.036 | [−0.133, +0.054] | 21% |
| LLM structured zero-shot − LLM direct with reasoning | −0.052 | [−0.180, +0.069] | 20% |
| logreg + LLM qualia (annotate few-shot) − logreg context only | −0.024 | [−0.080, +0.041] | 21% |
| logreg + gold qualia − logreg context only | +0.052 | [−0.037, +0.149] | 87% |
| logreg + gold qualia − logreg + LLM qualia (annotate few-shot) | +0.077 | [−0.007, +0.166] | 96% |
| logreg + LLM qualia (annotate few-shot) − LLM direct few-shot | +0.073 | [−0.060, +0.203] | 88% |

What this shows:

1. **Direct LLM prediction is close to chance** (0.50–0.56) and below a
   logistic regression on the context features alone (0.657). Zero-shot, the
   model answers BN for 129 of the 155 items.
2. **Annotating first did not help the LLM.** The structured prompts score
   below the direct prompts in both settings and below direct prediction with
   free-text reasoning. They also flip the bias towards YI-CL-N (BN recall
   0.16–0.29). The differences are within noise, but none points in the
   expected direction.
3. **The LLM's qualia labels do not improve the classifier.** Adding them to
   the context features gives 0.633–0.645, slightly below context only.
4. **The gold qualia labels do improve it**, by +0.05 over context only. The
   interval includes zero, so on 155 items this is a consistent direction
   (87% of resamples), not an established effect. The gap between gold and
   LLM labels (+0.08, 96% of resamples) is what the annotation errors cost.
5. Feeding the LLM's labels to a classifier does better than asking the LLM
   for the form directly (+0.07), but this interval also includes zero.

A coincidence worth noting: the classifiers using annotate zero-shot and
structured zero-shot labels have identical aggregate scores. Their
item-level predictions differ on 26 items; the errors happen to balance out.

### 5.3 Licensing asymmetry

| Qualia labels from | BN rate with NONE | BN rate with a match (TELIC/AGENTIVE/HAVE) |
|---|---:|---:|
| gold (expert) | 7/71 (0.10) | 24/84 (0.29) |
| annotate, zero-shot | 10/63 (0.16) | 21/92 (0.23) |
| annotate, few-shot | 11/68 (0.16) | 20/87 (0.23) |
| structured, zero-shot | 8/48 (0.17) | 23/107 (0.21) |
| structured, few-shot | 14/63 (0.22) | 17/92 (0.18) |

The gold labels show the asymmetry the theory predicts: a bare noun is about
three times as likely when the verb matches a qualia or possessive relation.
The LLM labels do not reproduce it: the difference shrinks to a few points
and reverses with structured few-shot labels. Accordingly, the licensing rule
drops from 0.645 with gold labels to 0.47–0.55 with LLM labels.

### 5.4 Limits and next steps

- **Small sample.** 155 items with 31 BN give intervals of about ±0.09, so
  only large differences are detectable. None of the differences between LLM
  conditions can be distinguished from noise.
- **One annotator.** κ is measured against my own labels. Without a second
  annotator there is no human agreement ceiling, so I cannot yet say how much
  of the disagreement is model error and how much reflects genuinely hard
  cases.
- **One model and one prompt set.** A quantised 7B model was chosen so the
  whole experiment runs free on a laptop. Larger models or revised prompts
  may do better; the pipeline accepts any Ollama model
  (`--model`), and each model gets its own report folder.
- **Next:** a qualitative error analysis of the LLM–gold disagreements
  (especially TELIC vs. NONE), kept private because it quotes the source
  sentences; then the open, double-annotated parallel-corpus set (see the
  README roadmap), where the same comparison can be published item by item.
