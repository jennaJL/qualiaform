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
beats one that predicts the form directly. See `GOLD_LLM_EXPERIMENT.md`.
