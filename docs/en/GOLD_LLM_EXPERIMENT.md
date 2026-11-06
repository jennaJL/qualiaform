# Part 3 – Gold-set LLM evaluation

## Question

> Does explicit linguistic structure improve an LLM's prediction of Mandarin
> nominal form, compared with direct end-to-end prediction?

The experiment breaks this into three sub-questions:

1. **Annotation.** Can an LLM reproduce the expert qualia labels (TELIC /
   AGENTIVE / HAVE / NONE) on the 156-item gold set? Measured by Cohen's κ
   against the gold labels.
2. **Prediction.** Which predicts BN vs. YI-CL-N better: an LLM asked directly,
   an LLM that annotates qualia first, or a pipeline in which the LLM
   annotates and a small classifier predicts?
3. **Theory check.** Do LLM-annotated qualia show the same *licensing
   asymmetry* as the gold labels, with BN rare under NONE?

## Design

**Inputs.** Each item becomes `{verb, noun, context}`:

- the Chinese sentence only (interlinear glosses removed);
- the whole annotated target NP replaced by `[TARGET_NP]`. For YI-CL-N items,
  the mask always starts at `一`, so the numeral and classifier are never
  visible;
- the verb and the head noun, derived by rule and corrected by hand in
  `configs/gold_head_overrides.csv` (62 items, each with the reason). This table is
  the one place where my own judgement entered the inputs, so it is kept
  separate and documented item by item.

The gold form, the gold qualia label and the annotator's notes are never sent
for the item itself. A test confirms that changing an item's gold labels
leaves its prompt unchanged.

**Codebook** (`src/qualiaform_part2/gold/codebook.py`). This is written from
my original analysis: the qualia roles, the lexical rule in
which BN is licensed when the verb doubles the noun's TELIC or AGENTIVE
relation, and the possessive extension motivated by exceptions with yǒu. Its
example pairs (`穿西装`, `写小说`, `有姐姐`, `看见汽车`, …) were chosen so
that none occurs in the gold set; a test checks this.

**Conditions** (7 runs × 155 items):

| Condition | Model is asked for | Few-shot examples |
|---|---|---|
| `annotate_zero` / `annotate_few` | qualia label (+ one-sentence reason) | 2 per label |
| `direct_zero` / `direct_few` | form only | 4 BN + 4 YI |
| `direct_reason_zero` | short reason + form (ablation) | – |
| `structured_zero` / `structured_few` | reason + qualia + form, with the codebook and the licensing theory | 1 BN + 1 YI per label |

**Out-of-fold few-shot examples.** Part 1 split the gold set into 5 grouped
folds. An item in fold *k* only sees examples from the other four folds, the
same rule that keeps Part 1's classifier honest. The reasoning shown in the
examples is the first sentence of my own `qualia_notes` from the gold annotation.

**Decoding.** Greedy (temperature 0, fixed seed). Ollama constrains the JSON
output to the schema, so labels are always valid enum values.

**Comparisons** (all on the same 155 items and Part 1's folds):

- majority; logistic regression on context features (`neg_dis`, `modifier`,
  `aspect`); the same model + **gold** qualia (oracle ceiling);
- the same model + **LLM** qualia (the automatic pipeline);
- the licensing rule (BN iff TELIC/AGENTIVE/HAVE) with gold and with LLM
  qualia;
- the LLM's own form predictions in every condition.

The primary metric is balanced accuracy, since BN is the minority class here.
Uncertainty comes from item-level bootstrap 95% intervals, and from paired
bootstrap intervals for the key differences: structured vs. direct, and
LLM-qualia pipeline vs. context only.

## How to run (macOS, free, about 1–2 hours)

1. Install **Ollama** from <https://ollama.com>, open it, then in Terminal:

   ```bash
   ollama pull qwen2.5:7b-instruct      # ~4.7 GB, one-off
   ```

2. In Terminal, go to the project folder and create a small environment
   (only `numpy` and `scikit-learn` are needed; no PyTorch):

   ```bash
   cd path/to/qualiaform
   python3 -m venv .venv-gold
   .venv-gold/bin/pip install numpy scikit-learn
   .venv-gold/bin/python scripts/run_gold_llm.py --limit 5   # 1–2 min: checks everything works
   .venv-gold/bin/python scripts/run_gold_llm.py             # full run
   ```

   The run can be stopped with Ctrl-C and resumed with the same command.
   Every answer is written to disk as soon as it arrives.

3. Outputs:

   - `reports/gold_llm/qwen2.5_7b-instruct/report.md` and `summary.json`:
     labels and scores only, safe to share;
   - `data/private/gold_runs/qwen2.5_7b-instruct/`: every request and raw
     response, plus `analysis/items_with_predictions_PRIVATE.csv` for error
     analysis. This folder contains the novel's sentences, so keep it
     private.

A different local model can be tried with `--model qwen3:8b` (or any Ollama
model). Each model gets its own folder.

## What to look at afterwards

Results for `qwen2.5:7b-instruct` are in [RESULTS.md, section 5](RESULTS.md#5-part-3-llm-annotation-on-the-gold-set).

- **κ for `annotate_few`.** Conventionally, 0.4–0.6 is moderate and 0.6–0.8
  substantial. The confusion matrix shows which distinctions the model
  misses; TELIC vs. NONE (use vs. mere handling) is the most likely
  candidate.
- **Pipeline vs. context only.** If LLM qualia improve the classifier, the
  annotation is useful as an automatic feature. The gap to the gold oracle
  measures what annotation errors cost.
- **Structured vs. direct, and `direct_reason`.** If structured beats direct
  but not direct_reason, the gain comes from "thinking first", not from the
  qualia framework.
- **Licensing table.** Whether the LLM labels reproduce the low BN rate under
  NONE.
- **The private item table.** Disagreements between the LLM and gold are the
  material for a qualitative error analysis. Some may turn out to be gold
  labels worth revisiting.
