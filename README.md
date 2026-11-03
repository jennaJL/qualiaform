# QualiaForm

**Can explicit linguistic structure help a model predict Mandarin noun-phrase form?**

> **Status (Oct 2026):** Parts 1 and 2 complete with results. Part 3 (LLM
> annotation on the expert gold set) is built and tested; the full 7B run is
> next.

QualiaForm is an NLP evaluation project built on my own research on Mandarin
nominal expressions. Mandarin can express "a book" as a bare noun (`看书`,
**BN**) or with the numeral *yi* and a classifier (`看一本书`, **YI-CL-N**). My
earlier analysis argues that an *indefinite* bare noun is licensed when the verb matches a
relation already stored in the noun's meaning (its *qualia* structure): its
typical use (TELIC, `穿西装`), its typical creation (AGENTIVE, `写小说`), or
possession (HAVE, `有姐姐`).

The project turns that hypothesis into a reproducible pipeline and tests it
three ways:

| Part | Data | Question | Methods |
|---|---|---|---|
| 1 | 156 expert-annotated indefinite occurrences | Do gold qualia features predict form? | scikit-learn, grouped cross-validation, ablation |
| 2 | 9,153 candidates from Universal Dependencies Chinese GSD | How far do encoders and small LLMs get on naturally occurring data? | CoNLL-U extraction, Hugging Face encoder + PyTorch heads, Qwen prompting, LoRA, FastAPI |
| 3 | The 156 gold examples | Can an LLM reproduce the expert annotation, and does annotating first beat predicting directly? | local LLM (Ollama), out-of-fold few-shot, agreement (κ), paired bootstrap |

This repository contains Parts 2 and 3. Part 1 (the gold-set baseline) is not
published: its example sentences come from a copyrighted translation, so the
gold data stays private. Its headline results are summarised below, and Part 3
reuses its exact cross-validation protocol (reproduced to 0.6512 / 0.7147
balanced accuracy on the 156 items).

## Data and methodology

**Task.** Each example is a verb–noun pair in a sentence. The whole target
noun phrase is replaced by `[TARGET_NP]`, and the model predicts whether the
original form was BN or YI-CL-N. Masking keeps the answer (e.g. `一本`) out of
the input.

**Data.**

- *Gold set (Parts 1 and 3, private):* 156 indefinite occurrences I annotated in my
  earlier research with qualia relation (TELIC / AGENTIVE / HAVE / NONE),
  typicality and context features: 32 BN, 124 YI-CL-N. Part 3 uses the 155
  items that have Chinese text (31 BN, 124 YI-CL-N).
- *UD corpus (Part 2, open):* 9,153 verb–noun candidates extracted by
  conservative rules from UD Chinese GSD Simplified (BN 8,897 / YI-CL-N 256,
  official train/dev/test split). Labels are observed surface forms, not
  expert judgements: the BN class mixes definite, generic and indefinite
  bare nouns.

**Methods.**

- Part 1: logistic regression with and without the gold qualia features,
  5-fold cross-validation grouped by verb–noun and context.
- Part 2: frozen multilingual MiniLM features with PyTorch linear/MLP heads
  (3 seeds), a character TF-IDF baseline, and Qwen2.5-0.5B with
  direct vs. structured prompts and a small LoRA run. Thresholds and model
  choice on dev only; test touched once.
- Part 3: a local 7B LLM (Ollama) in 7 conditions: annotate qualia, predict
  form directly, or annotate then predict; zero- and few-shot, with
  out-of-fold examples. Compared by Cohen's κ against the gold labels and
  balanced accuracy on the same folds as Part 1.
- Evaluation throughout: balanced accuracy and macro-F1 (BN is 97% of the
  corpus), bootstrap intervals, invalid outputs counted as errors.

## Results so far

**Part 1 (gold features, 156 items, out-of-fold).** Context features alone
reach 0.651 balanced accuracy. Adding the gold qualia label raises this to
0.715, and BN recall rises from 0.38 to 0.69.

**Licensing asymmetry in the gold data.** 24 of the 31 bare nouns (77%) occur
with a TELIC, AGENTIVE or HAVE match. The BN rate is 0.29 with a match and
0.10 without one. This fits the original analysis: a match *licenses*
a bare noun but does not *force* one: most matched nouns still appear as
YI-CL-N.

**Part 2 (UD corpus, weak labels, 913-item test set, 26 YI-CL-N).**

| Model | Balanced acc. | Macro-F1 | YI recall |
|---|---:|---:|---:|
| Majority (always BN) | 0.500 | 0.493 | 0.00 |
| Char TF-IDF + logistic regression | 0.514 | 0.517 | 0.04 |
| Frozen MiniLM + MLP head (selected on dev) | **0.542** | **0.538** | 0.12 |
| Qwen2.5-0.5B, four prompt conditions | 0.499–0.500 | 0.49 | 0.00 |
| Qwen2.5-0.5B + LoRA (256 examples) | 0.517 | 0.128 | 0.92 |

The best supervised model is only slightly above the baseline, and its
bootstrap interval (macro-F1 0.49–0.60) includes the majority baseline. The
0.5B LLM is not useful for this task: it predicts one class almost every
time. I report these negative results in full.

**How to read Part 2.** The qualia hypothesis is about *indefinite* noun
phrases. The UD labels record surface form only, so the BN class also
contains definite, generic and non-referential bare nouns. Part 2 therefore
measures how predictable the surface form is in encyclopedic text; it is not
a test of the qualia hypothesis, and its scores are not comparable with
Part 1. Separating form from interpretation is left for future work.
[Results and analysis →](docs/en/RESULTS.md)

**Part 3 (LLM annotation on the gold set).** The pipeline is built and
tested, and is ready to run on a local 7B model. Results will be added here.
[Design and how to run →](docs/en/GOLD_LLM_EXPERIMENT.md)

## Documentation

[Results](docs/en/RESULTS.md) · [Part 3 design and how to run](docs/en/GOLD_LLM_EXPERIMENT.md) ·
[Local LLM details](docs/en/LOCAL_LLM.md) · [Refactor and verification notes](docs/en/REFACTOR_NOTES.md)

## What the project demonstrates

- **Language data engineering:** a CoNLL-U parser with tree validation and
  conservative, audited extraction rules. Every excluded noun is counted with
  a reason. Snapshots are frozen and checksummed, and re-extraction must be
  byte-identical.
- **Evaluation design:** model and threshold selection on dev only;
  cross-validation folds grouped by verb–noun and context; out-of-fold few-shot
  demonstrations; sentence-level and item-level bootstrap; balanced metrics
  for imbalanced classes; invalid model outputs counted as errors, never
  dropped.
- **Leakage control:** the whole target NP is masked as `[TARGET_NP]`. Only
  whitelisted fields reach a model, and tests check that a changed gold label
  never changes a prompt.
- **Modelling:** Hugging Face encoders, PyTorch training loops, PEFT/LoRA,
  constrained JSON decoding, and scikit-learn baselines.
- **Engineering:** a FastAPI service with schema validation and metrics, 68
  automated tests, ruff/black, CI configuration, resumable LLM runs, and
  pinned model revisions with SHA-256 checks.

## Quick start

```bash
make test                      # 68 tests (some skip without optional deps / private data)
make verify                    # corpus snapshot: hashes + byte-identical re-extraction

# Part 2 supervised benchmark (needs requirements-portfolio.txt + pinned MiniLM)
python3.12 scripts/setup_portfolio.py
.venv-portfolio/bin/python scripts/run_portfolio.py --out results/my_reproduction
.venv-portfolio/bin/python scripts/serve.py          # http://127.0.0.1:8765

# Part 3 gold-set LLM experiment (needs Ollama + the private gold data)
ollama pull qwen2.5:7b-instruct
python3 scripts/run_gold_llm.py --limit 5            # smoke test
python3 scripts/run_gold_llm.py                      # full run, resumable
```

Nothing calls a paid API. All models run locally on a CPU or a laptop GPU.

## Repository map

| Path | Contents |
|---|---|
| `src/qualiaform_part2/extract.py` | UD CoNLL-U parsing and candidate extraction |
| `src/qualiaform_part2/portfolio.py` | Frozen encoder features, PyTorch heads, TF-IDF baseline, bootstrap |
| `src/qualiaform_part2/local_cpu.py` | Local Qwen inference with constrained decoding; LoRA training |
| `src/qualiaform_part2/service.py` | FastAPI inference service |
| `src/qualiaform_part2/gold/` | Part 3: codebook, input preparation, LLM runs, analysis, report |
| `configs/` | Pinned model manifests, experiment protocols, gold head-noun corrections |
| `data/processed/ud_candidates_v1/` | Frozen corpus snapshot (CC BY-SA, from UD) |
| `results/` | All predictions, metrics and trained weights from the reported runs |
| `reports/` | Execution logs, verification evidence, generated reports |
| `docs/en/` | English documentation |
| `archive/v2_original_code/` | Pre-refactor code, kept because result manifests reference its hashes |

## Limitations

- The UD labels are **observed surface forms from rules**, not expert
  judgements of indefiniteness. Many bare nouns in the corpus are definite,
  generic or part of fixed expressions.
- The UD test set was used for the earlier LLM experiments and then reused
  for the supervised models, so it is not a fresh held-out set.
- The gold set is small (156 items, one novel, one annotator). No
  inter-annotator agreement has been measured yet.
- Docker and CI files are provided, but the service has not been deployed to
  a cloud platform.

## Licence and data

Code: [MIT License](LICENSE). Data: UD Chinese GSD Simplified and every file
derived from it are CC BY-SA 4.0, with attribution to the upstream treebank. Model licences are
listed in `docs/model_sources/`. The Part 1 gold data and every file derived
from its sentences stay in the git-ignored `data/private/` folder. See
[DATA_LICENSE.md](DATA_LICENSE.md).

## Author

Built and maintained by [jennaJL](https://github.com/jennaJL), a linguist working
on language data and LLM evaluation. The research question, annotation scheme,
gold data and design decisions are my own; the code was developed with AI coding
assistance and reviewed by me. Questions and feedback are welcome through
[GitHub issues](https://github.com/jennaJL/qualiaform/issues).
