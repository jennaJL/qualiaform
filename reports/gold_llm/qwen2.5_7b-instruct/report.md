# Gold-set LLM evaluation

Model: **qwen2.5:7b-instruct** (ollama, 7.6B, Q4_K_M). Items: 155 (31 BN / 124 YI_CL_N).

Protocol: Part 1 grouped folds (seed 42); out-of-fold logistic regression (balanced, C=1, threshold 0.5); LLM few-shot demos drawn only from the other four folds.

## 1. Can the LLM reproduce the expert qualia annotation?

| Condition | Accuracy | Cohen's κ | Macro-F1 | Typicality κ | TELIC F1 | AGENTIVE F1 | HAVE F1 | NONE F1 | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| annotate_zero | 0.600 | 0.436 | 0.585 | 0.556 | 0.453 | 0.456 | 0.727 | 0.701 | 0 |
| annotate_few | 0.606 | 0.439 | 0.591 | 0.465 | 0.467 | 0.560 | 0.689 | 0.647 | 0 |
| structured_zero | 0.542 | 0.375 | 0.534 | 0.524 | 0.414 | 0.514 | 0.635 | 0.571 | 0 |
| structured_few | 0.613 | 0.454 | 0.615 | 0.516 | 0.486 | 0.561 | 0.756 | 0.657 | 0 |

Confusion matrix for **structured_few** (highest κ):

| gold \ LLM | TELIC | AGENTIVE | HAVE | NONE | INVALID |
|---|---:|---:|---:|---:|---:|
| **TELIC** | 18 | 2 | 0 | 5 | 0 |
| **AGENTIVE** | 11 | 16 | 0 | 6 | 0 |
| **HAVE** | 0 | 1 | 17 | 8 | 0 |
| **NONE** | 20 | 5 | 2 | 44 | 0 |

## 2. Form prediction: direct vs. linguistically informed

Balanced accuracy is the primary metric (BN is the minority class). *oracle* = uses the expert's gold qualia labels; *pipeline* = uses LLM-predicted qualia.

| System | Type | Balanced acc. | 95% CI | Macro-F1 | BN recall | YI recall | Invalid |
|---|---|---:|---|---:|---:|---:|---:|
| logreg: context + gold qualia (oracle) | oracle | 0.710 | [0.616, 0.799] | 0.645 | 0.710 | 0.710 | 0 |
| logreg: context only | classifier | 0.657 | [0.566, 0.748] | 0.677 | 0.387 | 0.927 | 0 |
| rule: gold qualia licenses BN (oracle) | oracle | 0.645 | [0.556, 0.729] | 0.537 | 0.774 | 0.516 | 0 |
| logreg: context + LLM qualia (structured_few) | pipeline | 0.645 | [0.546, 0.743] | 0.631 | 0.484 | 0.806 | 0 |
| logreg: context + LLM qualia (annotate_zero) | pipeline | 0.633 | [0.536, 0.728] | 0.597 | 0.548 | 0.718 | 0 |
| logreg: context + LLM qualia (annotate_few) | pipeline | 0.633 | [0.535, 0.728] | 0.615 | 0.484 | 0.782 | 0 |
| logreg: context + LLM qualia (structured_zero) | pipeline | 0.633 | [0.536, 0.730] | 0.597 | 0.548 | 0.718 | 0 |
| LLM direct_few | llm | 0.560 | [0.461, 0.657] | 0.465 | 0.677 | 0.444 | 0 |
| rule: LLM qualia licenses BN (annotate_zero) | pipeline | 0.552 | [0.462, 0.644] | 0.454 | 0.677 | 0.427 | 0 |
| rule: LLM qualia licenses BN (annotate_few) | pipeline | 0.552 | [0.457, 0.644] | 0.466 | 0.645 | 0.460 | 0 |
| rule: LLM qualia licenses BN (structured_zero) | pipeline | 0.532 | [0.439, 0.618] | 0.399 | 0.742 | 0.323 | 0 |
| LLM direct_reason_zero | llm | 0.520 | [0.428, 0.614] | 0.425 | 0.645 | 0.395 | 0 |
| LLM direct_zero | llm | 0.504 | [0.427, 0.574] | 0.302 | 0.839 | 0.169 | 0 |
| majority (always YI_CL_N) | baseline | 0.500 | [0.500, 0.500] | 0.444 | 0.000 | 1.000 | 0 |
| LLM structured_few | llm | 0.500 | [0.414, 0.593] | 0.494 | 0.290 | 0.710 | 0 |
| rule: LLM qualia licenses BN (structured_few) | pipeline | 0.472 | [0.374, 0.566] | 0.400 | 0.548 | 0.395 | 0 |
| LLM structured_zero | llm | 0.468 | [0.395, 0.545] | 0.468 | 0.161 | 0.774 | 0 |

### Paired differences (bootstrap over items)

| Comparison | Δ balanced acc. | 95% CI | Resamples with Δ > 0 |
|---|---:|---|---:|
| LLM structured_few  minus  LLM direct_few | -0.060 | [-0.195, +0.077] | 19% |
| LLM structured_zero  minus  LLM direct_zero | -0.036 | [-0.133, +0.054] | 21% |
| LLM structured_zero  minus  LLM direct_reason_zero | -0.052 | [-0.180, +0.069] | 20% |
| logreg: context + LLM qualia (annotate_few)  minus  logreg: context only | -0.024 | [-0.080, +0.041] | 21% |
| logreg: context + gold qualia (oracle)  minus  logreg: context only | +0.052 | [-0.037, +0.149] | 87% |
| logreg: context + gold qualia (oracle)  minus  logreg: context + LLM qualia (annotate_few) | +0.077 | [-0.007, +0.166] | 96% |
| logreg: context + LLM qualia (annotate_few)  minus  LLM direct_few | +0.073 | [-0.060, +0.203] | 88% |

## 3. Licensing asymmetry

The theory predicts that BN needs a qualia/possessive match: the BN rate should be clearly lower for NONE than for TELIC, AGENTIVE or HAVE.

| Qualia | gold: BN / n (rate) | annotate_zero: BN / n (rate) | annotate_few: BN / n (rate) | structured_zero: BN / n (rate) | structured_few: BN / n (rate) |
|---|---|---|---|---|---|
| TELIC | 6 / 25 (0.240) | 11 / 50 (0.220) | 8 / 35 (0.229) | 7 / 33 (0.212) | 7 / 49 (0.143) |
| AGENTIVE | 10 / 33 (0.303) | 7 / 24 (0.292) | 5 / 17 (0.294) | 10 / 37 (0.270) | 6 / 24 (0.250) |
| HAVE | 8 / 26 (0.308) | 3 / 18 (0.167) | 7 / 35 (0.200) | 6 / 37 (0.162) | 4 / 19 (0.211) |
| NONE | 7 / 71 (0.099) | 10 / 63 (0.159) | 11 / 68 (0.162) | 8 / 48 (0.167) | 14 / 63 (0.222) |
| any match (TELIC/AGENTIVE/HAVE) | 24 / 84 (0.286) | 21 / 92 (0.228) | 20 / 87 (0.230) | 23 / 107 (0.215) | 17 / 92 (0.185) |

## Caveats

- 155 items from one translated novel; 31 BN. Intervals are wide, so small differences are not evidence of a real effect.
- The gold qualia labels come from a single annotator; there is no inter-annotator agreement yet, so the LLM is compared with one expert, not with a consensus.
- Source sentences are copyrighted; this report contains labels and scores only.
