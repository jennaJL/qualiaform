---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
language:
- zh
tags:
- lora
- peft
- mandarin
- universal-dependencies
---

# QualiaForm LoRA adapter (Qwen2.5-0.5B-Instruct, CPU)

A small LoRA adapter trained as part of [QualiaForm](../../README.md) Part 2.
It predicts the observed surface form of a masked Mandarin noun phrase:
bare noun (`BN`) or *yi* + classifier + noun (`YI_CL_N`).

**This is a negative result kept for transparency, not a usable classifier.**

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (frozen) |
| Adapter | rank 4, alpha 8, dropout 0, on `q_proj` / `v_proj` of the last 4 layers; 45,056 trainable parameters |
| Training data | 256 class-balanced UD Chinese GSDSimp train candidates (128 BN + 128 YI_CL_N); labels are observed forms from rule-based extraction, not expert judgements |
| Training | 1 epoch, 64 steps, batch 4, AdamW, learning rate 2e-4, loss on answer tokens only, final checkpoint kept |
| Validation | 40 dev items (20 + 20); loss 0.614 → 0.069 |
| Test (913 items, 26 YI_CL_N) | balanced accuracy 0.517, macro-F1 0.128, accuracy 0.134, YI_CL_N recall 0.92, BN recall 0.11 |

The adapter learned to over-predict the minority class, and the falling
validation loss mostly reflects learning the output format. See
`docs/en/RESULTS.md` §3 for the analysis.

Files: `adapter_model.safetensors` (weights), `adapter_config.json`,
`protocol.json` (frozen training protocol), `training_result.json` (hashes of
inputs, code and weights), `training_history.jsonl` (loss per step).

Licence: the base model is Apache-2.0 (`docs/model_sources/QWEN_LICENSE`).
Training data is derived from UD Chinese GSDSimp (CC BY-SA 4.0).
