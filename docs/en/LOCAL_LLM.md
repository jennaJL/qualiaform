# Part 2 local LLM experiments (Qwen2.5-0.5B, CPU)

The results are summarised in `RESULTS.md` §3.

**Model.** `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision
`7ae557604adf67be50417f59c2c2f167def9a775`, float32 on CPU. Every downloaded
file is checked against `configs/cpu_model_manifest.json`. No remote code is
executed. A 4B model was downloaded first but could not run, because MLX
needs Metal and Metal was not available in that environment. I did not count
the download as a result.

**Inference** (`src/qualiaform_part2/local_cpu.py infer`).

- The input contains only `masked_context`, `noun_lemma`, `verb_lemma` and
  `relation`.
- Constrained decoding: a token trie of all valid JSON answers (2 for direct,
  24 for structured) restricts generation, and the model's logits choose
  between the allowed branches.
- Predictions are written and flushed per batch. A run resumes only if its
  identity matches exactly: model revision, requests, code and adapter
  hashes.

**LoRA training** (`local_cpu.py train`).

- Rank 4, alpha 8, on `q_proj`/`v_proj` of the last four layers: 45,056
  trainable parameters, with the base model frozen.
- 256 class-balanced train examples, 1 epoch (64 steps), AdamW, learning rate
  2e-4.
- Loss is computed on the answer tokens only. The final checkpoint is kept,
  with no selection on test.

**Reproduce.**

```bash
python3.12 scripts/setup_local_cpu.py
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-local/bin/python scripts/run_local_suite.py --model models/base_qwen05 --split pilot --prefix my_local
.venv-local/bin/python -m qualiaform_part2.local_cpu infer --model models/base_qwen05 \
    --adapter models/lora_cpu_v2 --out results/my_lora_test
.venv-local/bin/python -m qualiaform_part2.evaluate --run results/my_lora_test
```

Delivered runs in `results/local_v2_test_*` are records. They were produced
by the pre-refactor code (archived in `archive/v2_original_code/`, which is
what their manifests' code hashes point to), so write new runs to new
folders.
