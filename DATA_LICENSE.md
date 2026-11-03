# Data provenance and attribution

The three `data/ud/zh_gsdsimp-ud-*.conllu` files come from the UD Chinese GSDSimp treebank. Their bytes are preserved and SHA256 values recorded in the frozen manifest. The exact UD release number was not recorded at download time, so I do not claim that these are the current UD release; the SHA-256 values identify the exact files used.

Upstream: [Universal Dependencies Chinese GSDSimp](https://universaldependencies.org/treebanks/zh_gsdsimp/), [repository](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp). Contributors credited on the official treebank page: Peng Qi and Koichi Yasuoka; derived from the Google Chinese GSD treebank. GSDSimp converts the traditional treebank to simplified Chinese with corrections. Official current license: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). Official metadata checked 2026-10-07.

Raw corpus and corpus-derived candidate files are distributed under CC BY-SA 4.0 with attribution to those upstream contributors. Modifications in derived datasets: candidate filtering, extracted dependency/context fields, masking, deterministic sampling, machine-added review flags. No endorsement by the upstream authors is implied. Preserve this attribution and ShareAlike terms when redistributing derivatives.

Part 3 uses the 156-example Part 1 gold set (sentences from a copyrighted translation). It and every file derived from its sentences (prepared inputs, LLM requests and responses, item-level tables) live only in `data/private/`, which is git-ignored and must not be published. Only labels and aggregate scores (`reports/gold_llm/`) are shareable. `configs/gold_head_overrides.csv` contains short verb–noun lemma pairs only.

## Portfolio encoder

The portfolio experiments use [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), Apache-2.0, at revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Base weights are not bundled; the setup script downloads fixed files and verifies their SHA-256 hashes. The encoder stays frozen; trained classifier heads are separately saved in `results/portfolio_v1/`.

## Software licence

The source code in this repository is released under the MIT License (see `LICENSE`). The MIT License covers the code only. Corpus files and corpus-derived data keep CC BY-SA 4.0 as described above, and model weights keep their own licences (`docs/model_sources/`).

## LLM-assisted provisional labels

In Part 2, two small sets of labels were produced with LLM assistance and are marked as such wherever they appear (`llm_assisted` in provenance fields):

- the semantic labels of the 4 UD training sentences used as few-shot demonstrations (`data/local_llm_v1/corpus_demos.jsonl`);
- the 40-item spot review of extracted dev candidates (`reports/pilot_review.csv`).

They are provisional and not expert-checked. They were never used as gold labels or as evaluation targets. Expert-annotated data comes only from my Part 1 gold set (kept private) and, in future, the hand-annotated UD subset.
