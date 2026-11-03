"""Gold-set LLM evaluation (Part 3).

Does an LLM that first reproduces the qualia annotation (TELIC / AGENTIVE /
HAVE / NONE) predict Mandarin nominal form (BN vs. YI-CL-N) better than an
LLM asked for the form directly?

Modules:

* :mod:`.codebook` – annotation guidelines and prompts (from the original analysis)
* :mod:`.prepare`  – turn the 156-item Part 1 gold set into leakage-safe inputs
* :mod:`.llm`      – local Ollama / mock clients and resumable runs
* :mod:`.analyze`  – agreement, form-prediction comparison and report
"""
