"""Annotation codebook, task prompts and output schemas.

The qualia definitions follow dissertation Chapter 5, §5.6 (overview of the
qualia roles in (45), the lexical rule in (46), and the possessive / HAVE
extension motivated by the yǒu exceptions in (51)–(52)).

The illustrative verb–noun pairs in the codebook were chosen so that none of
them occurs in the 156-item gold set (checked in tests), so the zero-shot
prompts contain no test labels.
"""

from __future__ import annotations

from typing import Any

QUALIA = ("TELIC", "AGENTIVE", "HAVE", "NONE")
FORMS = ("BN", "YI_CL_N")
TYPICAL_QUALIA = ("TELIC", "AGENTIVE")  # typicality = 1 exactly for these

# Codebook examples. Each pair must NOT appear in the gold set.
CODEBOOK_EXAMPLES = {
    "TELIC": [("穿", "西装"), ("读", "报纸"), ("骑", "自行车"), ("弹", "钢琴")],
    "AGENTIVE": [("写", "小说"), ("盖", "房子"), ("做", "蛋糕"), ("拍", "照片")],
    "HAVE": [("有", "姐姐"), ("有", "汽车"), ("长着", "翅膀")],
    "NONE": [("看见", "汽车"), ("搬", "桌子"), ("扔", "石头"), ("需要", "钱")],
}

CODEBOOK = """\
## Qualia codebook (verb–noun relation)

Judge the LEXICAL relation between the verb V and the noun N: does the verb
spell out a relation that is already stored in the noun's meaning? Do not
judge the sentence as a whole; negation, plurality or distributivity in the
context do not change the label.

Following Pustejovsky's qualia structure, a noun's meaning specifies, among
other things:
- its TELIC role: what objects of this kind are designed or used for;
- its AGENTIVE role: how objects of this kind come into existence.
A verb "matches" a role when the verb itself expresses that relation.

Labels:
- TELIC    – V expresses the typical use or function of N: using N the way it
             is meant to be used. Examples: 穿西装, 读报纸, 骑自行车, 弹钢琴.
- AGENTIVE – V expresses how N typically comes into existence: V creates or
             produces N (including sounds, events and texts that exist only
             through the action). Examples: 写小说, 盖房子, 做蛋糕, 拍照片.
- HAVE     – V expresses possession or an inherent relation between the
             subject and N (ownership, kinship, body part, attribute), as with
             有 'have'. Examples: 有姐姐, 有汽车, 长着翅膀.
- NONE     – none of the above: perceiving, moving, handling, transferring,
             needing or wanting N without using it as intended, creating it,
             or possessing it. Examples: 看见汽车, 搬桌子, 扔石头, 需要钱.

Decision order: AGENTIVE (does V create N?) → TELIC (does V use N as
intended?) → HAVE (possession / inherent relation?) → otherwise NONE.
Choose exactly one label.
"""

TASK_INPUT = """\
Input fields:
- verb: the verb V
- noun: the head noun N of the target noun phrase
- context: the Chinese sentence, in which the whole target noun phrase has
  been replaced by [TARGET_NP]
"""

FORM_TASK = """\
In the original sentence, the target noun phrase at [TARGET_NP] had one of
two surface forms:
- BN      – a bare noun, with no numeral and no classifier (e.g. 看书, 穿西装);
- YI_CL_N – the numeral 一 + a classifier + the noun (e.g. 看一本书, 穿一件西装),
            possibly with modifiers.
Predict which form the original sentence used. Both forms are grammatical in
many contexts; choose the more likely one for this sentence.
"""

SYSTEM_PROMPTS = {
    "annotate": (
        "You are a linguist annotating Mandarin verb–noun pairs.\n\n"
        + CODEBOOK
        + "\n"
        + TASK_INPUT
        + '\nAnswer with JSON: {"reasoning": <one short sentence>, "qualia": <label>}.'
    ),
    "direct": (
        "You are a linguist analysing Mandarin noun phrases.\n\n"
        + FORM_TASK
        + "\n"
        + TASK_INPUT
        + '\nAnswer with JSON: {"form": "BN" or "YI_CL_N"}.'
    ),
    # Ablation: direct prediction with a free-text reasoning step but no
    # linguistic codebook. Separates "thinking first" from "qualia structure".
    "direct_reason": (
        "You are a linguist analysing Mandarin noun phrases.\n\n"
        + FORM_TASK
        + "\n"
        + TASK_INPUT
        + '\nThink briefly, then answer with JSON: {"reasoning": <one short sentence>, '
        '"form": "BN" or "YI_CL_N"}.'
    ),
    "structured": (
        "You are a linguist analysing Mandarin noun phrases.\n\n"
        + FORM_TASK
        + "\nFirst annotate the verb–noun relation with the codebook below, then "
        "use it, together with the context, to predict the form. According to the "
        "theory, a bare noun is licensed when the verb matches the noun's TELIC or "
        "AGENTIVE role, and sometimes with HAVE; with NONE, YI_CL_N is expected. A "
        "matching relation makes BN possible, not obligatory.\n\n"
        + CODEBOOK
        + "\n"
        + TASK_INPUT
        + '\nAnswer with JSON: {"reasoning": <one short sentence>, "qualia": <label>, '
        '"form": "BN" or "YI_CL_N"}.'
    ),
}

_REASONING = {"type": "string"}
SCHEMAS: dict[str, dict[str, Any]] = {
    "annotate": {
        "type": "object",
        "properties": {"reasoning": _REASONING, "qualia": {"type": "string", "enum": list(QUALIA)}},
        "required": ["reasoning", "qualia"],
    },
    "direct": {
        "type": "object",
        "properties": {"form": {"type": "string", "enum": list(FORMS)}},
        "required": ["form"],
    },
    "direct_reason": {
        "type": "object",
        "properties": {"reasoning": _REASONING, "form": {"type": "string", "enum": list(FORMS)}},
        "required": ["reasoning", "form"],
    },
    "structured": {
        "type": "object",
        "properties": {
            "reasoning": _REASONING,
            "qualia": {"type": "string", "enum": list(QUALIA)},
            "form": {"type": "string", "enum": list(FORMS)},
        },
        "required": ["reasoning", "qualia", "form"],
    },
}
TASKS = tuple(SYSTEM_PROMPTS)


def validate_output(task: str, obj: Any) -> dict[str, Any]:
    """Raise ``ValueError`` unless ``obj`` has the fields and values ``task`` needs."""
    if not isinstance(obj, dict):
        raise ValueError("output is not a JSON object")
    if task in ("annotate", "structured") and obj.get("qualia") not in QUALIA:
        raise ValueError(f"invalid qualia: {obj.get('qualia')!r}")
    if task in ("direct", "direct_reason", "structured") and obj.get("form") not in FORMS:
        raise ValueError(f"invalid form: {obj.get('form')!r}")
    return obj
