"""Persist the LLM-assisted spot-review notes; no human-gold certification."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qualiaform_part2.evaluate import metrics
from qualiaform_part2.extract import dump, read_rows

root = pathlib.Path(__file__).resolve().parents[1]
notes = {
    "dev-s100:13": "南方的江淮官话为特定语言类别；非典型不定论元。",
    "dev-s102:1": "市区为话题/地点，可能定指；不定性待审。",
    "dev-s103:14": "专名温泉列表与并列；不适宜直接充当indefinite gold。",
    "dev-s107:14": "以陋卷为介词性论元；BN表面合理，保留obl边界标记。",
    "dev-s107:2": "有人上告中的人；表面BN，存在结构和依存分析需复核。",
    "dev-s108:1": "中校加人物姓名，明显特指。",
    "dev-s109:20": "香山区中部为特定地点；不是典型不定名词。",
    "dev-s109:6": "台湾北部为特定地点。",
    "dev-s10:17": "博物馆中的恐龙化石；有关系修饰，可能定指。",
    "dev-s110:32": "媒体为集合/泛指或语境定指，需额外上下文。",
    "dev-s110:45": "激动人心为固定表达，形式选择受词汇化约束。",
    "dev-s110:9": "中石油为机构专名但中心词标NOUN，抽取需人工排除。",
    "dev-s111:10": "教育行政机构可能定指集合；不定性未确认。",
    "dev-s111:12": "陷入瘫痪为抽象状态/固定搭配。",
    "dev-s111:18": "恢复发展中的发展为事件名词；量化替换未必自然。",
    "dev-s129:6": "一座小亭：清晰yi+量词+名词。",
    "dev-s12:13": "一个范围之内：抽象边界、介词结构。",
    "dev-s147:14": "一个天主教家庭：清晰表面结构，地名修饰。",
    "dev-s158:9": "只有一个女儿：数量焦点且后接姓名同位语。",
    "dev-s162:16": "一座城堡及其他名词并列；量词作用域需审。",
    "dev-s196:17": "一对尾须：成对量词和或并列，不等于单体个数。",
    "dev-s19:26": "一个报告：在否定存在语境中。",
    "dev-s203:19": "寻找一个反例：典型候选。",
    "dev-s240:15": "根据一个预测：附加语与减少的远距离依存；论元资格可疑。",
    "dev-s270:15": "一个人形雕像：带关系从句。",
    "dev-s275:14": "一件金缕玉衣：清晰候选。",
    "dev-s292:20": "一张桌与椅子、折叠床：并列量词作用域。",
    "dev-s300:22": "一座寺院：带命名关系从句。",
    "dev-s312:8": "一条大道：存在引入后作分成的主语；依存需语境解释。",
    "dev-s315:14": "一种偏激和错位：种类量词与抽象名词。",
    "dev-s397:28": "一块蛋糕：比喻和表语结构。",
    "dev-s408:49": "一名军人：带关系从句，被动语境。",
    "dev-s428:36": "一种物质形式与过程：种类量词及并列。",
    "dev-s442:18": "一部电影：带命名修饰。",
    "dev-s443:12": "一颗新星：比喻及被视为结构。",
    "dev-s44:31": "一个门洞：修饰中有45米、50米，数值不属于目标数量。",
    "dev-s454:42": "其中一个团体：部分选取结构，不宜自动作为indefinite gold。",
    "dev-s479:8": "一个灯塔小岛上：地点存在结构。",
    "dev-s5:28": "一个交流平台：抽象/比喻但表面形式明确。",
    "dev-s89:15": "一批零部件和纪念品：集合量词与并列。",
}
rows = read_rows(root / "data/pilot_v2/candidates.jsonl")
assert len(notes) == len(rows) == 40
with (root / "reports/pilot_review.csv").open("w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "candidate_id",
            "observed_form",
            "sentence",
            "surface_review",
            "reviewer",
            "indefinite_gold",
            "notes",
        ],
    )
    w.writeheader()
    for r in rows:
        w.writerow(
            {
                "candidate_id": r["candidate_id"],
                "observed_form": r["observed_form"],
                "sentence": r["sentence"],
                "surface_review": "plausible_candidate",
                "reviewer": "llm_assisted_spot_review_2026-10-07",
                "indefinite_gold": "NOT_ESTABLISHED",
                "notes": notes[r["candidate_id"].removeprefix("dev:")],
            }
        )
full = read_rows(root / "data/processed/ud_candidates_v1/candidates.jsonl")
test = [r for r in full if r["source_split"] == "test"]
m = metrics([(r["observed_form"], "BN") for r in test])
m["status"] = "ACTUALLY_RUN deterministic majority baseline; NOT LLM"
m["reference"] = "extracted observed form, not gold indefinite"
m["training_rule"] = "majority label in source train = BN"
dump(root / "results/majority_baseline.json", m)
