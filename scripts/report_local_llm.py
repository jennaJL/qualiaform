"""Validate all real runs and report exactly the saved results."""

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qualiaform_part2.experiment import validate_response
from qualiaform_part2.extract import digest, dump, read_rows

root = pathlib.Path(__file__).resolve().parents[1]
names = [
    ("Direct zero", "direct_zero"),
    ("Direct corpus few", "direct_corpus_few"),
    ("Structured zero", "structured_zero"),
    ("Structured corpus few", "structured_corpus_few"),
    ("LoRA direct", "lora_direct"),
]
results = []
checks = {}
for label, name in names:
    folder = root / f"results/local_v2_test_{name}"
    m = json.loads((folder / "manifest.json").read_text())
    pred = read_rows(folder / "predictions.jsonl")
    req = read_rows(folder / "requests.jsonl")
    metrics = json.loads((folder / "metrics.json").read_text())["overall"]
    condition = "structured" if name.startswith("structured") else "direct"
    assert len(pred) == len(req) == metrics["n"] == 913
    assert m["scientific_model_experiment_run"] and m["paid_api_calls"] == 0
    # Runs were produced by the pre-refactor code, archived byte-for-byte for provenance.
    original_code = root / "archive/v2_original_code/src/qualiaform_part2/local_cpu.py"
    assert m["config"]["code_sha256"] == digest(original_code.read_bytes())
    for p, r in zip(pred, req):
        assert p["candidate_id"] == r["candidate_id"]
        assert p["request_sha256"] == digest(
            json.dumps(r["request"], ensure_ascii=False, sort_keys=True).encode()
        )
        if p["prediction"] is not None:
            assert validate_response(json.loads(p["raw_response"]), condition) == p["prediction"]
        assert set(r["request"]["input"]) == {"masked_context", "noun_lemma", "verb_lemma", "relation"}
    semantic = (
        {
            f: dict(collections.Counter((p["prediction"] or {}).get(f, "INVALID") for p in pred))
            for f in ("typicality", "qualia")
        }
        if condition == "structured"
        else None
    )
    results.append(
        {
            "condition": label,
            "metrics": metrics,
            "semantic_counts": semantic,
            "prediction_counts": dict(
                collections.Counter((p["prediction"] or {}).get("form_prediction", "INVALID") for p in pred)
            ),
        }
    )
    checks[name] = (
        "PASS: complete, real local model, correct code/request hashes, raw predictions agree, safe input fields"
    )
training = json.loads((root / "models/lora_cpu_v2/training_result.json").read_text())
assert training["changed_parameter_tensors"] > 0 and training["steps"] == 64
assert training["adapter_sha256"] == digest(
    (root / "models/lora_cpu_v2/adapter_model.safetensors").read_bytes()
)
dump(
    root / "reports/local_llm_quality.json",
    {
        "status": "PASS",
        "checks": checks,
        "training_weight_updates_verified": True,
        "inference_calls": 4565,
        "paid_api_calls": 0,
    },
)
dump(
    root / "results/local_comparison.json",
    {
        "results": results,
        "training": training,
        "reference": "extracted observed form, not expert indefinite gold",
        "fewshot": "UD train + LLM-assisted provisional semantic annotations, not Part1 gold",
    },
)
lines = [
    "# 真实本地 LLM 与轻量训练结果",
    "",
    "实验执行日期：2026-10-07；报告核验日期：2026-10-08。已在本机 CPU 完成 Qwen2.5-0.5B-Instruct 的四种提示条件和 LoRA direct，共 **4,565 条 test 预测**。没有使用 API key 或付费模型服务。",
    "",
    "## 完整 test 结果",
    "",
    "每组相同 913 条（887 BN、26 YI_CL_N）。下表为抽取 observed-form 弱标签的一致率，不是专家 indefinite gold 准确率。",
    "",
    "| 条件 | Accuracy | Balanced accuracy | Macro-F1 | BN recall | YI recall | Invalid |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
majority = json.loads((root / "results/majority_baseline.json").read_text())
for label, m in [("Majority BN", majority)] + [(r["condition"], r["metrics"]) for r in results]:
    lines.append(
        f"| {label} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | {m['per_class']['BN']['recall']:.4f} | {m['per_class']['YI_CL_N']['recall']:.4f} | {m['invalid']} |"
    )
lines += [
    "",
    "每个条件的 per-class precision/recall/F1、混淆矩阵和完整错误表位于对应 results 目录。未报告显著性检验或置信区间；少数类只有26条，且同句候选相关。",
    "",
    "## 实际训练",
    "",
    f"256 条 train（128/128平衡采样），40 条 dev（20/20），1 epoch、64步。实际改变 {training['changed_parameter_tensors']} 个参数张量，共 {training['trainable_parameters']:,} 个可训练参数。CPU训练耗时 {training['elapsed_seconds']:.1f} 秒。",
    f"验证 loss 从 {training['validation_loss_before']:.6f} 降至 {training['validation_loss_after']:.6f}。这是答案token的交叉熵，格式学习也会降低loss，不能直接解释为语言学理解提升。",
    "",
    "本次训练只更新最后4层q/v的LoRA，基础模型冻结。训练目标是 observed form；没有用 test 更新权重、选 checkpoint 或修改提示。训练权重已附在 models/lora_cpu_v2/。",
    "",
    "## 条件解释及限制",
    "",
    "- free local 模型为约0.49B的小型生成模型。原拟4B模型因当前环境没有Metal GPU访问而未运行，未将其下载冒充推理。",
    "- 先在dev发现structured输出格式错误，再固定JSON枚举解码和local_v2提示，随后跑test。枚举值由模型选择；evidence固定为空，不宣称模型生成了语义解释。",
    "- Few-shot用4条train示例，语义为LLM辅助暂定标注。不是Part1 gold示例；真正的Part1 gold few-shot仍待原数据。",
    "- 未微调语义标签，没有测量typicality/qualia相对于专家标注的准确率。结构化输出合法不等于语义正确。",
    "- 完整目标子树已遮蔽；这同时删除了部分修饰信息。任务是遮蔽后的表面形式恢复，不是唯一正确形式判定。",
    "- 语料包含定指、专名、地点和固定搭配候选；不能把此处结果推广为汉语不定性理论的证明。",
    "- 未实现lemma-disjoint或近重复控制；预训练语料是否见过相关文本也未知。",
    "",
    "## 预测分布",
    "",
]
for r in results:
    lines.append("- " + r["condition"] + ": " + json.dumps(r["prediction_counts"], ensure_ascii=False))
    if r["semantic_counts"]:
        lines.append("  模型估计语义分布：" + json.dumps(r["semantic_counts"], ensure_ascii=False))
lines += [
    "",
    "## 结果判断与错误分析",
    "",
    "本次没有得到可靠的形式选择模型。原模型几乎全部预测BN，structured并未改善结果；其中structured-zero还将全部typicality判为atypical、全部qualia判为NONE，显示语义输出退化。few-shot改变了typicality分布，但QUALIA仍全部为NONE，形式选择仍未改善。",
    "",
    "LoRA将913条中的813条预测为YI_CL_N。它找到了26条YI中的24条，但同时把887条BN中的789条误判为YI，YI precision仅约2.95%。微调后的macro-F1从多数类基线的0.493降到0.128。虽然balanced accuracy略高于0.50，不能据此称微调有效或推荐部署。",
    "",
    "训练样本经过类别平衡，而test保持自然不平衡，二者分布不同。这可能与偏置变化有关，但目前没有消融实验，不能认定为唯一原因。模型容量、弱标签噪声、遮蔽信息丢失和提示/解码设置也可能影响结果。降低验证loss主要反映token预测改善，不能替代按类别的评估。",
    "",
    "本次到此冻结，不依据已看到的test结果重训挑选更好分数。后续改进需先在train/dev设计独立方案，再使用新的独立评估或明确标注已使用的test；专家确认的indefinite与语义gold仍未获得。",
    "",
    "另行验证过的fast_decode优化在80条dev样本上与原解码输出完全一致，但没有用于本次任何正式test结果；保留为开发实验，不能将其运行耗时混入正式实验统计。",
    "",
    "## 软件与复现",
    "",
    "34项测试已通过；另外对每组913条真实预测逐条验证请求哈希、raw response、结果及输入字段。详见 local_llm_quality.json。",
    "",
    "安装/复现步骤见 docs/en/LOCAL_LLM.md。基础权重使用固定revision与SHA256；压缩包附已训练的小型adapter，基础模型通过脚本免费下载。完整执行日志位于 reports/local_execution/。",
]
(root / "reports/local_llm_report_zh.md").write_text("\n".join(lines) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
