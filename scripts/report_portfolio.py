"""Generate a summary report from the saved benchmark results."""

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qualiaform_part2.extract import dump

root = pathlib.Path(__file__).resolve().parents[1]
bench = root / "results/portfolio_v1"
m = json.loads((bench / "test_metrics.json").read_text())
selection = json.loads((bench / "selection.json").read_text())
selected = selection["selected"]
ci = json.loads((bench / "selected_bootstrap.json").read_text())
service = json.loads((root / "reports/service_verification.json").read_text())
rows = [
    "# QualiaForm：应用型 NLP／LLM 工程作品报告",
    "",
    "核验日期：2026-10-08。新增模型、训练、接口与测试均已在本地实际运行；没有付费API调用。",
    "",
    "## 这次增加了什么",
    "",
    "- 使用 Hugging Face Transformers 加载固定版本多语言 MiniLM，将9,153条遮蔽输入转换为384维向量；冻结encoder，不冒称全量微调Transformer。",
    "- 在全部7,265条原始train候选上训练PyTorch线性头和MLP头，各跑3个随机种子；TensorDataset/DataLoader、加权交叉熵、AdamW、梯度裁剪、权重持久化均实际使用。",
    "- 加入字符TF-IDF＋logistic regression基线。所有词表、统计量和模型参数只从train拟合；threshold与主模型由975条dev选择，test不参与选择。",
    "- 保留上阶段真实Qwen提示实验、LoRA权重及失败结果。没有为了简历删去负面结果。",
    "- 实现FastAPI单条/批量预测、OpenAPI、输入校验、健康检查、计数/耗时指标、无原文日志，以及只检索train的相似例句；本地网页连接真实模型。",
    "",
    "## 固定test结果",
    "",
    "该913条test（887 BN / 26 YI）已在上阶段使用，属于同一基准的后续比较，不是新收集的独立外部测试。所有reference为规则抽取的observed form弱标签。",
    "",
    "| 条件 | Accuracy | Balanced accuracy | Macro-F1 | YI recall |",
    "|---|---:|---:|---:|---:|",
]
for name in ["majority_BN", "char_tfidf_logreg", "linear_1729", "mlp_1729", "mlp_1730", "mlp_1731"]:
    r = m[name]
    rows.append(
        f"| {name} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['per_class']['YI_CL_N']['recall']:.4f} |"
    )
vals = [m[f"mlp_{s}"]["macro_f1"] for s in (1729, 1730, 1731)]
rows += [
    "",
    f"服务使用 **{selected}**，按固定primary seed=1729下的dev macro-F1选择，而不是选择test分数最高的seed。三种seed的MLP test macro-F1均值 {statistics.mean(vals):.4f}，标准差 {statistics.pstdev(vals):.4f}。",
    f"选定模型的句子级bootstrap（500次）macro-F1 95%区间为 [{ci['macro_f1_95'][0]:.4f}, {ci['macro_f1_95'][1]:.4f}]；balanced accuracy区间为 [{ci['balanced_accuracy_95'][0]:.4f}, {ci['balanced_accuracy_95'][1]:.4f}]。少数类支持量很小，不能声称显著优于基线或已可生产部署。",
    "",
    "这是一个可运行、可审计的研究原型。主模型找到26条YI中的3条，YI recall为11.54%；实际质量仍有明显限制。",
    "",
    "## 验证与服务",
    "",
    "- 53项自动化测试全部通过，其中API合同测试使用测试替身；另外单独用真实模型检查20条单条请求和1批共4条样本，与离线预测一致。",
    "- 第二次独立运行完整监督训练，6组权重和指标逐字节相同（11项比较通过）。原语料抽取冻结校验仍通过。",
    "- 实际启动Uvicorn，loopback的health、OpenAPI和predict请求成功；服务仅绑定127.0.0.1。",
    f"- 20条本机顺序TestClient请求的p50={service['latency_ms']['p50']:.1f}ms、p95={service['latency_ms']['p95']:.1f}ms。仅是特定CPU和小样本的原型测量，没有并发压测或生产SLO承诺。",
    "- 提供Dockerfile和GitHub Actions配置；二者均未在对应平台执行。",
]
(root / "reports/portfolio/portfolio_report_zh.md").write_text("\n".join(rows) + "\n")
dump(
    root / "reports/portfolio/summary.json",
    {
        "selected": selected,
        "selected_metrics": m[selected],
        "mlp_macro_f1_mean": statistics.mean(vals),
        "mlp_macro_f1_std": statistics.pstdev(vals),
        "unit_tests_passed": 53,
        "real_service_examples": 20,
        "training_repeat_checks": 11,
        "docker_executed": False,
        "github_actions_executed": False,
        "cloud_deployed": False,
    },
)
print("Portfolio report generated")
