# 实际评测与消融

安装：`uv sync --extra evaluation`。沿用 .env 的生成模型，显式配置独立 judge_model。示例使用 Qwen 生成、qwen-plus 自动评审，没有人工标注步骤。

```powershell
python -m ai_agent_template.evaluation.runner --config configs/eval/ragbench-smoke.yaml --dataset ragbench --split test --max-samples 2
python -m ai_agent_template.evaluation.runner --config configs/eval/qasper-retrieval.yaml
python -m ai_agent_template.evaluation.runner --config configs/eval/ragbench-smoke.yaml --input .runtime/datasets/ragbench-covidqa-test.jsonl
```

支持 output/resume/concurrency/requests-per-second/storage-root/judge-model 参数。并发按样本，共享模型限流器控制规划、生成、校验、引用评审及 RAGAS HTTP 请求。可单跑 retrieval、generation 或 both；provided 模式重新生成参考上下文上的回答，不能作为检索成绩；retrieved 调用实际项目检索与 Graph。数据集 response 只作参考答案保存。

## 标注与公式

RAGBench 使用官方原生划分与 documents_sentences、all_relevant_sentence_keys 等可用标注；按上下文指纹、原生句子键映射来源。QASPER 保留论文/段落身份、多参考答案及替代证据集合；图表或重复段落无法唯一映射时不可评估。QASPER 使用已知论文范围，RAGBench 在本次所选样本上下文组成的语料库检索，Smoke Test 不是完整数据集成绩。

Chunk 保存精确来源字符区间。前 5/10 个去重证据的区间并集覆盖至少 95% 才算单元召回。多证据 Recall 按单元计，替代标注取最佳集合。MRR@10 是首次完成任一正确证据单元覆盖的前缀排名倒数；HitRate@5 是前 5 个至少召回一个正确单元。各分块策略共用原生标注粒度。

Faithfulness 调用安装的 RAGAS 0.4 `metrics.collections.Faithfulness.ascore`，输入为系统最终回答及实际检索上下文。Citation Precision 逐个 claim-reference 对判断，汇总 supported 对数 / 可可靠评审对数，同时保存未评估数量及原因；这是自动评审，与 Context Precision 不同。uncertain、API 错误、缺失标注和拒答显示 null/not_evaluable，不默认为正确。参考答案、实际上下文、生成答案分别保存。拒答率按全部样本及已知有/无答案组统计；RAGBench 没有可靠 answerable 时不推断标签。

## 可复现结果

每次唯一 `experiments/runs/{run_id}`：config.json、dataset_snapshot.json、manifest.json、comparison.csv、总 metrics.json，以及各配置的 effective_settings.json、predictions.jsonl、metrics.json、errors.jsonl。记录数据指纹/原生划分、Git Commit/dirty、源代码指纹、Prompt 版本、生成/校验/评审模型、依赖与索引设置、调用次数、累计决策/生成/校验 Token 和逐条评审，排除 API Key。

续跑使用保存的数据快照，跳过成功样本，拒绝变更的代码、应用设置和配置。远端首次加载解析 HF revision；本地 JSONL 保存字节 SHA-256，支持 .provenance.json 来源侧车。模型服务变化及采样不能由 seed 完全固定。usage 不可用或失败尝试的成本未完整获取；未配置单价时不编造货币费用。

消融相对基线一次只改一个轴，Embedding 名称和维度视为一个轴。真实支持 recursive/section/parent_child、Dense/BM25/Hybrid、精排、静态/规则/LLM 路由、既有反馈重检、off/deterministic/semantic 校验。未知模式、维度或忽略的编码参数报错。共用测试问题、原生划分、Top-K 和候选预算；调参另用 validation。

## 性能脚本

```powershell
python -m ai_agent_template.evaluation.index_benchmark --blocks 10 --repeats 3 --threads 1 --alternate-model sentence-transformers/all-MiniLM-L6-v2
# 可为两组指定同一个存储环境
python -m ai_agent_template.evaluation.index_benchmark --blocks 10 --repeats 3 --threads 1 --storage-root "$env:TEMP/adative-rag-bench" --alternate-model sentence-transformers/all-MiniLM-L6-v2
```

全量组从空缓存真实解析、分块、编码与 Qdrant 写入；增量组先索引原文后更新。两组同目标文本、配置、已加载编码器，模型启动与原版准备不计入更新时间。变化比例按预定义研究内容块，覆盖 0%/10%/30%/80%、位置元数据、中断恢复和显式替代模型切换。中断在实际 Upsert 后注入，验证旧版可查，分别记录失败时间和恢复时间。performance.json 保存每次时间、计数、向量计算次数、均值与波动。生成小文档结果不外推为扫描 PDF、生产网络或大语料效果。
