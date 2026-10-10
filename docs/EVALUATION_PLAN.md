# RAG 评测方案（规划，尚未运行）

## 公开数据集

- [QASPER](https://huggingface.co/datasets/allenai/qasper)：科研论文全文问答，适用于分块策略与证据召回评测。
- [SciFact / BEIR](https://huggingface.co/datasets/BeIR/scifact)：科学文献标准检索，可计算 Recall@K、MRR、nDCG。
- [ViDoRe / ArxivQA](https://huggingface.co/datasets/vidore/arxivqa_test_subsampled_beir)：科研图像检索，待图像解析开发后接入。
- [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench)：事实支持性评估，不能直接等同于本项目的引用准确率。

## 实验顺序

1. 固定语料、开发/测试划分、Embedding 模型及评测脚本。
2. 开发集对比 Chunk Size、Overlap、Recursive 与 Structure-aware 分块；证据标注映射到原始内容，不依赖固定 Chunk ID。
3. 在相同条件下比较 Dense、BM25、Hybrid+RRF、Hybrid+RRF+Rerank、Adaptive-RAG。
4. 对生成答案评估 Faithfulness、人工核验的 Citation Precision 与证据不足拒答能力。
5. 最终冻结配置，在独立测试集记录 Recall@5/10、MRR@10、P95 延迟、索引时间和 Token 成本。

## 结果

当前只有 Dense RAG 基线；上述新增检索与评测尚未实现，**没有实测数字**。
