# 跨境电商多模态 Adaptive-RAG 智能运营诊断平台

面向跨境电商卖家内部运营人员，整合商品与供应商资料、Listing、广告、销售、退货和质检报表，识别异常并生成可追溯的调查报告。系统区分事实、相关性、假设和人工审核建议，支持通用品类及明确标记的模拟数据。

## 技术栈

Python、FastAPI、SQLAlchemy、PostgreSQL、LangGraph、Qwen、Qdrant、FastEmbed、BM25、RRF、BGE Reranker、PDF OCR/VLM、RAGAS、React、Docker。

## 核心功能

- **多模态商品知识与 Listing 诊断：** 商品映射、版本化供应商资料、快照及问题报告；确定性规格对比和修改草稿，保留原文、页码与版本定位。
- **广告表现与搜索词分析：** 按币种、广告类型、归因窗口及投放维度计算指标，比较历史表现，提示高消耗低销售、样本量与归因滞后；结合检索资料辅助人工审核搜索词匹配。
- **退货与质量分析：** 基于销售与退货件数、明确的时间和 cohort 口径计算退货率，关联原因分布、包装变更及质检时间线，明确因果关系尚待验证。
- **LangGraph 综合诊断：** Qwen 结构化规划必要模块，调用实际业务分析服务和 Adaptive-RAG，执行有限补检、跨来源关联、数值重算及引用核验，通过 SSE 展示执行过程。
- **证据、增量索引与评测：** SQL 报表行与计算溯源，复用文档版本、持久化解析及 Embedding 缓存，保留 QASPER/RAGBench/RAGAS，新增隔离 Ground Truth 的模拟业务评测。

## 项目目录

```text
src/ai_agent_template/
├── commerce/       业务实体、报表导入、诊断分析与调查图
├── agent/          原有问题规划、消歧、检索反馈与答案校验
├── rag/            多模态解析、Qdrant、版本索引、缓存与证据
├── api/            运营接口、SSE 与兼容接口
├── evaluation/     公开数据集与运营业务评测
├── core/           配置
└── observability/  日志与追踪
frontend/           React 卖家运营工作台
examples/commerce/  明确标记的模拟资料与报表
tests/              原有回归、业务与权限测试
docs/               配置说明、统计口径与验收记录
```

工程边界与实际验证范围见 [工程说明](docs/COMMERCE.md)。保留原 LICENSE 及第三方版权归属。
