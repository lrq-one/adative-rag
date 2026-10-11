# 多模态 Adaptive-RAG 知识库问答系统

面向科研论文、实验资料和技术文档的智能知识管理场景，构建集**多模态内容解析、自适应混合检索、可溯源问答与文档增量索引**于一体的 RAG 应用。以 LC-MS/MS、分子建模及保留时间预测等科研知识为典型应用场景，支持科研资料的结构化检索与多文档信息查询。

## 技术栈

**Python、FastAPI、LangChain、LangGraph、Qdrant、BM25、RRF、FastEmbed、Cross-Encoder、OCR/VLM、React、Docker**

## 核心功能

- **科研资料解析与分块：** 支持 PDF、Markdown、TXT、JSON 与图片；提取段落、章节和页码元数据，提供章节分块及 Parent-Child 分块。扫描文档 OCR 与图像 VLM 理解采用可配置组件。
- **Adaptive-RAG 检索：** 在 Qdrant 中实现 Dense 向量与 BM25 Sparse 双路索引，支持混合召回与 RRF 融合；通过 Query 分类动态选择检索方式，并提供可选 BGE Cross-Encoder 精排。
- **可信问答与证据溯源：** 基于 LangGraph 组织问题规划、多问题检索、证据不足重试、答案生成和引用校验；关联来源及页码，缺少有效证据时返回拒答结果。
- **增量索引与资料管理：** 基于稳定内容哈希复用未变化的 Chunk，支持资料更新时清理旧索引及按来源删除。
- **评测与工程能力：** 提供独立的检索评测代码、基础自动化测试、FastAPI 接口、流式问答与前端知识库管理。

## 项目目录

```text
frontend/                          React 问答与知识库界面
src/ai_agent_template/
├── api/                            FastAPI 接口与服务入口
├── agent/                          LangGraph 自适应问答工作流
├── rag/                            文档解析、分块、混合检索、精排与证据校验
├── evaluation/                     Recall、MRR 等离线评测代码
├── core/                           系统配置
└── observability/                  日志与链路追踪
tests/                              单元测试与回归测试
examples/                           科研知识库示例资料
docker-compose.yml                  服务组件定义
```

项目中的 OCR、VLM 与精排模块可按部署资源和需求单独开启；实际检索效果及性能以评测运行结果为准。
