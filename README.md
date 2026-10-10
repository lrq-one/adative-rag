# Scientific Adaptive-RAG｜科研资料知识库（Baseline）

面向科研论文、实验笔记与技术文档的 RAG 问答项目，以 **LC-MS/MS、保留时间（RT）预测和计算化学文献**为演示场景。基于 [phantran/ai-agent-template](https://github.com/phantran/ai-agent-template)（MIT）进行轻量二次开发，保留原有代码结构和许可证。

> **当前状态：基础 Dense RAG 可用，Adaptive-RAG 尚未实现。** README 区分已具备的能力与后续方案；不含未经实测的性能指标。

## 技术栈

Python 3.12、FastAPI、LangChain、LangGraph、Qdrant、FastEmbed、pypdf、React、Docker Compose。

## 现有功能

- **资料导入**：上传 PDF、Markdown、TXT、JSON；PDF 使用 pypdf 提取文本（不支持扫描 PDF OCR、图像理解）。
- **索引检索**：LangChain 递归字符分块 → FastEmbed Embedding → Qdrant Dense 向量检索；默认分块 **1000 字符、重叠 150 字符**，参数可通过环境变量调整。
- **问答与来源**：LangGraph 运行 `retrieve → agent` 流程，支持 LLM 回答、SSE 流式输出和来源标签展示；来源标签不等于逐条事实校验。
- **基础工程**：React 前端、FastAPI 接口、Docker Compose、简单自动化测试与 JSONL 示例评测。
- **重复导入**：对相同来源和内容生成确定性 Chunk ID，减少重复写入；修改文件后的旧 Chunk 删除和完善的增量索引尚未实现。

## 目录

```text
frontend/                      React 问答、上传和检索界面
src/ai_agent_template/
  api/                         FastAPI 接口、鉴权、SSE
  agent/                       LangGraph 工作流、LLM 调用、状态
  rag/                         文件加载、分块、Embedding、Qdrant
  core/                        环境配置、日志
  observability/               可选追踪
tests/                         单测、基础评测脚本
docs/EVALUATION_PLAN.md        公共数据集与实验设计（待运行）
examples/scientific_notes.md   科研场景模拟笔记
docker-compose.yml             API、前端、Qdrant
.env.example                   配置模板
```

## 启动（Windows / Docker Desktop）

先安装 Docker Desktop，复制示例环境变量，并在本地 `.env` 配置 Groq 或 OpenAI 的 API Key（**不要提交密钥**）。

```powershell
Copy-Item .env.example .env
notepad .env
docker compose up --build
```

- 前端：<http://localhost:5173>
- API 文档：<http://localhost:8000/docs>
- Qdrant：<http://localhost:6333/dashboard>

在前端 Knowledge 区域上传 `examples/scientific_notes.md`，再提问“比较 RT 预测模型时为什么需要核对数据划分？”等问题。首次启动需要拉取依赖及 Embedding 模型。

运行后端测试：`uv sync --all-extras --dev`，然后 `uv run pytest`。

## 后续技术方案（未实现）

1. OCR/VLM + 章节/图表关联 + Parent-Child Chunking。
2. BM25 + Dense 双路召回、RRF 融合、BGE Reranker。
3. LangGraph Query 分类、Adaptive Routing、查询改写与有界多步检索。
4. 事实级证据核验、拒答逻辑、带失效删除的增量索引。
5. 使用 **QASPER、SciFact/BEIR、ViDoRe、RAGBench** 做公开基准评测，比较分块参数、Recall@K、MRR、Faithfulness、引用准确性和延迟。详见 [评测计划](docs/EVALUATION_PLAN.md)。

**尚未产生可核验的 Recall、MRR、Faithfulness 或性能提升结果。**

## 上游与开源协议

本仓库由 [AI Agent Template](https://github.com/phantran/ai-agent-template) 改编；原始代码及版权声明按仓库保留的 [MIT LICENSE](LICENSE) 执行。本仓库当前改动集中在科研问答提示词、界面示例、项目说明及示例资料。
