# LLM 驱动的科研 Adaptive-RAG

## 执行流程

```text
原始问题 + 最近对话 + 文档范围 + 上次相关来源
    → QueryPlan：指代消解、意图、独立查询、路线、必需事实
    → 需要澄清：直接询问用户，不做检索
    → Dense / Sparse / Hybrid 检索（Hybrid 沿用 Qdrant RRF）
    → 可选 Cross-Encoder 精排
    → EvidenceGrade：相关性、覆盖度、条件一致性、冲突
    → 充分：生成答案 → 引用/数值校验 → 返回
    → 不充分：只针对缺失事实改写、补检
    → 达到预算、重复查询或无增益：部分回答或拒答
```

`agent/decisions.py` 使用 `with_structured_output(..., method="function_calling")` 和 Pydantic 模型。无效结构化输出或无效证据引用会停止对应流程，不会降级为“检索结果非空就回答”。问题理解和首次查询改写合并为一次模型调用。

问答路径显式把 `QueryPlan.route` 传入检索器；`GET /v1/rag/search` 保留原有规则路由，供独立检索测试和不依赖 LLM 的基线使用。模型没有执行工具或改变重试预算的权限。

规划器保留没有在上下文中定义的缩写，不自行扩展 `RT` 等术语；代码额外移除缺少上下文依据的括号释义。评估器的澄清必须说明具体原因（指代、文献或比较条件有歧义），且引用至少两项相关证据中的冲突选项；“资料缺少原因解释”会作为补检缺口处理，不会转成询问用户是否需要更多细节。补检规划器无法提出有用的新查询时，转入部分回答或拒答，不重新要求用户澄清已经明确的问题。

## 模型配置

OpenAI 兼容服务（例如千问）可配置：

```dotenv
AI_AGENT_MODEL_PROVIDER=openai
AI_AGENT_MODEL_NAME=qwen3.7-flash
AI_AGENT_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=填入你自己的密钥
```

`AI_AGENT_QUERY_MODEL_NAME` 留空时复用回答模型；如果已部署支持工具调用和结构化输出的小模型，可单独设置该名字。此升级不自动下载或部署 Qwen 4B，也不假设云服务支持某个本地模型名称。OpenAI 兼容地址也支持现有 `OPENAI_BASE_URL` / `OPENAI_API_BASE` 配置。Qwen 调用关闭 thinking，回答及决策输出分别受 token 上限约束。

本地启动示例（先将 `.env.example` 复制为 `.env` 并完成配置）：

```powershell
uv sync --locked
# .env 中将 AI_AGENT_QDRANT_URL 留空，使用项目目录内的本地持久化存储。
uv run uvicorn ai_agent_template.main:app --host 127.0.0.1 --port 8001
# 另开终端，在 frontend/.env.local 写入 VITE_API_BASE_URL=http://127.0.0.1:8001。
cd frontend
npm ci
npm run dev -- --port 5174 --strictPort
```

## 对话与文档范围

`thread_id` 关联最近对话，最多给规划模型提供最近四轮；持久化的消息窗口也会裁剪。同一会话会把上次相关来源交给问题理解层，支持“它的 MAE 呢？”等追问。没有足够上下文时由规划器提出澄清，不凭空猜测模型或数据集。前端为每次页面会话生成独立 ID。

`active_doc_ids` 是导入时的 source 名（通常是文件名），并不是 Qdrant point UUID：

```json
{
  "message": "它的 MAE 呢？",
  "thread_id": "my-research-thread",
  "active_doc_ids": ["paper.pdf"]
}
```

省略该字段会保留同一会话的范围；传 `[]` 会取消范围限制。API 允许最多八个 source。Qdrant 在 top-k 检索之前过滤 source 和文档类型，工作流再检查一次范围，避免其他文件挤占候选位置。前端右侧可输入文件名，以逗号分隔。

## 执行预算与终止

| 配置 | 默认值 | 控制内容 |
| --- | --- | --- |
| `AI_AGENT_RAG_EVIDENCE_RETRY_LIMIT` | 2 | 初始一轮加最多两轮补检 |
| `AI_AGENT_RAG_MAX_SEARCH_CALLS` | 9 | 每次用户请求的检索调用总数；一轮最多三个子问题 |
| `AI_AGENT_RAG_DECISION_TOKEN_BUDGET` | 16000 | 规划、评估、反馈改写累计 token 用量 |
| `AI_AGENT_QUERY_MODEL_MAX_TOKENS` | 1200 | 单次决策模型输出上限 |
| `AI_AGENT_MODEL_MAX_TOKENS` | 2048 | 单次答案输出上限 |
| `AI_AGENT_RAG_CONTEXT_MAX_CHARS` | 12000 | 提供给评估/回答模型的证据文字窗口 |
| `AI_AGENT_AGENT_HISTORY_TURNS` | 4 | 最近对话轮数 |

累计决策用量优先使用服务端 usage；缺失时按输入估计与最大输出记账。调用前预留输入 token 的估计并限制输出，预算耗尽后停止继续决策。这是执行预算，不是精确计费上限；不同模型的 tokenization、失败请求与技术重试的费用需要服务端监测。

补检保留已相关的证据，优先加入新的证据，并用原始问题的完整要求重新评估。相同查询（忽略大小写、空白和标点）不再执行。补检未增加证据时直接终止；技术异常沿用模型客户端重试，不消耗一次“业务补检”来重复猜测。

普通响应和 SSE `done` 包含 `query_plan`、`retrieval_rounds`、`search_calls`、`decision_tokens`、`stop_reason`、`verification_reason` 和 `missing_facts`。可据此观察路由、补检次数及停止原因。模型规划、评估文本及未经校验的答案不会作为 SSE delta 发送。

## 验证与评测边界

运行 `uv run pytest`、前端构建及 lint。测试覆盖非空无关结果、条件不匹配、定向补检、证据合并、重复查询、无增益停止、轮次/token/调用预算、跨会话隔离、文档过滤、结构化输出异常及流式草稿隔离。

证据评估是 LLM 判断。最终校验现已升级为 Claim/Evidence ID 绑定、确定性检查与可配置语义支持验证；数值校验仍忽略 Markdown 列表序号。版本化缓存和实际公开数据评测见 [证据与增量索引](EVIDENCE_AND_INDEXING.md)、[评测说明](EVALUATION.md)。Cross-Encoder 沿用可选配置。

此升级不声称已有 Recall/MRR 改善。效果比较需在相同独立测试集上重新运行静态 Hybrid、规则路由、仅 LLM 规划和完整反馈流程，并同时记录 P95 延迟、模型调用次数与 token 用量。离线测试和少量真实服务冒烟检查不能代替检索基准评测。
