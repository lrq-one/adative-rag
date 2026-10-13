# 运营诊断工程说明

## 结构与授权

`operations_models.py` 定义 ListingSnapshot/Issue、SupplierDocument、ProductRevision、Sales/Return/QualityRecord、AdsCampaign/SearchTerm/Performance、BusinessEvidence、InvestigationPlan/Report。原 Product/Variant 和兼容 API 保留。SKU/ASIN 映射在供应商记录中包含 seller、marketplace、product 和版本；多 ASIN 时必须明确选择。SQL 查询先过滤 seller/marketplace，再过滤 SKU/ASIN。前端是内部运营工作台，默认入口为综合诊断。

生产设置 `AI_AGENT_COMMERCE_MODE=production`，数据库使用 `postgresql+psycopg://...`。受信任 Bearer 令牌在服务端 `AI_AGENT_COMMERCE_PRINCIPALS` 映射至 store_id（seller_id）、operator 和 marketplaces；客户端不能自行扩大权限。旧 `/v1/agent` 与 `/v1/rag` 独立采用 API Key。生产未配置 API Key 时关闭其访问。

运营数据库复用 CommerceRepository 的 engine。`operations_repository.py` 创建增量 v2 表（operations_records/import_batches/investigations/migrations），采用事务和跨进程初始化锁，不删除 v1 数据。运营实体以 seller/marketplace/kind/id 复合键存储，SKU 与生效时间有 SQL 索引；JSON 保存类型校验后的实体。PostgreSQL 与 SQLite 使用同一 SQLAlchemy 适配器；SQLite 仅为 Demo/测试用途。

## 报表与统计口径

`POST /v1/operations/imports/{kind}` 接受 CSV/XLSX/JSON、seller_id、marketplace_id、可选 field_mapping（原表头到标准字段的 JSON）。类型：supplier/listing/listing_issue/revision/sales/returns/quality/campaign/ads/search_terms/rules。必要字段见相应 Pydantic 实体和 `examples/commerce/operations/*.json`。所有数据校验完再原子提交；错误返回行号。整文件批次按内容去重，记录经日期/数字归一化后跨文件格式去重。

提供源系统稳定 `id` 时，同一 seller/marketplace/SKU/ASIN/product 的后续行可更新该记录。没有 id 时采用规范化内容指纹，只能识别内容完全相同的重复，修正数字的报告必须保留稳定 id，避免被误认作新事件。导入需确认报告本身的颗粒度，不能将汇总和明细重复叠加。

Listing 比较最新有效供应商规格与快照，对缺字段、容量、颜色、尺寸、包装数量和版本差异逐项生成证据。同时间同版本冲突时停止生成修改草稿。导入 rules 必须有来源 reference 和 authority；默认结果只声明内部一致性检查，不宣称通过 Amazon 合规审核。

退货 numerator=退货件数，denominator=销售件数。event_period 按各自事件日期窗口，不等同于订单归属期，可能存在滞后；sales_cohort 按 sold_date 窗口、明确 cohort_id 及 as_of 观测截止，缺归属信息时不给退货率。零/缺少分母返回 null。包装变更只提供时间上的共现和待验证假设，建议对照批次、运输及质检。

广告 CTR=clicks/impressions、CPC=spend/clicks、CVR=attributed_orders/clicks、ACOS=spend/attributed_sales、ROAS=attributed_sales/spend。零或缺少分母返回 null；部分行缺少值时不将其当零。币种、广告类型、归因窗口、campaign/adgroup/search-term 分组，不跨组混算。Campaign 报告和搜索词报告分别分析，不合并两个粒度的总数。等长窗口才比较点击总量；比率只在相同口径下比较。归因尚未成熟和 clicks<20 明确提示不确定性。异常阈值是内部诊断规则，不代表 Amazon 规则。

## 调查图与知识检索

每个批次另存规范化导入行；稳定 id 的后续修正不会覆盖原批次证据。`GET /v1/operations/rows` 按授权站点、row_ids 和 batch_ids 读取相应来源。旧本机批次若尚无归档，仅在当前行的批次精确一致时返回当前快照，不把其他批次的修正值当作旧证据。

`POST /v1/operations/investigations` 或 `/investigations/stream` 使用选择范围、question、use_llm、modules。离线必须明确选择模块；不通过假模型或关键词猜测业务意图。use_llm=true 实际调用 Qwen/OpenAI 兼容结构化模型；缺密钥、非法输出或调用失败明确报错。模型计划不能改变选定范围。

LangGraph：规划 → 参数验证 → 按需 Listing/Returns/Ads/Knowledge → EvidenceAssessment → CrossSourceAnalysis → ClaimVerification → Report。最多四个业务工具；知识工具复用 LLMQueryPolicy 的 EvidenceGrade/Rewrite，继承检索调用、决策 token 和补检次数预算。结果包含执行轨迹、失败工具、统计口径、证据和人工审核建议。每项结构化结果重新计算并比较完整计算/来源对象；期间数据变化会拒绝发布。模型解释/搜索词匹配必须经过已有 LLMEvidenceVerifier，未支持的解释不进入结论，生成的业务影响始终作为待验证假设。

`POST /documents` 绑定 seller/marketplace/SKU/ASIN/product/version，使用原 Loader → VersionedIndexer → EvidenceRepository，保留原文件、Parent-Child、增量索引及解析缓存。真实检索配置复用多语言 FastEmbed、Qdrant Dense/BM25/Hybrid/RRF 和可选 BGE；断网 Demo 使用真实 BM25 在持久化证据上计算词法相关性，不生成模拟 dense 向量，也不使用 Fake LLM。两种索引路径独立。未建立资料索引时报告资料缺失；OCR/VLM 依赖不可用时明确失败。

原文可通过 `/documents/{id}/{version}/file?marketplace_id=...&sku=...&asin=...` 定位。当前检索仅采用活动且符合所选生效日期/版本的文档；历史原文可查看，但活动新版在所选历史期尚未生效时不会冒充旧版知识，需明确导入/选择相应历史资料。

## 运行与评测

安装：`uv sync --locked --all-extras --dev`；后端入口：`uv run uvicorn ai_agent_template.main:app --host 127.0.0.1 --port 8000`。前端：`npm ci`、`npm run dev -- --host 127.0.0.1 --port 5174`；本地 `VITE_API_BASE_URL=http://127.0.0.1:8000`。Compose 使用 PostgreSQL/Qdrant 与持久化卷，`.env` 可选；production 需自行配置真实授权和数据库凭据。

`python -m ai_agent_template.commerce.operations_demo` 重建 20 SKU、两个时间窗口及多版本报表。生成器里的异常设定只用于造数据，分析代码不按 SKU 写死结论。Ground Truth 文件只由评测器加载，绝不提交给规划模型。`python -m ai_agent_template.evaluation.operations` 输出实际 predictions、metrics 和数据 SHA256 至独立 `experiments/runs/operations-*`；旧公开数据集评测入口保留。此合成评测验证已知字段/数字/引用与异常场景，不能当作真实运营或通用 LLM 准确率。原因字段直接来自报表，尚未加入独立原因分类器，因此不冒报分类 F1。

实际运行和外部依赖验证范围见 [验收记录](VALIDATION.md)。不连接未经授权的 Amazon API，不执行广告或 Listing 写操作，不向买家发消息。所有输出用于运营人员审核。
