# 实际验收记录

日期：2026-10-13。样本均为明确标记的模拟商品/报表；真实模型调用使用已授权的 Qwen 配置。密钥、数据库、日志、模型和原始评测运行目录均不提交。

## 代码与回归

- `uv sync --locked --all-extras --dev`：成功，182 个解析包。
- `uv run --no-sync pytest`：153 项通过（保留原有回归），整体覆盖率 76%。
- Ruff E4/E7/E9/F：通过；Ruff 格式检查 91 个文件通过；mypy 70 个源文件通过。
- Node 22.23.3 下 `npm ci --offline --no-audit --no-fund`：成功，371 个包；前端 lint、TypeScript 和 Vite build 通过。普通 npm ci 曾因联网审计停滞，使用缓存执行锁定依赖安装。
- 新测试覆盖字段错误/正常、源版本冲突、零/缺失分母、归因窗口/币种/广告类型隔离、缺 SKU、跨 seller/marketplace、幂等和跨 CSV/XLSX/JSON 去重、原文权限、单/多模块图、无知识、非法模型结果/未配置模型、有限结构修复、拒绝因果臆测、修正报表后仍能读取原批次证据，以及相同 SKU 的异常数据被替换后结论改变。

## 实际 HTTP 与模型

自然语言答复升级：全量回归 159 项通过，整体覆盖率 76%，mypy 71 个源文件及前端 lint/类型检查/build 通过。新增验证覆盖部分报表仍给答复、缺分母不虚构退货率、HTTP/SSE/历史报告保存答复、单模块不掺入其他模块，以及模型答复的数值/语义核验、各模块证据覆盖和无依据行业基准拦截。本机 8000/5174 与 Docker 8800/5175 实际检查答复、段落引用、来源范围和已服务的前端源码通过。真实 Qwen 四模块答复曾通过逐段语义与数值核验；因漏引对照期、无依据数值或核验预算不足而未通过的生成答复会保留数据整理版本。模型结果有波动，不保证每次都采用生成版本。

原生：API `127.0.0.1:8000`，React `127.0.0.1:5174`。`/health/live`、`/health/ready`、`/openapi.json`、前端 HTML、CORS 预检均通过。完整流程实际上传 Markdown → 导入 Listing/广告/搜索词/销售/退货/变更/质检 → 四模块 LangGraph 调查 → 结构化统计/文档证据 → SSE → 原文件字节一致性校验通过。

模拟 SKU-A102 的 9 月结果：销售 1000 件，退货 80 件，事件期退货率 0.08；广告 Spend=300 USD、归因销售=100 USD、ACOS=3.0；Listing 容量 60ml 与供应商 30ml 不一致。日期、分组和计算输入都保留在证据中。

真实 Qwen `qwen3.7-flash` 已验证：单广告问题仅选择 ads；跨模块问题选择 listing/returns/ads/knowledge 并返回报告。一次完整运行约 24.3 秒，实际决策 token=12253，失败工具为空。搜索词 `sunscreen spf50` 与供应商明确不具备 SPF 声明的匹配风险通过了语义核验；另一条将关联直接写成导致结果的解释被校验器拒绝并移除。报告明确列出缺失资料，未把拒绝的解释发布成事实。运行结果可随模型输出变化，不作为固定性能指标。

多语言 FastEmbed ONNX 下载完成并经远端 SHA-256 校验，实际编码返回 384 维向量；使用实际 Qdrant Hybrid 检索返回了引用。离线模式使用持久化 BM25，运行时不使用 Fake LLM。

## Docker 与多模态

Docker Compose 的 PostgreSQL 16、Qdrant 1.19、API 和 React 容器已启动；实际 PostgreSQL schema/导入/读取及完整 HTTP 流程通过。容器内 Tesseract 5.5.0 对没有文本层的模拟扫描 PDF 实际识别出 Capacity 30ml，检索证据定位到第 1 页，并保留原文件；OCR 开关在 Compose 默认启用。

本机集成使用忽略目录中的端口覆盖：API 8800、前端 5175、PostgreSQL 55434、Qdrant 6335。由于 Docker Hub Node 24 镜像鉴权域名网络异常，前端容器本次使用缓存 Node 20.20.2 Alpine；仓库默认仍为 Node 24。该限制不伪装成默认镜像的成功验证。

## 业务评测与边界

实际执行 `python -m ai_agent_template.evaluation.operations`，20 个合成 SKU、两个时间窗口。运行目录 `experiments/runs/operations-9aaabcaf38e5` 保存数据哈希、预测及指标。Listing 字段定位 20/20，广告五项公式 100/100，退货数值 20/20，合成高消耗异常 precision/recall、引用有效性和三模块任务完成率均为 1.0；平均执行约 59.5ms。该成绩只说明指定确定性合成案例，不能当作真实运营、通用模型准确率或公开数据集成绩。未建立独立退货原因分类器，不声称分类 F1。

原 QASPER/RAGBench/RAGAS、Recall/MRR/Faithfulness、增量索引与缓存模块及测试保留；本次未重跑大规模公开基准或引用历史成绩作为新业务成绩。

报告滚动修正使用本机 Chrome headless 实测并检查截图：1280×800 桌面和 390×844 手机宽度实际发起诊断，明细默认折叠。将答复扩展至原来的 16 倍并展开明细后，文档高度/宽度仍等于视口；答复高度分别为 416/439px，明细高度为 480/506px，其余内容在对应区域内滚动。前端 lint、类型检查和 build 通过，本机及 Docker 前端已更新。

未进行真实 VLM 图像模型及 BGE 模型实测；原实现与可选配置保留。原生 Windows 未安装 Tesseract，扫描 PDF OCR 已在 Docker 环境验证。当前工作流使用授权报表导入；不声明真实 Amazon 账户 API 接入或执行平台修改。
