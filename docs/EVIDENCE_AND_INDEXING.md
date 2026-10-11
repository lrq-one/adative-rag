# 证据与版本化索引

功能接入现有 RagService、Agent Graph、FastAPI 和 React，共用原有问题规划、证据评估与有限补检，没有另建问答系统。

## 证据与校验

EvidenceRecord 保存 document_id、version、chunk_id、parent_chunk_id、文件名、页码、章节、原文、SHA-256 和可选坐标。同名不同文档可通过上传的 document_id 区分；未指定 ID 时，同命名空间、类型、来源名视为同一逻辑文档的更新。更新可传回上次响应的 document_id。跨版本 Evidence ID 不复用；历史引用解析其旧版本。

检索后按请求分配 E1 等短引用。生成器返回逐条 claim 与 evidence_ids；校验只使用绑定的具体 Child 原文，不能借用 Parent 其他内容。默认先检查版本、引用、数值、时间单位换算和明显实体冲突，再调用结构化语义判断检查模型、数据集、条件和事实支持。错误、缺失或格式异常的模型结果不通过。

`AI_AGENT_RAG_VERIFICATION_LEVEL=off|deterministic|semantic`，响应 verification 明确级别。deterministic 只能发现确定性错误，不能证明事实蕴含；semantic 仍是自动模型判断，有误判可能。`AI_AGENT_RAG_VERIFIER_MODEL_NAME` 可单独指定模型。答案最多修正一次，证据缺失沿用现有 Rewrite/补检预算，最终部分回答或拒答。SSE 仅推送最终校验后的回答。

React 点击 [E1] 展开版本、页码、章节及原文；查看原文取回上传的具体版本，PDF 跳至页码。文本 API 上传没有二进制原文件，以证据文本定位。当前 PDF 逐页提取，bbox 为 null，没有声称双栏阅读顺序或扫描页坐标可靠。OCR、VLM 默认关闭，需要额外组件及模型配置。

## 缓存、版本与恢复

`AI_AGENT_RAG_DATA_PATH` 保存 SQLite Manifest、证据、缓存、原文件，Qdrant 独立配置。WAL、事务与跨进程文件锁支持重启和同文档并发。全文件解析、PDF 页文本、OCR、VLM、分块、Dense/Sparse 编码分别缓存，键包含内容及影响输出的模型、提示词、解析和分块/编码参数。相同文本共享向量缓存，各原文位置保留独立身份。

Parent-Child 按段落单元分块，开头插段落能复用后文；Recursive 可能因边界变化影响更多 Chunk。位置、章节、文件名变化时复用已有 Qdrant 点，更新版本成员和 Manifest 证据映射；新/改文本才 Upsert 向量。vector_reuses 与 embedding_cache_hits 分开计数，不把未发生的编码缓存查询计为命中。

更新先建 building Manifest，写入证据和向量，检查点数量与 Dense/Sparse 完整性，最后以一次 SQLite 事务发布活动指针。查询读取活动快照，在 Top-K 前过滤 doc/version，按快照映射返回正确元数据，避免 building/failed 混入。失败保留旧活动版本并记录任务；下次持锁更新将遗留 building 标为 interrupted。保留历史索引，可校验后回滚。

集合名包含完整编码配置、实际下载的模型/分词器文件指纹与 schema 指纹，不混用不同模型向量。FastEmbed 不支持任意 HF revision 参数，本项目对此报错，请使用已注册模型并保存实际配置和依赖版本。历史点尚无自动保留期限清理；BM25 集合 IDF 仍包含历史点，版本过滤保证返回内容隔离，不保证历史完全不影响排序。删除屏蔽全部版本、移除缓存引用并清理该文档 Qdrant 点；其他文档使用的共享缓存保留。原上传文件暂作离线保留，删除后原文 API 不再访问。

## 接口

原有上传、检索、invoke/stream 保留。上传新增 document_id、document_version、stats；问答新增 answer、citations、verification、answer_repairs。

| 接口 | 结果 |
| --- | --- |
| GET `/v1/rag/evidence/{id}` | 指定版本原文和 active/historical；已删除、失败版本 404 |
| GET `/v1/rag/document-versions/{doc}` | 版本 Manifest |
| GET `/v1/rag/document-versions/{doc}/{version}/file` | 原上传文件 |
| POST `/v1/rag/document-versions/{doc}/{version}/activate` | 同模型且完整的历史索引恢复；不兼容 409 |
| GET `/v1/rag/index-jobs?limit=50` | 当前命名空间的任务、缓存计数、时间及失败类型 |

本机 D 盘曾出现 SQLite 写入长时间等待，C 盘临时目录探测正常；性能脚本支持 `--storage-root` 并记录实际条件，这不是通用硬件结论。
