import { useCallback, useEffect, useState } from "react";
import { Layers } from "lucide-react";
import {
  api,
  stream,
  original,
  setToken,
  type Data,
  type Product,
  type Report,
  type Source,
} from "./lib/operations";
import "./commerce.css";
const tabs: [string, string][] = [
  ["overview", "运营诊断总览"],
  ["listing", "商品资料与 Listing 诊断"],
  ["returns", "退货与质量分析"],
  ["ads", "广告异常分析"],
  ["investigation", "综合 AI 诊断"],
  ["knowledge", "知识资料管理"],
];
function Detail({ name, value }: { name: string; value: unknown }) {
  return (
    <details>
      <summary>{name}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
function Analysis({ name, data }: { name: string; data: Data }) {
  const records = (data.issues ?? data.groups ?? []) as Data[];
  return (
    <section className="panel">
      <h2>{name}</h2>
      {"returned_units" in data && (
        <div className="metric-grid">
          {["returned_units", "sold_units", "return_rate"].map((key) => (
            <div className="metric" key={key}>
              <small>{key}</small>
              <strong>
                {data[key] === null ? "缺少分母" : String(data[key])}
              </strong>
            </div>
          ))}
        </div>
      )}
      {records.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {Object.keys(records[0])
                  .filter((key) => key !== "evidence_id")
                  .map((key) => (
                    <th key={key}>{key}</th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {records.map((row, index) => (
                <tr key={index}>
                  {Object.entries(row)
                    .filter(([key]) => key !== "evidence_id")
                    .map(([key, value]) => (
                      <td key={key}>
                        {typeof value === "object"
                          ? JSON.stringify(value)
                          : String(value ?? "—")}
                      </td>
                    ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {Object.entries(data)
        .filter(
          ([key]) => !["groups", "issues", "actions_executed"].includes(key),
        )
        .map(([key, value]) => (
          <Detail
            key={key}
            name={
              key === "evidence" ? "统计证据 · 批次、行、公式与筛选条件" : key
            }
            value={value}
          />
        ))}
    </section>
  );
}
export default function OperationsApp() {
  const [tab, setTab] = useState("investigation"),
    [scope, setScope] = useState<Record<string, string>>({
      seller_id: "demo-store",
      marketplace_id: "US",
      sku: "SKU-A102",
      asin: "DEMO000102",
      start: "2026-09-01",
      end: "2026-09-30",
      comparison_start: "2026-08-01",
      comparison_end: "2026-08-31",
      as_of: "2026-10-13",
    }),
    [products, setProducts] = useState<Product[]>([]),
    [markets, setMarkets] = useState(["US"]);
  const [access, setAccess] = useState("demo-operator"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [result, setResult] = useState<Data | null>(null),
    [report, setReport] = useState<Report | null>(null),
    [steps, setSteps] = useState<string[]>([]),
    [sources, setSources] = useState<Source[]>([]),
    [evidence, setEvidence] = useState<Data | null>(null);
  const [sourceRows, setSourceRows] = useState<Data[]>([]);
  const [question, setQuestion] = useState(
      "分析这个商品 Listing 差异、退货增加与广告异常，说明证据、可能影响和建议。",
    ),
    [llm, setLlm] = useState(false),
    [modules, setModules] = useState([
      "listing",
      "returns",
      "ads",
      "knowledge",
    ]),
    [kind, setKind] = useState("supplier"),
    [mapping, setMapping] = useState("{}"),
    [attributes, setAttributes] = useState(
      '{"capacity":"30ml","color":"clear","package_quantity":"1","packaging":"carton-v2","category":"skincare"}',
    );
  const refresh = useCallback(async () => {
    const identity = await api<{ sellers: string[]; marketplaces: string[] }>(
      "/scope",
    );
    setMarkets(identity.marketplaces);
    setScope((value) => ({ ...value, seller_id: identity.sellers[0] }));
    setProducts(
      await api<Product[]>(`/products?marketplace_id=${scope.marketplace_id}`),
    );
  }, [scope.marketplace_id]);
  useEffect(() => {
    let active = true;
    Promise.all([
      api<{ sellers: string[]; marketplaces: string[] }>("/scope"),
      api<Product[]>(`/products?marketplace_id=${scope.marketplace_id}`),
    ])
      .then(([identity, catalog]) => {
        if (active) {
          setMarkets(identity.marketplaces);
          setScope((value) => ({ ...value, seller_id: identity.sellers[0] }));
          setProducts(catalog);
        }
      })
      .catch((err) => {
        if (active) setError(String(err));
      });
    return () => {
      active = false;
    };
  }, [scope.marketplace_id]);
  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await work();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }
  const request = () => ({ ...scope, question, modules, use_llm: llm });
  const field = (key: string, value: string) => {
    setScope((current) => ({ ...current, [key]: value }));
    setResult(null);
    setReport(null);
    setSources([]);
  };
  async function diagnose() {
    setSteps([]);
    setReport(null);
    for await (const event of stream(request())) {
      if (event.type === "error")
        throw new Error(`${event.message} (${event.error_type})`);
      if (event.type === "tool")
        setSteps((value) => [...value, `${event.node} · ${event.status}`]);
      if (event.report) {
        setReport(event.report);
        setSources(event.report.document_evidence);
      }
    }
  }
  function openEvidence(id: string) {
    if (!report) return;
    const ref = report.evidence.find((item) => item.evidence_id === id);
    if (ref) setEvidence(ref);
    else {
      const source = report.document_evidence.find(
        (item) => item.evidence.evidence_id === id,
      );
      if (source) void action(() => original(source, scope));
    }
  }
  async function importFile(file: File, document = false) {
    const body = new FormData();
    body.set("file", file);
    body.set("seller_id", scope.seller_id);
    body.set("marketplace_id", scope.marketplace_id);
    if (document) {
      const product = products.find(
        (item) => item.sku === scope.sku && item.asin === scope.asin,
      );
      if (!product) throw new Error("请先导入商品映射");
      body.set("sku", scope.sku);
      body.set("asin", scope.asin);
      body.set("product_id", product.product_id);
      body.set("product_version", product.product_version);
      body.set("effective_at", scope.start);
      body.set("attributes", attributes);
    } else body.set("field_mapping", mapping);
    const data = await api<Data>(
      document ? "/documents" : `/imports/${kind}`,
      body,
    );
    setNotice(`导入完成：${JSON.stringify(data)}`);
    await refresh();
  }
  return (
    <div className="commerce-shell">
      <aside className="sidebar">
        <div className="brand">
          <Layers />
          <div>
            运营诊断<span>ADAPTIVE RAG · SELLER OPS</span>
          </div>
        </div>
        <nav>
          {tabs.map(([key, name]) => (
            <button
              key={key}
              className={tab === key ? "active" : ""}
              onClick={() => {
                setTab(key);
                setResult(null);
              }}
            >
              {name}
            </button>
          ))}
        </nav>
        <div className="identity">
          <label>
            运营身份令牌
            <input
              type="password"
              value={access}
              onChange={(event) => setAccess(event.target.value)}
            />
          </label>
          <button
            onClick={() =>
              void action(async () => {
                setToken(access);
                setReport(null);
                setResult(null);
                setSources([]);
                setEvidence(null);
                setSourceRows([]);
                await refresh();
              })
            }
          >
            应用身份
          </button>
          <small>Demo: demo-operator。生产站点权限由服务端管理。</small>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <p className="eyebrow">OPERATIONS INTELLIGENCE</p>
            <h1>跨境电商智能运营诊断平台</h1>
            <p>连接商品资料与业务报表，让发现可以追溯</p>
          </div>
          <span className="status">
            {products.some((item) => item.synthetic)
              ? "模拟数据 · Demo"
              : "授权报表"}
          </span>
        </header>
        <section className="panel controls">
          <label>
            Seller
            <select
              value={scope.seller_id}
              onChange={(event) => field("seller_id", event.target.value)}
            >
              <option value={scope.seller_id}>{scope.seller_id}</option>
            </select>
          </label>
          <label>
            Marketplace
            <select
              value={scope.marketplace_id}
              onChange={(event) => field("marketplace_id", event.target.value)}
            >
              {markets.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            SKU / ASIN
            <select
              value={`${scope.sku}|${scope.asin}`}
              onChange={(event) => {
                const [sku, asin] = event.target.value.split("|");
                setScope((value) => ({ ...value, sku, asin }));
                setReport(null);
                setSources([]);
                setResult(null);
              }}
            >
              {products.map((item) => (
                <option
                  key={`${item.sku}|${item.asin}`}
                  value={`${item.sku}|${item.asin}`}
                >
                  {item.sku} / {item.asin}
                </option>
              ))}
            </select>
          </label>
          {[
            ["start", "开始"],
            ["end", "结束"],
            ["comparison_start", "对照开始"],
            ["comparison_end", "对照结束"],
            ["as_of", "观测截止"],
          ].map(([key, name]) => (
            <label key={key}>
              {name}
              <input
                type="date"
                value={scope[key]}
                onChange={(event) => field(key, event.target.value)}
              />
            </label>
          ))}
        </section>
        {error && (
          <div className="notice error" role="alert">
            {error}
          </div>
        )}
        {notice && (
          <details className="notice">
            <summary>导入结果</summary>
            <pre>{notice}</pre>
          </details>
        )}
        {tab === "overview" && (
          <section className="panel">
            <h2>运营诊断总览</h2>
            <p>
              所选站点 {products.length} 个商品映射。本次调查发现{" "}
              {report?.findings.length ?? 0} 项。
            </p>
            <p>
              模拟案例 SKU-A102 对比 2026 年 8 月与 9 月；结论由源报表计算。
            </p>
            <button onClick={() => setTab("investigation")}>
              进入综合诊断
            </button>
          </section>
        )}
        {["listing", "returns", "ads"].includes(tab) && (
          <>
            <section className="panel">
              <button
                disabled={busy}
                onClick={() =>
                  void action(async () =>
                    setResult(await api<Data>(`/analysis/${tab}`, scope)),
                  )
                }
              >
                执行{tabs.find(([key]) => key === tab)?.[1]}
              </button>
            </section>
            {result ? (
              <Analysis name={tab} data={result} />
            ) : (
              <p className="empty">选择范围后执行分析</p>
            )}
          </>
        )}
        {tab === "investigation" && (
          <>
            <section className="panel">
              <h2>综合 AI 诊断</h2>
              <textarea
                className="investigation-question"
                rows={4}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
              <div className="controls">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={llm}
                    onChange={(event) => setLlm(event.target.checked)}
                  />
                  真实千问语义规划
                </label>
                {["listing", "returns", "ads", "knowledge"].map((module) => (
                  <label key={module} className="check">
                    <input
                      type="checkbox"
                      disabled={llm}
                      checked={modules.includes(module)}
                      onChange={(event) =>
                        setModules((value) =>
                          event.target.checked
                            ? [...value, module]
                            : value.filter((item) => item !== module),
                        )
                      }
                    />
                    {module}
                  </label>
                ))}
              </div>
              <p className="notice">
                {llm
                  ? "模型规划必要模块；指标由 Python 计算并核验。"
                  : "确定性模式：明确选择模块，不使用模拟 LLM。"}
              </p>
              <button
                disabled={busy || (!llm && !modules.length)}
                onClick={() => void action(diagnose)}
              >
                {busy ? "诊断执行中…" : "发起综合诊断"}
              </button>
              <div className="execution">
                {steps.map((step, index) => (
                  <span key={index}>{step}</span>
                ))}
              </div>
            </section>
            {report && (
              <>
                <section
                  className="panel diagnosis-answer"
                  aria-label="诊断答复"
                >
                  <div className="answer-heading">
                    <h2>诊断答复</h2>
                    <span>
                      {report.answer_mode === "verified_llm"
                        ? "千问答复 · 已核验证据"
                        : "依据已核验数据整理"}
                    </span>
                  </div>
                  <p className="answer-scope">
                    {report.investigation.request.sku} ·{" "}
                    {report.investigation.request.marketplace_id} ·{" "}
                    {report.investigation.request.start} 至{" "}
                    {report.investigation.request.end}
                  </p>
                  <div className="answer-prose">
                    {(report.answer_paragraphs?.length
                      ? report.answer_paragraphs
                      : [
                          {
                            text:
                              report.answer ||
                              "这份历史报告没有保存自然语言答复，请重新发起诊断。",
                            kind: "limitation",
                            evidence_ids: [],
                          },
                        ]
                    ).map((paragraph, index) => (
                      <div
                        className={`answer-paragraph ${paragraph.kind}`}
                        key={index}
                      >
                        <p>{paragraph.text}</p>
                        {paragraph.evidence_ids.length > 0 && (
                          <div
                            className="answer-citations"
                            aria-label="本段依据"
                          >
                            <span>依据</span>
                            {paragraph.evidence_ids.map((id, sourceIndex) => (
                              <button
                                className="citation-button"
                                key={id}
                                onClick={() => openEvidence(id)}
                                title={id}
                              >
                                证据 {sourceIndex + 1}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
                <section className="panel">
                  <h2>证据与详细发现 · {report.elapsed_ms.toFixed(0)} ms</h2>
                  {Object.entries({
                    fact: "已确认事实",
                    correlation: "观察到的相关性",
                    hypothesis: "待验证假设",
                    suggestion: "人工审核建议",
                  }).map(([kind, name]) => (
                    <div key={kind}>
                      <h3>{name}</h3>
                      {report.findings
                        .filter((item) => item.kind === kind)
                        .map((item, index) => (
                          <article className="finding" key={index}>
                            <p>{item.text}</p>
                            {item.evidence_ids.map((id) => (
                              <button
                                className="citation-button"
                                key={id}
                                onClick={() => openEvidence(id)}
                              >
                                {id.slice(0, 14)}
                              </button>
                            ))}
                          </article>
                        ))}
                    </div>
                  ))}
                  {report.missing_evidence.map((gap, index) => (
                    <p className="notice" key={index}>
                      {gap}
                    </p>
                  ))}
                  <Detail
                    name="指标和引用核验结果"
                    value={{
                      mode: report.generation_mode,
                      verification: report.verification,
                    }}
                  />
                </section>
                {Object.entries(report.results)
                  .filter(([key]) => key !== "knowledge")
                  .map(([key, data]) => (
                    <Analysis key={key} name={key} data={data} />
                  ))}
              </>
            )}
          </>
        )}
        {tab === "knowledge" && (
          <>
            <section className="panel">
              <h2>授权报表导入</h2>
              <div className="controls">
                <label>
                  报告类型
                  <select
                    value={kind}
                    onChange={(event) => setKind(event.target.value)}
                  >
                    {[
                      "supplier",
                      "listing",
                      "listing_issue",
                      "revision",
                      "sales",
                      "returns",
                      "quality",
                      "ads",
                      "search_terms",
                      "campaign",
                      "rules",
                    ].map((item) => (
                      <option key={item}>{item}</option>
                    ))}
                  </select>
                </label>
                <input
                  type="file"
                  accept=".csv,.xlsx,.json"
                  disabled={busy}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void action(() => importFile(file));
                    event.target.value = "";
                  }}
                />
              </div>
              <label>
                原始表头 → 标准字段映射（JSON）
                <textarea
                  value={mapping}
                  onChange={(event) => setMapping(event.target.value)}
                />
              </label>
            </section>
            <section className="panel">
              <h2>供应商文档与版本索引</h2>
              <label>
                经资料确认的规格属性（JSON）
                <textarea
                  value={attributes}
                  onChange={(event) => setAttributes(event.target.value)}
                />
              </label>
              <p>绑定当前商品和版本，生效日期使用所选开始日期。</p>
              <input
                type="file"
                accept=".md,.txt,.pdf,.docx,.png,.jpg"
                disabled={busy}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void action(() => importFile(file, true));
                  event.target.value = "";
                }}
              />
              <div className="controls">
                <input
                  aria-label="检索问题"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                />
                <button
                  disabled={busy}
                  onClick={() =>
                    void action(async () =>
                      setSources(
                        (
                          await api<{ sources: Source[] }>(
                            "/knowledge/search",
                            request(),
                          )
                        ).sources,
                      ),
                    )
                  }
                >
                  检索商品资料
                </button>
              </div>
            </section>
          </>
        )}
        {sources.length > 0 && (
          <section className="panel">
            <h2>文档证据与原文定位</h2>
            {sources.map((source) => (
              <details key={source.evidence.evidence_id}>
                <summary>
                  {source.source} · v{source.evidence.document_version} · 页{" "}
                  {source.evidence.page_number ?? "—"}
                </summary>
                <p>{source.content}</p>
                <button
                  onClick={() => void action(() => original(source, scope))}
                >
                  查看原文件
                </button>
              </details>
            ))}
          </section>
        )}
        {busy && <p role="status">正在读取资料或执行分析，请稍候…</p>}
      </main>
      {evidence && (
        <div className="modal-backdrop" onClick={() => setEvidence(null)}>
          <section
            className="evidence-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <button onClick={() => setEvidence(null)}>关闭</button>
            <h2>统计证据与计算溯源</h2>
            <pre>{JSON.stringify(evidence, null, 2)}</pre>
            <button
              disabled={busy}
              onClick={() =>
                void action(async () => {
                  const query = new URLSearchParams({
                    marketplace_id: String(
                      evidence.marketplace ?? scope.marketplace_id,
                    ),
                    row_ids: (evidence.row_ids as string[]).join(","),
                    batch_ids: (evidence.import_batch_id as string[]).join(","),
                  });
                  setSourceRows(await api<Data[]>(`/rows?${query}`));
                })
              }
            >
              读取对应批次的导入行
            </button>
            <Detail name="导入记录" value={sourceRows} />
          </section>
        </div>
      )}
    </div>
  );
}
