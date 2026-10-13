import { API_BASE_URL } from "./api";
let token = "demo-operator";
export const setToken = (value: string) => {
  token = value;
};
export type Data = Record<string, unknown>;
export type Product = {
  sku: string;
  asin: string;
  product_id: string;
  product_version: string;
  synthetic: boolean;
};
export type Source = {
  content: string;
  source: string;
  evidence: {
    evidence_id: string;
    document_id: string;
    document_version: number;
    page_number: number | null;
  };
};
export type Report = {
  answer: string;
  answer_paragraphs: { text: string; kind: string; evidence_ids: string[] }[];
  answer_mode: "verified_data" | "verified_llm";
  investigation: {
    request: {
      sku: string;
      marketplace_id: string;
      start: string;
      end: string;
      question: string;
    };
  };
  results: Record<string, Data>;
  findings: { kind: string; text: string; evidence_ids: string[] }[];
  evidence: Data[];
  document_evidence: Source[];
  missing_evidence: string[];
  elapsed_ms: number;
  verification: Data;
  generation_mode: string;
};
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}/v1/operations${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body && !(body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
    },
    body:
      body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const data = (await response.json()) as { detail?: string };
    throw new Error(
      typeof data.detail === "string" ? data.detail : JSON.stringify(data),
    );
  }
  return response.json() as Promise<T>;
}
export async function* stream(body: unknown): AsyncGenerator<{
  type: string;
  node?: string;
  status?: string;
  message?: string;
  error_type?: string;
  report?: Report;
}> {
  const response = await fetch(
    `${API_BASE_URL}/v1/operations/investigations/stream`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    },
  );
  if (!response.ok || !response.body) throw new Error(await response.text());
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end = buffer.indexOf("\n\n");
    while (end >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5))
        .join("");
      if (data) yield JSON.parse(data) as { type: string; report?: Report };
      end = buffer.indexOf("\n\n");
    }
  }
}
export async function original(source: Source, scope: Record<string, string>) {
  const tab = window.open("about:blank", "_blank");
  if (!tab) throw new Error("请允许原文窗口");
  try {
    const e = source.evidence,
      query = new URLSearchParams({
        marketplace_id: scope.marketplace_id,
        sku: scope.sku,
        asin: scope.asin,
      });
    const response = await fetch(
      `${API_BASE_URL}/v1/operations/documents/${e.document_id}/${e.document_version}/file?${query}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    if (!response.ok) throw new Error("原文不可用");
    const url = URL.createObjectURL(await response.blob());
    tab.location.href = url + (e.page_number ? `#page=${e.page_number}` : "");
    window.setTimeout(() => URL.revokeObjectURL(url), 300000);
  } catch (error) {
    tab.close();
    throw error;
  }
}
