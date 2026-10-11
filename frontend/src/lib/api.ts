export type AgentInvokeRequest = {
  message: string;
  thread_id?: string;
  active_doc_ids?: string[];
};

export type AdaptiveDetails = {
  answer?: string;
  citations?: CitationRecord[];
  verification?: { level: string; supported: boolean; error: string } | null;
  answer_repairs?: number;
  query_plan?: {
    standalone_query: string;
    intent: "fact_lookup" | "explanation" | "comparison" | "multi_hop";
    route: "dense" | "sparse" | "hybrid";
    entities: string[];
    sub_queries: string[];
    required_facts: string[];
    needs_clarification: boolean;
    clarification_question?: string | null;
  } | null;
  retrieval_rounds?: number;
  search_calls?: number;
  decision_tokens?: number;
  stop_reason?: string;
  verification_reason?: string;
  missing_facts?: string[];
};

export type CitationRecord = {
  citation_id: string;
  snippet: string;
  status: string;
  evidence: {
    evidence_id: string;
    document_id: string;
    document_version: number;
    source_filename: string;
    page_number?: number | null;
    section_title?: string | null;
    bbox?: number[] | null;
    text: string;
  };
};

export type AgentInvokeResponse = AdaptiveDetails & {
  message: string;
  thread_id?: string;
  sources: RagSource[];
};

export type RagSource = {
  source: string;
  content: string;
  score?: number | null;
  metadata: Record<string, unknown>;
};

export type RagIngestResponse = {
  source: string;
  chunks: number;
  document_id?: string;
  document_version?: number;
  stats?: { update_status: string; vector_reuses: number; embedding_cache_hits: number; embedding_cache_misses: number; total_index_time_ms: number };
};

export type RagSearchResponse = {
  query: string;
  sources: RagSource[];
};

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
export const API_KEY = import.meta.env.VITE_API_KEY ?? "";

export async function openOriginal(citation: CitationRecord): Promise<void> {
  const preview = window.open("about:blank", "_blank");
  if (!preview) throw new Error("请允许浏览器打开原文窗口。");
  try {
    const evidence = citation.evidence;
    const response = await fetch(`${API_BASE_URL}/v1/rag/document-versions/${encodeURIComponent(evidence.document_id)}/${evidence.document_version}/file`, { headers: authHeaders() });
    if (!response.ok) throw new Error(await readErrorMessage(response));
    const url = URL.createObjectURL(await response.blob());
    preview.location.href = url + (evidence.page_number ? `#page=${evidence.page_number}` : "");
    window.setTimeout(() => URL.revokeObjectURL(url), 300000);
  } catch (error) {
    preview.close();
    throw error;
  }
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return API_KEY ? { ...extra, "x-api-key": API_KEY } : extra;
}

export type StreamEvent =
  | { type: "sources"; sources: RagSource[] }
  | { type: "delta"; text: string; node?: string | null }
  | (AdaptiveDetails & { type: "done"; thread_id?: string });

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text();
  if (!body) return `Request failed with ${response.status}`;
  try {
    const parsed = JSON.parse(body) as { detail?: string; title?: string };
    return parsed.detail ?? parsed.title ?? body;
  } catch {
    return body;
  }
}

export async function invokeAgent(request: AgentInvokeRequest): Promise<AgentInvokeResponse> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/v1/agent/invoke`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
    });
  } catch (error) {
    throw new Error(
      `Could not reach the API at ${API_BASE_URL}. Check that FastAPI is running and CORS allows this frontend origin.`,
      { cause: error },
    );
  }

  if (!response.ok) throw new Error(await readErrorMessage(response));

  return response.json() as Promise<AgentInvokeResponse>;
}

export async function* streamAgent(
  request: AgentInvokeRequest,
  options: { signal?: AbortSignal } = {},
): AsyncGenerator<StreamEvent, void, void> {
  const response = await fetch(`${API_BASE_URL}/v1/agent/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify(request),
    signal: options.signal,
  });

  if (!response.ok || !response.body) throw new Error(await readErrorMessage(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let separator = buffer.indexOf("\n\n");
    while (separator !== -1) {
      const block = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const event = parseSseBlock(block);
      if (event) yield event;
      separator = buffer.indexOf("\n\n");
    }
  }
}

function parseSseBlock(block: string): StreamEvent | null {
  let eventName = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  const payload = JSON.parse(data) as Record<string, unknown>;
  return { type: eventName, ...payload } as StreamEvent;
}

export async function getReadiness(): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/health/ready`);
  return response.ok;
}

export async function ingestText(input: {
  text: string;
  source: string;
  documentType?: "knowledge" | "voice_sample";
}): Promise<RagIngestResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/rag/documents`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      text: input.text,
      source: input.source,
      document_type: input.documentType ?? "knowledge",
      metadata: { uploaded_from: "frontend" },
    }),
  });

  if (!response.ok) throw new Error(await readErrorMessage(response));

  return response.json() as Promise<RagIngestResponse>;
}

export async function ingestFile(
  file: File,
  documentType: "knowledge" | "voice_sample" = "knowledge",
): Promise<RagIngestResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const path = documentType === "voice_sample" ? "/v1/rag/voice-files" : "/v1/rag/files";

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  if (!response.ok) throw new Error(await readErrorMessage(response));

  return response.json() as Promise<RagIngestResponse>;
}

export async function searchRag(query: string): Promise<RagSearchResponse> {
  const params = new URLSearchParams({ q: query });
  const response = await fetch(`${API_BASE_URL}/v1/rag/search?${params.toString()}`, {
    headers: authHeaders(),
  });

  if (!response.ok) throw new Error(await readErrorMessage(response));

  return response.json() as Promise<RagSearchResponse>;
}
