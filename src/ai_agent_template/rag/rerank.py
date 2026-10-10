"""Optional FastEmbed cross-encoder reranking for retrieved scientific evidence."""

from collections.abc import Sequence

from ai_agent_template.rag.schemas import RagSource


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, sources: Sequence[RagSource]) -> list[RagSource]:
        candidates = list(sources)
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [s.content for s in candidates]))
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [item.model_copy(update={"score": float(score)}) for item, score in ranked]
