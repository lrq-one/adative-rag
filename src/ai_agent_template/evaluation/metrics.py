"""Reproducible source-level retrieval metrics for JSONL benchmark inputs."""

from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    evaluated: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mrr_at_k": self.mrr_at_k,
            "evaluated": self.evaluated,
        }


def evaluate_retrieval(rows: list[dict[str, Any]], k: int = 5) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError("k must be positive")
    recalls, hits, reciprocals = [], [], []
    for row in rows:
        gold = {str(s) for s in row["gold_sources"]}
        if not gold:  # Unanswerable examples are measured separately as abstention.
            continue
        retrieved = [str(s) for s in row["retrieved_sources"][:k]]
        relevant = gold.intersection(retrieved)
        recalls.append(len(relevant) / len(gold))
        hits.append(float(bool(relevant)))
        rank = next((i for i, s in enumerate(retrieved, start=1) if s in gold), None)
        reciprocals.append(1.0 / rank if rank else 0.0)
    return RetrievalMetrics(
        recall_at_k=mean(recalls) if recalls else 0.0,
        hit_rate_at_k=mean(hits) if hits else 0.0,
        mrr_at_k=mean(reciprocals) if reciprocals else 0.0,
        evaluated=len(recalls),
    )
