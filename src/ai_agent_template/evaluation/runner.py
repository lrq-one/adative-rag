"""Offline evaluation helpers. No benchmark results are stored in the repository."""

import argparse
import json
from pathlib import Path
from typing import Any

from ai_agent_template.evaluation.metrics import evaluate_retrieval


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_ragas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional generation metrics; require ragas and a configured evaluator LLM."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness
    except ImportError as error:
        raise RuntimeError("Install the optional ragas package to evaluate generation.") from error

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row.get("reference_answer", ""),
        )
        for row in rows
    ]
    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=[Faithfulness()])
    return result.to_pandas().to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recorded scientific RAG outputs.")
    parser.add_argument("results", help="JSONL with gold_sources and retrieved_sources")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ragas", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.results)
    metrics = evaluate_retrieval(rows, k=args.k).as_dict()
    if args.ragas:
        metrics["ragas"] = evaluate_ragas(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
