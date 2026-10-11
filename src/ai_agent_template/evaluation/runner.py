"""Offline evaluation helpers. No benchmark results are stored in the repository."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ai_agent_template.evaluation.metrics import evaluate_retrieval


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_ragas(
    rows: list[dict[str, Any]], judge_model: str | None = None
) -> list[dict[str, Any]]:
    """Compatibility entry; uses the installed official RAGAS collections API."""
    if not judge_model:
        raise ValueError("Explicit --judge-model is required; no implicit evaluation model")
    from ai_agent_template.core.settings import Settings
    from ai_agent_template.evaluation.scoring import GenerationScorer

    scorer = GenerationScorer(Settings(), judge_model, True)
    metric = scorer.faithfulness
    assert metric is not None

    async def evaluate() -> Any:
        results = []
        for row in rows:
            score = await metric.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["retrieved_contexts"],
            )
            results.append({"faithfulness": float(score.value)})
        return results

    return asyncio.run(evaluate())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible project evaluations or score recorded retrieval outputs."
    )
    parser.add_argument(
        "results", nargs="?", help="Legacy JSONL with gold_sources and retrieved_sources"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=["qasper", "ragbench"])
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--input")
    parser.add_argument("--storage-root")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--requests-per-second", type=float)
    args = parser.parse_args()
    if args.config:
        from ai_agent_template.core.settings import Settings
        from ai_agent_template.evaluation.config import EvaluationConfig, load_config
        from ai_agent_template.evaluation.experiments import run_experiments

        config = load_config(args.config).model_dump()
        for argument, field in (
            ("dataset", "dataset"),
            ("split", "split"),
            ("input", "input_path"),
            ("storage_root", "storage_root"),
            ("max_samples", "max_samples"),
            ("concurrency", "concurrency"),
            ("requests_per_second", "requests_per_second"),
            ("judge_model", "judge_model"),
        ):
            value = getattr(args, argument)
            if value is not None:
                config[field] = value
        validated = EvaluationConfig.model_validate(config)
        asyncio.run(
            run_experiments(
                validated,
                Settings(),
                Path(args.output) if args.output else None,
                Path(args.resume) if args.resume else None,
            )
        )
        return
    if not args.results:
        parser.error("Provide --config or a recorded results JSONL")
    rows = load_jsonl(args.results)
    metrics: dict[str, Any] = dict(evaluate_retrieval(rows, k=args.k).as_dict())
    if args.ragas:
        metrics["ragas"] = evaluate_ragas(rows, args.judge_model)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
