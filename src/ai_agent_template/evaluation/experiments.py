"""Resumable, rate-limited experiments with append-only predictions and provenance."""

import asyncio
import csv
import json
import random
import subprocess
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Any

from ai_agent_template.evaluation.config import EvaluationConfig
from ai_agent_template.evaluation.datasets import load_samples
from ai_agent_template.evaluation.execution import ProjectExecutor
from ai_agent_template.evaluation.scoring import GenerationScorer, retrieval_scores
from ai_agent_template.rag.evidence import PROMPT_VERSION, GroundedClaim
from ai_agent_template.rag.repository import fingerprint, timestamp
from ai_agent_template.rag.schemas import EvidenceRecord


def write_json(path: Any, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def append_json(path: Any, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Any) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.exists()
        else []
    )


def aggregate(rows: Any) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    metrics: dict[str, Any] = {
        "successful_samples": len(success),
        "failed_samples": len(rows) - len(success),
    }
    for metric in (
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "hit_rate_at_5",
        "citation_precision",
        "faithfulness",
    ):
        values = [row.get("scores", {}).get(metric) for row in success]
        values = [value for value in values if value is not None]
        metrics[metric] = mean(values) if values else None
        metrics[metric + "_evaluated"] = len(values)
    latencies = sorted(row["latency_ms"] for row in success)
    for percentile in (50, 95):
        metrics[f"latency_p{percentile}_ms"] = (
            latencies[
                min(len(latencies) - 1, max(0, int((len(latencies) - 1) * percentile / 100 + 0.5)))
            ]
            if latencies
            else None
        )
    generated = [row for row in success if row.get("answer") is not None]
    metrics["refusal_rate"] = (
        mean(row.get("scores", {}).get("refused", False) for row in generated)
        if generated
        else None
    )
    for answerable, label in ((True, "answerable"), (False, "unanswerable")):
        subset = [row for row in generated if row["answerable"] is answerable]
        metrics[label + "_refusal_rate"] = (
            mean(row["scores"]["refused"] for row in subset) if subset else None
        )
        metrics[label + "_samples"] = len(subset)
    metrics["mean_search_calls"] = (
        mean(len(row["retrieval_calls"]) for row in success) if success else None
    )
    metrics["decision_tokens"] = sum(row.get("decision_tokens", 0) for row in success)
    metrics["verification_tokens"] = sum(row.get("verification_tokens", 0) for row in success)
    metrics["generation_tokens"] = sum(row.get("generation_tokens", 0) for row in success)
    for field in (
        "citation_pairs_total",
        "citation_pairs_evaluated",
        "citation_pairs_supported",
        "citation_pairs_not_evaluable",
        "citation_judge_tokens",
    ):
        metrics[field] = sum(row.get("scores", {}).get(field, 0) for row in success)
    if metrics["citation_pairs_evaluated"]:
        metrics["citation_precision"] = (
            metrics["citation_pairs_supported"] / metrics["citation_pairs_evaluated"]
        )
    metrics["mean_verification_calls"] = (
        mean(row.get("verification_calls", 0) for row in success) if success else None
    )
    return metrics


async def run_experiments(
    config: EvaluationConfig,
    settings: Any,
    output: Path | None = None,
    resume: Path | None = None,
    executor_factory: Any = ProjectExecutor,
    scorer_factory: Any = GenerationScorer,
) -> Path:
    if config.phase != "retrieval" and config.faithfulness and not settings.openai_api_key:
        raise ValueError("Faithfulness needs a configured independent judge API key")
    random.seed(config.seed)
    directory = resume or output or Path("experiments/runs") / ("eval-" + uuid.uuid4().hex[:12])
    config_hash = fingerprint(config.model_dump())
    if resume:
        previous = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if fingerprint(previous) != config_hash:
            raise ValueError("Resume config differs from the persisted config")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", config.model_dump())
    samples, provenance = load_samples(
        config.dataset,
        config.subset,
        config.split,
        config.max_samples,
        config.input_path,
        config.revision,
        json.loads((directory / "dataset_snapshot.json").read_text(encoding="utf-8"))
        if resume
        else None,
    )
    write_json(directory / "dataset_snapshot.json", provenance)
    manifest = {
        "run_id": directory.name,
        "created_at": timestamp(),
        "config_sha256": config_hash,
        "dataset": {key: value for key, value in provenance.items() if key != "raw_rows"},
        "split": config.split,
        "sample_ids": [sample.sample_id for sample in samples],
        "judge_model": config.judge_model,
        "generator_model": settings.model_name,
        "semantic_verifier_model": settings.rag_verifier_model_name or settings.model_name,
        "judge_prompt_version": PROMPT_VERSION,
        "source_of_citation_scores": "independent automatic judge",
        "retrieval_definition": "95% character coverage of annotated units; exact origin spans; best alternative annotation; first-appearance ranking across calls",
        "dependencies": {},
        "application_settings": settings.model_dump(
            exclude={
                "openai_api_key",
                "groq_api_key",
                "qdrant_api_key",
                "auth_api_keys",
                "commerce_principals",
                "commerce_database_url",
            }
        ),
    }
    code_files = sorted(Path(__file__).parents[1].rglob("*.py"))
    manifest["source_sha256"] = fingerprint(
        [
            (str(path.relative_to(Path(__file__).parents[1])), path.read_text(encoding="utf-8"))
            for path in code_files
        ]
    )
    if not resume:
        write_json(
            directory / "application_source.json",
            {
                str(path.relative_to(Path(__file__).parents[1])): path.read_text(encoding="utf-8")
                for path in code_files
            },
        )
    if resume:
        original_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if original_manifest.get("application_settings") != manifest["application_settings"]:
            raise ValueError("Application settings changed; start a new run")
        if (
            original_manifest.get("source_sha256")
            and original_manifest["source_sha256"] != manifest["source_sha256"]
        ):
            raise ValueError("Application source changed; use a new run rather than mixing results")
        manifest["created_at"] = original_manifest["created_at"]
        manifest["resumed_at"] = timestamp()
    for package in ("ragas", "datasets", "langchain", "langgraph", "qdrant-client", "fastembed"):
        try:
            manifest["dependencies"][package] = version(package)
        except PackageNotFoundError:
            manifest["dependencies"][package] = "not_installed"
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        manifest["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = "unavailable"
    write_json(directory / "manifest.json", manifest)
    scorer = (
        scorer_factory(settings, config.judge_model, config.faithfulness)
        if config.phase != "retrieval"
        else None
    )
    from langchain_core.rate_limiters import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(
        requests_per_second=config.requests_per_second,
        max_bucket_size=1,
        check_every_n_seconds=0.05,
    )
    if scorer and hasattr(scorer, "set_rate_limiter"):
        scorer.set_rate_limiter(limiter)
    comparison = []
    for experiment in config.experiments:
        exp_dir = directory / experiment.name
        exp_dir.mkdir(exist_ok=True)
        predictions = exp_dir / "predictions.jsonl"
        existing = read_jsonl(predictions)
        completed = {row["sample_id"] for row in existing if row["status"] == "success"}
        executor = await asyncio.to_thread(
            executor_factory, settings, config, experiment, samples, exp_dir
        )
        if hasattr(executor, "set_rate_limiter"):
            executor.set_rate_limiter(limiter)
        semaphore, rate_lock = asyncio.Semaphore(config.concurrency), asyncio.Lock()
        next_start = 0.0

        async def process(sample: Any) -> None:
            nonlocal next_start
            if sample.sample_id in completed:
                return
            async with semaphore:
                for attempt in range(config.retries + 1):
                    async with rate_lock:
                        delay = max(0, next_start - time.monotonic())
                        if delay:
                            await asyncio.sleep(delay)
                        next_start = time.monotonic() + 1 / config.requests_per_second
                    try:
                        prediction = await executor.run_sample(sample)
                        retrieval = (
                            retrieval_scores(sample, prediction["retrieved"])
                            if config.context_mode == "retrieved"
                            else {"status": "not_evaluable", "reason": "provided_context_mode"}
                        )
                        scores = {
                            key: value
                            for key, value in retrieval.items()
                            if key not in {"status", "reason"}
                        }
                        if scorer:
                            # Score only final exported claims, not rejected drafts.
                            cited = {
                                citation["citation_id"]: EvidenceRecord.model_validate(
                                    citation["evidence"]
                                )
                                for citation in prediction["citations"]
                            }
                            claims = [
                                GroundedClaim.model_validate(claim)
                                for claim in prediction["claims"]
                                if all(label in cited for label in claim["evidence_ids"])
                            ]
                            refused = (
                                prediction["stop_reason"]
                                in {
                                    "verification_failed",
                                    "insufficient_evidence",
                                    "clarification_required",
                                }
                                or not prediction["citations"]
                            )
                            generation = await scorer.score(
                                sample.question,
                                prediction["answer"],
                                claims,
                                cited,
                                [source["content"] for source in prediction["retrieved"]],
                                refused,
                            )
                            scores.update(generation)
                        row = {
                            "status": "success",
                            "sample_id": sample.sample_id,
                            "question": sample.question,
                            "dataset": sample.dataset,
                            "subset": sample.subset,
                            "split": sample.split,
                            "data_version": sample.data_version,
                            "reference_answers": sample.reference_answers,
                            "gold_sets": sample.gold_sets,
                            "answerable": sample.answerable,
                            "retrieval_status": retrieval,
                            "scores": scores,
                            **prediction,
                        }
                        append_json(predictions, row)
                        return
                    except Exception as error:
                        append_json(
                            exp_dir / "errors.jsonl",
                            {
                                "sample_id": sample.sample_id,
                                "attempt": attempt,
                                "error_type": type(error).__name__,
                                "timestamp": timestamp(),
                            },
                        )
                        if attempt == config.retries:
                            append_json(
                                predictions,
                                {
                                    "status": "failed",
                                    "sample_id": sample.sample_id,
                                    "error_type": type(error).__name__,
                                },
                            )

        try:
            await asyncio.gather(*(process(sample) for sample in samples))
        finally:
            executor.close()
        latest = {row["sample_id"]: row for row in read_jsonl(predictions)}
        metrics = aggregate(list(latest.values()))
        write_json(exp_dir / "metrics.json", metrics)
        comparison.append({"experiment": experiment.name, **metrics})
        for path in (exp_dir / "errors.jsonl", predictions):
            path.touch(exist_ok=True)
    write_json(directory / "metrics.json", {row["experiment"]: row for row in comparison})
    if comparison:
        with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    manifest["completed_at"] = timestamp()
    write_json(directory / "manifest.json", manifest)
    print(
        json.dumps(
            {"run_directory": str(directory), "comparison": comparison},
            ensure_ascii=False,
            indent=2,
        )
    )
    return directory
