"""Real parser/chunker/encoder/Qdrant performance; generated blocks, no preset targets."""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import (
    CachedEmbeddings,
    CachedSparseEmbeddings,
    PersistentCache,
    embedding_identity,
)
from ai_agent_template.rag.loaders import load_upload_documents
from ai_agent_template.rag.schemas import IndexStats
from ai_agent_template.rag.service import RagService


async def run(
    output: Path,
    repeats: int = 3,
    blocks: int = 20,
    threads: int = 1,
    storage_root: Path | None = None,
    alternate_model: str | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(rag_embedding_parameters={"threads": threads})
    print("Loading cached encoders", flush=True)
    # Warm the exact same actual encoders for both arms; model loading is excluded.
    dense_model = FastEmbedEmbeddings(model_name=settings.rag_embedding_model_name, threads=threads)
    sparse_model = FastEmbedSparse(model_name=settings.rag_sparse_model_name, threads=threads)
    dense_model.embed_documents(["warm-up"])
    sparse_model.embed_documents(["warm-up"])
    print("Encoders ready", flush=True)
    original = [
        f"# Research block {i}\n\nModel{i} was evaluated on Dataset{i} with a fixed train/test split. The MAE was {i + 1}.25 seconds. Units and experimental conditions must match before comparisons."
        for i in range(blocks)
    ]
    rows = []

    alternate = (
        FastEmbedEmbeddings(model_name=alternate_model, threads=threads)
        if alternate_model
        else None
    )
    alternate_dimensions = len(alternate.embed_query("warm-up")) if alternate else None

    def service(directory: Any, incremental: Any, encoder: Any = None) -> Any:
        encoder = encoder or dense_model
        config = settings.model_copy(
            update={
                "rag_data_path": str(directory),
                "rag_chunk_size": 1000,
                "rag_chunk_overlap": 100,
                "rag_embedding_model_name": alternate_model
                if encoder is alternate
                else settings.rag_embedding_model_name,
                "rag_embedding_dimensions": alternate_dimensions
                if encoder is alternate
                else settings.rag_embedding_dimensions,
            }
        )
        cache = PersistentCache(directory / "cache.sqlite3")
        dense = (
            CachedEmbeddings(
                encoder, cache, embedding_identity(config), config.rag_embedding_dimensions
            )
            if incremental
            else encoder
        )
        sparse = (
            CachedSparseEmbeddings(sparse_model, cache, config.rag_sparse_model_name)
            if incremental
            else sparse_model
        )
        client = QdrantClient(":memory:")
        client.create_collection(
            "bench",
            vectors_config={
                "dense": models.VectorParams(
                    size=config.rag_embedding_dimensions, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        store = QdrantVectorStore(
            client,
            "bench",
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        return RagService(config, store), client

    async def ingest(rag: Any, text: Any, metadata_only: Any = False) -> Any:
        stats = IndexStats()
        started = time.perf_counter()
        docs = await load_upload_documents(
            UploadFile(file=BytesIO(text.encode()), filename="benchmark.md"),
            settings,
            rag.indexer.cache,
            stats,
        )
        if metadata_only:
            for i, doc in enumerate(docs):
                doc.metadata["page"] = i + 10
        stats.parsing_time_ms = (time.perf_counter() - started) * 1000
        result = await asyncio.to_thread(
            rag.ingest_sections, sections=docs, source="benchmark.md", preprocessing_stats=stats
        )
        return result, (time.perf_counter() - started) * 1000

    for scenario, fraction in [
        ("unchanged", 0),
        ("10_percent_blocks", 0.1),
        ("30_percent_blocks", 0.3),
        ("80_percent_blocks", 0.8),
        ("metadata_only", 0),
        ("interrupted_recovery", 0.1),
        *([("embedding_model_change", 0)] if alternate else []),
    ]:
        print("Scenario: " + scenario, flush=True)
        modified = list(original)
        changed = round(blocks * fraction)
        for i in range(changed):
            modified[i] = (
                modified[i]
                .replace("fixed train/test split", "a revised independent test split")
                .replace("seconds.", "seconds, with corrected units.")
            )
        target = "\n\n".join(modified)
        for repeat in range(repeats):
            base = (storage_root / output.name if storage_root else output) / scenario / str(repeat)
            full, full_client = service(
                base / "full", False, alternate if scenario == "embedding_model_change" else None
            )
            inc, inc_client = service(base / "incremental", True)
            await ingest(inc, "\n\n".join(original))
            previous_client = None
            interrupted_ms = 0.0
            interrupted_vector_inputs = 0
            if scenario == "embedding_model_change":
                previous_client = inc_client
                inc, inc_client = service(base / "incremental", True, alternate)
            if scenario == "interrupted_recovery":
                upsert = inc.indexer.store.add_documents

                def interrupted(*args: Any, **kwargs: Any) -> None:
                    upsert(*args, **kwargs)
                    raise RuntimeError("Injected interruption after vector staging")

                inc.indexer.store.add_documents = interrupted
                failure_start = time.perf_counter()
                try:
                    await ingest(inc, target)
                    raise AssertionError("Interruption did not execute")
                except RuntimeError:
                    interrupted_ms = (time.perf_counter() - failure_start) * 1000
                    interrupted_vector_inputs = inc.repository.jobs()[0]["stats"][
                        "embedding_cache_misses"
                    ]
                    assert all(
                        source.evidence.document_version == 1 for source in inc.search("Research")
                    )
                finally:
                    inc.indexer.store.add_documents = upsert
            # Both arms use the same target, encoders, parser, chunker and fresh in-memory server.
            full_result, full_ms = await ingest(full, target, scenario == "metadata_only")
            inc_result, inc_ms = await ingest(inc, target, scenario == "metadata_only")
            rows.append(
                {
                    "scenario": scenario,
                    "repeat": repeat,
                    "blocks": blocks,
                    "changed_blocks": changed,
                    "change_ratio_basis": "predefined research content blocks",
                    "model_state": "warm in both arms",
                    "full_cache_state": "empty",
                    "incremental_cache_state": "original document indexed",
                    "full_ms": full_ms,
                    "incremental_ms": inc_ms,
                    "time_reduction": (full_ms - inc_ms) / full_ms if full_ms else None,
                    "full_vector_inputs": full_result.chunks,
                    "incremental_stats": inc_result.stats.model_dump(),
                    "target_embedding_model": full._settings.rag_embedding_model_name,
                    "interrupted_attempt_ms": interrupted_ms,
                    "interrupted_attempt_vector_inputs": interrupted_vector_inputs,
                    "incremental_ms_including_interruption": inc_ms + interrupted_ms,
                }
            )
            full_client.close()
            inc_client.close()
            if previous_client:
                previous_client.close()
    summary = {
        scenario: {
            "full_mean_ms": statistics.mean(
                row["full_ms"] for row in rows if row["scenario"] == scenario
            ),
            "incremental_mean_ms": statistics.mean(
                row["incremental_ms"] for row in rows if row["scenario"] == scenario
            ),
            "time_reduction_mean": statistics.mean(
                row["time_reduction"] for row in rows if row["scenario"] == scenario
            ),
            "time_reduction_stdev": statistics.stdev(
                [row["time_reduction"] for row in rows if row["scenario"] == scenario]
            )
            if repeats > 1
            else None,
        }
        for scenario in dict.fromkeys(row["scenario"] for row in rows)
    }
    artifact = {
        "threads": threads,
        "storage_root": str((storage_root / output.name if storage_root else output).resolve()),
        "run_id": output.name,
        "embedding_model": settings.rag_embedding_model_name,
        "alternate_embedding_model": alternate_model,
        "repeats": repeats,
        "results": rows,
        "summary": summary,
        "limitations": [
            "Generated text documents; not an OCR or production network benchmark.",
            "Interruption scenario times recovery separately and also records the failed attempt; model loading and original-version seeding are excluded equally.",
            "Model-change timing is not_run unless --alternate-model is provided; restart recovery also has offline integration tests.",
        ],
    }
    (output / "performance.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--alternate-model", default=None)
    args = parser.parse_args()
    if args.repeats < 1 or args.blocks < 10 or args.threads < 1:
        parser.error("repeats must be positive; blocks must be at least 10")
    asyncio.run(
        run(
            Path(args.output or ("experiments/runs/index-" + uuid.uuid4().hex[:12])),
            args.repeats,
            args.blocks,
            args.threads,
            Path(args.storage_root) if args.storage_root else None,
            args.alternate_model,
        )
    )


if __name__ == "__main__":
    main()
