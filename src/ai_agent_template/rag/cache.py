"""SQLite-backed parsing, chunking, dense and sparse caches with OS compute locks."""

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

from filelock import FileLock
from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.repository import digest, fingerprint

_OWNER: ContextVar[str | None] = ContextVar("rag_cache_owner", default=None)


class PersistentCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache (kind TEXT, key TEXT, data TEXT, PRIMARY KEY(kind,key))"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=120)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS owners (owner TEXT,kind TEXT,key TEXT,PRIMARY KEY(owner,kind,key))"
        )
        try:
            with db:
                yield db
        finally:
            db.close()

    def get_many(self, kind: str, keys: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        with self.connect() as db:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                rows = db.execute(
                    "SELECT key,data FROM cache WHERE kind=? AND key IN ("
                    + ",".join("?" * len(batch))
                    + ")",
                    [kind, *batch],
                ).fetchall()
                result.update({key: json.loads(data) for key, data in rows})
        self.claim(kind, list(result))
        return result

    def get(self, kind: str, key: str) -> Any | None:
        return self.get_many(kind, [key]).get(key)

    def put_many(self, kind: str, values: dict[str, Any]) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                [
                    (kind, key, json.dumps(value, ensure_ascii=False, allow_nan=False))
                    for key, value in values.items()
                ],
            )

        self.claim(kind, list(values))

    def compute_lock(self, kind: str) -> FileLock:
        return FileLock(
            str(self.path) + "." + digest(kind)[:16] + ".lock", timeout=180, thread_local=False
        )

    def claim(self, kind: str, keys: list[str]) -> None:
        if _OWNER.get() and keys:
            with self.connect() as db:
                db.executemany(
                    "INSERT OR IGNORE INTO owners VALUES (?,?,?)",
                    [(_OWNER.get(), kind, key) for key in keys],
                )

    @contextmanager
    def bind(self, owner: str) -> Iterator[None]:
        token = _OWNER.set(owner)
        try:
            yield
        finally:
            _OWNER.reset(token)

    def remove_owner(self, owner: str) -> None:
        with self.connect() as db:
            candidates = db.execute(
                "SELECT kind,key FROM owners WHERE owner=?", (owner,)
            ).fetchall()
            db.execute("DELETE FROM owners WHERE owner=?", (owner,))
            for kind, key in candidates:
                db.execute(
                    "DELETE FROM cache WHERE kind=? AND key=? AND NOT EXISTS (SELECT 1 FROM owners WHERE kind=? AND key=?)",
                    (kind, key, kind, key),
                )


def embedding_identity(settings: Settings) -> str:
    return fingerprint(
        {
            "model": settings.rag_embedding_model_name,
            "revision": settings.rag_embedding_model_revision,
            "dimensions": settings.rag_embedding_dimensions,
            "preprocessing": "lf-strip-v1",
            "encoding_parameters": settings.rag_embedding_parameters,
            "fastembed_version": version("fastembed"),
        }
    )


@lru_cache(maxsize=64)
def _artifact_sha(path: str, size: int, modified_ns: int) -> str:
    hashed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hashed.update(block)
    return hashed.hexdigest()


def model_artifacts(model: Any) -> list[tuple[str, str]]:
    directory = getattr(model, "_model_dir", None)
    if not directory:
        raise ValueError("Cannot establish the actual FastEmbed artifact identity")
    root = Path(directory)
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".onnx", ".json", ".txt", ".data"}:
            info = path.stat()
            artifacts.append(
                (
                    str(path.relative_to(root)),
                    _artifact_sha(str(path), info.st_size, info.st_mtime_ns),
                )
            )
    if not artifacts:
        raise ValueError("Embedding artifacts are unavailable")
    return artifacts


def resolved_embedding_identity(settings: Settings, encoder: Any) -> str:
    """Include actual downloaded model and tokenizer bytes in cache compatibility."""
    model = getattr(getattr(encoder, "model", None), "model", None)
    return fingerprint([embedding_identity(settings), model_artifacts(model)])


class CachedEmbeddings(Embeddings):
    def __init__(
        self, inner: Embeddings, cache: PersistentCache, model_id: str, dimensions: int
    ) -> None:
        self.inner, self.cache, self.model_id, self.dimensions = inner, cache, model_id, dimensions
        self.hits = self.misses = 0
        self.compute_time_ms = 0.0
        self.owner_usage: dict[str, tuple[int, int, float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized = [text.replace("\r\n", "\n").strip() for text in texts]
        keys = [fingerprint([self.model_id, digest(text)]) for text in normalized]
        with self.cache.compute_lock("dense-" + self.model_id):
            found = self.cache.get_many("dense", keys)
            hits = sum(key in found for key in keys)
            unique_missing = {
                key: text for key, text in zip(keys, normalized, strict=True) if key not in found
            }
            started = time.perf_counter()
            if unique_missing:
                vectors = self.inner.embed_documents(list(unique_missing.values()))
                if len(vectors) != len(unique_missing):
                    raise ValueError("Embedding batch has incorrect length")
                values = dict(zip(unique_missing, vectors, strict=True))
                self._validate(list(values.values()))
                self.cache.put_many("dense", values)
                found.update(values)
            elapsed = (time.perf_counter() - started) * 1000
            self.compute_time_ms += elapsed
            self.misses += len(unique_missing)
            self.hits += hits
            owner = _OWNER.get() or ""
            prior = self.owner_usage.get(owner, (0, 0, 0.0))
            self.owner_usage[owner] = (
                prior[0] + hits,
                prior[1] + len(unique_missing),
                prior[2] + elapsed,
            )
            result = [found[key] for key in keys]
            self._validate(result)
            return result

    def _validate(self, vectors: list[list[float]]) -> None:
        if any(
            len(vector) != self.dimensions or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError("Cached embedding dimension or values are incompatible")

    def usage_for(self, owner: str) -> tuple[int, int, float]:
        return self.owner_usage.get(owner, (0, 0, 0.0))

    def embed_query(self, text: str) -> list[float]:
        # Query/document encoders may differ; never borrow a document vector for a query.
        result = self.inner.embed_query(text)
        self._validate([result])
        return result


class CachedSparseEmbeddings(SparseEmbeddings):
    def __init__(self, inner: SparseEmbeddings, cache: PersistentCache, model_id: str) -> None:
        self.inner, self.cache, self.model_id = inner, cache, model_id
        self.hits = self.misses = 0
        self.owner_usage: dict[str, tuple[int, int, float]] = {}

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        keys = [fingerprint([self.model_id, text]) for text in texts]
        with self.cache.compute_lock("sparse-" + self.model_id):
            found = self.cache.get_many("sparse", keys)
            hits = sum(key in found for key in keys)
            missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in found}
            if missing:
                vectors = self.inner.embed_documents(list(missing.values()))
                if len(vectors) != len(missing):
                    raise ValueError("Sparse embedding batch has incorrect length")
                values = {
                    key: vector.model_dump() for key, vector in zip(missing, vectors, strict=True)
                }
                self.cache.put_many("sparse", values)
                found.update(values)
            self.hits += hits
            self.misses += len(missing)
            owner = _OWNER.get() or ""
            prior = self.owner_usage.get(owner, (0, 0, 0.0))
            self.owner_usage[owner] = (prior[0] + hits, prior[1] + len(missing), 0.0)
            return [SparseVector.model_validate(found[key]) for key in keys]

    def embed_query(self, text: str) -> SparseVector:
        return self.inner.embed_query(text)

    def usage_for(self, owner: str) -> tuple[int, int, float]:
        return self.owner_usage.get(owner, (0, 0, 0.0))
