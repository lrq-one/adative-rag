"""Durable evidence snapshots; SQLite commits publish one complete document version."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from ai_agent_template.rag.schemas import DocumentManifest, EvidenceRecord, RagIngestResponse


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def fingerprint(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class EvidenceRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.path.parent / "locks"
        self.lock_dir.mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS versions (
                    document_id TEXT NOT NULL, version INTEGER NOT NULL,
                    namespace TEXT NOT NULL, source TEXT NOT NULL, kind TEXT NOT NULL,
                    status TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(document_id, version));
                CREATE TABLE IF NOT EXISTS active (
                    document_id TEXT PRIMARY KEY, version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    version INTEGER NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS evidence_document ON evidence(document_id, version);
                CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT, created_at TEXT, data TEXT);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=60)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def document_lock(self, document_id: str) -> Iterator[None]:
        # OS locks are released on process death; SQLite uniqueness is a second barrier.
        with FileLock(str(self.lock_dir / (digest(document_id) + ".lock")), timeout=120):
            yield

    def active(self, document_id: str) -> DocumentManifest | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT v.data FROM versions v JOIN active a ON "
                "v.document_id=a.document_id AND v.version=a.version WHERE v.document_id=?",
                (document_id,),
            ).fetchone()
        return DocumentManifest.model_validate_json(row[0]) if row else None

    def versions(self, document_id: str) -> list[DocumentManifest]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT data FROM versions WHERE document_id=? ORDER BY version", (document_id,)
            ).fetchall()
        return [DocumentManifest.model_validate_json(row[0]) for row in rows]

    def snapshot(self, namespace: str = "default") -> list[DocumentManifest]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT v.data FROM versions v JOIN active a ON "
                "v.document_id=a.document_id AND v.version=a.version WHERE v.namespace=?",
                (namespace,),
            ).fetchall()
        return [DocumentManifest.model_validate_json(row[0]) for row in rows]

    def managed_sources(self) -> list[str]:
        with self.connect() as db:
            rows = db.execute("SELECT DISTINCT source FROM versions").fetchall()
        return [row[0] for row in rows]

    def save(self, manifest: DocumentManifest, records: list[EvidenceRecord] | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?,?,?)",
                (
                    manifest.document_id,
                    manifest.document_version,
                    manifest.namespace,
                    manifest.source_filename,
                    manifest.document_type,
                    manifest.index_status,
                    manifest.model_dump_json(),
                ),
            )
            if records:
                db.executemany(
                    "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?)",
                    [
                        (r.evidence_id, r.document_id, r.document_version, r.model_dump_json())
                        for r in records
                    ],
                )

    def publish(self, manifest: DocumentManifest) -> None:
        # No Qdrant payload flag is used as an atomic switch. Readers take this snapshot
        # and apply the exact (document_id, version) pairs before top-k retrieval.
        with self.connect() as db:
            old = db.execute(
                "SELECT version FROM active WHERE document_id=?", (manifest.document_id,)
            ).fetchone()
            if old:
                row = db.execute(
                    "SELECT data FROM versions WHERE document_id=? AND version=?",
                    (manifest.document_id, old[0]),
                ).fetchone()
                previous = DocumentManifest.model_validate_json(row[0])
                previous.index_status = "historical"
                db.execute(
                    "UPDATE versions SET status=?, data=? WHERE document_id=? AND version=?",
                    (
                        "historical",
                        previous.model_dump_json(),
                        previous.document_id,
                        previous.document_version,
                    ),
                )
            manifest.index_status = "active"
            manifest.activated_at = timestamp()
            manifest.stats.update_status = "active"
            db.execute(
                "UPDATE versions SET status=?, data=? WHERE document_id=? AND version=?",
                (
                    "active",
                    manifest.model_dump_json(),
                    manifest.document_id,
                    manifest.document_version,
                ),
            )
            db.execute(
                "INSERT OR REPLACE INTO active VALUES (?,?)",
                (manifest.document_id, manifest.document_version),
            )

    def resolve(self, evidence_id: str) -> tuple[EvidenceRecord, str] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT e.data, v.status FROM evidence e JOIN versions v ON e.document_id=v.document_id AND e.version=v.version WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
        if not row or row[1] in {"deleted", "building", "failed"}:
            return None
        return EvidenceRecord.model_validate_json(row[0]), row[1]

    def retire(self, document_id: str, *, lock_held: bool = False) -> int:
        with nullcontext() if lock_held else self.document_lock(document_id), self.connect() as db:
            count = db.execute(
                "SELECT COUNT(*) FROM evidence WHERE document_id=?", (document_id,)
            ).fetchone()[0]
            rows = db.execute(
                "SELECT data FROM versions WHERE document_id=?", (document_id,)
            ).fetchall()
            for row in rows:
                manifest = DocumentManifest.model_validate_json(row[0])
                manifest.index_status = "deleted"
                db.execute(
                    "UPDATE versions SET status='deleted',data=? WHERE document_id=? AND version=?",
                    (manifest.model_dump_json(), document_id, manifest.document_version),
                )
            db.execute("DELETE FROM active WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM evidence WHERE document_id=?", (document_id,))
        return int(count)

    def record_job(self, result: RagIngestResponse) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs(document_id,created_at,data) VALUES (?,?,?)",
                (result.document_id, timestamp(), result.model_dump_json()),
            )

    def jobs(self, limit: int = 50, namespace: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,created_at,data FROM jobs WHERE (? IS NULL OR document_id IN (SELECT document_id FROM versions WHERE namespace=?)) ORDER BY id DESC LIMIT ?",
                (namespace, namespace, limit),
            ).fetchall()
        return [{"job_id": row[0], "created_at": row[1], **json.loads(row[2])} for row in rows]
