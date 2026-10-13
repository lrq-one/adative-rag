"""Additive SQL migration v2, with scope enforced before fetching report rows."""

from __future__ import annotations

import builtins
from typing import Any, TypeVar

from sqlalchemy import JSON, Column, Date, Integer, MetaData, String, Table, select, text
from sqlalchemy.engine import Connection

from ai_agent_template.commerce.operations_models import (
    AdsCampaign,
    AdsPerformance,
    AdsSearchTerm,
    ListingIssue,
    ListingSnapshot,
    OperationsRecord,
    ProductRevision,
    QualityRecord,
    ReturnRecord,
    RuleRecord,
    SalesRecord,
    SupplierDocument,
)
from ai_agent_template.commerce.repository import CommerceRepository

T = TypeVar("T", bound=OperationsRecord)
RECORD_TYPES: dict[str, type[OperationsRecord]] = {
    "supplier": SupplierDocument,
    "listing": ListingSnapshot,
    "listing_issue": ListingIssue,
    "revision": ProductRevision,
    "returns": ReturnRecord,
    "sales": SalesRecord,
    "quality": QualityRecord,
    "campaign": AdsCampaign,
    "ads": AdsPerformance,
    "search_terms": AdsSearchTerm,
    "rules": RuleRecord,
}
metadata = MetaData()
rows = Table(
    "operations_records",
    metadata,
    Column("seller_id", String(128), primary_key=True),
    Column("marketplace_id", String(128), primary_key=True),
    Column("kind", String(32), primary_key=True),
    Column("id", String(128), primary_key=True),
    Column("sku", String(128), index=True, nullable=False),
    Column("effective_at", Date, index=True, nullable=False),
    Column("payload", JSON, nullable=False),
)
batches = Table(
    "operations_import_batches",
    metadata,
    Column("seller_id", String(128), primary_key=True),
    Column("id", String(128), primary_key=True),
    Column("payload", JSON, nullable=False),
)
import_rows = Table(
    "operations_import_rows",
    metadata,
    Column("seller_id", String(128), primary_key=True),
    Column("marketplace_id", String(128), primary_key=True),
    Column("batch_id", String(128), primary_key=True),
    Column("kind", String(32), primary_key=True),
    Column("id", String(128), primary_key=True),
    Column("payload", JSON, nullable=False),
)
reports = Table(
    "operations_investigations",
    metadata,
    Column("seller_id", String(128), primary_key=True),
    Column("id", String(128), primary_key=True),
    Column("payload", JSON, nullable=False),
)
migrations = Table("operations_migrations", metadata, Column("version", Integer, primary_key=True))


class OperationsRepository:
    def __init__(self, repository: CommerceRepository) -> None:
        self.engine = repository.engine
        # Reuse the same cross-worker migration lock as schema v1.
        from contextlib import nullcontext

        from filelock import FileLock

        database = self.engine.url.database
        lock = (
            FileLock(database + ".schema.lock")
            if self.engine.dialect.name == "sqlite" and database and database != ":memory:"
            else nullcontext()
        )
        with lock, self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(14796821)"))
            metadata.create_all(conn)
            if conn.execute(select(migrations.c.version)).scalar_one_or_none() is None:
                conn.execute(migrations.insert().values(version=2))

    def _upsert(self, conn: Connection, table: Table, values: dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        statement = insert(table).values(**values)
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=list(table.primary_key.columns), set_=values
            )
        )

    def import_rows(self, kind: str, items: list[OperationsRecord], batch: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            for item in items:
                self._upsert(
                    conn,
                    import_rows,
                    dict(
                        seller_id=item.seller_id,
                        marketplace_id=item.marketplace_id,
                        batch_id=batch["id"],
                        kind=kind,
                        id=item.id,
                        payload=item.model_dump(mode="json"),
                    ),
                )
                self._upsert(
                    conn,
                    rows,
                    dict(
                        seller_id=item.seller_id,
                        marketplace_id=item.marketplace_id,
                        kind=kind,
                        id=item.id,
                        sku=item.sku,
                        effective_at=item.effective_at,
                        payload=item.model_dump(mode="json"),
                    ),
                )
            self._upsert(
                conn, batches, dict(seller_id=batch["seller_id"], id=batch["id"], payload=batch)
            )

    def list(self, cls: type[T], seller: str, market: str, sku: str | None = None) -> list[T]:
        kind = next(key for key, value in RECORD_TYPES.items() if value is cls)
        statement = select(rows.c.payload).where(
            rows.c.seller_id == seller, rows.c.marketplace_id == market, rows.c.kind == kind
        )
        if sku is not None:
            statement = statement.where(rows.c.sku == sku)
        with self.engine.connect() as conn:
            return [
                cls.model_validate(value)
                for value in conn.execute(
                    statement.order_by(rows.c.effective_at, rows.c.id)
                ).scalars()
            ]

    def batch(self, seller: str, batch_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            return conn.execute(
                select(batches.c.payload).where(
                    batches.c.seller_id == seller, batches.c.id == batch_id
                )
            ).scalar_one_or_none()

    def save_report(self, seller: str, report_id: str, payload: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            self._upsert(conn, reports, dict(seller_id=seller, id=report_id, payload=payload))

    def report(self, seller: str, report_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            return conn.execute(
                select(reports.c.payload).where(
                    reports.c.seller_id == seller, reports.c.id == report_id
                )
            ).scalar_one_or_none()

    def source_rows(
        self,
        seller: str,
        market: str,
        ids: builtins.list[str],
        batch_ids: builtins.list[str] | None = None,
    ) -> builtins.list[dict[str, Any]]:
        if len(ids) > 500:
            raise ValueError("At most 500 evidence rows per request")
        with self.engine.connect() as conn:
            if batch_ids:
                archived = list(
                    conn.execute(
                        select(import_rows.c.payload).where(
                            import_rows.c.seller_id == seller,
                            import_rows.c.marketplace_id == market,
                            import_rows.c.id.in_(ids),
                            import_rows.c.batch_id.in_(batch_ids),
                        )
                    ).scalars()
                )
                found = {(item["id"], item["import_batch_id"]) for item in archived}
                # Compatibility with local v2 rows imported before archival was introduced.
                current = conn.execute(
                    select(rows.c.payload).where(
                        rows.c.seller_id == seller,
                        rows.c.marketplace_id == market,
                        rows.c.id.in_(ids),
                    )
                ).scalars()
                return archived + [
                    item
                    for item in current
                    if item.get("import_batch_id") in batch_ids
                    and (item["id"], item["import_batch_id"]) not in found
                ]
            return list(
                conn.execute(
                    select(rows.c.payload).where(
                        rows.c.seller_id == seller,
                        rows.c.marketplace_id == market,
                        rows.c.id.in_(ids),
                    )
                ).scalars()
            )
