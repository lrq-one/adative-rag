"""The same SQLAlchemy schema for PostgreSQL production and SQLite demo/tests.

Schema v1 is additive: separate tables, composite store/id keys and JSON entity payloads.
Catalog SKU and order ownership are indexed relational columns. No business row enters Qdrant.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypeVar

from filelock import FileLock
from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine, delete, select, text
from sqlalchemy.engine import Engine

from ai_agent_template.commerce.schemas import (
    Entity,
    ListingDraft,
    Order,
    Product,
    ProductDocument,
    ProductVariant,
    ReturnCase,
    Shipment,
    StorePolicy,
)

T = TypeVar("T", bound=Entity)
metadata = MetaData()
KINDS: dict[type[Entity], str] = {
    Product: "products",
    ProductVariant: "variants",
    ProductDocument: "documents",
    ListingDraft: "listings",
    Order: "orders",
    Shipment: "shipments",
    ReturnCase: "returns",
    StorePolicy: "policies",
}
TABLES = {
    cls: Table(
        name,
        metadata,
        Column("store_id", String(128), primary_key=True),
        Column("id", String(128), primary_key=True),
        Column("lookup", String(128), index=True),
        Column("owner", String(128), index=True),
        Column("payload", JSON, nullable=False),
    )
    for cls, name in KINDS.items()
}
sessions = Table(
    "commerce_sessions",
    metadata,
    Column("store_id", String(128), primary_key=True),
    Column("id", String(256), primary_key=True),
    Column("payload", JSON, nullable=False),
)
schema = Table("commerce_schema", metadata, Column("version", String(20), primary_key=True))


class CommerceRepository:
    def __init__(self, url: str, *, production: bool = False) -> None:
        if production and not url.startswith("postgresql+psycopg://"):
            raise ValueError("Production commerce requires PostgreSQL (postgresql+psycopg://)")
        if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"pool_pre_ping": True, "hide_parameters": True}
        if url.startswith("sqlite:"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if url == "sqlite:///:memory:":
                from sqlalchemy.pool import StaticPool

                kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(url, **kwargs)
        database = self.engine.url.database
        lock = (
            FileLock(database + ".schema.lock")
            if self.engine.dialect.name == "sqlite" and database and database != ":memory:"
            else nullcontext()
        )
        with lock, self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(text("SELECT pg_advisory_xact_lock(14796821)"))
            metadata.create_all(connection)
            versions = connection.execute(select(schema.c.version)).scalars().all()
            if versions and versions != ["1"]:
                raise ValueError("Unsupported commerce schema version; migrate before startup")
            if not versions:
                connection.execute(schema.insert().values(version="1"))

    def put(self, item: Entity) -> None:
        self.put_many([item])

    def put_many(self, items: list[Entity]) -> None:
        # Portable dialect upsert. Composite key guarantees tenant-local identity.
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        with self.engine.begin() as connection:
            for item in items:
                table = TABLES[type(item)]
                values = dict(
                    store_id=item.store_id,
                    id=item.id,
                    lookup=getattr(item, "sku", getattr(item, "order_id", None)),
                    owner=getattr(item, "customer_id", getattr(item, "product_id", None)),
                    payload=item.model_dump(mode="json"),
                )
                insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
                statement = insert(table).values(**values)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[table.c.store_id, table.c.id], set_=values
                    )
                )

    def get(self, cls: type[T], store: str, item_id: str) -> T | None:
        table = TABLES[cls]
        with self.engine.connect() as connection:
            value = connection.execute(
                select(table.c.payload).where(table.c.store_id == store, table.c.id == item_id)
            ).scalar_one_or_none()
        return cls.model_validate(value) if value is not None else None

    def list(self, cls: type[T], store: str) -> list[T]:
        table = TABLES[cls]
        with self.engine.connect() as connection:
            values = (
                connection.execute(
                    select(table.c.payload).where(table.c.store_id == store).order_by(table.c.id)
                )
                .scalars()
                .all()
            )
        return [cls.model_validate(value) for value in values]

    def session(self, store: str, key: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(sessions.c.payload).where(sessions.c.store_id == store, sessions.c.id == key)
            ).scalar_one_or_none()
        return dict(value or {})

    def save_session(self, store: str, key: str, value: dict[str, Any]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                delete(sessions).where(sessions.c.store_id == store, sessions.c.id == key)
            )
            connection.execute(sessions.insert().values(store_id=store, id=key, payload=value))

    def health(self) -> bool:
        with self.engine.connect() as connection:
            return bool(connection.execute(select(schema.c.version)).scalar_one() == "1")


def migrate(url: str) -> None:
    CommerceRepository(url, production=url.startswith("postgresql")).engine.dispose()


if __name__ == "__main__":
    from ai_agent_template.core.settings import Settings

    migrate(Settings().commerce_database_url)
