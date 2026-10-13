"""Atomic, typed and idempotent authorized CSV/XLSX/JSON report imports."""

import csv
import json
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from pydantic import ValidationError

from ai_agent_template.commerce.operations_models import OperationsRecord
from ai_agent_template.commerce.operations_repository import RECORD_TYPES, OperationsRepository
from ai_agent_template.rag.repository import fingerprint

ALIASES = {
    "SKU": "sku",
    "ASIN": "asin",
    "Seller SKU": "sku",
    "Date": "report_date",
    "Report Date": "report_date",
    "Campaign ID": "campaign_id",
    "Ad Group ID": "ad_group_id",
    "Search Term": "search_term",
    "Customer Search Term": "search_term",
    "Impressions": "impressions",
    "Clicks": "clicks",
    "Spend": "spend",
    "Sales": "attributed_sales",
    "Attributed Sales": "attributed_sales",
    "Orders": "attributed_orders",
    "Attributed Orders": "attributed_orders",
    "Currency": "currency",
    "Attribution Window": "attribution_window",
    "退货日期": "return_date",
    "退货数量": "units",
    "退货原因": "reason",
    "销量": "units",
}


def parse_rows(content: bytes, filename: str) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        data = json.loads(content.decode("utf-8-sig"))
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("JSON must be an array of objects")
        return data
    if suffix == ".csv":
        return list(csv.DictReader(StringIO(content.decode("utf-8-sig"))))
    if suffix == ".xlsx":
        book = load_workbook(BytesIO(content), read_only=True, data_only=True)
        try:
            values = iter(book.active.values) if book.active else iter([])
            headers = next(values, None)
            if not headers or len(set(headers)) != len(headers):
                raise ValueError("XLSX needs unique column headers")
            return [
                dict(zip(map(str, headers), row, strict=True))
                for row in values
                if any(value is not None for value in row)
            ]
        finally:
            book.close()
    raise ValueError("Supported reports: CSV, XLSX, JSON")


def import_report(
    repo: OperationsRepository,
    kind: str,
    seller: str,
    market: str,
    content: bytes,
    filename: str,
    mapping: dict[str, str] | None = None,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    if kind not in RECORD_TYPES:
        raise ValueError("Unknown report kind")
    raw_rows = parse_rows(content, filename)
    if not raw_rows:
        raise ValueError("Empty report")
    if len(raw_rows) > 100000:
        raise ValueError("Import exceeds 100000 rows")
    cls = RECORD_TYPES[kind]
    items: list[OperationsRecord] = []
    seen: dict[str, dict[str, Any]] = {}
    duplicates = 0
    batch_id = fingerprint([seller, market, kind, content.hex(), mapping or {}])
    existing = repo.batch(seller, batch_id)
    if existing:
        return {**existing, "idempotent": True}
    for number, raw in enumerate(raw_rows, 2):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            target = (mapping or {}).get(key, ALIASES.get(key, key))
            if isinstance(value, datetime):
                value = value.date().isoformat()
            elif isinstance(value, date):
                value = value.isoformat()
            if value is None or value == "":
                continue
            if target in row:
                raise ValueError(f"Row {number}: multiple columns map to {target}")
            row[target] = value
        if row.get("seller_id", seller) != seller or row.get("marketplace_id", market) != market:
            raise PermissionError(f"Row {number}: seller/marketplace mismatch")
        for field in ("attributes", "required_fields", "evidence_ids"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])
        row.update(
            seller_id=seller,
            marketplace_id=market,
            source_id=filename,
            import_batch_id=batch_id,
            synthetic=synthetic or row.get("synthetic", False),
        )
        try:
            item = cls.model_validate(row)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(map(str, part['loc']))}: {part['msg']}"
                for part in error.errors(include_input=False)
            )
            raise ValueError(f"Row {number}: {details}") from error
        # Normalize dates, numbers and defaults BEFORE cross-format deduplication.
        identity_data = item.model_dump(
            mode="json", exclude={"id", "source_id", "import_batch_id", "imported_at", "synthetic"}
        )
        for field, value in item.model_dump().items():
            if isinstance(value, Decimal):
                identity_data[field] = format(value.normalize(), "f")
        supplied_id = str(row.get("id", ""))
        row_id = (
            fingerprint(
                [
                    kind,
                    seller,
                    market,
                    row.get("sku"),
                    row.get("asin"),
                    row.get("product_id"),
                    supplied_id,
                ]
            )
            if supplied_id
            else fingerprint([kind, identity_data])
        )
        if row_id in seen:
            if seen[row_id] != identity_data:
                raise ValueError(f"Row {number}: conflicting duplicate record id")
            duplicates += 1
            continue
        seen[row_id] = identity_data
        item.id = row_id
        items.append(item)
    batch = dict(
        id=batch_id,
        seller_id=seller,
        marketplace_id=market,
        kind=kind,
        filename=filename,
        rows=len(items),
        duplicate_rows=duplicates,
        imported_at=datetime.now().isoformat(),
        idempotent=False,
    )
    repo.import_rows(kind, items, batch)
    return batch
