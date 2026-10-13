"""Deterministic spreadsheet fields stay in SQL; only descriptive columns become documents."""

import csv
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from ai_agent_template.commerce.schemas import CatalogImport, Product, ProductVariant

ALIASES = {
    "SKU": "sku",
    "商品编号": "sku",
    "商品ID": "product_id",
    "名称": "name_zh",
    "容量": "capacity",
    "颜色": "color",
    "价格": "price",
    "库存": "stock",
    "成分": "ingredients",
    "使用方法": "usage",
    "描述": "description",
}
TEXT_FIELDS = ("description", "ingredients", "usage", "faq")


def read_catalog(
    content: bytes, filename: str, store: str
) -> tuple[CatalogImport, list[dict[str, str]]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        rows: list[dict[str, Any]] = list(csv.DictReader(StringIO(content.decode("utf-8-sig"))))
    elif suffix == ".xlsx":
        book = load_workbook(BytesIO(content), read_only=True, data_only=True)
        try:
            values = iter(book.active.values) if book.active else iter(())
            headers = next(values, ())
            rows = [
                dict(zip((str(value or "").strip() for value in headers), row, strict=True))
                for row in values
            ]
        finally:
            book.close()
    else:
        raise ValueError("Catalog import supports .csv and .xlsx")
    if not rows or len(rows) > 10000:
        raise ValueError("Catalog must contain 1–10000 rows")
    products: dict[str, Product] = {}
    variants = []
    documents = []
    seen = set()
    for raw in rows:
        row = {
            ALIASES.get(key, key): str(value).strip()
            for key, value in raw.items()
            if value is not None and str(value).strip()
        }
        sku, product_id = row.get("sku", ""), row.get("product_id", "")
        names = {
            language: row["name_" + language]
            for language in ("zh", "en", "es")
            if "name_" + language in row
        }
        if not sku or not product_id or not names or sku in seen:
            raise ValueError("Every row needs unique sku, product_id and name_en/name_zh/name_es")
        seen.add(sku)
        product = Product(
            id=product_id,
            store_id=store,
            brand=row.get("brand", ""),
            category=row.get("category", ""),
            names=names,
            synthetic=row.get("synthetic", "").lower() == "true",
        )
        if product_id in products and products[product_id].names != names:
            raise ValueError("Rows sharing product_id must agree on product names")
        products[product_id] = product
        variants.append(
            ProductVariant(
                id=sku,
                store_id=store,
                product_id=product_id,
                sku=sku,
                attributes={
                    key.removeprefix("attr_"): value
                    for key, value in row.items()
                    if key in {"capacity", "color", "material"} or key.startswith("attr_")
                },
                price=row.get("price"),
                stock=row.get("stock"),
                currency=row.get("currency", "USD"),
                barcode=row.get("barcode"),
                markets=row.get("markets", "US").split("|"),
                synthetic=product.synthetic,
            )
        )
        descriptive = "\n\n".join(f"{key}: {row[key]}" for key in TEXT_FIELDS if key in row)
        if descriptive:
            documents.append(
                {
                    "sku": sku,
                    "text": descriptive,
                    "market": row.get("market", "US"),
                    "language": row.get("language", "en"),
                }
            )
    return CatalogImport(products=list(products.values()), variants=variants), documents
