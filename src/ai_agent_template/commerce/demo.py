"""Reproducible synthetic catalog, order and policy fixtures; no private customers."""

from datetime import datetime, timedelta, timezone
from typing import Any

from ai_agent_template.commerce.repository import CommerceRepository
from ai_agent_template.commerce.schemas import (
    ListingDraft,
    Order,
    OrderItem,
    Product,
    ProductVariant,
    ReturnCase,
    Shipment,
    StorePolicy,
)

NAMES = [
    ("SERUM", "补水精华", "Hydrating Serum", "Sérum hidratante", "skincare"),
    ("CREAM", "保湿面霜", "Moisturizing Cream", "Crema hidratante", "skincare"),
    ("WASH", "洁面液", "Face Wash", "Limpiador facial", "skincare"),
    ("TONER", "爽肤水", "Facial Toner", "Tónico facial", "skincare"),
    ("MASK", "面膜", "Face Mask", "Mascarilla facial", "skincare"),
    ("BRUSH", "化妆刷", "Makeup Brush", "Brocha de maquillaje", "beauty_tools"),
    ("SPONGE", "美妆海绵", "Makeup Sponge", "Esponja de maquillaje", "beauty_tools"),
    ("COMB", "梳子", "Hair Comb", "Peine", "beauty_tools"),
    ("BAG", "收纳包", "Travel Pouch", "Bolsa de viaje", "accessories"),
    ("BOTTLE", "分装瓶", "Travel Bottle", "Botella de viaje", "accessories"),
]


def fixture_catalog(store: str = "demo-store") -> tuple[list[Product], list[ProductVariant]]:
    products, variants = [], []
    for index, (code, zh, en, es, category) in enumerate(NAMES):
        product = Product(
            id="product-" + code.lower(),
            store_id=store,
            brand="Synthetic Demo",
            category=category,
            names={"zh": zh, "en": en, "es": es},
            synthetic=True,
        )
        products.append(product)
        for size in (30, 60):
            attributes = (
                {"capacity": f"{size} ml"}
                if category in {"skincare", "accessories"}
                else {
                    "size": "small" if size == 30 else "large",
                    "material": "synthetic fiber" if code == "BRUSH" else "polymer",
                }
            )
            sku = f"DEMO-{code}-{size}"
            variants.append(
                ProductVariant(
                    id=sku,
                    sku=sku,
                    product_id=product.id,
                    store_id=store,
                    attributes=attributes,
                    price=str(10 + index + size // 10),
                    stock=100 - index,
                    markets=["US", "DE", "ES"],
                    synthetic=True,
                )
            )
    return products, variants


def manual(variant: ProductVariant, product: Product) -> str:
    values = " / ".join(variant.attributes.values())
    return (
        f"""# Demo / Synthetic — {variant.sku}

## 产品 / Product / Producto
SKU: {variant.sku}. {product.names["zh"]} / {product.names["en"]} / {product.names["es"]}.
规格 / Specifications / Especificaciones: {values}.

## 成分与用法 / Ingredients and use / Ingredientes y uso
This is synthetic training material, not a real commercial product or a safety claim.
For skincare examples the declared ingredients are water and glycerin; no added fragrance is listed.
护肤示例的声明成分为水和甘油，资料未列出添加香精；不代表过敏或功效保证。
Ingredientes del ejemplo: agua y glicerina; no se enumera fragancia añadida.
Use according to the package instructions. Stop use if irritation occurs. No clinical efficacy or certification is provided.
使用前核对包装说明。资料不提供临床功效或认证信息。
"""
        if product.category == "skincare"
        else f"""# Demo / Synthetic — {variant.sku}

## 产品 / Product / Producto
SKU: {variant.sku}. {product.names["zh"]} / {product.names["en"]} / {product.names["es"]}.
规格 / Specifications / Especificaciones: {values}.

## 使用方法 / Instructions
Synthetic demonstration product. Clean after use and store dry. No efficacy or certification is asserted.
模拟商品资料：使用后清洁并保持干燥。没有提供功效或认证信息。
"""
    )


def seed_structured(repository: CommerceRepository, store: str = "demo-store") -> None:
    if repository.list(Product, store):
        return
    products, variants = fixture_catalog(store)
    at = datetime.now(timezone.utc)
    orders = [
        Order(
            id=f"DEMO-ORDER-{i}",
            store_id=store,
            customer_id="demo-customer" if i < 3 else "other-customer",
            market="US",
            payment_status="paid",
            fulfillment_status="delivered" if i == 1 else "shipped",
            status="completed" if i == 1 else "in_transit",
            items=[OrderItem(sku=variants[i - 1].sku, quantity=1)],
            placed_at=(at - timedelta(days=10)).isoformat(),
            delivered_at=(at - timedelta(days=3)).isoformat() if i == 1 else None,
            synthetic=True,
        )
        for i in (1, 2, 3)
    ]
    shipments = [
        Shipment(
            id="shipment-" + order.id,
            store_id=store,
            order_id=order.id,
            tracking_number="SYNTHETIC-" + order.id,
            carrier="Demo Carrier",
            status=order.fulfillment_status,
            events=[{"status": order.fulfillment_status, "at": order.updated_at}],
            synthetic=True,
        )
        for order in orders
    ]
    policies = [
        StorePolicy(
            id="demo-" + kind,
            store_id=store,
            kind=kind,
            market="US",
            language="en",
            effective_at="2020-01-01T00:00:00+00:00",
            text=text,
            synthetic=True,
        )
        for kind, text in (
            (
                "shipping",
                "Demo/Synthetic US shipping policy: estimated transit 5–10 business days after dispatch, not a guaranteed arrival date.",
            ),
            (
                "return",
                "Demo/Synthetic US return policy: within 30 days of recorded delivery, unopened items may be submitted for review. Damaged, wrong or missing items need human verification. No automatic refund.",
            ),
            (
                "refund",
                "Demo/Synthetic refunds require authorized human review. No refund amount or payment is executed by this assistant.",
            ),
        )
    ]
    listings = [
        ListingDraft(
            id="demo-listing-" + variant.sku,
            store_id=store,
            sku=variant.sku,
            platform="amazon",
            market="US",
            language="en",
            title="Synthetic demo " + variant.sku,
            bullets=["Demo only"],
            description="Historical synthetic draft, not reviewed or published.",
            facts=[],
            issues=["示例草稿尚未运行事实校验"],
            review_status="pending_review",
            generation_mode="synthetic_fixture",
            synthetic=True,
        )
        for variant in variants[:2]
    ]
    cases = [
        ReturnCase(
            id="demo-case-" + reason,
            store_id=store,
            order_id=orders[0].id,
            customer_id=orders[0].customer_id,
            reason=reason,
            suggestion="Synthetic case: human review required, no refund executed.",
            synthetic=True,
        )
        for reason in ("damaged", "changed_mind")
    ]
    repository.put_many([*products, *variants, *orders, *shipments, *policies, *listings, *cases])


def export_fixtures(directory: Any) -> None:
    import csv
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    products, variants = fixture_catalog()
    directory.joinpath("catalog.json").write_text(
        json.dumps(
            {
                "products": [item.model_dump(mode="json") for item in products],
                "variants": [item.model_dump(mode="json") for item in variants],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    by_id = {item.id: item for item in products}
    with directory.joinpath("catalog.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sku",
                "product_id",
                "brand",
                "category",
                "name_en",
                "name_zh",
                "name_es",
                "capacity",
                "price",
                "stock",
                "markets",
                "synthetic",
                "description",
            ],
        )
        writer.writeheader()
        for variant in variants:
            product = by_id[variant.product_id]
            writer.writerow(
                {
                    "sku": variant.sku,
                    "product_id": product.id,
                    "brand": product.brand,
                    "category": product.category,
                    **{"name_" + key: value for key, value in product.names.items()},
                    "capacity": variant.attributes.get("capacity", ""),
                    "price": variant.price,
                    "stock": variant.stock,
                    "markets": "|".join(variant.markets),
                    "synthetic": "true",
                    "description": manual(variant, product),
                }
            )
    for variant in variants[:4]:
        directory.joinpath(variant.sku + ".md").write_text(
            manual(variant, by_id[variant.product_id]), encoding="utf-8"
        )
    directory.joinpath("conversations.json").write_text(
        json.dumps(
            [
                {"messages": ["DEMO-SERUM-60 有香精吗？", "那小容量版本呢？"]},
                {"messages": ["What is the capacity of DEMO-SERUM-30?"]},
                {"messages": ["¿Cuál es la capacidad de DEMO-SERUM-60?"]},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    export_fixtures("examples/commerce")
