"""Read-only authorized mock-platform data and deterministic policy decisions."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ai_agent_template.commerce.repository import CommerceRepository
from ai_agent_template.commerce.schemas import (
    AfterSalesRequest,
    Order,
    Principal,
    ReturnCase,
    Shipment,
    StorePolicy,
    effective,
    now,
    parsed_date,
)


def authorized_order(repository: CommerceRepository, principal: Principal, order_id: str) -> Order:
    order = repository.get(Order, principal.store_id, order_id)
    if (
        not order
        or order.market not in principal.marketplaces
        or (principal.role != "operator" and order.customer_id != principal.customer_id)
    ):
        # Identical not-found response prevents cross-customer enumeration.
        raise LookupError("Order not found or inaccessible")
    return order


class MockOrderAdapter:
    source = "demo_sql_order_adapter"

    def __init__(self, repository: CommerceRepository) -> None:
        self.repository = repository

    def query(self, principal: Principal, order_id: str) -> dict[str, Any]:
        order = authorized_order(self.repository, principal, order_id)
        shipments = [
            item
            for item in self.repository.list(Shipment, principal.store_id)
            if item.order_id == order.id
        ]
        return {
            "order": order.model_dump(mode="json"),
            "shipments": [item.model_dump() for item in shipments],
            "business_source": {
                "adapter": self.source,
                "record_id": order.id,
                "updated_at": order.updated_at,
                "queried_at": now(),
                "result_id": str(uuid4()),
            },
            "actions_executed": [],
        }


def active_policy(
    repository: CommerceRepository, store: str, market: str, kind: str
) -> StorePolicy | None:
    items = [
        item
        for item in repository.list(StorePolicy, store)
        if item.kind == kind and item.market in {market, "GLOBAL"} and effective(item)
    ]
    # Specific site supersedes global; latest effective policy wins deterministically.
    return max(
        items,
        key=lambda item: (item.market == market, parsed_date(item.effective_at)),
        default=None,
    )


def after_sales(
    repository: CommerceRepository, principal: Principal, order_id: str, request: AfterSalesRequest
) -> dict[str, Any]:
    order = authorized_order(repository, principal, order_id)
    policy = active_policy(repository, principal.store_id, order.market, "return")
    eligible: bool | None = None
    reasons = []
    if not policy:
        reasons.append("没有当前有效退货政策，需要人工核实")
    else:
        # Client-provided received_at is untrusted and cannot overrule the authoritative order.
        if request.received_at and request.received_at != order.delivered_at:
            reasons.append("收货日期与订单记录不一致，需要人工核实")
        delivered = parsed_date(order.delivered_at) if order.delivered_at else None
        if delivered is None or delivered > datetime.now(timezone.utc):
            reasons.append("尚无有效签收记录，需要人工核实")
        else:
            days = (datetime.now(timezone.utc) - delivered).days
            if days > policy.return_days:
                eligible = False
                reasons.append("超过退货时间窗口")
            elif (
                request.reason in {"damaged", "wrong_item", "missing_item"}
                and policy.damaged_exception
            ):
                eligible = True
                reasons.append("损坏/错发/漏发适用人工核实例外")
            elif policy.requires_unopened and request.opened is None:
                reasons.append("需要确认是否开封")
            elif policy.requires_unopened and request.opened:
                eligible = False
                reasons.append("普通退货要求未开封")
            else:
                eligible = True
                reasons.append("满足时间窗口和包装条件，可提交人工审核")
    if any("不一致" in reason for reason in reasons):
        eligible = None
    suggestion = "；".join(reasons) + "。仅生成工单草稿，退款金额及实际退款由授权人员审核。"
    case = ReturnCase(
        id=str(uuid4()),
        store_id=principal.store_id,
        order_id=order.id,
        customer_id=order.customer_id,
        reason=request.reason,
        suggestion=suggestion,
        synthetic=order.synthetic,
    )
    repository.put(case)
    return {
        "case": case.model_dump(),
        "eligible_for_review": eligible,
        "policy": policy.model_dump() if policy else None,
        "business_source": {
            "adapter": "deterministic_policy_rules_v1",
            "record_id": order.id,
            "updated_at": order.updated_at,
            "queried_at": now(),
        },
        "actions_executed": [],
    }
