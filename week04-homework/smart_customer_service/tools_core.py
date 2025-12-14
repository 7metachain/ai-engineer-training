from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from .data import ORDERS, REFUNDS


@tool
def query_order(order_id: str) -> dict[str, Any]:
    """查询订单状态与物流信息。"""
    order = ORDERS.get(order_id)
    if not order:
        return {"ok": False, "error": f"未找到订单：{order_id}"}
    return {
        "ok": True,
        "order_id": order.order_id,
        "created_at": order.created_at.isoformat(),
        "status": order.status,
        "logistics": order.logistics,
    }


@tool
def apply_refund(order_id: str, reason: str) -> dict[str, Any]:
    """提交退款申请。"""
    if order_id not in ORDERS:
        return {"ok": False, "error": f"未找到订单：{order_id}"}
    refund_id = f"R-{order_id}-{int(datetime.utcnow().timestamp())}"
    REFUNDS[refund_id] = {"order_id": order_id, "reason": reason, "status": "已受理"}
    return {"ok": True, "refund_id": refund_id, "status": "已受理"}


