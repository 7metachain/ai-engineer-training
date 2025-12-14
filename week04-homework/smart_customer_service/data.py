from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Order:
    order_id: str
    created_at: datetime
    status: str
    logistics: str


# Simple in-memory "DB" for homework/demo.
ORDERS: dict[str, Order] = {
    "A1001": Order(
        order_id="A1001",
        created_at=datetime(2025, 12, 13, 10, 0, 0),
        status="已发货",
        logistics="顺丰：SF123456789CN（运输中）",
    ),
    "A1002": Order(
        order_id="A1002",
        created_at=datetime(2025, 12, 12, 18, 30, 0),
        status="已签收",
        logistics="中通：ZT987654321CN（已签收）",
    ),
}


REFUNDS: dict[str, dict] = {}


