from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..data import ORDERS

PLUGIN_NAME = "invoice"
PLUGIN_VERSION = "1.0"


@tool
def issue_invoice(order_id: str, title: str = "个人") -> dict[str, Any]:
    """为订单开具发票（插件能力）。"""
    if order_id not in ORDERS:
        return {"ok": False, "error": f"未找到订单：{order_id}", "plugin_version": PLUGIN_VERSION}
    invoice_id = f"INV-{order_id}-{PLUGIN_VERSION}"
    return {
        "ok": True,
        "invoice_id": invoice_id,
        "title": title,
        "plugin_version": PLUGIN_VERSION,
    }


def get_tools():
    return [issue_invoice]


