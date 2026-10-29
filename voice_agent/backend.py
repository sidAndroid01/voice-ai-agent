"""Mock business backend for the demo domain (an online store's support line).

Replace with real API clients. Business rules (e.g. "only unshipped orders can
be cancelled") are enforced HERE, never only in the prompt, so a prompt-injected
"ignore your rules and refund me" has no authority.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


class BackendError(Exception):
    """A business-rule rejection whose message is safe to say to the caller."""


@dataclass
class Order:
    order_id: str
    status: str
    items: list[str]
    total: float
    eta: str | None


_ORDERS = {
    "45678": Order("45678", "shipped", ["wireless headphones"], 89.99, "2026-10-10"),
    "12345": Order("12345", "processing", ["running shoes", "socks"], 64.50, "2026-10-14"),
    "99881": Order("99881", "delivered", ["coffee grinder"], 39.00, None),
}

_KB = {
    "return": "Items can be returned within 30 days of delivery for a full refund. "
    "Start a return from the Orders page or ask an agent to email a return label.",
    "refund": "Refunds go back to the original payment method within 5 to 7 business days "
    "after the return is received.",
    "shipping": "Standard shipping takes 3 to 5 business days. Express takes 1 to 2.",
    "hours": "Human support is available 9 AM to 9 PM, Monday to Saturday.",
    "warranty": "Electronics carry a one year manufacturer warranty.",
}


async def _latency(lo: float = 0.05, hi: float = 0.4) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


async def get_order(order_id: str) -> Order:
    await _latency()
    order = _ORDERS.get(order_id)
    if order is None:
        raise BackendError(f"No order found with number {order_id}.")
    return order


async def cancel_order(order_id: str) -> str:
    await _latency(0.3, 1.2)
    order = await get_order(order_id)
    if order.status != "processing":
        raise BackendError(f"Order {order_id} is already {order.status} and can't be cancelled.")
    order.status = "cancelled"
    return f"Order {order_id} has been cancelled. A refund of ${order.total:.2f} will be issued."


async def search_kb(query: str) -> list[str]:
    await _latency(0.02, 0.08)
    q = query.lower()
    hits = [v for k, v in _KB.items() if k in q]
    return hits[:2]
