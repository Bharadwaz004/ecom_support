"""The MCP server: four tools over DesiCart's policy docs and order database.

Docstrings here are the tool descriptions the model sees when choosing what to call, so
they say what the tool is *for* and when not to reach for it, not just what it returns.
Every tool returns a plain dict - including for failures - because a raised exception
reaches the model as an opaque protocol error it cannot recover from.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app import db, rag
from app.rag import RetrievalError

PINCODE_RE = re.compile(r"^[1-9][0-9]{5}$")
MAX_TOP_K = 10
MAX_SNIPPET_CHARS = 1200

# streamable_http_path is "/" because this app is mounted under /mcp by main.py; leaving
# the default would put the endpoint at /mcp/mcp.
mcp = FastMCP(
    "desicart-support",
    instructions=(
        "Tools for answering DesiCart customer-support questions. Policy wording comes "
        "from search_policies; order and delivery facts come from the other three tools. "
        "Nothing here invents data - if a lookup finds nothing it says so."
    ),
    streamable_http_path="/",
)


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **extra}


@mcp.tool()
async def search_policies(
    query: Annotated[str, Field(description="A natural-language question or phrase.")],
    top_k: Annotated[int, Field(description="How many chunks to return, 1-10.", ge=1, le=MAX_TOP_K)] = 5,
) -> dict[str, Any]:
    """Semantic search over DesiCart's published policy documents.

    Use this for anything about rules rather than a specific order: return windows,
    refund timelines, cancellation rights, cash-on-delivery limits, warranty terms,
    exchanges, damaged or missing items, gift cards, seller onboarding, payment
    failures, festival sale terms, account and privacy.

    Returns the matching chunks with their source `doc`, `section` and `anchor` so the
    answer can cite them, plus a cosine `score`. The corpus does not cover every topic;
    if the results are unrelated to the question, say the policy does not cover it rather
    than reasoning from a loosely related chunk.
    """
    query = (query or "").strip()
    if not query:
        return _error("query must not be empty")

    top_k = max(1, min(int(top_k), MAX_TOP_K))
    try:
        chunks = await rag.search(query, top_k)
    except RetrievalError as exc:
        return _error(str(exc))

    return {
        "ok": True,
        "query": query,
        "count": len(chunks),
        "chunks": [{**chunk, "text": chunk["text"][:MAX_SNIPPET_CHARS]} for chunk in chunks],
    }


@mcp.tool()
async def order_status(
    order_id: Annotated[int, Field(description="Numeric DesiCart order ID, e.g. 4412.")],
) -> dict[str, Any]:
    """Look up one order: its current status, timeline, value, payment method and items.

    Use this whenever the question names an order. It returns the recorded status
    (placed, packed, in_transit, delivered, returned, cancelled, refunded), every
    timestamp the order has reached so far, the shipping destination, and each line item
    with its category, warranty length and whether that item is returnable at all.

    `days_since_delivery` is included for delivered orders so return-window questions can
    be answered against the policy docs without doing date arithmetic.

    If the order does not exist this returns `found: false` - it does not guess. Never
    present an order ID to the customer that did not come back from this tool.
    """
    if order_id <= 0:
        return _error("order_id must be a positive integer", order_id=order_id)

    try:
        order = await db.fetch_order(order_id)
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured error, not a traceback
        return _error(f"order lookup failed: {type(exc).__name__}")

    if order is None:
        return {"ok": True, "found": False, "order_id": order_id}

    days_since_delivery = None
    if order.get("delivered_at"):
        delivered = datetime.fromisoformat(order["delivered_at"])
        days_since_delivery = (datetime.now() - delivered).days

    return {
        "ok": True,
        "found": True,
        "order": {**order, "days_since_delivery": days_since_delivery},
    }


@mcp.tool()
async def orders_summary(
    days: Annotated[int, Field(description="Look-back window in days, 1-365.", ge=1, le=365)] = 30,
    status: Annotated[
        str | None,
        Field(description="Optional filter: one of placed, packed, in_transit, delivered, returned, cancelled, refunded."),
    ] = None,
) -> dict[str, Any]:
    """Aggregate order counts and value over a recent period, optionally by status.

    Use this for questions about volume or trends ("how many orders were cancelled last
    month", "how many are in transit"), never to answer about a specific order - that is
    what order_status is for.

    Returns the total order count and value for the window plus a per-status breakdown.
    The window is measured from the order's placed date.
    """
    days = int(days)
    if not 1 <= days <= db.MAX_SUMMARY_DAYS:
        return _error(f"days must be between 1 and {db.MAX_SUMMARY_DAYS}", days=days)

    if status is not None:
        status = status.strip().lower()
        if status not in db.ORDER_STATUSES:
            return _error(
                f"unknown status '{status}'", valid_statuses=list(db.ORDER_STATUSES)
            )

    try:
        return {"ok": True, **await db.summarise_orders(days, status)}
    except Exception as exc:  # noqa: BLE001 - structured error, never a traceback
        return _error(f"summary failed: {type(exc).__name__}")


@mcp.tool()
async def check_delivery(
    pincode: Annotated[str, Field(description="Six-digit Indian pincode, e.g. 560001.")],
) -> dict[str, Any]:
    """Check whether DesiCart delivers to a pincode, and on what terms.

    Returns serviceability, the delivery zone, the estimated delivery window in business
    days, and whether cash on delivery is offered there. Use this for "do you deliver
    to...", "how long will it take to...", and "can I pay cash on delivery in..."
    questions.

    If the exact pincode is not in our serviceability table, the result says so and may
    include a `nearby` inference from pincodes in the same postal region - clearly marked
    as an inference. Present that as "likely", never as a confirmed serviceability.
    """
    pincode = (pincode or "").strip()
    if not PINCODE_RE.match(pincode):
        return _error("pincode must be six digits and not start with 0", pincode=pincode)

    try:
        row = await db.lookup_pincode(pincode)
        if row is not None:
            serviceable = bool(row["serviceable"])
            return {
                "ok": True,
                "known": True,
                "pincode": pincode,
                "city": row["city"],
                "state": row["state"],
                "serviceable": serviceable,
                "zone": row["zone"] if serviceable else None,
                "cod_available": bool(row["cod_available"]),
                "estimated_days": (
                    {"min": row["est_days_min"], "max": row["est_days_max"]} if serviceable else None
                ),
            }

        nearby = await db.state_for_unknown_pincode(pincode[:3])
    except Exception as exc:  # noqa: BLE001 - structured error, never a traceback
        return _error(f"serviceability lookup failed: {type(exc).__name__}")

    if nearby is None:
        return {
            "ok": True,
            "known": False,
            "pincode": pincode,
            "nearby": None,
            "note": "This pincode is not in our serviceability table and no nearby pincode matched.",
        }

    return {
        "ok": True,
        "known": False,
        "pincode": pincode,
        "nearby": {
            "state": nearby["state"],
            "zone": nearby["zone"],
            "cod_available": bool(nearby["cod_available"]),
            "estimated_days": {"min": nearby["est_days_min"], "max": nearby["est_days_max"]},
        },
        "note": (
            "Exact pincode not on file. The 'nearby' values are inferred from another "
            "pincode in the same postal region and are not a confirmed serviceability."
        ),
    }
