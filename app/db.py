"""Read-only SQLite access for order data.

Every query is parameterized and the connection is opened in SQLite's read-only URI mode,
so a bug or an injected string cannot write to the database. Queries run in a worker
thread because sqlite3 is blocking and the request handlers are async.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from app.config import get_settings

# Whitelist. A status is compared against this before it reaches a query, so the value in
# the SQL is always one of ours even though it is also passed as a bound parameter.
ORDER_STATUSES = (
    "placed",
    "packed",
    "in_transit",
    "delivered",
    "returned",
    "cancelled",
    "refunded",
)

MAX_SUMMARY_DAYS = 365


def _connect() -> sqlite3.Connection:
    """Open the DB read-only. as_uri() handles Windows drive letters and escaping."""
    settings = get_settings()
    conn = sqlite3.connect(f"{settings.db_file.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


async def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_rows, sql, params)


async def fetch_order(order_id: int) -> dict[str, Any] | None:
    """Full order record with its line items, or None if the order does not exist."""
    orders = await _query(
        """
        SELECT order_id, customer_name, status, payment_method, order_value_inr,
               shipping_fee_inr, cod_fee_inr, pincode, city, state, zone,
               placed_at, packed_at, shipped_at, expected_delivery, delivered_at,
               cancelled_at, returned_at, refunded_at, refund_amount_inr
        FROM orders WHERE order_id = ?
        """,
        (order_id,),
    )
    if not orders:
        return None

    order = orders[0]
    order["items"] = await _query(
        """
        SELECT p.name, p.category, p.warranty_months, p.returnable,
               i.quantity, i.unit_price_inr, i.line_total_inr
        FROM order_items i
        JOIN products p ON p.product_id = i.product_id
        WHERE i.order_id = ?
        ORDER BY i.item_id
        """,
        (order_id,),
    )
    return order


async def summarise_orders(days: int, status: str | None = None) -> dict[str, Any]:
    """Counts and order values over the last `days`, optionally for one status."""
    # The cutoff is computed here rather than with SQLite's datetime('now') because the
    # stored timestamps are local naive times and datetime('now') is UTC.
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(sep=" ", timespec="seconds")

    if status is None:
        rows = await _query(
            """
            SELECT status, COUNT(*) AS n, SUM(order_value_inr) AS value
            FROM orders WHERE placed_at >= ?
            GROUP BY status ORDER BY n DESC
            """,
            (cutoff,),
        )
    else:
        rows = await _query(
            """
            SELECT status, COUNT(*) AS n, SUM(order_value_inr) AS value
            FROM orders WHERE placed_at >= ? AND status = ?
            GROUP BY status
            """,
            (cutoff, status),
        )

    by_status = {row["status"]: {"orders": row["n"], "value_inr": row["value"] or 0} for row in rows}
    return {
        "days": days,
        "since": cutoff,
        "status_filter": status,
        "total_orders": sum(entry["orders"] for entry in by_status.values()),
        "total_value_inr": sum(entry["value_inr"] for entry in by_status.values()),
        "by_status": by_status,
    }


async def lookup_pincode(pincode: str) -> dict[str, Any] | None:
    """Serviceability row for a pincode, or None if we have no record of it."""
    rows = await _query(
        """
        SELECT pincode, city, state, zone, serviceable, cod_available,
               est_days_min, est_days_max
        FROM pincodes WHERE pincode = ?
        """,
        (pincode,),
    )
    return rows[0] if rows else None


async def state_for_unknown_pincode(prefix: str) -> dict[str, Any] | None:
    """Best-effort fallback: another pincode sharing the first 3 digits.

    Indian pincodes are regionally allocated, so a shared prefix is a reasonable proxy for
    'same sorting region' - enough to answer 'do you deliver near here' honestly, as long
    as the caller is told it is an inference and not a record.
    """
    rows = await _query(
        """
        SELECT state, zone, serviceable, cod_available, est_days_min, est_days_max
        FROM pincodes WHERE substr(pincode, 1, 3) = ? AND serviceable = 1
        LIMIT 1
        """,
        (prefix,),
    )
    return rows[0] if rows else None


def check_readable() -> dict[str, Any]:
    """Health probe: can we open the DB read-only and read a row?"""
    try:
        rows = _rows("SELECT COUNT(*) AS n FROM orders")
    except sqlite3.Error as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "orders": rows[0]["n"]}
