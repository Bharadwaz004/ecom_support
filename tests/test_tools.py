"""Tool-level tests: SQL injection against every tool, and MCP schema validity.

Calls go through FastMCP's own dispatch (mcp.call_tool), so argument validation is exercised
exactly as it is for a real MCP client rather than being bypassed by calling the Python
functions directly.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from app import db, rag
from app.mcp_tools import mcp

INJECTIONS = [
    "'; DROP TABLE orders;--",
    '" OR ""="',
    "1 OR 1=1",
    "'; DELETE FROM pincodes WHERE 1=1; --",
    "%' UNION SELECT name, category FROM products --",
    "4412; UPDATE orders SET status='refunded'",
    "\\'; ATTACH DATABASE '/tmp/evil.db' AS evil; --",
    "0 UNION ALL SELECT NULL,NULL,NULL",
]

TOOL_NAMES = {"search_policies", "order_status", "orders_summary", "check_delivery"}


async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool and return its structured result."""
    _, structured = await mcp.call_tool(name, arguments)
    return structured


def table_counts() -> dict[str, int]:
    conn = sqlite3.connect(f"{db.get_settings().db_file.as_uri()}?mode=ro", uri=True)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("orders", "order_items", "products", "pincodes")
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("payload", INJECTIONS)
async def test_orders_summary_rejects_injected_status(payload: str) -> None:
    result = await call("orders_summary", {"days": 30, "status": payload})
    assert result["ok"] is False
    assert "unknown status" in result["error"]
    assert set(result["valid_statuses"]) == set(db.ORDER_STATUSES)


@pytest.mark.parametrize("payload", INJECTIONS)
async def test_check_delivery_rejects_injected_pincode(payload: str) -> None:
    result = await call("check_delivery", {"pincode": payload})
    assert result["ok"] is False
    assert "six digits" in result["error"]


@pytest.mark.parametrize("payload", INJECTIONS)
async def test_order_status_rejects_non_integer_ids(payload: str) -> None:
    # The schema is the first line of defence: a string never reaches our code at all.
    with pytest.raises(ToolError):
        await call("order_status", {"order_id": payload})


@pytest.mark.parametrize("payload", INJECTIONS)
async def test_search_policies_treats_the_query_as_data(payload: str, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        seen["query"] = query
        seen["top_k"] = top_k
        return [{"chunk_id": "x#y#0", "doc": "x.md", "section": "y", "anchor": "y",
                 "score": 0.5, "text": "chunk"}]

    monkeypatch.setattr(rag, "search", fake_search)
    result = await call("search_policies", {"query": payload, "top_k": 3})

    assert result["ok"] is True
    assert seen["query"] == payload  # passed through untouched, never interpolated
    assert seen["top_k"] == 3


async def test_injections_leave_the_database_untouched(monkeypatch) -> None:
    before = table_counts()

    async def fake_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(rag, "search", fake_search)

    for payload in INJECTIONS:
        await call("orders_summary", {"days": 30, "status": payload})
        await call("check_delivery", {"pincode": payload})
        await call("search_policies", {"query": payload})
        with pytest.raises(ToolError):
            await call("order_status", {"order_id": payload})

    assert table_counts() == before


def test_database_connection_is_read_only() -> None:
    conn = db._connect()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM orders")
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# Ordinary behaviour and bounds
# --------------------------------------------------------------------------------------


async def test_order_status_reports_missing_orders_without_raising() -> None:
    result = await call("order_status", {"order_id": 999_999})
    assert result == {"ok": True, "found": False, "order_id": 999_999}


async def test_order_status_rejects_non_positive_ids() -> None:
    result = await call("order_status", {"order_id": 0})
    assert result["ok"] is False


async def test_orders_summary_bounds_the_window() -> None:
    # Out-of-range values never reach our code: the schema stops them at dispatch.
    for days in (0, -5, 400):
        with pytest.raises(ToolError):
            await call("orders_summary", {"days": days})


async def test_orders_summary_bounds_are_also_enforced_in_code() -> None:
    # Defence in depth, for any caller that bypasses the schema.
    from app.mcp_tools import orders_summary

    assert (await orders_summary(days=0))["ok"] is False
    assert (await orders_summary(days=400))["ok"] is False


async def test_search_policies_clamps_top_k(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        seen["top_k"] = top_k
        return []

    monkeypatch.setattr(rag, "search", fake_search)
    with pytest.raises(ToolError):  # above the schema maximum
        await call("search_policies", {"query": "x", "top_k": 50})

    await call("search_policies", {"query": "x", "top_k": 10})
    assert seen["top_k"] == 10


async def test_search_policies_rejects_an_empty_query() -> None:
    result = await call("search_policies", {"query": "   "})
    assert result["ok"] is False


async def test_retrieval_failure_becomes_a_structured_error(monkeypatch) -> None:
    async def boom(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise rag.RetrievalError("Qdrant unreachable")

    monkeypatch.setattr(rag, "search", boom)
    result = await call("search_policies", {"query": "returns"})
    assert result["ok"] is False
    assert "Qdrant unreachable" in result["error"]


async def test_check_delivery_marks_inferred_results_as_inferred() -> None:
    known = await call("check_delivery", {"pincode": "560001"})
    assert known["known"] is True and known["serviceable"] is True

    unknown = await call("check_delivery", {"pincode": "560999"})
    assert unknown["known"] is False
    assert "not a confirmed serviceability" in unknown["note"]


async def test_non_serviceable_pincode_has_no_estimate() -> None:
    result = await call("check_delivery", {"pincode": "744101"})
    assert result["serviceable"] is False
    assert result["estimated_days"] is None
    assert result["zone"] is None


# --------------------------------------------------------------------------------------
# MCP schema validity
# --------------------------------------------------------------------------------------


async def test_every_tool_is_registered() -> None:
    tools = await mcp.list_tools()
    assert {tool.name for tool in tools} == TOOL_NAMES


async def test_schemas_are_valid_json_schema_objects() -> None:
    for tool in await mcp.list_tools():
        schema = tool.inputSchema
        assert schema["type"] == "object", tool.name
        assert schema.get("properties"), tool.name
        # Must survive the round trip into an OpenAI tool-calling payload.
        json.dumps(schema)


async def test_every_parameter_is_described() -> None:
    for tool in await mcp.list_tools():
        for name, spec in tool.inputSchema["properties"].items():
            assert spec.get("description"), f"{tool.name}.{name} has no description"


async def test_descriptions_are_written_for_the_model() -> None:
    for tool in await mcp.list_tools():
        description = (tool.description or "").strip()
        assert len(description) > 120, f"{tool.name} description is too thin to route on"


async def test_required_fields_match_parameters_without_defaults() -> None:
    required = {tool.name: set(tool.inputSchema.get("required", [])) for tool in await mcp.list_tools()}
    assert required["search_policies"] == {"query"}
    assert required["order_status"] == {"order_id"}
    assert required["check_delivery"] == {"pincode"}
    assert required["orders_summary"] == set()
