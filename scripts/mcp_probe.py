"""Talk to the MCP server over the wire - the same way the agent and Claude Desktop do.

This is a verification tool, not part of the app. It imports only the MCP client; it never
touches app.rag, app.db or app.mcp_tools.

Usage:
    python scripts/mcp_probe.py                                  # list tools, run a smoke suite
    python scripts/mcp_probe.py --tool order_status --args '{"order_id": 4412}'
    python scripts/mcp_probe.py --url http://127.0.0.1:8000/mcp/ --schemas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp/")

SMOKE_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("order_status", {"order_id": 4412}),
    ("order_status", {"order_id": 999999}),
    ("order_status", {"order_id": -1}),
    ("orders_summary", {"days": 30}),
    ("orders_summary", {"days": 90, "status": "cancelled"}),
    ("orders_summary", {"days": 30, "status": "'; DROP TABLE orders;--"}),
    ("check_delivery", {"pincode": "560001"}),
    ("check_delivery", {"pincode": "744101"}),
    ("check_delivery", {"pincode": "560999"}),
    ("check_delivery", {"pincode": "abc"}),
    ("search_policies", {"query": "how many days do I have to return a shirt", "top_k": 3}),
]


def render(result: Any) -> str:
    blocks = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text is None:
            blocks.append(f"<{type(item).__name__}>")
            continue
        try:
            blocks.append(json.dumps(json.loads(text), indent=2)[:1200])
        except json.JSONDecodeError:
            blocks.append(text[:1200])
    return "\n".join(blocks)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the DesiCart MCP server.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--tool", help="call one tool instead of the smoke suite")
    parser.add_argument("--args", default="{}", help="JSON arguments for --tool")
    parser.add_argument("--schemas", action="store_true", help="print full input schemas")
    options = parser.parse_args()

    async with streamablehttp_client(options.url) as (read, write, _):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"connected to {info.serverInfo.name} v{info.serverInfo.version} at {options.url}\n")

            listing = await session.list_tools()
            print(f"tools ({len(listing.tools)}):")
            for tool in listing.tools:
                first_line = (tool.description or "").strip().splitlines()[0]
                print(f"  - {tool.name}: {first_line}")
                if options.schemas:
                    print(json.dumps(tool.inputSchema, indent=4))
            print()

            calls = (
                [(options.tool, json.loads(options.args))] if options.tool else SMOKE_CALLS
            )
            for name, arguments in calls:
                started = time.perf_counter()
                try:
                    result = await session.call_tool(name, arguments)
                    elapsed = (time.perf_counter() - started) * 1000
                    flag = "ERR " if result.isError else "ok  "
                    print(f"{flag}{name}({json.dumps(arguments)})  {elapsed:.0f}ms")
                    print("    " + render(result).replace("\n", "\n    "))
                except Exception as exc:  # noqa: BLE001 - a probe reports, it does not crash
                    print(f"FAIL {name}({json.dumps(arguments)}): {type(exc).__name__}: {exc}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
