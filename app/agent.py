"""The agent loop and its MCP client session.

Architectural rule: this module talks to the tools ONLY over the MCP protocol. It does not
import app.rag, app.db or app.mcp_tools, and it must never start to - the whole point of
the trace pane is that what you see on the wire is genuinely all that passes between the
agent and the tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import date
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app import inference, trace
from app.config import get_settings
from app.inference import InferenceError

log = logging.getLogger("support-copilot.agent")

# Tool results go back into the message history every round, so an unbounded result would
# grow the prompt quadratically. Policy chunks are ~1.5k chars, so this fits several.
MAX_TOOL_RESULT_CHARS = 4000
CONNECT_TIMEOUT_S = 30.0

SYSTEM_PROMPT = """You are the DesiCart customer support copilot.

You have no knowledge of DesiCart other than what the tools return. Follow these rules:

1. Answer only from tool results. If you did not get it from a tool, you do not know it.
2. Cite every policy statement with the document and section exactly as returned by
   search_policies, like: (returns-policy.md > Return window).
3. Cite the section that actually states the rule you are applying. A chunk that merely
   points at another policy - "this is handled under the X policy instead" - is a signpost,
   not an answer. Search again for that topic and cite what you find there.
4. Before using a chunk, check it governs this customer's situation. A section about items
   that can never be returned says nothing about an item that is returnable.
5. If the tools do not answer the question, say plainly that DesiCart's documentation does
   not cover it and suggest contacting support. Never fall back on general knowledge of how
   online stores usually work.
6. Never invent order IDs, dates, amounts, pincodes or policy terms. If order_status
   reports found: false, say the order was not found.
7. If a question has both an order part and a policy part, look up both before answering.
   A question spanning two policy topics needs a search for each - one search is rarely
   enough.
8. The conversation may run over several turns. Resolve references to earlier turns - "it",
   "that order", "the same pincode" - from the messages above. But do not carry a fact
   forward on memory alone: if the new question turns on a detail, call the tool again
   rather than trusting what was said earlier.
9. Today's date is {today}. Use it for any date arithmetic, such as whether a delivered
   order is still inside its return window.
10. Be concise: a few sentences, or short bullets. Lead with the answer."""


class AgentError(RuntimeError):
    """The agent could not produce an answer."""


class MCPConnection:
    """One long-lived MCP ClientSession, owned by a dedicated task.

    The transport's context managers must be entered and exited in the same task, so a
    background task holds them open and everything else borrows the session. The session
    is opened lazily on first use rather than in the FastAPI lifespan: the server it dials
    is this same process, which is not accepting connections yet during startup.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._task: asyncio.Task[None] | None = None
        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._lock = asyncio.Lock()
        self._tools: list[dict[str, Any]] | None = None

    async def _run(self) -> None:
        started = time.perf_counter()
        try:
            trace.emit("out", "initialize", {"url": self._url})
            async with streamablehttp_client(self._url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    info = await session.initialize()
                    self._session = session
                    self._error = None
                    trace.emit(
                        "in",
                        "result",
                        {
                            "of": "initialize",
                            "server": info.serverInfo.name,
                            "version": info.serverInfo.version,
                            "protocol": info.protocolVersion,
                        },
                        ms=(time.perf_counter() - started) * 1000,
                    )
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:  # noqa: BLE001 - stored and re-raised to the caller
            self._error = exc
            trace.emit("in", "error", {"of": "initialize", "error": f"{type(exc).__name__}: {exc}"})
            log.warning("MCP session ended: %s: %s", type(exc).__name__, exc)
        finally:
            self._session = None
            self._tools = None
            self._ready.set()  # never leave a waiter hanging

    async def session(self) -> ClientSession:
        async with self._lock:
            if self._task is None or self._task.done():
                self._ready.clear()
                self._stop.clear()
                self._error = None
                self._task = asyncio.create_task(self._run(), name="mcp-session")
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=CONNECT_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise AgentError(f"MCP server at {self._url} did not respond in time.") from exc

        if self._session is None:
            reason = f"{type(self._error).__name__}: {self._error}" if self._error else "unknown"
            raise AgentError(f"Could not open an MCP session to {self._url} ({reason}).")
        return self._session

    async def tools(self) -> list[dict[str, Any]]:
        """MCP tool schemas converted to OpenAI tool-calling format, cached per session."""
        session = await self.session()
        if self._tools is not None:
            return self._tools

        started = time.perf_counter()
        trace.emit("out", "tools/list", {})
        listing = await session.list_tools()
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in listing.tools
        ]
        trace.emit(
            "in",
            "result",
            {"of": "tools/list", "tools": [tool.name for tool in listing.tools]},
            ms=(time.perf_counter() - started) * 1000,
        )
        return self._tools

    async def call(self, name: str, arguments: dict[str, Any], request_id: str) -> str:
        """Call one tool and return its text content."""
        session = await self.session()
        started = time.perf_counter()
        trace.emit("out", "tools/call", {"tool": name, "arguments": arguments}, request_id=request_id)

        try:
            result = await session.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 - reported to the model, not raised at it
            elapsed = (time.perf_counter() - started) * 1000
            message = f"{type(exc).__name__}: {exc}"
            trace.emit("in", "error", {"tool": name, "error": message}, ms=elapsed, request_id=request_id)
            return json.dumps({"ok": False, "error": message})

        elapsed = (time.perf_counter() - started) * 1000
        text = "\n".join(
            item.text for item in result.content if getattr(item, "text", None) is not None
        )

        # A tool the server does not know comes back as a normal response with isError set,
        # not as an exception. It is still a failure, so it is traced as one.
        if result.isError:
            trace.emit(
                "in",
                "error",
                {"of": "tools/call", "tool": name, "error": trace.preview(text)},
                ms=elapsed,
                request_id=request_id,
            )
            return text

        trace.emit(
            "in",
            "result",
            {"of": "tools/call", "tool": name, "chars": len(text), "preview": trace.preview(text)},
            ms=elapsed,
            request_id=request_id,
        )
        return text

    async def aclose(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None


_connection: MCPConnection | None = None


def connection() -> MCPConnection:
    global _connection
    if _connection is None:
        _connection = MCPConnection(get_settings().mcp_server_url)
    return _connection


async def aclose() -> None:
    global _connection
    if _connection is not None:
        await _connection.aclose()
        _connection = None
    await inference.aclose()


def _truncate(text: str, request_id: str, tool: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    log.info("truncated %s result: %d chars dropped", tool, dropped)
    trace.emit(
        "in",
        "result",
        {"of": "truncation", "tool": tool, "kept": MAX_TOOL_RESULT_CHARS, "dropped": dropped},
        request_id=request_id,
    )
    return text[:MAX_TOOL_RESULT_CHARS] + f"\n... [truncated, {dropped} chars omitted]"


def _assistant_message(message: Any) -> dict[str, Any]:
    """The SDK message object as a plain dict for the next request's history."""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return payload


def _history_messages(history: list[dict[str, str]] | None, max_turns: int) -> list[dict[str, str]]:
    """Sanitise client-supplied history before it becomes part of the prompt.

    The history arrives from the browser, so only user and assistant turns are accepted -
    a client must not be able to inject a system instruction or forge a tool result.
    """
    if not history:
        return []
    clean = [
        {"role": turn["role"], "content": turn["content"].strip()}
        for turn in history
        if turn.get("role") in ("user", "assistant") and (turn.get("content") or "").strip()
    ]
    return clean[-(max_turns * 2) :]


async def answer(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Run the agent loop and return the final answer plus how many rounds it took."""
    request_id = uuid.uuid4().hex[:8]
    settings = get_settings()
    max_rounds = settings.max_tool_rounds
    mcp = connection()
    tools = await mcp.tools()

    prior = _history_messages(history, settings.max_history_turns)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
        *prior,
        {"role": "user", "content": question},
    ]
    if prior:
        trace.emit("out", "llm", {"of": "history", "prior_turns": len(prior)}, request_id=request_id)

    rounds = 0
    for round_number in range(1, max_rounds + 1):
        rounds = round_number
        started = time.perf_counter()
        trace.emit(
            "out",
            "llm",
            {"round": round_number, "model": settings.llm_model, "messages": len(messages),
             "tools": len(tools)},
            request_id=request_id,
        )
        try:
            message = await inference.complete(messages, tools)
        except InferenceError as exc:
            trace.emit("in", "error", {"of": "llm", "error": str(exc)}, request_id=request_id)
            raise AgentError(f"The language model call failed: {exc}") from exc

        elapsed = (time.perf_counter() - started) * 1000
        tool_calls = message.tool_calls or []
        trace.emit(
            "in",
            "llm",
            {
                "round": round_number,
                "tool_calls": [call.function.name for call in tool_calls],
                "content": trace.preview(message.content or ""),
            },
            ms=elapsed,
            request_id=request_id,
        )

        if not tool_calls:
            return {"answer": (message.content or "").strip(), "rounds": rounds}

        messages.append(_assistant_message(message))
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
                trace.emit(
                    "in",
                    "error",
                    {"tool": call.function.name, "error": "model sent unparseable arguments"},
                    request_id=request_id,
                )
            result = await mcp.call(call.function.name, arguments, request_id)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _truncate(result, request_id, call.function.name),
                }
            )

    # Round budget spent and the model still wants tools. One final call with no tools
    # offered, so it has to answer from what it already has instead of looping forever.
    trace.emit(
        "out",
        "llm",
        {"round": "final", "note": f"tool budget of {max_rounds} rounds exhausted"},
        request_id=request_id,
    )
    started = time.perf_counter()
    try:
        message = await inference.complete(messages, tools=None)
    except InferenceError as exc:
        trace.emit("in", "error", {"of": "llm", "error": str(exc)}, request_id=request_id)
        raise AgentError(f"The language model call failed: {exc}") from exc

    trace.emit(
        "in",
        "llm",
        {"round": "final", "content": trace.preview(message.content or "")},
        ms=(time.perf_counter() - started) * 1000,
        request_id=request_id,
    )
    return {"answer": (message.content or "").strip(), "rounds": rounds}
