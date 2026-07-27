"""FastAPI application: MCP server at /mcp, agent at /chat, live trace at /trace, UI at /."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import agent, db, rag, trace
from app.config import ROOT, get_settings, validate_or_exit
from app.mcp_tools import mcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("support-copilot")

MAX_MESSAGE_CHARS = 500
HEARTBEAT_SECONDS = 15


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    rounds: int


class RateLimiter:
    """Per-IP and global caps, in memory. Rolling windows, no Redis, no background task.

    Rolling rather than calendar windows: a calendar-day cap lets someone burn the whole
    budget at 23:59 and again at 00:01, and a rolling window needs no reset scheduling.
    """

    def __init__(self, per_ip_hourly: int, daily_cap: int) -> None:
        self.per_ip_hourly = per_ip_hourly
        self.daily_cap = daily_cap
        self._by_ip: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()

    @staticmethod
    def _prune(stamps: deque[float], window: float, now: float) -> None:
        while stamps and now - stamps[0] > window:
            stamps.popleft()

    def check(self, ip: str, now: float | None = None) -> tuple[bool, str]:
        now = time.time() if now is None else now

        self._prune(self._global, 86_400, now)
        if len(self._global) >= self.daily_cap:
            return False, (
                "This demo has hit its daily message cap. It resets on a rolling 24-hour "
                "window - please try again later."
            )

        stamps = self._by_ip[ip]
        self._prune(stamps, 3_600, now)
        if len(stamps) >= self.per_ip_hourly:
            wait_minutes = int((3_600 - (now - stamps[0])) / 60) + 1
            return False, (
                f"You have used all {self.per_ip_hourly} messages for this hour. "
                f"Try again in about {wait_minutes} minute(s)."
            )

        stamps.append(now)
        self._global.append(now)

        # Cheap housekeeping so idle IPs cannot accumulate forever in a long-lived process.
        if len(self._by_ip) > 5_000:
            for key in [k for k, v in self._by_ip.items() if not v]:
                del self._by_ip[key]
        return True, ""


limiter: RateLimiter | None = None


def client_ip(request: Request) -> str:
    """Real client IP behind Railway's proxy, falling back to the socket peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global limiter
    settings = validate_or_exit()
    limiter = RateLimiter(settings.per_ip_hourly_cap, settings.daily_message_cap)
    log.info(
        "model=%s embed=%s collection=%s caps=%s/hour-per-ip %s/day",
        settings.llm_model,
        settings.embed_model,
        settings.qdrant_collection,
        settings.per_ip_hourly_cap,
        settings.daily_message_cap,
    )

    # The MCP session manager must run for the whole app lifetime; mounting its ASGI app
    # does not start it, because FastAPI does not run a mounted sub-app's lifespan.
    async with mcp.session_manager.run():
        try:
            info = await rag.assert_dimensions()
            log.info("qdrant ok: %s", info)
        except rag.RetrievalError as exc:
            # A real dimension mismatch is fatal and says so in the message. Anything else
            # is a transient free-tier hiccup (cold model, network), which should not stop
            # the app from booting - /health keeps reporting it.
            if "dimension mismatch" in str(exc):
                raise
            log.warning("startup retrieval check skipped: %s", exc)
        yield

    await agent.aclose()
    await rag.aclose()


app = FastAPI(title="DesiCart support copilot", version="0.3.0", lifespan=lifespan)

app.mount("/mcp", mcp.streamable_http_app())


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Message is {len(message)} characters; the limit is {MAX_MESSAGE_CHARS}.",
        )

    assert limiter is not None  # set in lifespan, which runs before any request
    allowed, reason = limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    try:
        result = await agent.answer(message)
    except agent.AgentError as exc:
        trace.emit("in", "error", {"of": "chat", "error": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(**result)


@app.get("/trace")
async def trace_stream(request: Request) -> StreamingResponse:
    """SSE stream of MCP protocol events. Rate limiting never applies here."""

    async def events() -> AsyncIterator[str]:
        async with trace.subscribe() as queue:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keeps proxies from dropping an idle stream
                    continue
                yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering, otherwise SSE arrives in bursts
        },
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Per-dependency status. Never raises; a failing dependency shows up as ok: false."""
    settings = get_settings()

    database = db.check_readable()

    try:
        qdrant: dict[str, Any] = {"ok": True, **await rag.collection_info()}
    except rag.RetrievalError as exc:
        qdrant = {"ok": False, "error": str(exc)}

    llm = {
        "ok": bool(settings.hf_token),
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "token_present": bool(settings.hf_token),
    }

    dependencies = {"db": database, "qdrant": qdrant, "llm": llm}
    return {
        "status": "ok" if all(dep["ok"] for dep in dependencies.values()) else "degraded",
        "dependencies": dependencies,
        "trace": trace.stats(),
    }


# Mounted last: a StaticFiles mount at "/" matches everything, so it must not shadow the
# routes above.
STATIC_DIR = ROOT / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:  # pragma: no cover - only when the checkout is incomplete
    log.warning("static directory %s not found; UI will not be served", STATIC_DIR)
