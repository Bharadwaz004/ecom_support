"""Offline evaluation of the deployed copilot.

Runs from a laptop against a running instance. Nothing is deployed for this, and nothing
imports the app: retrieval is measured through the public MCP endpoint, behaviour through
POST /chat, and tool selection by watching the live /trace stream.

Usage:
    python evals/run_evals.py --url http://127.0.0.1:8000
    python evals/run_evals.py --url https://your-app.up.railway.app --limit 5
    python evals/run_evals.py --url ... --no-readme

Environment:
    HF_TOKEN      required, for the judge model
    JUDGE_MODEL   optional, defaults to Qwen/Qwen2.5-72B-Instruct
    LLM_BASE_URL  optional, defaults to the HF router
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI, OpenAIError

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "evals" / "history"
README = ROOT / "README.md"
MARKER_START = "<!-- EVAL_RESULTS_START -->"
MARKER_END = "<!-- EVAL_RESULTS_END -->"

TOP_K = 5
JUDGE_TEMPERATURE = 0.0  # a judge that varies between runs cannot detect a regression
TRACE_SETTLE_S = 0.4  # let the last events of a turn arrive before closing the window

JUDGE_SYSTEM = """You are a strict evaluator of a customer support assistant for an online \
store called DesiCart.

You are given a customer QUESTION, the assistant's ANSWER, and the EVIDENCE that was \
actually returned to it by its tools. The evidence is the complete set of facts the \
assistant had. Anything in the answer that is not in the evidence is unsupported, no \
matter how plausible it sounds or how true it might be of other online stores.

Judge only what is stated. Do not reward or penalise style, tone or length.

Reply with a single JSON object and nothing else:

{
  "supported_claims": ["..."],
  "unsupported_claims": ["..."],
  "fabricated_specifics": true|false,
  "refused": true|false,
  "grounded_score": 0.0
}

Definitions:
- supported_claims: factual assertions in the answer that the evidence backs up.
- unsupported_claims: factual assertions the evidence does not back up. Citing a document \
section that does not actually state the claim counts as unsupported.
- fabricated_specifics: true if the answer invents a concrete detail - an order ID, date, \
amount, pincode, timeframe or named policy term - that appears nowhere in the evidence.
- refused: true if the answer says the information is not available, not covered by \
DesiCart's documentation, or directs the customer to support instead of answering.
- grounded_score: supported_claims / (supported_claims + unsupported_claims), or 1.0 if \
the answer makes no factual claims at all (a clean refusal scores 1.0)."""


@dataclass
class CaseResult:
    question: str
    answerable: bool
    expected_tool: str | None
    expected_chunk_ids: list[str]
    notes: str = ""

    retrieved_chunk_ids: list[str] = field(default_factory=list)
    recall_at_k: float | None = None
    hit_at_k: bool | None = None

    answer: str = ""
    rounds: int = 0
    latency_s: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    tool_arguments: list[dict[str, Any]] = field(default_factory=list)
    correct_tool: bool | None = None

    grounded_score: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    fabricated_specifics: bool | None = None
    refused: bool | None = None

    error: str | None = None


# --------------------------------------------------------------------------------------
# Trace consumption
# --------------------------------------------------------------------------------------


async def consume_trace(url: str, sink: list[dict[str, Any]], stop: asyncio.Event) -> None:
    """Mirror the live trace stream into `sink`, so tool selection can be observed."""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{url}/trace") as response:
                if response.status_code != 200:
                    return
                async for line in response.aiter_lines():
                    if stop.is_set():
                        return
                    if line.startswith("data: "):
                        try:
                            sink.append(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            continue
    except httpx.HTTPError:
        return  # tool-selection metrics degrade to "unmeasured"; everything else survives


def tool_calls_in(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls from one turn. Uses the request id so a stray event cannot leak in."""
    calls = [e for e in events if e.get("kind") == "tools/call"]
    if not calls:
        return []
    request_ids = {call.get("req") for call in calls}
    if len(request_ids) > 1:
        # Another client was chatting at the same time. Keep the busiest turn and warn.
        busiest = max(request_ids, key=lambda rid: sum(1 for c in calls if c.get("req") == rid))
        print(f"    ! concurrent traffic on /trace; keeping request {busiest}", file=sys.stderr)
        calls = [call for call in calls if call.get("req") == busiest]
    return [call["payload"] for call in calls]


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply, fenced or not."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text[:200]}")
    return json.loads(match.group(0))


async def judge(
    client: AsyncOpenAI, model: str, question: str, answer: str, evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {
        "QUESTION": question,
        "ANSWER": answer,
        "EVIDENCE": evidence if evidence else "No tools returned any evidence.",
    }
    response = await client.chat.completions.create(
        model=model,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=900,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
        ],
    )
    verdict = extract_json(response.choices[0].message.content or "")

    supported = verdict.get("supported_claims") or []
    unsupported = verdict.get("unsupported_claims") or []
    total = len(supported) + len(unsupported)
    if "grounded_score" in verdict:
        score = float(verdict["grounded_score"])
    else:
        score = 1.0 if total == 0 else len(supported) / total
    verdict["grounded_score"] = max(0.0, min(1.0, score))
    verdict["unsupported_claims"] = unsupported
    return verdict


# --------------------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------------------


async def run_case(
    case: dict[str, Any],
    session: ClientSession,
    http: httpx.AsyncClient,
    url: str,
    events: list[dict[str, Any]],
    judge_client: AsyncOpenAI,
    judge_model: str,
) -> CaseResult:
    result = CaseResult(
        question=case["question"],
        answerable=bool(case.get("answerable", True)),
        expected_tool=case.get("expected_tool"),
        expected_chunk_ids=list(case.get("expected_chunk_ids") or []),
        notes=case.get("notes", ""),
    )

    # 1. Retrieval, measured directly against the retriever rather than through the agent.
    if result.expected_chunk_ids:
        retrieval = await session.call_tool(
            "search_policies", {"query": result.question, "top_k": TOP_K}
        )
        payload = json.loads(retrieval.content[0].text)
        if payload.get("ok"):
            result.retrieved_chunk_ids = [chunk["chunk_id"] for chunk in payload["chunks"]]
            found = set(result.retrieved_chunk_ids) & set(result.expected_chunk_ids)
            result.recall_at_k = len(found) / len(result.expected_chunk_ids)
            result.hit_at_k = bool(found)
        else:
            result.error = f"retrieval failed: {payload.get('error')}"

    # 2. The agent's own answer.
    mark = len(events)
    started = time.perf_counter()
    try:
        response = await http.post(f"{url}/chat", json={"message": result.question})
    except httpx.HTTPError as exc:
        result.error = f"chat request failed: {type(exc).__name__}: {exc}"
        return result
    result.latency_s = round(time.perf_counter() - started, 2)

    if response.status_code == 429:
        raise RateLimited(response.json().get("detail", "rate limited"))
    if response.status_code != 200:
        result.error = f"chat returned {response.status_code}: {response.text[:200]}"
        return result

    body = response.json()
    result.answer = body["answer"]
    result.rounds = body["rounds"]

    # 3. Which tools it chose, read off the live trace.
    await asyncio.sleep(TRACE_SETTLE_S)
    calls = tool_calls_in(events[mark:])
    result.tool_arguments = calls
    result.tools_called = [call["tool"] for call in calls]
    if result.expected_tool:
        result.correct_tool = result.expected_tool in result.tools_called

    # 4. Replay those exact calls to reconstruct the evidence the agent actually had.
    evidence: list[dict[str, Any]] = []
    for call in calls:
        try:
            replay = await session.call_tool(call["tool"], call["arguments"])
            evidence.append({
                "tool": call["tool"],
                "arguments": call["arguments"],
                "result": json.loads(replay.content[0].text) if replay.content else None,
            })
        except Exception as exc:  # noqa: BLE001 - a replay failure must not abort the run
            evidence.append({"tool": call["tool"], "error": f"{type(exc).__name__}: {exc}"})

    # 5. Judge.
    try:
        verdict = await judge(judge_client, judge_model, result.question, result.answer, evidence)
    except (OpenAIError, ValueError, json.JSONDecodeError) as exc:
        result.error = f"judge failed: {type(exc).__name__}: {exc}"
        return result

    result.grounded_score = verdict["grounded_score"]
    result.unsupported_claims = verdict["unsupported_claims"]
    result.fabricated_specifics = bool(verdict.get("fabricated_specifics"))
    result.refused = bool(verdict.get("refused"))
    return result


class RateLimited(RuntimeError):
    """The target refused further messages; the run cannot continue meaningfully."""


# --------------------------------------------------------------------------------------
# Aggregation and output
# --------------------------------------------------------------------------------------


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    done = [r for r in results if r.error is None]
    answerable = [r for r in done if r.answerable]
    unanswerable = [r for r in done if not r.answerable]

    recalls = [r.recall_at_k for r in done if r.recall_at_k is not None]
    hits = [r.hit_at_k for r in done if r.hit_at_k is not None]
    tool_checks = [r.correct_tool for r in done if r.correct_tool is not None]
    grounded = [r.grounded_score for r in answerable if r.grounded_score is not None]

    hallucinated = [
        r for r in answerable
        if r.fabricated_specifics is not None and (r.fabricated_specifics or r.unsupported_claims)
    ]
    judged_answerable = [r for r in answerable if r.fabricated_specifics is not None]
    refused_unanswerable = [r for r in unanswerable if r.refused]
    refused_answerable = [r for r in answerable if r.refused]

    return {
        "cases": len(results),
        "completed": len(done),
        "failed": len(results) - len(done),
        "retrieval_recall_at_5": mean([float(value) for value in recalls]),
        "retrieval_hit_rate_at_5": mean([float(hit) for hit in hits]),
        "correct_tool_rate": mean([float(ok) for ok in tool_checks]),
        "avg_rounds": mean([float(r.rounds) for r in done]),
        "avg_latency_s": mean([r.latency_s for r in done]),
        "groundedness": mean([float(score) for score in grounded]),
        "hallucination_rate": (
            round(len(hallucinated) / len(judged_answerable), 3) if judged_answerable else None
        ),
        "refusal_accuracy": (
            round(len(refused_unanswerable) / len(unanswerable), 3) if unanswerable else None
        ),
        "false_refusal_rate": (
            round(len(refused_answerable) / len(answerable), 3) if answerable else None
        ),
    }


ROWS = [
    ("Retrieval recall@5", "retrieval_recall_at_5", "higher", "Fraction of expected chunks in the top 5"),
    ("Retrieval hit rate@5", "retrieval_hit_rate_at_5", "higher", "At least one expected chunk retrieved"),
    ("Correct-tool rate", "correct_tool_rate", "higher", "Expected tool actually called"),
    ("Groundedness", "groundedness", "higher", "Supported claims / all claims, judged"),
    ("Hallucination rate", "hallucination_rate", "lower", "Answers with an unsupported or invented claim"),
    ("Refusal accuracy", "refusal_accuracy", "higher", "Undocumented topics correctly declined"),
    ("False-refusal rate", "false_refusal_rate", "lower", "Answerable questions wrongly declined"),
    ("Average rounds", "avg_rounds", "-", "Tool-calling rounds per answer"),
    ("Average latency", "avg_latency_s", "-", "Seconds per answer, end to end"),
]


def format_value(key: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if key in {"avg_rounds"}:
        return f"{value:.2f}"
    if key in {"avg_latency_s"}:
        return f"{value:.1f}s"
    return f"{value:.0%}" if value <= 1 else f"{value:.2f}"


def print_table(summary: dict[str, Any], meta: dict[str, Any]) -> None:
    print()
    print(f"  target        {meta['target_url']}")
    print(f"  answer model  {meta['answer_model']}")
    print(f"  judge model   {meta['judge_model']} (temperature {JUDGE_TEMPERATURE})")
    print(f"  cases         {summary['completed']} completed, {summary['failed']} failed")
    print()
    print(f"  {'metric':<24}{'value':>9}   {'better'}")
    print(f"  {'-' * 24}{'-' * 9}   {'-' * 6}")
    for label, key, direction, _ in ROWS:
        print(f"  {label:<24}{format_value(key, summary[key]):>9}   {direction}")
    print()


def markdown_table(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [
        f"_Last run {meta['timestamp']} against `{meta['target_url']}`._",
        "",
        f"Answers: `{meta['answer_model']}` · Judge: `{meta['judge_model']}` at temperature "
        f"{JUDGE_TEMPERATURE} · {summary['completed']} of {summary['cases']} cases completed.",
        "",
        "| Metric | Value | Better | What it measures |",
        "|---|---|---|---|",
    ]
    for label, key, direction, description in ROWS:
        lines.append(f"| {label} | {format_value(key, summary[key])} | {direction} | {description} |")
    lines += ["", f"Full history in [`evals/history/`](evals/history/). Re-run with `python evals/run_evals.py --url <target>`."]
    return "\n".join(lines)


def rewrite_readme(table: str) -> bool:
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    if MARKER_START not in text or MARKER_END not in text:
        print("! README markers not found; skipping README update", file=sys.stderr)
        return False
    pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END), re.DOTALL
    )
    README.write_text(
        pattern.sub(f"{MARKER_START}\n\n{table}\n\n{MARKER_END}", text), encoding="utf-8"
    )
    return True


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a running support copilot.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="target base URL")
    parser.add_argument("--dataset", default=str(ROOT / "evals" / "dataset.jsonl"))
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--no-readme", action="store_true", help="do not rewrite the README table")
    options = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; the judge needs it.", file=sys.stderr)
        return 1

    judge_model = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    base_url = os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1")
    url = options.url.rstrip("/")

    cases = [json.loads(line) for line in Path(options.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if options.limit:
        cases = cases[: options.limit]

    async with httpx.AsyncClient(timeout=180) as http:
        try:
            health = (await http.get(f"{url}/health")).json()
        except httpx.HTTPError as exc:
            print(f"Cannot reach {url}/health: {exc}", file=sys.stderr)
            return 1
        answer_model = health.get("dependencies", {}).get("llm", {}).get("model", "unknown")
        if health.get("status") != "ok":
            print(f"! target reports status={health.get('status')}; results may be skewed", file=sys.stderr)

        events: list[dict[str, Any]] = []
        stop = asyncio.Event()
        trace_task = asyncio.create_task(consume_trace(url, events, stop))
        await asyncio.sleep(0.5)  # let the stream attach before the first question

        judge_client = AsyncOpenAI(base_url=base_url, api_key=token, timeout=120)
        results: list[CaseResult] = []
        rate_limited = False

        async with streamablehttp_client(f"{url}/mcp/") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                for index, case in enumerate(cases, start=1):
                    print(f"[{index}/{len(cases)}] {case['question'][:70]}")
                    try:
                        result = await run_case(
                            case, session, http, url, events, judge_client, judge_model
                        )
                    except RateLimited as exc:
                        print(f"\nStopped: the target rate-limited this run ({exc}).", file=sys.stderr)
                        print(
                            "Raise PER_IP_HOURLY_CAP and DAILY_MESSAGE_CAP on the target, or use\n"
                            "--limit to stay under the cap.",
                            file=sys.stderr,
                        )
                        rate_limited = True
                        break
                    results.append(result)
                    flag = "!" if result.error else " "
                    print(
                        f"   {flag} rounds={result.rounds} tools={','.join(result.tools_called) or '-'} "
                        f"grounded={result.grounded_score} refused={result.refused}"
                        + (f" error={result.error}" if result.error else "")
                    )

        stop.set()
        trace_task.cancel()
        await judge_client.close()

    if not results:
        print("No cases completed.", file=sys.stderr)
        return 1

    summary = summarise(results)
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "target_url": url,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "judge_temperature": JUDGE_TEMPERATURE,
        "dataset": Path(options.dataset).name,
        "partial_run": rate_limited or bool(options.limit),
    }

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = HISTORY_DIR / f"{stamp}.json"
    out_path.write_text(
        json.dumps({"meta": meta, "summary": summary, "cases": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )

    print_table(summary, meta)
    print(f"  written to {out_path.relative_to(ROOT)}")

    # A partial run would overwrite a full run's numbers with a smaller sample.
    if options.no_readme or meta["partial_run"]:
        if meta["partial_run"] and not options.no_readme:
            print("  README not updated: partial run")
    elif rewrite_readme(markdown_table(summary, meta)):
        print("  README table updated")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
