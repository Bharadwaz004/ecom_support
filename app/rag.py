"""Retrieval: embed a query via the HF inference API, then search Qdrant over REST.

Deliberately no `sentence-transformers` and no `qdrant-client` here. The embedding model
is the same one scripts/index_docs.py used at index time, but called remotely; Qdrant is
reached with plain httpx because the app only needs two endpoints and qdrant-client would
drag grpcio and protobuf into a 512MB container for no benefit.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from app.config import get_settings

# Must match scripts/index_docs.py. bge models want this prefix on queries only; passages
# were embedded bare. Dropping it costs several points of recall.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_QUERY_CHARS = 512

_client: httpx.AsyncClient | None = None


class RetrievalError(RuntimeError):
    """Anything that went wrong reaching the embedding API or Qdrant."""


def http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=get_settings().request_timeout_s)
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _flatten_embedding(payload: Any) -> list[float]:
    """Normalise the several shapes the feature-extraction endpoint can return.

    Depending on the backend the response is a bare vector, a batch of one vector, or
    token-level vectors. bge uses CLS pooling, so the first token vector is the right one
    to take in the token-level case.
    """
    node = payload
    depth = 0
    while isinstance(node, list) and node and isinstance(node[0], list):
        node = node[0]
        depth += 1
    if not isinstance(node, list) or not node or not isinstance(node[0], (int, float)):
        raise RetrievalError(f"Unexpected embedding response shape (depth {depth}).")
    return [float(x) for x in node]


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise RetrievalError("Embedding API returned a zero vector.")
    return [x / norm for x in vector]


async def embed_query(text: str) -> list[float]:
    """Embed one query string. The single place the app talks to the embedding model."""
    settings = get_settings()
    text = text.strip()[:MAX_QUERY_CHARS]
    if not text:
        raise RetrievalError("Cannot embed an empty query.")

    try:
        response = await http_client().post(
            settings.embed_endpoint,
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            json={"inputs": BGE_QUERY_PREFIX + text},
        )
    except httpx.HTTPError as exc:
        raise RetrievalError(f"Embedding API unreachable: {type(exc).__name__}: {exc}") from exc

    if response.status_code == 503:
        raise RetrievalError(
            f"Embedding model {settings.embed_model} is loading on the provider. Retry shortly."
        )
    if response.status_code >= 400:
        raise RetrievalError(
            f"Embedding API returned {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RetrievalError("Embedding API returned a non-JSON body.") from exc

    return _l2_normalise(_flatten_embedding(payload))


def _qdrant_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    return headers


async def collection_info() -> dict[str, Any]:
    """Vector size, point count and status for the configured collection."""
    settings = get_settings()
    url = f"{settings.qdrant_base}/collections/{settings.qdrant_collection}"
    try:
        response = await http_client().get(url, headers=_qdrant_headers())
    except httpx.HTTPError as exc:
        raise RetrievalError(f"Qdrant unreachable: {type(exc).__name__}: {exc}") from exc

    if response.status_code == 404:
        raise RetrievalError(
            f"Qdrant collection '{settings.qdrant_collection}' does not exist. "
            "Run: python scripts/index_docs.py"
        )
    if response.status_code >= 400:
        raise RetrievalError(f"Qdrant returned {response.status_code}: {response.text[:200]}")

    result = response.json().get("result", {})
    vectors = result.get("config", {}).get("params", {}).get("vectors", {})
    # Unnamed vectors give {"size": n, ...}; named vectors give {"name": {"size": n, ...}}.
    if "size" not in vectors and vectors:
        vectors = next(iter(vectors.values()))
    return {
        "collection": settings.qdrant_collection,
        "status": result.get("status"),
        "points": result.get("points_count"),
        "dim": vectors.get("size"),
    }


async def assert_dimensions() -> dict[str, Any]:
    """Startup check: the runtime embedder and the indexed collection must agree.

    A silent mismatch here is the worst possible failure - Qdrant would either error per
    query or, with the wrong model at the same width, return confidently wrong chunks.
    """
    info = await collection_info()
    probe = await embed_query("dimension probe")
    if info["dim"] != len(probe):
        raise RetrievalError(
            f"Embedding dimension mismatch: {get_settings().embed_model} produces "
            f"{len(probe)} dims but collection '{info['collection']}' expects {info['dim']}. "
            "Re-index with the same model the app is configured to use."
        )
    return {**info, "embed_dim": len(probe)}


async def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search. Returns chunks ordered by descending cosine score."""
    settings = get_settings()
    vector = await embed_query(query)
    url = f"{settings.qdrant_base}/collections/{settings.qdrant_collection}/points/query"

    try:
        response = await http_client().post(
            url,
            headers=_qdrant_headers(),
            json={"query": vector, "limit": top_k, "with_payload": True},
        )
    except httpx.HTTPError as exc:
        raise RetrievalError(f"Qdrant unreachable: {type(exc).__name__}: {exc}") from exc

    if response.status_code >= 400:
        raise RetrievalError(f"Qdrant search returned {response.status_code}: {response.text[:200]}")

    points = response.json().get("result", {}).get("points", [])
    return [
        {
            "chunk_id": point.get("payload", {}).get("chunk_id"),
            "doc": point.get("payload", {}).get("doc"),
            "section": point.get("payload", {}).get("section"),
            "anchor": point.get("payload", {}).get("anchor"),
            "score": round(float(point.get("score", 0.0)), 4),
            "text": point.get("payload", {}).get("text", ""),
        }
        for point in points
    ]
