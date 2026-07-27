"""Chunk the DesiCart policy docs, embed them locally, and upsert to Qdrant Cloud.

Runs on a laptop, never in the container: sentence-transformers is a heavy dependency and
is imported lazily inside embed_texts() so that `chunk_markdown` stays importable from
tests without pulling in torch.

Usage:
    python scripts/index_docs.py                 # chunk + embed + upsert
    python scripts/index_docs.py --dry-run       # chunk only, print the plan, no network

Environment (see .env.example):
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBED_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

# Chunking targets. Tokens are estimated, not tokenised - a real tokeniser here would mean
# shipping one more dependency for a heuristic that only decides where to split.
MAX_TOKENS = 400
OVERLAP_RATIO = 0.15
WORDS_PER_TOKEN = 0.75  # English prose averages ~1.33 tokens per word

# bge models expect this prefix on the QUERY side only; passages are embedded bare.
# rag.py must use the same prefix at query time or recall drops.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION = "desicart_policies"

# Namespace for deterministic point IDs, so re-running replaces points instead of adding.
POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    doc_title: str
    section: str
    anchor: str
    text: str
    n_words: int
    est_tokens: int


def anchor_for(heading: str) -> str:
    """GitHub-style heading anchor - must match the one make_corpus.py writes."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s_]+", "-", slug).strip("-")


def est_tokens(text: str) -> int:
    return int(len(text.split()) / WORDS_PER_TOKEN)


def _atomic_blocks(text: str, max_words: int) -> list[str]:
    """Break text into pieces that each fit the budget: paragraphs, then lines, then words.

    A long bullet list has no blank lines in it, so paragraph splitting alone leaves an
    oversized block; falling back to lines keeps list items intact.
    """
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            blocks.append(para)
            continue
        buffer: list[str] = []
        buffered_words = 0
        for line in para.splitlines():
            line_words = len(line.split())
            if line_words > max_words:
                if buffer:
                    blocks.append("\n".join(buffer))
                    buffer, buffered_words = [], 0
                words = line.split()
                for start in range(0, len(words), max_words):
                    blocks.append(" ".join(words[start : start + max_words]))
                continue
            if buffer and buffered_words + line_words > max_words:
                blocks.append("\n".join(buffer))
                buffer, buffered_words = [], 0
            buffer.append(line)
            buffered_words += line_words
        if buffer:
            blocks.append("\n".join(buffer))
    return blocks


def split_long(text: str, max_tokens: int = MAX_TOKENS, overlap: float = OVERLAP_RATIO) -> list[str]:
    """Split on the largest safe boundary, carrying ~15% of the previous part as overlap."""
    if est_tokens(text) <= max_tokens:
        return [text]

    # Headroom for the doc title and section heading prepended in chunk_markdown().
    max_words = int(max_tokens * WORDS_PER_TOKEN) - 15
    overlap_words = int(max_words * overlap)

    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    # Blocks are capped below the budget so a block landing on top of an overlap tail
    # still fits.
    for block in _atomic_blocks(text, max_words - overlap_words):
        block_words = len(block.split())
        if current and current_words + block_words > max_words:
            joined = "\n\n".join(current)
            parts.append(joined)
            tail = joined.split()[-overlap_words:] if overlap_words else []
            current = [" ".join(tail)] if tail else []
            current_words = len(tail)
        current.append(block)
        current_words += block_words

    if current and "".join(current).strip():
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(markdown: str, doc: str) -> list[Chunk]:
    """One chunk per `##` section, splitting sections over MAX_TOKENS.

    Each chunk text is prefixed with the doc title and section heading so an embedded
    chunk carries its own context and a retrieved chunk reads standalone in a citation.
    """
    lines = markdown.splitlines()
    doc_title = doc
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break

    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    buffer: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, buffer))
            current_heading = line[3:].strip()
            buffer = []
        elif current_heading is not None:
            buffer.append(line)
    if current_heading is not None:
        sections.append((current_heading, buffer))

    chunks: list[Chunk] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        anchor = anchor_for(heading)
        for i, part in enumerate(split_long(body)):
            text = f"# {doc_title}\n## {heading}\n\n{part.strip()}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}#{anchor}#{i}",
                    doc=doc,
                    doc_title=doc_title,
                    section=heading,
                    anchor=anchor,
                    text=text,
                    n_words=len(text.split()),
                    est_tokens=est_tokens(text),
                )
            )
    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """Local embedding. Heavy import stays inside the function on purpose."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return [v.tolist() for v in vectors]


def upsert(chunks: list[Chunk], vectors: list[list[float]], collection: str) -> dict[str, object]:
    from qdrant_client import QdrantClient  # noqa: PLC0415
    from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: PLC0415

    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url:
        raise SystemExit("QDRANT_URL is not set. Copy .env.example to .env and fill it in.")

    client = QdrantClient(url=url, api_key=api_key, timeout=60)
    dim = len(vectors[0])

    # Recreate rather than diff: the corpus is small and a full rebuild is the only way to
    # be sure a removed section leaves no orphan point behind.
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, chunk.chunk_id)),
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "doc": chunk.doc,
                "doc_title": chunk.doc_title,
                "section": chunk.section,
                "anchor": chunk.anchor,
                "text": chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)

    info = client.get_collection(collection)
    return {
        "collection": collection,
        "status": str(info.status),
        "points": info.points_count,
        "dim": dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DesiCart policy docs into Qdrant.")
    parser.add_argument("--docs-dir", default="data/docs")
    parser.add_argument("--manifest", default="data/chunks.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="chunk only; no model, no network")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass  # .env is a convenience; real env vars work fine without python-dotenv

    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.is_dir():
        raise SystemExit(f"{docs_dir} not found. Run scripts/make_corpus.py first.")

    chunks = load_chunks(docs_dir)
    if not chunks:
        raise SystemExit(f"No markdown found in {docs_dir}.")

    # The manifest gives evals a stable list of chunk_ids to write expectations against.
    manifest_path = Path(args.manifest).resolve()
    manifest_path.write_text(
        "\n".join(json.dumps(asdict(c)) for c in chunks) + "\n", encoding="utf-8"
    )

    tokens = [c.est_tokens for c in chunks]
    print(f"docs      : {len(set(c.doc for c in chunks))}")
    print(f"chunks    : {len(chunks)}")
    print(f"tokens    : avg {sum(tokens) / len(tokens):.0f}, min {min(tokens)}, max {max(tokens)}")
    print(f"split     : {sum(1 for c in chunks if c.chunk_id.endswith('#1'))} section(s) needed a split")
    print(f"manifest  : {manifest_path}")

    if args.dry_run:
        print("\n--dry-run: stopping before embedding.")
        for c in chunks[:5]:
            print(f"  {c.chunk_id:<50} {c.est_tokens:>4} tok  {c.section}")
        return

    model_name = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    collection = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    print(f"\nembedding with {model_name} ...")
    vectors = embed_texts([c.text for c in chunks], model_name)

    result = upsert(chunks, vectors, collection)
    print(
        f"\nqdrant    : collection={result['collection']} status={result['status']} "
        f"points={result['points']} dim={result['dim']}"
    )


if __name__ == "__main__":
    main()
