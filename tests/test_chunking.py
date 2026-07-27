"""Chunking behaviour: section boundaries, anchors, the token budget, and overlap."""

from __future__ import annotations

from pathlib import Path

from scripts.index_docs import (
    MAX_TOKENS,
    OVERLAP_RATIO,
    anchor_for,
    chunk_markdown,
    est_tokens,
    split_long,
)

SAMPLE = """# Sample Policy

Preamble that belongs to no section.

## First section

Body of the first section.

## Second section, with punctuation!

Body of the second section.
"""


def test_one_chunk_per_short_section() -> None:
    chunks = chunk_markdown(SAMPLE, "sample.md")
    assert [chunk.section for chunk in chunks] == [
        "First section",
        "Second section, with punctuation!",
    ]


def test_preamble_before_first_heading_is_dropped() -> None:
    # Text above the first `##` has no section to cite, so it is not indexed.
    chunks = chunk_markdown(SAMPLE, "sample.md")
    assert all("Preamble" not in chunk.text for chunk in chunks)


def test_chunk_carries_doc_title_and_heading() -> None:
    chunk = chunk_markdown(SAMPLE, "sample.md")[0]
    assert chunk.text.startswith("# Sample Policy\n## First section")
    assert chunk.doc == "sample.md"
    assert chunk.doc_title == "Sample Policy"


def test_anchor_is_a_github_style_slug() -> None:
    chunks = chunk_markdown(SAMPLE, "sample.md")
    assert chunks[0].anchor == "first-section"
    assert chunks[1].anchor == "second-section-with-punctuation"
    assert anchor_for("COD order limits") == "cod-order-limits"


def test_chunk_ids_are_stable_and_unique() -> None:
    first = chunk_markdown(SAMPLE, "sample.md")
    second = chunk_markdown(SAMPLE, "sample.md")
    ids = [chunk.chunk_id for chunk in first]
    assert ids == [chunk.chunk_id for chunk in second]
    assert len(ids) == len(set(ids))
    assert ids[0] == "sample.md#first-section#0"


def test_empty_sections_produce_no_chunks() -> None:
    assert chunk_markdown("# T\n\n## Empty\n\n## Also empty\n", "t.md") == []


def test_short_text_is_not_split() -> None:
    assert split_long("one short paragraph") == ["one short paragraph"]


def _long_section(paragraphs: int = 40) -> str:
    return "\n\n".join(
        f"Paragraph {i} discusses topic {i} in enough words to consume a real budget."
        for i in range(paragraphs)
    )


def test_long_section_is_split() -> None:
    parts = split_long(_long_section())
    assert len(parts) > 1
    assert all(est_tokens(part) <= MAX_TOKENS for part in parts)


def test_split_parts_overlap() -> None:
    parts = split_long(_long_section())
    for previous, following in zip(parts, parts[1:]):
        tail = previous.split()[-5:]
        assert " ".join(tail) in following, "consecutive parts should share an overlap"


def test_overlap_is_roughly_the_configured_ratio() -> None:
    parts = split_long(_long_section())
    budget_words = MAX_TOKENS * 0.75
    previous_words = parts[0].split()
    shared = 0
    for size in range(1, len(previous_words)):
        if " ".join(previous_words[-size:]) in parts[1]:
            shared = size
    assert 0.05 * budget_words <= shared <= 0.30 * budget_words, (
        f"shared {shared} words, expected around {OVERLAP_RATIO:.0%} of {budget_words:.0f}"
    )


def test_unsplittable_paragraph_still_fits_budget() -> None:
    # A bullet list has no blank lines, so paragraph splitting alone cannot break it.
    bullets = "\n".join(f"- item {i} with several words of description here" for i in range(120))
    parts = split_long(bullets)
    assert len(parts) > 1
    assert all(est_tokens(part) <= MAX_TOKENS for part in parts)


def test_real_corpus_respects_the_budget(docs_dir: Path) -> None:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))

    assert len(chunks) > 50
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    over_budget = [(chunk.chunk_id, chunk.est_tokens) for chunk in chunks if chunk.est_tokens > MAX_TOKENS]
    assert not over_budget, f"chunks over {MAX_TOKENS} tokens: {over_budget}"
    assert all(chunk.anchor and chunk.section and chunk.doc for chunk in chunks)
