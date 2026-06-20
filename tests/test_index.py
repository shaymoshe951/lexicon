from pathlib import Path

import pytest

from lexicon.index import HashingEmbedder, LexiconIndex


def test_index_search_returns_cited_passages(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    policy = docs / "policy.txt"
    policy.write_text(
        "Remote work is allowed with manager approval.\n\nLunch expenses are not reimbursed.\n",
        encoding="utf-8",
    )

    index = LexiconIndex(tmp_path / "lexicon.db")
    try:
        count = index.rebuild([docs], HashingEmbedder())
        results = index.search("remote work manager approval", HashingEmbedder(), limit=1)
    finally:
        index.close()

    assert count == 1
    assert results[0].passage.doc_path == policy
    assert results[0].passage.locator == "lines 1-3"
    assert "Remote work" in results[0].passage.text


def test_search_rejects_embedding_dimension_mismatch(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.txt").write_text("Remote work is allowed.\n", encoding="utf-8")

    index = LexiconIndex(tmp_path / "lexicon.db")
    try:
        index.rebuild([docs], HashingEmbedder(dimensions=8))
        with pytest.raises(ValueError, match="do not match the index"):
            index.search("remote work", HashingEmbedder(dimensions=16), limit=1)
    finally:
        index.close()

