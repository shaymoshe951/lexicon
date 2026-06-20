from pathlib import Path

from lexicon.ingest import iter_document_paths, load_passages


def test_load_passages_keeps_line_citations(tmp_path: Path) -> None:
    document = tmp_path / "policy.md"
    document.write_text("Title\n\nRemote work is allowed.\n\nExpenses need approval.\n", encoding="utf-8")

    passages = load_passages(document, max_chars=80)

    assert passages
    assert passages[0].citation == "policy.md (lines 1-5)"
    assert "Remote work is allowed." in passages[0].text


def test_iter_document_paths_filters_supported_files(tmp_path: Path) -> None:
    supported = tmp_path / "notes.txt"
    ignored = tmp_path / "image.png"
    supported.write_text("hello", encoding="utf-8")
    ignored.write_bytes(b"png")

    paths = list(iter_document_paths([tmp_path]))

    assert paths == [supported]

