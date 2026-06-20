from pathlib import Path

from lexicon.models import Passage, SearchResult
from lexicon.reconcile import appears_contested, reconcile


def test_appears_contested_for_opposing_passages() -> None:
    results = [
        SearchResult(Passage(Path("a.md"), "lines 1-1", "Remote work is allowed."), 0.9),
        SearchResult(Passage(Path("b.md"), "lines 1-1", "Remote work is prohibited."), 0.8),
    ]

    assert appears_contested(results)


def test_reconcile_without_command_does_not_synthesize(monkeypatch) -> None:
    monkeypatch.delenv("LEXICON_LLM_COMMAND", raising=False)
    results = [
        SearchResult(Passage(Path("a.md"), "lines 1-1", "Remote work is allowed."), 0.9),
        SearchResult(Passage(Path("b.md"), "lines 1-1", "Remote work is prohibited."), 0.8),
    ]

    output = reconcile("Is remote work allowed?", results)

    assert output.contested
    assert "No LLM call was made" in output.explanation

