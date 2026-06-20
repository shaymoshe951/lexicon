from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    """A retrievable source passage with enough location data to cite it."""

    doc_path: Path
    locator: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_path.name} ({self.locator})"


@dataclass(frozen=True)
class SearchResult:
    passage: Passage
    score: float


@dataclass(frozen=True)
class Reconciliation:
    contested: bool
    explanation: str

