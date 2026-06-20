from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lexicon.models import Passage

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".txt",
}


def iter_document_paths(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if _is_supported_file(child):
                    yield child
        elif _is_supported_file(path):
            yield path


def load_passages(path: Path, max_chars: int = 1400, overlap_chars: int = 180) -> list[Passage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = _paragraph_blocks(text)
    passages: list[Passage] = []
    current: list[tuple[int, int, str]] = []
    current_size = 0

    for start_line, end_line, block in blocks:
        if current and current_size + len(block) + 2 > max_chars:
            passages.append(_passage_from_blocks(path, current))
            current = _overlap_tail(current, overlap_chars)
            current_size = sum(len(item[2]) for item in current)

        current.append((start_line, end_line, block))
        current_size += len(block) + 2

    if current:
        passages.append(_passage_from_blocks(path, current))

    return passages


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _paragraph_blocks(text: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    active: list[str] = []
    active_start = 1

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not active:
                active_start = line_number
            active.append(line.rstrip())
            continue

        if active:
            blocks.append((active_start, line_number - 1, "\n".join(active).strip()))
            active = []

    if active:
        blocks.append((active_start, active_start + len(active) - 1, "\n".join(active).strip()))

    return blocks


def _passage_from_blocks(path: Path, blocks: list[tuple[int, int, str]]) -> Passage:
    start_line = blocks[0][0]
    end_line = blocks[-1][1]
    body = "\n\n".join(block[2] for block in blocks).strip()
    return Passage(doc_path=path, locator=f"lines {start_line}-{end_line}", text=body)


def _overlap_tail(blocks: list[tuple[int, int, str]], target_chars: int) -> list[tuple[int, int, str]]:
    tail: list[tuple[int, int, str]] = []
    total = 0
    for block in reversed(blocks):
        if tail and total >= target_chars:
            break
        tail.append(block)
        total += len(block[2])
    return list(reversed(tail))

