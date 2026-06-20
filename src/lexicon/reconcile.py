from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Sequence

from lexicon.models import Reconciliation, SearchResult

NEGATION_CUES = {"not", "never", "no", "cannot", "can't", "must not", "prohibited", "forbidden"}
AFFIRMATION_CUES = {"can", "may", "must", "should", "required", "allowed", "supported"}
OPPOSITION_PAIRS = [
    ("allowed", "prohibited"),
    ("required", "optional"),
    ("enabled", "disabled"),
    ("included", "excluded"),
    ("supported", "unsupported"),
    ("must", "must not"),
    ("can", "cannot"),
]


def appears_contested(results: Sequence[SearchResult]) -> bool:
    if len(results) < 2:
        return False

    normalized = [f" {result.passage.text.lower()} " for result in results]
    for left, right in OPPOSITION_PAIRS:
        if any(left in text for text in normalized) and any(right in text for text in normalized):
            return True

    polarity = [_polarity(text) for text in normalized]
    return any(value > 0 for value in polarity) and any(value < 0 for value in polarity)


def reconcile(query: str, results: Sequence[SearchResult]) -> Reconciliation:
    command = os.environ.get("LEXICON_LLM_COMMAND")
    if not command:
        return Reconciliation(
            contested=True,
            explanation=(
                "Potentially divergent passages were found, but LEXICON_LLM_COMMAND is not set. "
                "No LLM call was made; inspect the cited passages above."
            ),
        )

    payload = {
        "task": (
            "You are Lexicon's reconciliation layer. Use only the supplied passages. "
            "Do not answer the user's question directly. Surface divergent positions, "
            "keep both sides, and cite each position with its passage citation."
        ),
        "query": query,
        "passages": [
            {
                "citation": result.passage.citation,
                "text": result.passage.text,
            }
            for result in results
        ],
    }
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        return Reconciliation(
            contested=True,
            explanation=(
                "Reconciliation command failed. No fallback answer was synthesized.\n"
                f"{completed.stderr.strip()}"
            ).strip(),
        )

    return Reconciliation(contested=True, explanation=completed.stdout.strip())


def _polarity(text: str) -> int:
    score = 0
    for cue in AFFIRMATION_CUES:
        if f" {cue} " in text:
            score += 1
    for cue in NEGATION_CUES:
        if f" {cue} " in text:
            score -= 1
    return score

