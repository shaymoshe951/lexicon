from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lexicon.ingest import iter_document_paths, load_passages
from lexicon.models import Passage, SearchResult

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)


class Embedder:
    name = "base"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Small local fallback. Prefer sentence-transformers for real semantic quality."""

    name = "hashing"

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_number, text in enumerate(texts):
            tokens = [token.lower() for token in TOKEN_RE.findall(text)]
            features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self.dimensions
                sign = 1.0 if value & 1 else -1.0
                rows[row_number, index] += sign

        return _normalize(rows)


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(backend: str = "auto", model_name: str | None = None) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()

    if backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            if backend == "sentence-transformers":
                raise
            return HashingEmbedder()

    raise ValueError(f"Unknown embedding backend: {backend}")


class LexiconIndex:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path.expanduser().resolve()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.index_path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def rebuild(self, source_paths: Sequence[Path], embedder: Embedder) -> int:
        passages: list[Passage] = []
        for document_path in iter_document_paths(source_paths):
            passages.extend(load_passages(document_path))

        with self.connection:
            self.connection.execute("delete from passages")
            self.connection.execute("delete from metadata")
            self.connection.execute(
                "insert into metadata(key, value) values (?, ?)",
                ("embedder", embedder.name),
            )

        if not passages:
            return 0

        vectors = embedder.embed([passage.text for passage in passages])
        with self.connection:
            self.connection.executemany(
                """
                insert into passages(doc_path, locator, text, embedding)
                values (?, ?, ?, ?)
                """,
                [
                    (
                        str(passage.doc_path),
                        passage.locator,
                        passage.text,
                        json.dumps(vector.tolist()),
                    )
                    for passage, vector in zip(passages, vectors)
                ],
            )
        return len(passages)

    def search(self, query: str, embedder: Embedder, limit: int = 5) -> list[SearchResult]:
        rows = self.connection.execute(
            "select doc_path, locator, text, embedding from passages"
        ).fetchall()
        if not rows:
            return []

        query_vector = embedder.embed([query])[0]
        scored: list[SearchResult] = []
        for row in rows:
            passage_vector = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
            if query_vector.shape != passage_vector.shape:
                raise ValueError(
                    "Query embedding dimensions do not match the index. "
                    "Use the same --backend/--model used for indexing, or rebuild the index."
                )
            score = float(np.dot(query_vector, passage_vector))
            scored.append(
                SearchResult(
                    passage=Passage(
                        doc_path=Path(row["doc_path"]),
                        locator=row["locator"],
                        text=row["text"],
                    ),
                    score=score,
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def _ensure_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                create table if not exists metadata (
                  key text primary key,
                  value text not null
                )
                """
            )
            self.connection.execute(
                """
                create table if not exists passages (
                  id integer primary key autoincrement,
                  doc_path text not null,
                  locator text not null,
                  text text not null,
                  embedding text not null
                )
                """
            )


def _normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms

