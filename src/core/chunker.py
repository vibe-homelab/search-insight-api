"""Recursive character text splitting for document chunking."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class TextChunker:
    """Split text into overlapping chunks suitable for embedding.

    Uses recursive character text splitting: tries to split on the first
    separator that produces chunks within *max_chunk_size*, then falls back
    to the next separator, and so on.
    """

    max_chunk_size: int = 1000
    overlap: int = 200
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "]
    )

    def chunk(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Return a list of chunk dicts with ``text``, ``metadata``, and ``chunk_id``."""
        if metadata is None:
            metadata = {}

        raw_chunks = self._split_recursive(text, list(self.separators))
        merged = self._merge_with_overlap(raw_chunks)

        results: list[dict] = []
        for idx, chunk_text in enumerate(merged):
            results.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": {**metadata, "chunk_index": idx},
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the ordered list of separators."""
        if len(text) <= self.max_chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # No separators left — hard-split by character count.
            return self._hard_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the single part itself exceeds the limit, recurse deeper.
                if len(part) > self.max_chunk_size:
                    chunks.extend(self._split_recursive(part, remaining_seps))
                else:
                    current = part
                    continue
                current = ""

        if current and current.strip():
            chunks.append(current)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Character-level fallback split."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end
        return chunks

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Re-merge chunks so that consecutive pieces share *overlap* characters."""
        if not chunks or self.overlap <= 0:
            return chunks

        merged: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged.append(chunk)
            else:
                # Prepend the tail of the previous chunk as overlap context.
                prev = merged[i - 1]
                overlap_text = prev[-self.overlap :] if len(prev) > self.overlap else prev
                combined = overlap_text + chunk
                # Trim back to max size if needed.
                if len(combined) > self.max_chunk_size:
                    combined = combined[: self.max_chunk_size]
                merged.append(combined)
        return merged
