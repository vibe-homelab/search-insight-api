"""Client for the Language Insight API embedding endpoint."""

from __future__ import annotations

import httpx


class EmbeddingClient:
    """Generate embeddings by calling the Language Insight API."""

    def __init__(self, endpoint: str, model: str) -> None:
        self.endpoint = endpoint
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each input text.

        Calls the Language Insight API ``/v1/embeddings`` endpoint which
        follows the OpenAI-compatible format.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.endpoint,
                json={
                    "input": texts,
                    "model": self.model,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    async def embed_single(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single text."""
        results = await self.embed([text])
        return results[0]
