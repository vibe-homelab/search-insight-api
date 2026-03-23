"""Client for the Language Insight API embedding endpoint."""

from __future__ import annotations

import httpx

MAX_BATCH_SIZE = 100


class EmbeddingClient:
    """Generate embeddings by calling the Language Insight API."""

    def __init__(self, endpoint: str, model: str) -> None:
        self.endpoint = endpoint
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0, read=30.0)
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each input text.

        Calls the Language Insight API ``/v1/embeddings`` endpoint which
        follows the OpenAI-compatible format.  Large inputs are automatically
        split into batches of ``MAX_BATCH_SIZE``.
        """
        if len(texts) <= MAX_BATCH_SIZE:
            return await self._embed_batch(texts)

        # Split into batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch to the embedding endpoint."""
        client = await self._get_client()
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
