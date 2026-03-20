"""LanceDB vector store wrapper."""

from __future__ import annotations

import uuid
from pathlib import Path

import lancedb
import pyarrow as pa


class VectorStore:
    """Thin wrapper around LanceDB for collection-based vector storage."""

    def __init__(self, db_path: str, dimensions: int = 768) -> None:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.dimensions = dimensions

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, name: str) -> None:
        """Create an empty collection (LanceDB table) if it does not exist."""
        existing = self.list_collections()
        if name in existing:
            return

        schema = pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field("metadata", pa.utf8()),  # JSON-encoded
                pa.field(
                    "vector", pa.list_(pa.float32(), list_size=self.dimensions)
                ),
            ]
        )
        self.db.create_table(name, schema=schema)

    def delete_collection(self, name: str) -> None:
        """Drop a collection entirely."""
        self.db.drop_table(name, ignore_missing=True)

    def list_collections(self) -> list[str]:
        """Return the names of all existing collections."""
        return self.db.table_names()

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        collection: str,
        documents: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """Insert documents with their embeddings into *collection*.

        Each document dict must contain at least ``text``.  An optional
        ``metadata`` key (dict or JSON string) is stored alongside it.

        Returns the number of rows added.
        """
        import json

        table = self.db.open_table(collection)

        rows: list[dict] = []
        for doc, vec in zip(documents, embeddings):
            meta = doc.get("metadata", {})
            if isinstance(meta, dict):
                meta = json.dumps(meta, ensure_ascii=False)
            rows.append(
                {
                    "id": doc.get("chunk_id", str(uuid.uuid4())),
                    "text": doc["text"],
                    "metadata": meta,
                    "vector": vec,
                }
            )

        table.add(rows)
        return len(rows)

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[dict]:
        """Return the top-*limit* nearest documents in *collection*."""
        import json

        table = self.db.open_table(collection)
        results = table.search(query_embedding).limit(limit).to_list()

        out: list[dict] = []
        for row in results:
            meta = row.get("metadata", "{}")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            out.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": meta,
                    "score": float(row.get("_distance", 0.0)),
                }
            )
        return out
