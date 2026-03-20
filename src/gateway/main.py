"""Search Insight API — FastAPI gateway for semantic search and RAG."""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.core.chunker import TextChunker
from src.core.config import load_config
from src.core.embeddings import EmbeddingClient
from src.core.vectorstore import VectorStore

logger = logging.getLogger("search-insight-api")

# ---------------------------------------------------------------------------
# Global singletons (initialised at startup)
# ---------------------------------------------------------------------------
config = load_config()
embedding_client: EmbeddingClient | None = None
vector_store: VectorStore | None = None
chunker: TextChunker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_client, vector_store, chunker
    embedding_client = EmbeddingClient(
        endpoint=config.embedding.endpoint,
        model=config.embedding.model,
    )
    vector_store = VectorStore(
        db_path=config.storage.path,
        dimensions=config.embedding.dimensions,
    )
    chunker = TextChunker(
        max_chunk_size=config.chunking.max_chunk_size,
        overlap=config.chunking.overlap,
    )
    logger.info("Search Insight API started on port %s", config.gateway.port)
    yield
    logger.info("Search Insight API shutting down")


app = FastAPI(
    title="Search Insight API",
    description="Semantic search and RAG API for Vibe Homelab",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional API-key auth
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(api_key_header)):
    if config.gateway.api_key:
        if not key or key != config.gateway.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class DocumentInput(BaseModel):
    text: str = Field(..., max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[DocumentInput]


class IndexResponse(BaseModel):
    indexed: int
    collection: str


class SearchRequest(BaseModel):
    query: str
    collection: str
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    collection: str


class RAGRequest(BaseModel):
    query: str
    collection: str
    model: str = "llm-fast"
    limit: int = Field(default=5, ge=1, le=20)
    system_prompt: str | None = None


class RAGSource(BaseModel):
    id: str
    text: str
    score: float


class RAGResponse(BaseModel):
    answer: str
    sources: list[RAGSource]
    model: str
    query: str


class CollectionCreateRequest(BaseModel):
    name: str


class CollectionInfo(BaseModel):
    name: str


class CollectionsListResponse(BaseModel):
    collections: list[CollectionInfo]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Collection endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/collections", response_model=CollectionInfo, dependencies=[Security(verify_api_key)])
def _validate_collection_name(name: str) -> None:
    if not re.match(r'^[a-zA-Z0-9_-]{1,100}$', name):
        raise HTTPException(
            status_code=400,
            detail="Invalid collection name. Use alphanumeric, hyphens, underscores only (max 100 chars)",
        )


async def create_collection(body: CollectionCreateRequest):
    assert vector_store is not None
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name must not be empty")
    _validate_collection_name(name)
    vector_store.create_collection(name)
    return CollectionInfo(name=name)


@app.get("/v1/collections", response_model=CollectionsListResponse, dependencies=[Security(verify_api_key)])
async def list_collections():
    assert vector_store is not None
    names = vector_store.list_collections()
    return CollectionsListResponse(collections=[CollectionInfo(name=n) for n in names])


@app.delete("/v1/collections/{name}", dependencies=[Security(verify_api_key)])
async def delete_collection(name: str):
    assert vector_store is not None
    _validate_collection_name(name)
    if name not in vector_store.list_collections():
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    vector_store.delete_collection(name)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Document indexing
# ---------------------------------------------------------------------------
@app.post("/v1/collections/{name}/documents", response_model=IndexResponse, dependencies=[Security(verify_api_key)])
async def index_documents(name: str, body: IndexRequest):
    assert vector_store is not None
    assert embedding_client is not None
    assert chunker is not None

    _validate_collection_name(name)
    if name not in vector_store.list_collections():
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    if not body.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    if len(body.documents) > 1000:
        raise HTTPException(status_code=413, detail="Too many documents (max 1000 per request)")

    # Chunk all documents
    all_chunks: list[dict] = []
    for doc in body.documents:
        chunks = chunker.chunk(doc.text, metadata=doc.metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="Documents produced no chunks")

    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    try:
        embeddings = await embedding_client.embed(texts)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service error: {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service unreachable: {exc}",
        )

    # Store in LanceDB
    count = vector_store.add_documents(name, all_chunks, embeddings)
    return IndexResponse(indexed=count, collection=name)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------
@app.post("/v1/search", response_model=SearchResponse, dependencies=[Security(verify_api_key)])
async def search(body: SearchRequest):
    assert vector_store is not None
    assert embedding_client is not None

    if body.collection not in vector_store.list_collections():
        raise HTTPException(status_code=404, detail=f"Collection '{body.collection}' not found")

    try:
        query_vec = await embedding_client.embed_single(body.query)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service error: {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service unreachable: {exc}",
        )

    results = vector_store.search(body.collection, query_vec, limit=body.limit)
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query=body.query,
        collection=body.collection,
    )


# ---------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------
@app.post("/v1/rag", response_model=RAGResponse, dependencies=[Security(verify_api_key)])
async def rag(body: RAGRequest):
    assert vector_store is not None
    assert embedding_client is not None

    if body.collection not in vector_store.list_collections():
        raise HTTPException(status_code=404, detail=f"Collection '{body.collection}' not found")

    # 1. Embed the query
    try:
        query_vec = await embedding_client.embed_single(body.query)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {exc}")

    # 2. Retrieve context
    results = vector_store.search(body.collection, query_vec, limit=body.limit)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    context_block = "\n\n---\n\n".join(
        f"[Source {i+1}] {r['text']}" for i, r in enumerate(results)
    )

    # 3. Build augmented prompt
    system = body.system_prompt or (
        "You are a helpful assistant. Answer the user's question based on the "
        "provided context. Cite sources using [Source N] notation. If the context "
        "does not contain enough information, say so."
    )
    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {body.query}"
    )

    # 4. Call Language Insight API chat completion
    parsed = urlparse(config.embedding.endpoint)
    llm_endpoint = urlunparse((parsed.scheme, parsed.netloc, "/v1/chat/completions", "", "", ""))
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                llm_endpoint,
                json={
                    "model": body.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                },
            )
            resp.raise_for_status()
            llm_data = resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}")

    answer = llm_data["choices"][0]["message"]["content"]
    sources = [
        RAGSource(id=r["id"], text=r["text"], score=r["score"])
        for r in results
    ]

    return RAGResponse(
        answer=answer,
        sources=sources,
        model=body.model,
        query=body.query,
    )
