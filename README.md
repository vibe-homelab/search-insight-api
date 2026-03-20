# Search Insight API

**시맨틱 검색 및 RAG (Retrieval-Augmented Generation) API** for [Vibe Homelab](https://github.com/vibe-homelab).

Self-hosted semantic search service that uses LanceDB for vector storage and the Language Insight API for embeddings and LLM inference.

## Features

| Feature | Description |
|---|---|
| Document Indexing | 문서를 청크로 분할하고 임베딩을 생성하여 벡터 DB에 저장 |
| Semantic Search | 자연어 쿼리로 의미 기반 문서 검색 |
| RAG | 검색된 컨텍스트를 기반으로 LLM이 답변 생성 |
| Collections | 문서를 컬렉션 단위로 분리 관리 |
| Embedded DB | LanceDB 사용 — 별도 벡터 DB 서버 불필요 |

## Architecture

```
Client
  │
  ▼
┌──────────────────────┐
│  Search Insight API  │ :8600
│  (FastAPI Gateway)   │
└──────┬───────┬───────┘
       │       │
       ▼       ▼
  ┌────────┐  ┌──────────────────┐
  │ LanceDB│  │ Language Insight  │
  │ (local)│  │ API :8400        │
  └────────┘  │  - /v1/embeddings│
              │  - /v1/chat/...  │
              └──────────────────┘
```

- **Gateway** — FastAPI 서버로 문서 인덱싱, 검색, RAG 엔드포인트 제공
- **Language Insight API** — 임베딩 생성 및 RAG용 LLM 추론 담당
- **LanceDB** — 임베디드 벡터 데이터베이스 (별도 서버 없이 로컬 파일로 동작)

## Quick Start

### Docker Compose

```bash
docker compose up -d
```

서비스가 `http://localhost:8600`에서 시작됩니다.

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
uvicorn src.gateway.main:app --host 0.0.0.0 --port 8600 --reload

# Run tests
pytest
```

## API Reference

### Health Check

```
GET /healthz
```

### Collections

```bash
# Create collection
curl -X POST http://localhost:8600/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs"}'

# List collections
curl http://localhost:8600/v1/collections

# Delete collection
curl -X DELETE http://localhost:8600/v1/collections/my-docs
```

### Document Indexing

```bash
curl -X POST http://localhost:8600/v1/collections/my-docs/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"text": "FastAPI is a modern web framework for Python.", "metadata": {"source": "docs"}},
      {"text": "LanceDB is an embedded vector database.", "metadata": {"source": "docs"}}
    ]
  }'
```

문서는 자동으로 청크로 분할되고 임베딩이 생성되어 저장됩니다.

### Semantic Search

```bash
curl -X POST http://localhost:8600/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vector database",
    "collection": "my-docs",
    "limit": 5
  }'
```

### RAG (Retrieval-Augmented Generation)

```bash
curl -X POST http://localhost:8600/v1/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LanceDB?",
    "collection": "my-docs",
    "model": "llm-fast",
    "limit": 5
  }'
```

## Configuration

설정은 `config.yaml`에서 관리됩니다.

| Section | Key | Default | Description |
|---|---|---|---|
| `embedding` | `endpoint` | `http://localhost:8400/v1/embeddings` | Language Insight API 임베딩 엔드포인트 |
| `embedding` | `model` | `embedding` | 임베딩 모델 이름 |
| `embedding` | `dimensions` | `768` | 임베딩 벡터 차원 |
| `storage` | `path` | `./data/lancedb` | LanceDB 데이터 경로 |
| `gateway` | `port` | `8600` | API 서버 포트 |
| `gateway` | `api_key` | `""` | API 키 (빈 값이면 인증 비활성화) |
| `chunking` | `max_chunk_size` | `1000` | 최대 청크 크기 (문자 수) |
| `chunking` | `overlap` | `200` | 청크 간 오버랩 크기 |

## Model Info

이 서비스는 자체적으로 모델을 실행하지 않습니다. Language Insight API에 의존합니다:

- **Embeddings** — Language Insight API의 `/v1/embeddings` 엔드포인트 사용
- **LLM (RAG용)** — Language Insight API의 `/v1/chat/completions` 엔드포인트 사용

## Port

| Service | Port |
|---|---|
| Search Insight API | 8600 |

## License

MIT
