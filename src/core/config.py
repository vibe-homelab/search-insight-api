"""Application configuration loaded from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    endpoint: str = "http://localhost:8400/v1/embeddings"
    model: str = "embedding"
    dimensions: int = 768


@dataclass
class StorageConfig:
    type: str = "lancedb"
    path: str = "./data/lancedb"


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8600
    api_key: str = ""


@dataclass
class ChunkingConfig:
    max_chunk_size: int = 1000
    overlap: int = 200
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file.

    Falls back to defaults when the file is missing or a section is absent.
    Environment variable ``CONFIG_PATH`` can override the default location.
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")

    path = Path(path)
    raw: dict = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    return AppConfig(
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        storage=StorageConfig(**raw.get("storage", {})),
        gateway=GatewayConfig(**raw.get("gateway", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
    )
