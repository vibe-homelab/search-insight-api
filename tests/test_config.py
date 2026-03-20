"""Tests for configuration loading."""

import textwrap
from pathlib import Path

from src.core.config import AppConfig, load_config


def test_load_defaults_when_no_file(tmp_path: Path):
    """Loading from a non-existent path should return all defaults."""
    cfg = load_config(tmp_path / "missing.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.embedding.dimensions == 768
    assert cfg.storage.type == "lancedb"
    assert cfg.gateway.port == 8600
    assert cfg.chunking.max_chunk_size == 1000


def test_load_partial_yaml(tmp_path: Path):
    """Partial config should override only specified values."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
        embedding:
          dimensions: 1024
        gateway:
          port: 9000
        """)
    )
    cfg = load_config(yaml_file)
    assert cfg.embedding.dimensions == 1024
    assert cfg.gateway.port == 9000
    # defaults preserved
    assert cfg.storage.type == "lancedb"
    assert cfg.chunking.overlap == 200


def test_load_full_yaml(tmp_path: Path):
    """Full config round-trips correctly."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
        embedding:
          endpoint: "http://custom:8400/v1/embeddings"
          model: "custom-embed"
          dimensions: 512
        storage:
          type: "lancedb"
          path: "/data/vectors"
        gateway:
          host: "127.0.0.1"
          port: 7777
          api_key: "secret"
        chunking:
          max_chunk_size: 500
          overlap: 100
          separators: ["\\n", " "]
        """)
    )
    cfg = load_config(yaml_file)
    assert cfg.embedding.endpoint == "http://custom:8400/v1/embeddings"
    assert cfg.embedding.model == "custom-embed"
    assert cfg.embedding.dimensions == 512
    assert cfg.storage.path == "/data/vectors"
    assert cfg.gateway.host == "127.0.0.1"
    assert cfg.gateway.port == 7777
    assert cfg.gateway.api_key == "secret"
    assert cfg.chunking.max_chunk_size == 500
    assert cfg.chunking.overlap == 100
    assert cfg.chunking.separators == ["\n", " "]
