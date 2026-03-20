"""Tests for the text chunker."""

from src.core.chunker import TextChunker


def test_short_text_single_chunk():
    """Text shorter than max_chunk_size should produce one chunk."""
    chunker = TextChunker(max_chunk_size=100, overlap=20)
    result = chunker.chunk("Hello, world!")
    assert len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert "chunk_id" in result[0]
    assert result[0]["metadata"]["chunk_index"] == 0


def test_metadata_passthrough():
    """Supplied metadata should appear in every chunk."""
    chunker = TextChunker(max_chunk_size=100, overlap=0)
    result = chunker.chunk("Short text.", metadata={"source": "test.pdf"})
    assert result[0]["metadata"]["source"] == "test.pdf"
    assert result[0]["metadata"]["chunk_index"] == 0


def test_splits_on_paragraph():
    """Long text with paragraph breaks should split on \\n\\n."""
    text = ("A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80)
    chunker = TextChunker(max_chunk_size=100, overlap=0)
    result = chunker.chunk(text)
    assert len(result) >= 3
    # First chunk should contain only A's
    assert "A" in result[0]["text"]


def test_overlap_present():
    """Chunks after the first should contain overlap from the previous chunk."""
    text = "word " * 200  # ~1000 chars
    chunker = TextChunker(max_chunk_size=100, overlap=20)
    result = chunker.chunk(text)
    assert len(result) > 1
    # Second chunk should start with text from the end of the first chunk
    first_tail = result[0]["text"][-20:]
    assert result[1]["text"].startswith(first_tail)


def test_empty_text():
    """Empty text should produce no chunks."""
    chunker = TextChunker(max_chunk_size=100, overlap=0)
    result = chunker.chunk("")
    assert result == []


def test_whitespace_only_text():
    """Whitespace-only text should produce no chunks."""
    chunker = TextChunker(max_chunk_size=100, overlap=0)
    result = chunker.chunk("   \n\n  \n  ")
    assert result == []


def test_chunk_indices_sequential():
    """chunk_index metadata should be sequential starting from 0."""
    text = "\n\n".join(["Paragraph " + str(i) + " " * 50 for i in range(10)])
    chunker = TextChunker(max_chunk_size=80, overlap=0)
    result = chunker.chunk(text)
    indices = [c["metadata"]["chunk_index"] for c in result]
    assert indices == list(range(len(result)))


def test_hard_split_no_separators():
    """Text without any separators should still be split by character limit."""
    text = "A" * 500
    chunker = TextChunker(max_chunk_size=100, overlap=0)
    result = chunker.chunk(text)
    assert len(result) >= 5
    for c in result:
        assert len(c["text"]) <= 100
