import pytest
from app.rag.chunking import chunk_text


def test_chunking_preserves_overlap():
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)
    assert chunks[0][-3:] == chunks[1][:3]
    assert len(chunks) >= 3


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hi", chunk_size=5, overlap=5)
