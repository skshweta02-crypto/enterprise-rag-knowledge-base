import os
import tempfile
import pytest
from app.loaders.text_cleaner import clean_text
from app.loaders.document_loader import is_supported_file, load_document
from app.loaders.chunker import chunk_document
from docx import Document as DocxDocument


def test_clean_text():
    raw_text = "  Hello \x00 World!  \r\n\r\n\r\n  This is a   test.\xa0 "
    cleaned = clean_text(raw_text)
    assert "\x00" not in cleaned
    assert "\xa0" not in cleaned
    assert "Hello World!" in cleaned
    assert "This is a test." in cleaned


def test_is_supported_file():
    assert is_supported_file("report.pdf") is True
    assert is_supported_file("notes.txt") is True
    assert is_supported_file("doc.docx") is True
    assert is_supported_file("guide.md") is True
    assert is_supported_file("script.py") is False
    assert is_supported_file("archive.zip") is False


def test_txt_ingestion_and_chunking():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False, encoding="utf-8") as f:
        f.write("Line 1: Enterprise RAG Service.\n" * 50)
        temp_path = f.name

    try:
        processed = chunk_document(temp_path, filename="sample.txt", chunk_size=200, chunk_overlap=20)
        assert processed.filename == "sample.txt"
        assert processed.file_type == ".txt"
        assert processed.total_chunks > 0
        assert len(processed.chunks) == processed.total_chunks

        first_chunk = processed.chunks[0]
        assert first_chunk.metadata["filename"] == "sample.txt"
        assert first_chunk.metadata["document_id"] == processed.document_id
        assert "chunk_id" in first_chunk.metadata
        assert first_chunk.metadata["chunk_index"] == 0
    finally:
        os.remove(temp_path)


def test_markdown_ingestion():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w+", delete=False, encoding="utf-8") as f:
        f.write("# Title\n\nThis is a markdown document used for RAG testing.\n\n## Section 1\nDetails here.")
        temp_path = f.name

    try:
        processed = chunk_document(temp_path, filename="readme.md", chunk_size=100, chunk_overlap=10)
        assert processed.filename == "readme.md"
        assert processed.file_type == ".md"
        assert processed.total_chunks >= 1
    finally:
        os.remove(temp_path)


def test_docx_ingestion():
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        temp_path = f.name

    doc = DocxDocument()
    doc.add_heading("Enterprise Document", level=1)
    doc.add_paragraph("This is a paragraph inside a Word document for testing RAG processing.")
    doc.save(temp_path)

    try:
        processed = chunk_document(temp_path, filename="test_doc.docx", chunk_size=200, chunk_overlap=20)
        assert processed.filename == "test_doc.docx"
        assert processed.file_type == ".docx"
        assert processed.total_chunks >= 1
        assert "Word document" in processed.chunks[0].content
    finally:
        os.remove(temp_path)


def test_unsupported_file_raises_error():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"\x00\x01\x02")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            chunk_document(temp_path, filename="data.bin")
    finally:
        os.remove(temp_path)
