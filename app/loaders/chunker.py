import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models.document import DocumentChunk, ProcessedDocument
from app.loaders.document_loader import load_document
from app.core.config import settings
from app.core.logging import logger


def chunk_document(
    file_path: str,
    filename: Optional[str] = None,
    document_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> ProcessedDocument:
    actual_filename = filename or os.path.basename(file_path)
    doc_id = document_id or str(uuid.uuid4())
    c_size = chunk_size or settings.CHUNK_SIZE
    c_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    raw_sections = load_document(file_path, actual_filename)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[DocumentChunk] = []
    global_chunk_idx = 0

    for section in raw_sections:
        content = section["content"]
        page_number = section.get("page_number")

        split_texts = text_splitter.split_text(content)

        for text in split_texts:
            if not text.strip():
                continue

            chunk_id = f"{doc_id}_{global_chunk_idx}"
            metadata = {
                "filename": actual_filename,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": global_chunk_idx,
                "source": actual_filename,
            }
            if page_number is not None:
                metadata["page_number"] = page_number

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=text,
                    metadata=metadata,
                )
            )
            global_chunk_idx += 1

    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    ext = os.path.splitext(actual_filename)[1].lower()

    logger.info(f"Processed '{actual_filename}': generated {len(chunks)} chunks")

    return ProcessedDocument(
        document_id=doc_id,
        filename=actual_filename,
        file_type=ext,
        file_size_bytes=file_size,
        chunks=chunks,
        total_chunks=len(chunks),
    )
