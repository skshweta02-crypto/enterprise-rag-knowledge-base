from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunks: List[DocumentChunk]
    total_chunks: int
