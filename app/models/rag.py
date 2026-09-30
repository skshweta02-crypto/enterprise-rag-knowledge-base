from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_display: str


class RetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    formatted_context: str
    retrieval_latency_ms: float
