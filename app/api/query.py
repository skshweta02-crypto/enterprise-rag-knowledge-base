from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.rag.pipeline import RAGPipeline

router = APIRouter(tags=["Query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the RAG service")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of top chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float


def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty or whitespace only")

    try:
        rag_response = pipeline.query(question=request.question, top_k=request.top_k)

        chunk_dicts = [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "score": chunk.score,
                "source_display": chunk.source_display,
                "metadata": chunk.metadata,
            }
            for chunk in rag_response.retrieved_chunks
        ]

        return QueryResponse(
            question=rag_response.question,
            answer=rag_response.answer,
            sources=rag_response.sources,
            retrieved_chunks=chunk_dicts,
            latency_ms=rag_response.total_latency_ms,
            retrieval_latency_ms=rag_response.retrieval_latency_ms,
            llm_latency_ms=rag_response.llm_latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
