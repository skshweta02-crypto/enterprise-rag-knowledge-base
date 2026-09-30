import time
from typing import List, Optional
from pydantic import BaseModel
from app.models.rag import RetrievedChunk
from app.rag.retriever import RAGRetriever
from app.rag.llm import LLMService
from app.rag.prompts import NO_CONTEXT_FALLBACK
from app.core.logging import logger


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float


class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.retriever = retriever or RAGRetriever()
        self.llm_service = llm_service or LLMService()

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        start_time = time.perf_counter()
        logger.info(f"Executing RAG pipeline for question: '{question}'")

        # 1. Retrieve relevant context chunks
        retrieval_result = self.retriever.retrieve(question, top_k=top_k)

        # Extract unique sources
        sources = list(
            {chunk.source_display for chunk in retrieval_result.chunks}
        )

        # 2. Check if context was found
        if not retrieval_result.chunks:
            total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                question=question,
                answer=NO_CONTEXT_FALLBACK,
                sources=[],
                retrieved_chunks=[],
                retrieval_latency_ms=retrieval_result.retrieval_latency_ms,
                llm_latency_ms=0.0,
                total_latency_ms=round(total_elapsed_ms, 2),
            )

        # 3. Generate grounded LLM answer
        answer, llm_latency_ms = self.llm_service.generate_answer(
            question=question,
            formatted_context=retrieval_result.formatted_context,
        )

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"RAG query finished in {total_elapsed_ms:.2f} ms (Retrieval: {retrieval_result.retrieval_latency_ms} ms, LLM: {llm_latency_ms} ms)"
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieval_result.chunks,
            retrieval_latency_ms=retrieval_result.retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=round(total_elapsed_ms, 2),
        )
