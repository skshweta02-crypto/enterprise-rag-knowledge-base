import time
from typing import List, Optional
from app.models.rag import RetrievedChunk, RetrievalResult
from app.rag.vector_store import VectorStoreManager
from app.core.config import settings
from app.core.logging import logger


class RAGRetriever:
    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        self.vector_store_manager = vector_store_manager or VectorStoreManager()

    def _format_source_display(self, metadata: dict) -> str:
        filename = metadata.get("filename", "Unknown Document")
        page = metadata.get("page_number")
        if page is not None:
            return f"{filename}, page {page}"
        return filename

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        start_time = time.perf_counter()
        k = top_k or settings.TOP_K

        logger.info(f"Retrieving top {k} context chunks for query: '{query}'")
        raw_results = self.vector_store_manager.similarity_search_with_score(query, top_k=k)

        retrieved_chunks: List[RetrievedChunk] = []
        context_blocks: List[str] = []

        for idx, (doc, score) in enumerate(raw_results, 1):
            metadata = doc.metadata or {}
            source_str = self._format_source_display(metadata)
            chunk_id = metadata.get("chunk_id", f"chunk_{idx}")

            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk_id,
                content=doc.page_content,
                score=float(score),
                metadata=metadata,
                source_display=source_str,
            )
            retrieved_chunks.append(retrieved_chunk)

            context_blocks.append(
                f"[Document Chunk {idx}]\n"
                f"Source: {source_str}\n"
                f"Content: {doc.page_content}"
            )

        formatted_context = "\n\n---\n\n".join(context_blocks)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Retrieval completed in {elapsed_ms:.2f} ms with {len(retrieved_chunks)} chunks"
        )

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            formatted_context=formatted_context,
            retrieval_latency_ms=round(elapsed_ms, 2),
        )
