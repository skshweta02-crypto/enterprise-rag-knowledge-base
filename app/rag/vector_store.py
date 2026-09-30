import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS
from app.models.document import DocumentChunk
from app.rag.embeddings import get_embeddings
from app.core.config import settings
from app.core.logging import logger


class VectorStoreManager:
    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = index_dir or settings.VECTOR_DB_PATH
        self.embeddings = get_embeddings()
        self.vector_store: Optional[FAISS] = None
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(os.path.join(self.index_dir, "index.faiss")) and os.path.exists(
            os.path.join(self.index_dir, "index.pkl")
        ):
            try:
                logger.info(f"Loading existing FAISS index from '{self.index_dir}'")
                self.vector_store = FAISS.load_local(
                    folder_path=self.index_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {self.index_dir}: {e}")
                self.vector_store = None
        return False

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0

        lc_documents = [
            LCDocument(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]

        if self.vector_store is None:
            logger.info(f"Initializing new FAISS index with {len(lc_documents)} chunks")
            self.vector_store = FAISS.from_documents(
                documents=lc_documents,
                embedding=self.embeddings,
            )
        else:
            logger.info(f"Adding {len(lc_documents)} chunks to existing FAISS index")
            self.vector_store.add_documents(lc_documents)

        self.save()
        return len(chunks)

    def similarity_search_with_score(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[LCDocument, float]]:
        if self.vector_store is None:
            logger.warning("Similarity search attempted on empty vector store")
            return []

        k = top_k or settings.TOP_K
        logger.info(f"Performing similarity search for query '{query}' (top_k={k})")
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def save(self, dir_path: Optional[str] = None) -> None:
        target_dir = dir_path or self.index_dir
        if self.vector_store is None:
            logger.warning("Cannot save empty vector store")
            return

        os.makedirs(target_dir, exist_ok=True)
        self.vector_store.save_local(target_dir)
        logger.info(f"FAISS vector store saved to '{target_dir}'")

    def get_indexed_document_count(self) -> int:
        if self.vector_store is None:
            return 0
        return len(self.vector_store.docstore._dict)
