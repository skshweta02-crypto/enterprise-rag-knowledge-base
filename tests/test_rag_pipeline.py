import shutil
import tempfile
import pytest
from app.models.document import DocumentChunk
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import RAGRetriever
from app.rag.llm import LLMService
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import NO_CONTEXT_FALLBACK


@pytest.fixture
def temp_vector_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="travel_0",
            document_id="doc_travel",
            content="Enterprise travel budget allows up to $150 per night for hotel accommodation.",
            metadata={"filename": "travel_policy.pdf", "page_number": 4},
        )
    ]


def test_rag_pipeline_empty_context(temp_vector_dir):
    vs_manager = VectorStoreManager(index_dir=temp_vector_dir)
    retriever = RAGRetriever(vector_store_manager=vs_manager)
    pipeline = RAGPipeline(retriever=retriever, llm_service=LLMService())

    response = pipeline.query("What is the hotel allowance?")
    assert response.answer == NO_CONTEXT_FALLBACK
    assert response.sources == []
    assert response.retrieved_chunks == []
    assert response.total_latency_ms >= 0


def test_rag_pipeline_successful_grounded_answer(temp_vector_dir, sample_chunks):
    vs_manager = VectorStoreManager(index_dir=temp_vector_dir)
    vs_manager.add_chunks(sample_chunks)
    retriever = RAGRetriever(vector_store_manager=vs_manager)
    pipeline = RAGPipeline(retriever=retriever, llm_service=LLMService())

    response = pipeline.query("What is the hotel allowance per night?", top_k=1)
    assert response.question == "What is the hotel allowance per night?"
    assert len(response.sources) == 1
    assert response.sources[0] == "travel_policy.pdf, page 4"
    assert len(response.retrieved_chunks) == 1
    assert response.retrieval_latency_ms > 0
    assert response.total_latency_ms >= response.retrieval_latency_ms
