import shutil
import tempfile
import pytest
from app.models.document import DocumentChunk
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import RAGRetriever


@pytest.fixture
def temp_vector_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="policy_1",
            document_id="doc_1",
            content="Employees get 20 paid vacation days per calendar year.",
            metadata={"filename": "employee_handbook.pdf", "page_number": 14},
        ),
        DocumentChunk(
            chunk_id="policy_2",
            document_id="doc_1",
            content="Sick leave entitlement is 10 days per year with medical certificate.",
            metadata={"filename": "employee_handbook.pdf", "page_number": 15},
        ),
    ]


def test_retriever_empty_store(temp_vector_dir):
    vs_manager = VectorStoreManager(index_dir=temp_vector_dir)
    retriever = RAGRetriever(vector_store_manager=vs_manager)

    result = retriever.retrieve("What is the vacation policy?")
    assert result.query == "What is the vacation policy?"
    assert len(result.chunks) == 0
    assert result.formatted_context == ""
    assert result.retrieval_latency_ms >= 0


def test_retriever_pipeline(temp_vector_dir, sample_chunks):
    vs_manager = VectorStoreManager(index_dir=temp_vector_dir)
    vs_manager.add_chunks(sample_chunks)

    retriever = RAGRetriever(vector_store_manager=vs_manager)
    result = retriever.retrieve("How many paid vacation days do employees get?", top_k=1)

    assert len(result.chunks) == 1
    assert "20 paid vacation days" in result.chunks[0].content
    assert result.chunks[0].source_display == "employee_handbook.pdf, page 14"
    assert "[Document Chunk 1]" in result.formatted_context
    assert "Source: employee_handbook.pdf, page 14" in result.formatted_context
    assert result.retrieval_latency_ms > 0
