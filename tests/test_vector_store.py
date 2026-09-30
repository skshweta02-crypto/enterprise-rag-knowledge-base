import os
import shutil
import tempfile
import pytest
from app.models.document import DocumentChunk
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import VectorStoreManager


@pytest.fixture
def temp_vector_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="doc1_0",
            document_id="doc1",
            content="The company password policy requires passwords to be at least 12 characters long.",
            metadata={"filename": "security_policy.pdf", "document_id": "doc1", "chunk_id": "doc1_0", "page_number": 3},
        ),
        DocumentChunk(
            chunk_id="doc1_1",
            document_id="doc1",
            content="Employees must complete security awareness training annually.",
            metadata={"filename": "security_policy.pdf", "document_id": "doc1", "chunk_id": "doc1_1", "page_number": 5},
        ),
        DocumentChunk(
            chunk_id="doc2_0",
            document_id="doc2",
            content="Remote work requests must be approved by the department manager 2 weeks in advance.",
            metadata={"filename": "hr_guide.docx", "document_id": "doc2", "chunk_id": "doc2_0"},
        ),
    ]


def test_embedding_initialization():
    embeddings = get_embeddings()
    assert embeddings is not None
    vec = embeddings.embed_query("Test query")
    assert isinstance(vec, list)
    assert len(vec) > 0


def test_vector_store_add_and_search(temp_vector_dir, sample_chunks):
    vs_manager = VectorStoreManager(index_dir=temp_vector_dir)
    added_count = vs_manager.add_chunks(sample_chunks)
    assert added_count == 3
    assert vs_manager.get_indexed_document_count() == 3

    # Similarity search for password query
    results = vs_manager.similarity_search_with_score("What is the password requirement?", top_k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert "password" in top_doc.page_content.lower()
    assert top_doc.metadata["filename"] == "security_policy.pdf"
    assert top_doc.metadata["page_number"] == 3


def test_vector_store_persistence(temp_vector_dir, sample_chunks):
    vs_manager = VectorStoreManager(index_dir=temp_vector_dir)
    vs_manager.add_chunks(sample_chunks)

    assert os.path.exists(os.path.join(temp_vector_dir, "index.faiss"))
    assert os.path.exists(os.path.join(temp_vector_dir, "index.pkl"))

    # Reload vector store from disk
    reloaded_manager = VectorStoreManager(index_dir=temp_vector_dir)
    assert reloaded_manager.get_indexed_document_count() == 3

    results = reloaded_manager.similarity_search_with_score("remote work policy", top_k=1)
    assert len(results) == 1
    assert "Remote work" in results[0][0].page_content
    assert results[0][0].metadata["filename"] == "hr_guide.docx"
