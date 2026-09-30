import os
import time
import shutil
import tempfile
import pytest
from app.loaders.chunker import chunk_document
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import RAGRetriever
from app.rag.pipeline import RAGPipeline
from app.rag.llm import LLMService


@pytest.fixture
def benchmark_dataset():
    temp_dir = tempfile.mkdtemp()
    sample_path = os.path.join(temp_dir, "enterprise_handbook.txt")

    content_blocks = [
        "Section 1: Work Hours. Standard work hours are 9:00 AM to 5:00 PM EST from Monday to Friday.",
        "Section 2: Health Insurance. Comprehensive health insurance coverage starts on the first day of employment.",
        "Section 3: Remote Work. Employees can work remotely up to 2 days per week with manager consent.",
        "Section 4: Equipment Allowance. Employees receive a $1,000 stipend for home office setup.",
        "Section 5: Travel Policy. All business flights over 6 hours qualify for premium economy booking.",
    ]

    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(content_blocks * 10))

    yield sample_path, temp_dir
    shutil.rmtree(temp_dir)


def test_end_to_end_performance_and_latency(benchmark_dataset):
    file_path, vector_dir = benchmark_dataset

    # 1. Ingestion Performance
    ingest_start = time.perf_counter()
    processed_doc = chunk_document(file_path, filename="enterprise_handbook.txt")
    vs_manager = VectorStoreManager(index_dir=vector_dir)
    vs_manager.add_chunks(processed_doc.chunks)
    ingest_elapsed_ms = (time.perf_counter() - ingest_start) * 1000.0

    assert processed_doc.total_chunks > 0
    assert ingest_elapsed_ms > 0

    # 2. Retrieval Latency Benchmark
    retriever = RAGRetriever(vector_store_manager=vs_manager)

    retrieval_times = []
    for _ in range(5):
        res = retriever.retrieve("What is the home office equipment allowance?", top_k=3)
        retrieval_times.append(res.retrieval_latency_ms)

    avg_retrieval_ms = sum(retrieval_times) / len(retrieval_times)
    assert avg_retrieval_ms > 0

    # 3. Full Pipeline Latency Benchmark
    pipeline = RAGPipeline(retriever=retriever, llm_service=LLMService())

    pipeline_times = []
    for _ in range(5):
        pipeline_res = pipeline.query("How many remote work days are allowed per week?")
        pipeline_times.append(pipeline_res.total_latency_ms)

    avg_pipeline_ms = sum(pipeline_times) / len(pipeline_times)

    print(f"\n--- EMPIRICAL PERFORMANCE BENCHMARK ---")
    print(f"Total Chunks Indexed: {processed_doc.total_chunks}")
    print(f"Document Ingestion Time: {ingest_elapsed_ms:.2f} ms")
    print(f"Average Retrieval Latency: {avg_retrieval_ms:.2f} ms")
    print(f"Average Full RAG Query Latency (Mock LLM): {avg_pipeline_ms:.2f} ms")

    assert len(pipeline_res.sources) > 0
    assert "Remote Work" in pipeline_res.answer or "remotely" in pipeline_res.answer or "documents" in pipeline_res.answer
