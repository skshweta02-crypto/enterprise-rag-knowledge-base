from typing import Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
from app.core.logging import logger

_embedding_instance = None


def get_embeddings():
    global _embedding_instance
    if _embedding_instance is None:
        model_name = settings.EMBEDDING_MODEL
        logger.info(f"Initializing embedding model '{model_name}' using HuggingFaceEmbeddings")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_instance
