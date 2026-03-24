# rag/__init__.py
# Exposes the public interface of the RAG memory module.

from .vector_db import (
    RAGMemoryStore,
    initialize_vector_store,
    seed_memory_if_empty,
    add_error_fix_record,
    query_similar_errors,
)

__all__ = [
    "RAGMemoryStore",
    "initialize_vector_store",
    "seed_memory_if_empty",
    "add_error_fix_record",
    "query_similar_errors",
]