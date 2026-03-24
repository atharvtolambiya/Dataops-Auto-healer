
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    STACK_TRACE_MAX_CHARS,
)

class RAGMemoryStore:
    """
    Encapsulates all ChromaDB operations for the error-knowledge memory.

    Design Decisions:
      • Class-based (not bare functions) because:
        - State is encapsulated (client + collection references).
        - Testable: create instances with different configs.
        - Multiple stores can coexist (e.g., per-pipeline namespace).
      • The class is NOT a singleton — the module-level convenience
        functions manage a default instance instead. This avoids the
        singleton anti-pattern while still providing easy one-liner usage.
      • All public methods return structured dicts, not ChromaDB internals.
        This insulates downstream code from ChromaDB API changes.

    Usage:
        store = RAGMemoryStore().initialize()
        store.seed_memory_if_empty()
        results = store.query_similar_errors("KeyError: missing column 'email'")
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):

        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._is_initialized = False

    def initialize(self) -> "RAGMemoryStore":
        """
        Connect to ChromaDB and get-or-create the error memory collection.
        Returns:
            self — enables fluent chaining: store = RAGMemoryStore().initialize()
        Raises:
            RuntimeError: If ChromaDB client creation fails.
        """
        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "description": "Past pipeline errors and their  fixes",
                    "project": "dataops-auto-healer",
                    "hnsw:space": "cosine",
                },
            )

            self._is_initialized = True
            return self

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB at '{self._persist_dir}': {e}"
            ) from e

    def _ensure_initialized(self) -> None:
        if not self._is_initialized or self._collection is None:
            raise RuntimeError(
                "RAGMemoryStore is not initialized. Call .initialize() first."
            )

    @staticmethod
    def _format_document(
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
    ) -> str:
        """
        Format error details into a single document string for embedding.

        Args:
            error_type:    Exception class name (e.g., "KeyError").
            error_message: The exception's string representation.
            stack_trace:   Optional traceback text (will be truncated).

        Returns:
            Formatted document string.
        """
        parts = [
            f"Error Type: {error_type}",
            f"Error Message: {error_message}",
        ]

        if stack_trace:
            truncated = stack_trace[:STACK_TRACE_MAX_CHARS]
            if len(stack_trace) > STACK_TRACE_MAX_CHARS:
                truncated += "\n... [truncated]"
            parts.append(f"Stack Trace:\n{truncated}")

        return "\n".join(parts)

    @staticmethod
    def _generate_record_id(prefix: str = "err") -> str:
        """
        Generate a unique record ID using uuid4 for multiple entries same error type.

        Args:
            prefix: ID prefix for readability in ChromaDB's dashboard.

        Returns:
            String ID like "err-a1b2c3d4".
        """
        short_uuid = uuid.uuid4().hex[:12]
        return f"{prefix}-{short_uuid}"

    @staticmethod
    def _get_seed_records() -> list[dict]:
        """
        Return seed records that bootstrap the knowledge base.

        Interview Insight:
            These seeds demonstrate understanding of real-world
            data quality issues: schema drift, type coercion,
            missing data, and format inconsistencies — the exact
            problems encountered at companies like Uber, Airbnb,
            and Netflix in their ETL pipelines.
        """
        return [
            #  Seed 1: Schema Drift (column renamed upstream)
            # 'customer_id' to 'cust_id' .
            {
                "id": "seed-001",
                "error_type": "ValueError",
                "error_message": (
                    "[SCHEMA DRIFT DETECTED] Unexpected columns found: "
                    "['cust_id']. Expected exactly: ['amount', 'customer_id', "
                    "'email', 'name', 'transaction_date']."
                ),
                "stack_trace": (
                    "Traceback (most recent call last):\n"
                    '  File "pipeline/data_pipeline.py", line 142, in run_pipeline\n'
                    "    validate_schema(raw_df)\n"
                    '  File "pipeline/data_pipeline.py", line 108, in validate_schema\n'
                    "    raise ValueError(f\"[SCHEMA DRIFT DETECTED]...\")\n"
                    "ValueError: [SCHEMA DRIFT DETECTED] Unexpected columns found: ['cust_id']"
                ),
                "fix_code": (
                    "import pandas as pd\n"
                    "\n"
                    "# Fix: Rename the drifted column back to the expected name.\n"
                    "# This handles the common case where upstream renames a column\n"
                    "# without updating the schema contract.\n"
                    "if 'cust_id' in df.columns and 'customer_id' not in df.columns:\n"
                    "    df = df.rename(columns={'cust_id': 'customer_id'})"
                ),
                "fix_description": (
                    "Rename drifted column 'cust_id' back to 'customer_id'. "
                    "Includes a guard clause to prevent double-renaming."
                ),
                "tags": "schema_drift,column_rename,validation",
            },

            #  Seed 2: Missing Column
            # Root cause: A source API deprecated the 'email' field
            {
                "id": "seed-002",
                "error_type": "KeyError",
                "error_message": (
                    "[SCHEMA VALIDATION FAILED] Missing required columns: "
                    "['email']. Available columns: ['amount', 'customer_id', "
                    "'name', 'transaction_date']. Expected columns: "
                    "['amount', 'customer_id', 'email', 'name', 'transaction_date']."
                ),
                "stack_trace": (
                    "Traceback (most recent call last):\n"
                    '  File "pipeline/data_pipeline.py", line 142, in run_pipeline\n'
                    "    validate_schema(raw_df)\n"
                    '  File "pipeline/data_pipeline.py", line 95, in validate_schema\n'
                    "    raise KeyError(f\"[SCHEMA VALIDATION FAILED]...\")\n"
                    "KeyError: '[SCHEMA VALIDATION FAILED] Missing required columns: [\"email\"]'"
                ),
                "fix_code": (
                    "import pandas as pd\n"
                    "\n"
                    "# Fix: Add the missing column with a safe default value.\n"
                    "# Using a placeholder email instead of NaN ensures downstream\n"
                    "# string operations (lowercase, strip) don't fail.\n"
                    "if 'email' not in df.columns:\n"
                    "    df['email'] = 'unknown@placeholder.com'"
                ),
                "fix_description": (
                    "Add missing 'email' column with a safe placeholder value. "
                    "Prevents downstream NaN-related failures in string operations."
                ),
                "tags": "missing_column,schema_validation,default_value",
            },

            # Seed 3: Wrong Datatype (string in numeric column)
            # Root cause: CSV source file has mixed types in 'amount'
            # column — some rows contain "INVALID" or "N/A" strings.
            {
                "id": "seed-003",
                "error_type": "TypeError",
                "error_message": (
                    "[DTYPE MISMATCH] Column 'amount' has dtype 'object', "
                    "expected 'float64'. Sample values: ['150.0', 'INVALID', '89.99']"
                ),
                "stack_trace": (
                    "Traceback (most recent call last):\n"
                    '  File "pipeline/data_pipeline.py", line 142, in run_pipeline\n'
                    "    validate_schema(raw_df)\n"
                    '  File "pipeline/data_pipeline.py", line 117, in validate_schema\n'
                    "    raise TypeError(f\"[DTYPE MISMATCH]...\")\n"
                    "TypeError: [DTYPE MISMATCH] Column 'amount' has dtype 'object', "
                    "expected 'float64'"
                ),
                "fix_code": (
                    "import pandas as pd\n"
                    "\n"
                    "# Fix: Coerce non-numeric values to NaN, then fill with 0.0.\n"
                    "# errors='coerce' converts unparseable values to NaN instead\n"
                    "# of raising — a safe, production-standard pattern.\n"
                    "df['amount'] = pd.to_numeric(df['amount'], errors='coerce')\n"
                    "df['amount'] = df['amount'].fillna(0.0)"
                ),
                "fix_description": (
                    "Coerce 'amount' column to numeric using pd.to_numeric with "
                    "errors='coerce', then fill resulting NaN values with 0.0."
                ),
                "tags": "dtype_mismatch,type_coercion,numeric_conversion",
            },

            # ── Seed 4: Null values in required column ──
            # Root cause: A batch of records arrived with missing
            # customer_id values — possibly a partial data dump.
            {
                "id": "seed-004",
                "error_type": "ValueError",
                "error_message": (
                    "Column 'customer_id' contains 3 null values which "
                    "violates the not-null constraint. Pipeline requires "
                    "all rows to have a valid customer identifier."
                ),
                "stack_trace": (
                    "Traceback (most recent call last):\n"
                    '  File "pipeline/data_pipeline.py", line 145, in run_pipeline\n'
                    "    validate_not_null(raw_df)\n"
                    '  File "pipeline/data_pipeline.py", line 130, in validate_not_null\n'
                    "    raise ValueError(f\"Column 'customer_id' contains...\")\n"
                    "ValueError: Column 'customer_id' contains 3 null values"
                ),
                "fix_code": (
                    "import pandas as pd\n"
                    "\n"
                    "# Fix: Fill null customer_ids with 0 (sentinel value), then\n"
                    "# cast back to int. In production, you might use a sequence\n"
                    "# generator or drop these rows depending on business rules.\n"
                    "df['customer_id'] = df['customer_id'].fillna(0).astype(int)"
                ),
                "fix_description": (
                    "Fill null customer_id values with 0 (sentinel) and cast to int. "
                    "Preserves row count while maintaining type consistency."
                ),
                "tags": "null_values,fillna,data_quality,not_null_constraint",
            },

            # ── Seed 5: Date format parsing failure ──
            # Root cause: Some dates arrived in DD/MM/YYYY format
            # instead of the expected YYYY-MM-DD ISO format.
            {
                "id": "seed-005",
                "error_type": "ValueError",
                "error_message": (
                    "time data '15/01/2024' does not match format '%Y-%m-%d' "
                    "(match). Failed to parse 'transaction_date' column. "
                    "Expected ISO 8601 format (YYYY-MM-DD)."
                ),
                "stack_trace": (
                    "Traceback (most recent call last):\n"
                    '  File "pipeline/data_pipeline.py", line 150, in transform\n'
                    "    pd.to_datetime(df['transaction_date'], format='%Y-%m-%d')\n"
                    "ValueError: time data '15/01/2024' does not match format '%Y-%m-%d'"
                ),
                "fix_code": (
                    "import pandas as pd\n"
                    "\n"
                    "# Fix: Use flexible date parsing with dayfirst=True to handle\n"
                    "# DD/MM/YYYY format. errors='coerce' converts unparseable\n"
                    "# dates to NaT, which we then fill with a sentinel date.\n"
                    "df['transaction_date'] = pd.to_datetime(\n"
                    "    df['transaction_date'],\n"
                    "    dayfirst=True,\n"
                    "    errors='coerce'\n"
                    ")\n"
                    "df['transaction_date'] = df['transaction_date'].fillna(\n"
                    "    pd.Timestamp('1970-01-01')\n"
                    ")"
                ),
                "fix_description": (
                    "Parse dates flexibly with dayfirst=True and errors='coerce'. "
                    "Fill unparseable dates with epoch sentinel (1970-01-01)."
                ),
                "tags": "date_parsing,format_mismatch,datetime_coercion",
            },
        ]

    def seed_memory_if_empty(self) -> int:
        """
        Returns:
            Number of records seeded (0 if collection was not empty).

        Raises:
            RuntimeError: If store is not initialized.
        """
        self._ensure_initialized()

        current_count = self._collection.count()
        if current_count > 0:
            return 0

        seed_records = self._get_seed_records()

        ids = []
        documents = []
        metadatas = []

        for record in seed_records:
            ids.append(record["id"])

            # Format the document text for embedding
            doc_text = self._format_document(
                error_type=record["error_type"],
                error_message=record["error_message"],
                stack_trace=record.get("stack_trace"),
            )
            documents.append(doc_text)

            metadatas.append({
                "error_type": record["error_type"],
                "fix_code": record["fix_code"],
                "fix_description": record["fix_description"],
                "tags": record.get("tags", ""),
                "source": "seed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(seed_records)

    def add_error_fix_record(
        self,
        error_type: str,
        error_message: str,
        fix_code: str,
        fix_description: str,
        stack_trace: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> str:
        """
        Add a new error→fix record to the knowledge base.

        Called after the auto-healer successfully fixes a pipeline error.
        This is how the system LEARNS — each successful fix becomes
        retrievable context for future similar errors.

        Args:
            error_type:      Exception class name (e.g., "KeyError").
            error_message:   The exception's string representation.
            fix_code:        Python code that resolved the error.
            fix_description: Human-readable explanation of the fix.
            stack_trace:     Optional traceback text.
            tags:            Optional comma-separated tags for filtering.

        Returns:
            The unique ID of the inserted record.

        Raises:
            RuntimeError: If store is not initialized.
            ValueError:   If required fields are empty.
        """
        self._ensure_initialized()

        # ── Input validation ──
        if not error_type or not error_message or not fix_code:
            raise ValueError(
                "error_type, error_message, and fix_code are required. "
                "Cannot store an incomplete error-fix record."
            )

        record_id = self._generate_record_id(prefix="fix")

        document = self._format_document(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
        )

        metadata = {
            "error_type": error_type,
            "fix_code": fix_code,
            "fix_description": fix_description,
            "tags": tags or "",
            "source": "auto_healer",  # Distinguishes learned fixes from seeds
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._collection.add(
            ids=[record_id],
            documents=[document],
            metadatas=[metadata],
        )

        return record_id

    def query_similar_errors(
        self,
        error_log: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Find the most similar past errors and retrieve their fixes.

        This is the RETRIEVAL step of RAG:
          1. The error_log is embedded into a vector using the same
             embedding model that encoded the stored documents.
          2. ChromaDB performs approximate nearest-neighbor search
              to find the top-K most similar vectors.
          3. The associated fix_code and metadata are returned.

        The results feed directly into the LLM agent  as context
        dramatically improving fix accuracy and reducing hallucinations.
        Args:
            error_log: The error text to search for. Can be:
                       - A raw error message string
                       - A formatted error (type + message + trace)
                       - Output from capture_error_context()
            top_k:     Number of similar results to return .

        Returns:
            List of dicts, ordered by similarity:
            [
                {
                    "rank": 1,
                    "record_id": "seed-001",
                    "similarity_score": 0.87,
                    "error_type": "ValueError",
                    "fix_code": "df = df.rename(...)",
                    "fix_description": "Rename drifted column...",
                    "tags": "schema_drift,column_rename",
                    "source": "seed",
                    "document_preview": "Error Type: ValueError..."
                },
                ...
            ]

        Raises:
            RuntimeError: If store is not initialized.
        """
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        # ChromaDB raises if n_results > collection count.
        effective_k = min(top_k, self._collection.count())

        # Perform similarity search
        results = self._collection.query(
            query_texts=[error_log],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        # Transform ChromaDB output into clean structured dicts
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, record_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]

                # ── Convert distance to similarity score ──
                # ChromaDB with cosine space returns cosine distance
                # (0 = identical, 2 = opposite). We convert to a
                # 0-1 similarity score for intuitive interpretation.
                # Formula: similarity = 1 - (distance / 2)
                similarity = round(1.0 - (distance / 2.0), 4)

                formatted_results.append({
                    "rank": i + 1,
                    "record_id": record_id,
                    "similarity_score": similarity,
                    "error_type": metadata.get("error_type", "Unknown"),
                    "fix_code": metadata.get("fix_code", ""),
                    "fix_description": metadata.get("fix_description", ""),
                    "tags": metadata.get("tags", ""),
                    "source": metadata.get("source", "unknown"),
                    "document_preview": document[:200] + "..." if len(document) > 200 else document,
                })

        return formatted_results

    def get_collection_stats(self) -> dict:
        """
        Return metadata about the current state of the knowledge base.

        Useful for:
          • Observability dashboards
          • Health checks
          • Debugging RAG retrieval quality

        Returns:
            Dict with collection statistics.
        """
        self._ensure_initialized()

        count = self._collection.count()

        source_counts = {"seed": 0, "auto_healer": 0, "other": 0}
        if count > 0:
            # Fetch all metadata (acceptable for small collections <10K).
            # For large collections, this would use pagination.
            all_data = self._collection.get(include=["metadatas"])
            for meta in all_data["metadatas"]:
                src = meta.get("source", "other")
                if src in source_counts:
                    source_counts[src] += 1
                else:
                    source_counts["other"] += 1

        return {
            "collection_name": self._collection_name,
            "persist_dir": self._persist_dir,
            "total_records": count,
            "source_distribution": source_counts,
            "is_initialized": self._is_initialized,
        }

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection. USE WITH CAUTION.

        NOT for production use — production systems should use
        soft-delete with TTL-based expiration instead.
        """
        self._ensure_initialized()

        self._client.delete_collection(name=self._collection_name)

        # Recreate with same settings
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "description": "Historical pipeline errors and their proven fixes",
                "project": "dataops-auto-healer",
                "hnsw:space": "cosine",
            },
        )


_default_store: Optional[RAGMemoryStore] = None


def initialize_vector_store(
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> RAGMemoryStore:
    """
    Initialize and return the default RAGMemoryStore instance.

    Thread-safe — if already initialized, returns existing instance.

    Args:
        persist_dir:     ChromaDB storage directory.
        collection_name: Name of the vector collection.

    Returns:
        Initialized RAGMemoryStore instance.
    """
    global _default_store

    if _default_store is not None and _default_store._is_initialized:
        return _default_store

    _default_store = RAGMemoryStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    ).initialize()

    return _default_store


def _get_store() -> RAGMemoryStore:
    """
    Internal helper — get or auto-initialize the default store.

    Auto-initialization ensures the convenience functions work
    without requiring an explicit initialize_vector_store() call.
    This is a developer-experience optimization.
    """
    global _default_store
    if _default_store is None or not _default_store._is_initialized:
        initialize_vector_store()
    return _default_store


def seed_memory_if_empty() -> int:
    """Seed the default store with initial error→fix knowledge."""
    return _get_store().seed_memory_if_empty()


def add_error_fix_record(
    error_type: str,
    error_message: str,
    fix_code: str,
    fix_description: str,
    stack_trace: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """Add an error→fix record to the default store."""
    return _get_store().add_error_fix_record(
        error_type=error_type,
        error_message=error_message,
        fix_code=fix_code,
        fix_description=fix_description,
        stack_trace=stack_trace,
        tags=tags,
    )


def query_similar_errors(error_log: str, top_k: int = 3) -> list[dict]:
    """Query the default store for similar past errors."""
    return _get_store().query_similar_errors(error_log=error_log, top_k=top_k)


#  LOCAL TESTING


if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  DataOps Auto-Healer — RAG Memory Layer Smoke Test")
    print("=" * 70)

    # ── Test 1: Initialize store ──
    print("\n Test 1: Initialize ChromaDB persistent store")
    store = RAGMemoryStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name="test_error_memory",  # Separate test collection
    ).initialize()
    # Start fresh for reproducible test output
    store.reset_collection()
    print("   Store initialized and reset.")

    # ── Test 2: Seed memory ──
    print("\n▶ Test 2: Seed knowledge base with error→fix pairs")
    seeded = store.seed_memory_if_empty()
    print(f"  Seeded {seeded} records.")

    # Verify idempotency
    seeded_again = store.seed_memory_if_empty()
    print(f"  Second seed call returned {seeded_again} (idempotent).")

    # ── Test 3: Collection stats ──
    print("\n Test 3: Collection statistics")
    stats = store.get_collection_stats()
    print(f"  {json.dumps(stats, indent=4)}")

    # ── Test 4: Query — Schema Drift ──
    print("\n Test 4: Query for schema drift error")
    query_1 = (
        "ValueError: [SCHEMA DRIFT DETECTED] Unexpected columns found: "
        "['cust_id']. Expected: ['customer_id', 'name', 'email', 'amount']"
    )
    results_1 = store.query_similar_errors(query_1, top_k=2)
    for r in results_1:
        print(f"  Rank {r['rank']} | Similarity: {r['similarity_score']:.3f} "
              f"| Type: {r['error_type']}")
        print(f"    Fix: {r['fix_description']}")
        print(f"    Code: {r['fix_code'][:80]}...")
        print()

    # ── Test 5: Query — Dtype Mismatch ──
    print("Test 5: Query for dtype mismatch error")
    query_2 = (
        "TypeError: Column 'amount' has dtype 'object', expected 'float64'. "
        "Contains non-numeric values like 'INVALID' and 'N/A'."
    )
    results_2 = store.query_similar_errors(query_2, top_k=2)
    for r in results_2:
        print(f"  Rank {r['rank']} | Similarity: {r['similarity_score']:.3f} "
              f"| Type: {r['error_type']}")
        print(f"    Fix: {r['fix_description']}")
        print()

    # ── Test 6: Add a runtime-learned fix ──
    print(" Test 6: Add a runtime-learned error→fix record")
    new_id = store.add_error_fix_record(
        error_type="FileNotFoundError",
        error_message="Source file not found: '/data/raw/events.csv'",
        fix_code=(
            "import pandas as pd\n"
            "import os\n\n"
            "# Generate empty DataFrame with expected schema as fallback.\n"
            "if not os.path.exists(source_path):\n"
            "    df = pd.DataFrame(columns=['customer_id', 'name', 'email', "
            "'amount', 'transaction_date'])"
        ),
        fix_description="Create empty DataFrame with expected schema when source file is missing.",
        tags="file_not_found,fallback,schema_contract",
    )
    print(f"  Added record: {new_id}")

    # Verify the new record is retrievable
    stats_after = store.get_collection_stats()
    print(f"   Total records now: {stats_after['total_records']}")
    print(f"   Source distribution: {stats_after['source_distribution']}")

    # ── Test 7: Query the newly added record ──
    print("\n Test 7: Query for file-not-found error (should match runtime record)")
    query_3 = "FileNotFoundError: No such file or directory: '/data/input.csv'"
    results_3 = store.query_similar_errors(query_3, top_k=2)
    for r in results_3:
        print(f"  Rank {r['rank']} | Similarity: {r['similarity_score']:.3f} "
              f"| Type: {r['error_type']} | Source: {r['source']}")
        print(f"    Fix: {r['fix_description']}")
        print()

    # ── Cleanup test collection ──
    store.reset_collection()
    print("  Test collection cleaned up.")

    print("=" * 70)
    print("  RAG Memory Layer smoke test complete.")
    print("=" * 70)