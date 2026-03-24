from enum import Enum
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


class FailureType(Enum):
    NONE = "none"
    SCHEMA_DRIFT = "schema_drift"
    MISSING_COLUMN = "missing_column"
    WRONG_DATATYPE = "wrong_datatype"


EXPECTED_SCHEMA: dict[str, str] = {
    "customer_id": "int64",
    "name": "object",
    "email": "object",
    "amount": "float64",
    "transaction_date": "object",
}

REQUIRED_COLUMNS: list[str] = list(EXPECTED_SCHEMA.keys())


@dataclass(frozen=True)
class PipelineConfig:
    source_path: str = str(DATA_DIR / "expanded_dataset.csv")
    output_path: str = str(DATA_DIR / "cleaned_output.csv")
    failure_type: FailureType = FailureType.NONE
    max_retries: int = 3

CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_store")
CHROMA_COLLECTION_NAME = "pipeline_error_memory"
STACK_TRACE_MAX_CHARS = 500

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
RAG_TOP_K = 3