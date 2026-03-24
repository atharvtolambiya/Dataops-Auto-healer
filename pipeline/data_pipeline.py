import pandas as pd
import traceback
from datetime import datetime, timezone
from typing import Optional

from config.settings import (
    FailureType,
    PipelineConfig,
    EXPECTED_SCHEMA,
    REQUIRED_COLUMNS,
    DATA_DIR,
)

def generate_sample_csv(path: Optional[str] = None) -> str:
    """
    Args:
        path: Optional file path. Defaults to config DATA_DIR.

    Returns:
        The absolute path of the generated CSV file.
    """
    output_path = path or str(DATA_DIR / "expanded_dataset.csv")
    df = pd.read_csv('../data/expanded_dataset.csv')
    return output_path

def inject_failure(df: pd.DataFrame, failure_type: FailureType) -> pd.DataFrame:
    """
    Mutate a DataFrame to simulate a specific data-engineering failure.

    This runs AFTER extraction but BEFORE validation, exactly where
    upstream schema changes or corrupt source files would manifest
    in a real Spark/Pandas pipeline.

    Args:
        df:           Clean DataFrame from the extract step.
        failure_type: The type of failure to inject.

    Returns:
        A corrupted DataFrame (or the original if NONE).
    """
    corrupted = df.copy()

    if failure_type == FailureType.SCHEMA_DRIFT:
        corrupted = corrupted.rename(columns={"customer_id": "cust_id"})

    elif failure_type == FailureType.MISSING_COLUMN:
        corrupted = corrupted.drop(columns=["email"])

    elif failure_type == FailureType.WRONG_DATATYPE:
        corrupted["amount"] = corrupted["amount"].astype(str)
        corrupted.loc[2, "amount"] = "INVALID"
        corrupted.loc[5, "amount"] = "N/A"

    return corrupted

def extract(source_path: str) -> pd.DataFrame:
    """
    Extract step: Read CSV into a DataFrame.

    In production, this would be:
      • A Spark read from S3/GCS
      • A Kafka consumer deserialization
      • A database query via SQLAlchemy

    Raises:
        FileNotFoundError: If the source CSV doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    try:
        df = pd.read_csv(source_path)

        if df.empty:
            raise pd.errors.EmptyDataError(
                f"Source file '{source_path}' contains no data rows."
            )

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"[EXTRACT FAILED] Source file not found: '{source_path}'. "
            f"Run generate_sample_csv() to create seed data."
        )

def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate the DataFrame against the expected schema contract

    Checks performed:
      1. All required columns are present
      2. No unexpected columns exist
      3. Column dtypes match expected types

    Raises:
        KeyError:     If required columns are missing.
        TypeError:    If column dtypes don't match the contract.
        ValueError:   If unexpected columns are detected.

    Why raise instead of return bool?
      • Exceptions carry rich context (column names, types).
      • The observability layer captures the full stack trace.
      • The LLM agent (Phase 3) needs the error message to diagnose.
    """
    actual_columns = set(df.columns.tolist())
    expected_columns = set(REQUIRED_COLUMNS)

    missing = expected_columns - actual_columns
    if missing:
        raise KeyError(
            f"[SCHEMA VALIDATION FAILED] Missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(actual_columns)}. "
            f"Expected columns: {sorted(expected_columns)}."
        )

    extra = actual_columns - expected_columns
    if extra:
        raise ValueError(
            f"[SCHEMA DRIFT DETECTED] Unexpected columns found: {sorted(extra)}. "
            f"Expected exactly: {sorted(expected_columns)}."
        )

    for col, expected_dtype in EXPECTED_SCHEMA.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            raise TypeError(
                f"[DTYPE MISMATCH] Column '{col}' has dtype '{actual_dtype}', "
                f"expected '{expected_dtype}'. "
                f"Sample values: {df[col].head(3).tolist()}"
            )

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform step: Apply business-logic transformations.

    Transformations:
      1. Standardize email to lowercase (data quality).
      2. Parse transaction_date to datetime.
      3. Add a computed 'amount_category' column.
      4. Remove duplicates on customer_id.

    Args:
        df: Validated DataFrame.

    Returns:
        Transformed DataFrame ready for loading.
    """
    transformed = df.copy()

    transformed["email"] = transformed["email"].str.lower().str.strip()

    transformed["transaction_date"] = pd.to_datetime(
        transformed["transaction_date"],
        format="%Y-%m-%d",
        errors="raise",
    )

    transformed["amount_category"] = pd.cut(
        transformed["amount"],
        bins=[0, 100, 300, float("inf")],
        labels=["low", "medium", "high"],
    )
 
    before_count = len(transformed)
    transformed = transformed.drop_duplicates(subset=["customer_id"], keep="last")
    after_count = len(transformed)

    if before_count != after_count:
        print(f"  [TRANSFORM] Removed {before_count - after_count} duplicate rows.")

    return transformed

def load(df: pd.DataFrame, destination: str) -> dict:
    """
    Load step: Write the transformed DataFrame to a CSV sink.

    In production, this would be a write to:
      • A data warehouse (BigQuery, Snowflake, Redshift)
      • A Delta Lake table
      • An analytics database

    Args:
        df:          Transformed DataFrame.
        destination: Output file path.

    Returns:
        Load metadata dict with row count and destination.
    """
    df.to_csv(destination, index=False)

    return {
        "rows_written": len(df),
        "columns_written": list(df.columns),
        "destination": destination,
    }

def run_pipeline(
    config: Optional[PipelineConfig] = None,
    failure_type: Optional[FailureType] = None,
) -> dict:
    """
    Execute the full ETL pipeline: Extract → Validate → Transform → Load.

    This function is the primary entry point consumed by the orchestrator
    (Phase 5) and the retry loop. It returns a structured result dict
    that downstream components (observability, LLM agent) can parse.

    Args:
        config:       PipelineConfig instance. Uses defaults if None.
        failure_type: Override failure type (takes precedence over config).

    Returns:
        dict with keys:
          - status: "success" or "failure"
          - records_processed: int
          - error_type: str or None
          - error_message: str or None
          - stack_trace: str or None
          - timestamp: ISO-8601 UTC string

    Design Note:
        This function catches exceptions and returns them as structured
        data instead of letting them propagate. This is intentional —
        the orchestrator needs the error details to feed to the LLM agent,
        and unhandled exceptions would bypass the telemetry layer.
    """
    config = config or PipelineConfig()
    active_failure = failure_type if failure_type is not None else config.failure_type

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # ── Step 1: Ensure source data exists ──
        import os
        if not os.path.exists(config.source_path):
            print(f"  [PIPELINE] Source file not found. Generating sample data...")
            generate_sample_csv(config.source_path)

        # ── Step 2: Extract ──
        print(f"  [PIPELINE] Extracting from: {config.source_path}")
        raw_df = extract(config.source_path)
        print(f"  [PIPELINE] Extracted {len(raw_df)} rows.")

        # ── Step 3: Inject failure (if configured) ──
        if active_failure != FailureType.NONE:
            print(f"  [PIPELINE] ⚠ Injecting failure: {active_failure.value}")
            raw_df = inject_failure(raw_df, active_failure)

        # ── Step 4: Validate ──
        print(f"  [PIPELINE] Validating schema...")
        validate_schema(raw_df)
        print(f"  [PIPELINE] Schema validation passed. ✓")

        # ── Step 5: Transform ──
        print(f"  [PIPELINE] Transforming data...")
        transformed_df = transform(raw_df)
        print(f"  [PIPELINE] Transformation complete. ✓")

        # ── Step 6: Load ──
        print(f"  [PIPELINE] Loading to: {config.output_path}")
        load_meta = load(transformed_df, config.output_path)
        print(f"  [PIPELINE] Loaded {load_meta['rows_written']} rows. ✓")

        return {
            "status": "success",
            "records_processed": load_meta["rows_written"],
            "error_type": None,
            "error_message": None,
            "stack_trace": None,
            "timestamp": timestamp,
        }

    except Exception as e:
        # ── Capture the FULL stack trace ──
        # This is critical for the LLM agent — it needs the exact
        # traceback to generate a targeted fix.
        full_trace = traceback.format_exc()

        return {
            "status": "failure",
            "records_processed": 0,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "stack_trace": full_trace,
            "timestamp": timestamp,
        }

if __name__ == "__main__":
    print("=" * 65)
    print("  DataOps Auto-Healer — Pipeline Smoke Test")
    print("=" * 65)

    print("\n▶ Test 1: Clean pipeline run")
    result = run_pipeline(failure_type=FailureType.NONE)
    print(f"  Result: {result['status']}\n")

    print("▶ Test 2: Schema drift injection")
    result = run_pipeline(failure_type=FailureType.SCHEMA_DRIFT)
    print(f"  Result: {result['status']}")
    print(f"  Error:  {result['error_type']}: {result['error_message'][:80]}...\n")

    print("▶ Test 3: Missing column injection")
    result = run_pipeline(failure_type=FailureType.MISSING_COLUMN)
    print(f"  Result: {result['status']}")
    print(f"  Error:  {result['error_type']}: {result['error_message'][:80]}...\n")

    print("▶ Test 4: Wrong datatype injection")
    result = run_pipeline(failure_type=FailureType.WRONG_DATATYPE)
    print(f"  Result: {result['status']}")
    print(f"  Error:  {result['error_type']}: {result['error_message'][:80]}...\n")

    print("=" * 65)
    print("  Smoke test complete.")
    print("=" * 65)