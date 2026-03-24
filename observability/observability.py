import json
import logging
import traceback
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Any
from config.settings import LOG_DIR

class JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        """
        Convert a LogRecord into a JSON string.

        We extract standard fields and merge any 'extra' data passed
        via logger.error("msg", extra={"key": "value"}).
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        reserved_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "filename", "module", "pathname", "processName", "process",
            "threadName", "thread", "levelname", "levelno", "msecs",
            "message", "taskName",
        }

        for key, value in record.__dict__.items():
            if key not in reserved_attrs and not key.startswith("_"):
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception_type"] = record.exc_info[0].__name__
            log_entry["exception_message"] = str(record.exc_info[1])
            log_entry["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

class StructuredLogger:

    _initialized_loggers: dict[str, logging.Logger] = {}
    _lock = threading.Lock()

    @classmethod
    def get_logger(
        cls,
        name: str = "autohealer",
        level: int = logging.DEBUG,
        log_file: Optional[str] = None,
    ) -> logging.Logger:
        with cls._lock:
            # ── Return existing logger if already configured ──
            if name in cls._initialized_loggers:
                return cls._initialized_loggers[name]

            logger = logging.getLogger(name)
            logger.setLevel(level)

            # Prevent log propagation to root logger (avoids duplicates)
            logger.propagate = False

            json_formatter = JSONFormatter()

            # ── Console Handler ──
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(json_formatter)
            logger.addHandler(console_handler)

            # ── File Handler ──
            file_path = log_file or str(LOG_DIR / "autohealer.log")
            file_handler = logging.FileHandler(file_path, mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(json_formatter)
            logger.addHandler(file_handler)

            cls._initialized_loggers[name] = logger
            return logger


class TelemetryCollector:
    """
    Thread-safe telemetry metrics collector.

    Tracks operational metrics for the auto-healing pipeline:
      • retry_attempts — how many retries were attempted
      • successes / failures — pipeline outcome counters
      • llm_latency_ms — LLM API call duration (Phase 3 placeholder)
      • total_healing_time_ms — end-to-end healing loop duration

    Why a class and not a global dict?
      • Encapsulation — metrics can't be accidentally overwritten.
      • Thread safety — Lock protects concurrent increment operations.
      • Snapshot method — returns an immutable copy for logging.
      • Reset method — clean state between pipeline runs.

    Production Analog:
      This would be replaced by a Prometheus Counter/Histogram or
      an OpenTelemetry metrics exporter in a production system.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "retry_attempts": 0,
            "successes": 0,
            "failures": 0,
            "llm_calls": 0,
            "llm_latency_ms": [],       # List of latencies for percentile calc
            "total_healing_time_ms": 0,
            "errors_by_type": {},        # Counter per exception class name
            "started_at": None,
            "finished_at": None,
        }

    def record_retry(self) -> None:
        """Increment the retry attempt counter."""
        with self._lock:
            self._metrics["retry_attempts"] += 1

    def record_success(self) -> None:
        """Record a successful pipeline execution."""
        with self._lock:
            self._metrics["successes"] += 1

    def record_failure(self, error_type: str) -> None:
        """
        Record a failed pipeline execution.

        Args:
            error_type: The exception class name (e.g., "KeyError").
        """
        with self._lock:
            self._metrics["failures"] += 1
            # Track frequency of each error type for pattern analysis
            counter = self._metrics["errors_by_type"]
            counter[error_type] = counter.get(error_type, 0) + 1

    def record_llm_latency(self, latency_ms: float) -> None:
        """
        Record an LLM API call latency.
        Placeholder for Phase 3 — will be called by the LLM agent.

        Args:
            latency_ms: Duration of the LLM call in milliseconds.
        """
        with self._lock:
            self._metrics["llm_calls"] += 1
            self._metrics["llm_latency_ms"].append(latency_ms)

    def start_timer(self) -> None:
        """Mark the start of a healing cycle."""
        with self._lock:
            self._metrics["started_at"] = time.monotonic()

    def stop_timer(self) -> None:
        """Mark the end of a healing cycle and compute total duration."""
        with self._lock:
            if self._metrics["started_at"] is not None:
                elapsed = time.monotonic() - self._metrics["started_at"]
                self._metrics["total_healing_time_ms"] = round(elapsed * 1000, 2)
                self._metrics["finished_at"] = datetime.now(timezone.utc).isoformat()

    def get_snapshot(self) -> dict:
        """
        Return an immutable snapshot of current metrics.

        Returns a deep-ish copy so the caller can't mutate internal state.
        The LLM latency list is summarized into avg/p95/max stats.
        """
        with self._lock:
            snapshot = {
                "retry_attempts": self._metrics["retry_attempts"],
                "successes": self._metrics["successes"],
                "failures": self._metrics["failures"],
                "llm_calls": self._metrics["llm_calls"],
                "errors_by_type": dict(self._metrics["errors_by_type"]),
                "total_healing_time_ms": self._metrics["total_healing_time_ms"],
                "finished_at": self._metrics["finished_at"],
            }

            # ── Compute LLM latency statistics ──
            latencies = self._metrics["llm_latency_ms"]
            if latencies:
                sorted_lat = sorted(latencies)
                p95_index = int(len(sorted_lat) * 0.95)
                snapshot["llm_latency_stats"] = {
                    "count": len(latencies),
                    "avg_ms": round(sum(latencies) / len(latencies), 2),
                    "p95_ms": round(sorted_lat[min(p95_index, len(sorted_lat) - 1)], 2),
                    "max_ms": round(max(latencies), 2),
                }
            else:
                snapshot["llm_latency_stats"] = None

            return snapshot

    def reset(self) -> None:
        """Reset all metrics to zero. Called between pipeline runs."""
        with self._lock:
            self._metrics = {
                "retry_attempts": 0,
                "successes": 0,
                "failures": 0,
                "llm_calls": 0,
                "llm_latency_ms": [],
                "total_healing_time_ms": 0,
                "errors_by_type": {},
                "started_at": None,
                "finished_at": None,
            }

def capture_error_context(
    exception: Exception,
    retry_count: int = 0,
    pipeline_result: Optional[dict] = None,
) -> dict:
    """
    Extract structured error context from an exception or pipeline result.

    This function produces the exact JSON structure that:
      • Gets logged by StructuredLogger
      • Gets embedded into ChromaDB for RAG retrieval (Phase 2)
      • Gets sent to the LLM agent as diagnostic input (Phase 3)

    Args:
        exception:       The caught exception instance.
        retry_count:     Current retry attempt number.
        pipeline_result: Optional result dict from run_pipeline().

    Returns:
        Structured error context dict.
    """
    error_context = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "stack_trace": traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ),
        "retry_count": retry_count,
    }

    # ── Merge pipeline result metadata if available ──
    if pipeline_result:
        error_context["pipeline_status"] = pipeline_result.get("status")
        error_context["records_processed"] = pipeline_result.get("records_processed", 0)

    return error_context

if __name__ == "__main__":
    print("=" * 65)
    print("  DataOps Auto-Healer — Observability Smoke Test")
    print("=" * 65)

    # ── Test 1: Structured Logger ──
    print("\n▶ Test 1: Structured JSON Logging")
    logger = StructuredLogger.get_logger("test")
    logger.info("Pipeline initialized", extra={"run_id": "smoke-001"})
    logger.warning("High retry count detected", extra={"retry_count": 3})
    logger.error(
        "Schema validation failed",
        extra={
            "error_type": "KeyError",
            "missing_columns": ["customer_id"],
        },
    )

    # ── Test 2: Telemetry Collector ──
    print("\n▶ Test 2: Telemetry Metrics")
    telemetry = TelemetryCollector()
    telemetry.start_timer()
    telemetry.record_retry()
    telemetry.record_failure("KeyError")
    telemetry.record_retry()
    telemetry.record_failure("TypeError")
    telemetry.record_retry()
    telemetry.record_success()
    telemetry.record_llm_latency(234.5)
    telemetry.record_llm_latency(189.2)
    telemetry.stop_timer()

    snapshot = telemetry.get_snapshot()
    print(json.dumps(snapshot, indent=2))

    # ── Test 3: Error Context Capture ──
    print("\n Test 3: Error Context Capture")
    try:
        raise KeyError("[SCHEMA VALIDATION FAILED] Missing columns: ['customer_id']")
    except KeyError as e:
        context = capture_error_context(e, retry_count=1)
        print(json.dumps(context, indent=2, default=str))

    print("\n" + "=" * 65)
    print("  Observability smoke test complete.")
    print("=" * 65)