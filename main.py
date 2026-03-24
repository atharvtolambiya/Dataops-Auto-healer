
import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from typing import Optional

# ── Project imports ──────────────────────────────────────────────────
from config.settings import (
    FailureType,
    PipelineConfig,
    GROQ_API_KEY_ENV_VAR,
)
from pipeline.data_pipeline import (
    generate_sample_csv,
    extract,
    inject_failure,
    validate_schema,
    transform,
    load,
)
from observability.observability import (
    StructuredLogger,
    TelemetryCollector,
    capture_error_context,
)
from rag.vector_db import (
    initialize_vector_store,
    seed_memory_if_empty,
    add_error_fix_record,
)
from agents.auto_healer_agent import AutoHealerAgent
from executor.safe_executor import SafeExecutor


# Constants
MAX_RETRIES = 3

# Scenarios to demonstrate — each injects a different failure type.
DEMO_SCENARIOS: list[dict] = [
    {
        "name": "Schema Drift",
        "description": "Upstream renamed 'customer_id' → 'cust_id'",
        "failure_type": FailureType.SCHEMA_DRIFT,
    },
    {
        "name": "Missing Column",
        "description": "Source dropped 'email' column without notice",
        "failure_type": FailureType.MISSING_COLUMN,
    },
    {
        "name": "Wrong Datatype",
        "description": "Amount column has 'INVALID' and 'N/A' strings",
        "failure_type": FailureType.WRONG_DATATYPE,
    },
]

#  AUTO-HEAL ORCHESTRATOR


class AutoHealOrchestrator:
    """
    Connects all system components into a self-healing pipeline loop.

    Responsibilities:
      1. Initialize all subsystems (RAG, Agent, Executor, Telemetry).
      2. Extract and corrupt data (simulating real-world failures).
      3. Run the heal loop: detect → diagnose → fix → verify → learn.
      4. Track telemetry and produce structured reports.

    Design Principles:
      • Each method has a single responsibility.
      • No method exceeds ~40 lines (readable at a glance).
      • All errors are caught and logged — the system never crashes.
      • The orchestrator does NOT contain business logic — it delegates
        to specialized components (agent, executor, pipeline).
    """

    def __init__(self):
        """
        Initialize all subsystem components.

        Order matters:
          1. Logger (needed by everything else for error reporting)
          2. Telemetry (tracks all operations from this point)
          3. RAG store (must be ready before agent is created)
          4. Agent (connects to LLM + RAG at init time)
          5. Executor (stateless, fast to create)
          6. Pipeline config (data paths, retry limits)
        """
        self._logger = StructuredLogger.get_logger("orchestrator")
        self._telemetry = TelemetryCollector()

        # RAG store initialization
        self._logger.info("Initializing RAG memory store...")
        initialize_vector_store()
        seeded = seed_memory_if_empty()
        if seeded > 0:
            self._logger.info(
                f"Seeded RAG memory with {seeded} error-fix records"
            )

        # Agent initialization
        self._logger.info("Initializing GenAI remediation agent...")
        self._agent = AutoHealerAgent()

        # Executor initialization
        self._executor = SafeExecutor()

        # Pipeline config
        self._config = PipelineConfig()

        self._logger.info("Orchestrator initialization complete")


    @staticmethod
    def _print_scenario_header(scenario: dict, index: int) -> None:
        """Print a visual header for each scenario."""
        print(f"\n{'━' * 66}")
        print(f"  SCENARIO {index}: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'━' * 66}")

    # ─── Core Pipeline Steps ────────────────────────────────────────

    def _extract_and_corrupt(
        self,
        failure_type: FailureType,
    ) -> tuple:
        """
        Extract raw data and inject the configured failure.

        Returns the corrupted DataFrame that the heal loop will fix.

        Args:
            failure_type: The type of failure to inject.

        Returns:
            Tuple of (corrupted_df, extraction_successful: bool)
        """
        # Ensure source data exists
        if not os.path.exists(self._config.source_path):
            print("  📄 Generating sample data...")
            generate_sample_csv(self._config.source_path)

        print(f"  📥 Extracting from: {self._config.source_path}")
        raw_df = extract(self._config.source_path)
        print(f"     Extracted {len(raw_df)} rows, {len(raw_df.columns)} columns")

        # Inject failure
        print(f"  ⚡ Injecting failure: {failure_type.value}")
        corrupted_df = inject_failure(raw_df, failure_type)

        return corrupted_df

    def _try_pipeline_steps(self, df):
        """
        Attempt validate → transform → load on the given DataFrame.

        This isolates the pipeline execution so the heal loop can
        retry cleanly after each fix attempt.

        Args:
            df: DataFrame to process (may be original or patched).

        Returns:
            dict with "success", "error_context", "records_processed"

        Note: Exceptions are caught and returned as structured data.
              This is intentional — the heal loop needs error details
              to feed to the LLM agent.
        """
        try:
            # Validate
            validate_schema(df)
            print("     ✅ Schema validation passed")

            # Transform
            transformed_df = transform(df)
            print(f"     ✅ Transformation complete ({len(transformed_df)} rows)")

            # Load
            load_result = load(transformed_df, self._config.output_path)
            print(f"      ✅ Loaded {load_result['rows_written']} rows → {self._config.output_path}")

            return {
                "success": True,
                "error_context": None,
                "records_processed": load_result["rows_written"],
            }

        except Exception as e:
            error_ctx = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stack_trace": traceback.format_exc(),
            }

            print(f"     ❌ Pipeline failed: {type(e).__name__}")
            print(f"        {str(e)[:100]}...")

            return {
                "success": False,
                "error_context": error_ctx,
                "records_processed": 0,
            }

    def _attempt_remediation(
        self,
        error_context: dict,
        current_df,
        attempt: int,
    ) -> dict:
        """
        Single remediation attempt: diagnose → patch → validate → execute.

        Args:
            error_context: Structured error from the failed pipeline step.
            current_df:    The DataFrame in its current (broken) state.
            attempt:       Current retry attempt number (for logging).

        Returns:
            {
                "success": bool,
                "fixed_df": pd.DataFrame or None,
                "patch_code": str,
                "agent_result": dict,
                "executor_result": dict
            }
        """
        print(f"\n  🔧 Auto-Heal Attempt {attempt}/{MAX_RETRIES}")

        # ── Step 1: Call the LLM Agent ──
        print("     🤖 Calling GenAI agent for diagnosis...")
        agent_result = self._agent.diagnose_and_fix(error_context)

        # Track LLM latency in telemetry
        self._telemetry.record_llm_latency(agent_result["llm_latency_ms"])

        if agent_result["status"] != "success":
            print(f"     ❌ Agent failed: {agent_result['error_details']}")
            return {
                "success": False,
                "fixed_df": None,
                "patch_code": "",
                "agent_result": agent_result,
                "executor_result": None,
            }

        print(f"     📡 LLM response ({agent_result['llm_latency_ms']}ms) "
              f"| RAG context: {agent_result['retrieved_context_count']} fixes")

        patch_code = agent_result["generated_patch"]
        print(f"     📝 Generated patch:")
        for line in patch_code.split('\n'):
            print(f"        {line}")

        # ── Step 2: Safe Execution ──
        print("     🛡️  Validating & executing in sandbox...")
        exec_result = self._executor.validate_and_execute(patch_code, current_df)

        if not exec_result["success"]:
            print(f"     ❌ Safe execution failed: {exec_result['error']}")
            if exec_result.get("rejection_reasons"):
                for reason in exec_result["rejection_reasons"]:
                    print(f"        🚫 {reason}")
            return {
                "success": False,
                "fixed_df": None,
                "patch_code": patch_code,
                "agent_result": agent_result,
                "executor_result": exec_result,
            }

        print(f"     ✅ Patch executed safely ({exec_result['execution_time_ms']}ms)")

        return {
            "success": True,
            "fixed_df": exec_result["modified_df"],
            "patch_code": patch_code,
            "agent_result": agent_result,
            "executor_result": exec_result,
        }

    def _store_learned_fix(
        self,
        error_context: dict,
        patch_code: str,
    ) -> None:
        """
        Store a successful fix in RAG memory for future retrieval.

        This closes the learning feedback loop:
          Error → Diagnose → Fix → Verify → ★ STORE-> RAG -> future similar errors


        The stored fix becomes retrievable context for future
        similar errors, making the system smarter over time.
        """
        try:
            record_id = add_error_fix_record(
                error_type=error_context["error_type"],
                error_message=error_context["error_message"],
                fix_code=patch_code,
                fix_description=(
                    f"Auto-generated fix for {error_context['error_type']} "
                    f"at {datetime.now(timezone.utc).isoformat()}"
                ),
                stack_trace=error_context.get("stack_trace"),
                tags="auto_generated,runtime_fix,verified",
            )
            print(f"     📚 Fix stored in RAG memory: {record_id}")

        except Exception as e:
            # RAG storage failure should NOT break the pipeline.
            # The fix was already applied successfully — this is just
            # a "nice to have" for future improvement.
            self._logger.warning(
                "Failed to store fix in RAG memory",
                extra={"error": str(e)},
            )

    # ─── Main Scenario Execution ────────────────────────────────────

    def run_scenario(self, scenario: dict) -> dict:
        """
        Execute a single failure scenario with auto-healing.

        This is the main heal loop. It follows the pattern:
          1. Extract + corrupt data
          2. Try pipeline → fails
          3. For each retry: diagnose → fix → retry pipeline
          4. Report final result

        Args:
            scenario: Dict with "name", "description", "failure_type".

        Returns:
            Scenario result dict with status, attempts, telemetry.
        """
        self._telemetry.reset()
        self._telemetry.start_timer()

        failure_type = scenario["failure_type"]
        scenario_result = {
            "scenario": scenario["name"],
            "failure_type": failure_type.value,
            "status": "unknown",
            "attempts": 0,
            "records_processed": 0,
            "healed": False,
            "final_patch": None,
        }

        try:
            # ── Extract and corrupt ──
            current_df = self._extract_and_corrupt(failure_type)

            # ── Initial pipeline attempt ──
            print(f"\n  🔄 Initial pipeline run...")
            pipeline_result = self._try_pipeline_steps(current_df)
            scenario_result["attempts"] = 1

            if pipeline_result["success"]:
                # This shouldn't happen with failure injection, but handle it
                print("  ✅ Pipeline succeeded on first run (no healing needed)")
                scenario_result["status"] = "success_no_heal"
                scenario_result["records_processed"] = pipeline_result["records_processed"]
                self._telemetry.record_success()
                self._telemetry.stop_timer()
                scenario_result["telemetry"] = self._telemetry.get_snapshot()
                return scenario_result

            # ── Pipeline failed — enter heal loop ──
            self._telemetry.record_failure(
                pipeline_result["error_context"]["error_type"]
            )
            last_error = pipeline_result["error_context"]

            for attempt in range(1, MAX_RETRIES + 1):
                self._telemetry.record_retry()
                scenario_result["attempts"] = attempt + 1  # +1 for initial run

                # ── Attempt remediation ──
                remediation = self._attempt_remediation(
                    error_context=last_error,
                    current_df=current_df,
                    attempt=attempt,
                )

                if not remediation["success"]:
                    print(f"     ⚠️  Remediation attempt {attempt} failed")

                    if attempt >= MAX_RETRIES:
                        break
                    continue

                # ── Re-run pipeline with fixed DataFrame ──
                print(f"\n  🔄 Re-running pipeline with patched data...")
                fixed_df = remediation["fixed_df"]
                retry_result = self._try_pipeline_steps(fixed_df)

                if retry_result["success"]:
                    # ── SUCCESS — Pipeline healed! ──
                    self._telemetry.record_success()
                    scenario_result["status"] = "healed"
                    scenario_result["healed"] = True
                    scenario_result["records_processed"] = retry_result["records_processed"]
                    scenario_result["final_patch"] = remediation["patch_code"]

                    # Store the successful fix in RAG memory
                    self._store_learned_fix(
                        error_context=last_error,
                        patch_code=remediation["patch_code"],
                    )

                    print(f"\n  ✨ PIPELINE HEALED after {attempt} remediation(s)!")
                    break

                else:
                    # Fix applied but pipeline still failing (different error?)
                    self._telemetry.record_failure(
                        retry_result["error_context"]["error_type"]
                    )
                    current_df = fixed_df  # Keep the partial fix
                    last_error = retry_result["error_context"]
                    print(f"     ⚠️  Pipeline still failing after fix — new error detected")

            else:
                # ── All retries exhausted ──
                scenario_result["status"] = "failed"
                print(f"\n  💀 MAX RETRIES EXCEEDED — Pipeline could not be healed")

            # If we broke out of the loop without setting status
            if scenario_result["status"] == "unknown":
                scenario_result["status"] = "failed"

        except Exception as e:
            scenario_result["status"] = "orchestrator_error"
            self._logger.error(
                "Orchestrator-level error",
                extra={
                    "scenario": scenario["name"],
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"\n  💥 Orchestrator error: {type(e).__name__}: {str(e)}")

        finally:
            self._telemetry.stop_timer()
            scenario_result["telemetry"] = self._telemetry.get_snapshot()

        return scenario_result

    # Full Demo Execution

    def run_all_scenarios(
        self,
        scenarios: Optional[list] = None,
    ) -> list[dict]:
        """
        Run all demo scenarios and produce a summary report.

        Args:
            scenarios: List of scenario dicts. Defaults to DEMO_SCENARIOS.

        Returns:
            List of scenario result dicts.
        """
        self._print_banner()
        scenarios = scenarios or DEMO_SCENARIOS
        all_results = []

        for i, scenario in enumerate(scenarios, 1):
            self._print_scenario_header(scenario, i)

            result = self.run_scenario(scenario)
            all_results.append(result)

            # Brief pause between scenarios for readable output
            if i < len(scenarios):
                time.sleep(0.5)

        # ── Print Summary ──
        self._print_summary(all_results)

        return all_results

    @staticmethod
    def _print_summary(results: list[dict]) -> None:
        """Print a formatted summary table of all scenarios."""
        print(f"\n{'═' * 66}")
        print("  SUMMARY REPORT")
        print(f"{'═' * 66}")
        print(f"  {'Scenario':<22} {'Status':<14} {'Attempts':<10} {'Records':<10}")
        print(f"  {'─' * 22} {'─' * 14} {'─' * 10} {'─' * 10}")

        healed_count = 0
        for r in results:
            icon = "✅" if r["healed"] else "❌"
            status = r["status"].upper()
            print(
                f"  {icon} {r['scenario']:<20} {status:<14} "
                f"{r['attempts']:<10} {r['records_processed']:<10}"
            )
            if r["healed"]:
                healed_count += 1

            # Show telemetry
            telemetry = r.get("telemetry", {})
            if telemetry:
                healing_time = telemetry.get("total_healing_time_ms", 0)
                llm_stats = telemetry.get("llm_latency_stats")
                llm_info = ""
                if llm_stats:
                    llm_info = f" | LLM avg: {llm_stats['avg_ms']}ms"
                print(f"     Total time: {healing_time}ms{llm_info}")

            if r.get("final_patch"):
                print(f"     Patch: {r['final_patch'][:70]}...")

        print(f"\n  {'─' * 56}")
        print(
            f"  Healed: {healed_count}/{len(results)} scenarios "
            f"({healed_count/len(results)*100:.0f}% success rate)"
        )
        print(f"{'═' * 66}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DataOps Auto-Healer: LLM-Driven Pipeline Remediation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run all demo scenarios
  python main.py --scenario schema_drift  # Run a specific scenario
  python main.py --scenario missing_column
  python main.py --scenario wrong_datatype

Environment:
  GROQ_API_KEY    Your Groq API key (required)
        """,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=["schema_drift", "missing_column", "wrong_datatype", "all"],
        default="all",
        help="Which failure scenario to run (default: all)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Maximum remediation attempts (default: {MAX_RETRIES})",
    )

    return parser.parse_args()


def main() -> None:
    """
    Application entry point.

    Performs pre-flight checks, then runs the auto-heal demo.
    """
    args = parse_args()

    # ── Pre-flight check: API key ──
    api_key = os.environ.get(GROQ_API_KEY_ENV_VAR)
    if not api_key:
        print(f"""

   GROQ API KEY NOT FOUND                                 
                                                              
  Set your API key using one of these methods:                
                                                              
  Option 1: Environment variable                             
    export {GROQ_API_KEY_ENV_VAR}=gsk_your_key_here              
                                                              
  Option 2: .env file                                        
    echo "{GROQ_API_KEY_ENV_VAR}=gsk_your_key" > .env             
                                                              
  Get your key at: https://console.groq.com/keys              

""")
        sys.exit(1)

    # ── Update global MAX_RETRIES if overridden ──
    global MAX_RETRIES
    MAX_RETRIES = args.max_retries

    # ── Initialize orchestrator ──
    try:
        orchestrator = AutoHealOrchestrator()
    except Exception as e:
        print(f"\n   Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # ── Select scenarios ──
    if args.scenario == "all":
        selected_scenarios = DEMO_SCENARIOS
    else:
        scenario_map = {
            "schema_drift": DEMO_SCENARIOS[0],
            "missing_column": DEMO_SCENARIOS[1],
            "wrong_datatype": DEMO_SCENARIOS[2],
        }
        selected_scenarios = [scenario_map[args.scenario]]

    # ── Run ──
    try:
        results = orchestrator.run_all_scenarios(selected_scenarios)

        # ── Exit code based on results ──
        all_healed = all(r["healed"] for r in results)
        sys.exit(0 if all_healed else 1)

    except KeyboardInterrupt:
        print("\n\n   Interrupted by user. Exiting gracefully.")
        sys.exit(130)

    except Exception as e:
        print(f"\n   Unhandled error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


# =====================================================================
if __name__ == "__main__":
    main()