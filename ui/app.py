import os
import sys
import time
import traceback

import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Project imports ──
from config.settings import FailureType, PipelineConfig, GROQ_API_KEY_ENV_VAR
from pipeline.data_pipeline import (
    generate_sample_csv,
    extract,
    inject_failure,
    validate_schema,
    transform,
    load,
)
from rag.vector_db import RAGMemoryStore, initialize_vector_store, seed_memory_if_empty
from agents.auto_healer_agent import AutoHealerAgent
from executor.safe_executor import SafeExecutor
from observability.observability import TelemetryCollector

from ui.state_manager import (
    initialize_session_state,
    add_step_log,
    reset_current_run,
)
from ui.components import (
    render_header,
    render_sidebar_config,
    render_dataframe_comparison,
    render_step_trace,
    render_agent_diagnostics,
    render_telemetry_dashboard,
    render_rag_explorer,
    render_safety_demo,
)

# ── Page Configuration ──
st.set_page_config(
    page_title="DataOps Auto-Healer",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize State ──
initialize_session_state()


SCENARIO_MAP = {
    "Schema Drift": FailureType.SCHEMA_DRIFT,
    "Missing Column": FailureType.MISSING_COLUMN,
    "Wrong Datatype": FailureType.WRONG_DATATYPE,
}


@st.cache_resource
def init_agent():
    """
    Initialize the AutoHealerAgent (cached across reruns).

    @st.cache_resource ensures the LLM client and RAG store
    are created only ONCE, even though Streamlit reruns the
    entire script on every interaction.
    """
    api_key = os.environ.get(GROQ_API_KEY_ENV_VAR)
    if not api_key:
        return None, "API key not found"

    try:
        initialize_vector_store()
        seed_memory_if_empty()
        agent = AutoHealerAgent()
        return agent, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def init_executor():
    """Initialize the SafeExecutor (cached)."""
    return SafeExecutor()


@st.cache_resource
def init_rag_store():
    """Initialize and return the RAG store for the explorer."""
    store = RAGMemoryStore().initialize()
    store.seed_memory_if_empty()
    return store

def run_heal_scenario(
    scenario_name: str,
    failure_type: FailureType,
    agent: AutoHealerAgent,
    executor: SafeExecutor,
    max_retries: int,
) -> dict:
    """
    Execute the full auto-heal flow for a given scenario.

    This is the Streamlit-adapted version of the CLI orchestrator.
    Instead of printing to console, it updates session state and
    renders progress in real-time using st.status().

    Args:
        scenario_name: Human-readable scenario label.
        failure_type:  Which failure to inject.
        agent:         Initialized AutoHealerAgent.
        executor:      Initialized SafeExecutor.
        max_retries:   Maximum remediation attempts.

    Returns:
        Scenario result dict.
    """
    reset_current_run()
    telemetry = TelemetryCollector()
    telemetry.start_timer()
    config = PipelineConfig()

    result = {
        "scenario": scenario_name,
        "failure_type": failure_type.value,
        "status": "unknown",
        "attempts": 0,
        "records_processed": 0,
        "healed": False,
        "final_patch": None,
        "telemetry": {},
    }

    with st.status(f"🔄 Running: {scenario_name}", expanded=True) as status:

        try:
            # ── Step 1: Extract ──
            st.write("📥 **Extracting data...**")
            if not os.path.exists(config.source_path):
                generate_sample_csv(config.source_path)

            raw_df = extract(config.source_path)
            st.session_state["raw_df"] = raw_df.copy()
            add_step_log("Extract Data", "success",
                        f"Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns",
                        {"shape": raw_df.shape, "columns": list(raw_df.columns)})

            # ── Step 2: Inject Failure ──
            st.write(f"⚡ **Injecting failure:** `{failure_type.value}`")
            corrupted_df = inject_failure(raw_df, failure_type)
            st.session_state["corrupted_df"] = corrupted_df.copy()
            add_step_log("Inject Failure", "warning",
                        f"Applied {failure_type.value} corruption",
                        {"columns_after": list(corrupted_df.columns)})

            # ── Step 3: Initial Pipeline Attempt ──
            st.write("🔄 **Running initial pipeline...**")
            result["attempts"] = 1

            try:
                validate_schema(corrupted_df)
                # If validation passes (shouldn't with injection), continue
                transformed = transform(corrupted_df)
                load_result = load(transformed, config.output_path)
                result["status"] = "success_no_heal"
                result["healed"] = True
                result["records_processed"] = load_result["rows_written"]
                add_step_log("Initial Run", "success", "Pipeline passed (no healing needed)")
                status.update(label="✅ Pipeline succeeded without healing", state="complete")
                telemetry.record_success()
                telemetry.stop_timer()
                result["telemetry"] = telemetry.get_snapshot()
                return result

            except Exception as e:
                error_context = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "stack_trace": traceback.format_exc(),
                }
                st.session_state["last_error_context"] = error_context
                telemetry.record_failure(error_context["error_type"])
                add_step_log("Initial Run", "failure",
                            f"{type(e).__name__}: {str(e)[:100]}",
                            error_context)
                st.write(f"❌ **Pipeline failed:** `{type(e).__name__}`")

            # ── Step 4: Heal Loop ──
            current_df = corrupted_df
            last_error = error_context

            for attempt in range(1, max_retries + 1):
                telemetry.record_retry()
                result["attempts"] = attempt + 1

                st.write(f"---")
                st.write(f"🔧 **Remediation Attempt {attempt}/{max_retries}**")

                # ── 4a: Call LLM Agent ──
                st.write("🤖 Calling GenAI agent...")
                agent_result = agent.diagnose_and_fix(last_error)
                st.session_state["last_agent_result"] = agent_result
                st.session_state["last_rag_results"] = agent_result.get("rag_results", [])
                telemetry.record_llm_latency(agent_result["llm_latency_ms"])
                st.session_state["latency_history"].append(agent_result["llm_latency_ms"])

                if agent_result["status"] != "success":
                    add_step_log(f"Agent (Attempt {attempt})", "failure",
                                f"Agent error: {agent_result.get('error_details', 'Unknown')}")
                    st.write(f"❌ Agent failed: {agent_result.get('error_details')}")
                    continue

                patch_code = agent_result["generated_patch"]
                st.session_state["last_patch_code"] = patch_code
                add_step_log(f"LLM Diagnosis (Attempt {attempt})", "success",
                            f"Generated patch ({agent_result['llm_latency_ms']:.0f}ms, "
                            f"RAG: {agent_result['retrieved_context_count']} fixes)",
                            patch_code)
                st.write(f"📝 **Patch generated** ({agent_result['llm_latency_ms']:.0f}ms)")
                st.code(patch_code, language="python")

                # ── 4b: Safe Execution ──
                st.write("🛡️ Validating & executing in sandbox...")
                exec_result = executor.validate_and_execute(patch_code, current_df)
                st.session_state["last_exec_result"] = exec_result

                if not exec_result["success"]:
                    add_step_log(f"Execution (Attempt {attempt})", "failure",
                                f"Blocked: {exec_result.get('error', 'Unknown')}",
                                exec_result)
                    st.write(f"❌ Execution failed: {exec_result.get('error')}")
                    continue

                fixed_df = exec_result["modified_df"]
                st.session_state["fixed_df"] = fixed_df.copy()
                add_step_log(f"Execution (Attempt {attempt})", "success",
                            f"Patch applied ({exec_result['execution_time_ms']:.1f}ms)")
                st.write(f"✅ **Patch applied** ({exec_result['execution_time_ms']:.1f}ms)")

                # ── 4c: Re-run pipeline ──
                st.write("🔄 Re-running pipeline with fixed data...")
                try:
                    validate_schema(fixed_df)
                    transformed = transform(fixed_df)
                    load_result = load(transformed, config.output_path)

                    # ── SUCCESS ──
                    telemetry.record_success()
                    result["status"] = "healed"
                    result["healed"] = True
                    result["records_processed"] = load_result["rows_written"]
                    result["final_patch"] = patch_code
                    st.session_state["final_df"] = transformed.copy()

                    add_step_log("Pipeline Retry", "success",
                                f"Pipeline healed! {load_result['rows_written']} rows processed")

                    # Store learned fix
                    try:
                        from rag.vector_db import add_error_fix_record
                        add_error_fix_record(
                            error_type=last_error["error_type"],
                            error_message=last_error["error_message"],
                            fix_code=patch_code,
                            fix_description=f"Auto-fix for {last_error['error_type']}",
                            tags="auto_generated,runtime_fix",
                        )
                        add_step_log("Store Fix", "success", "Fix saved to RAG memory")
                    except Exception:
                        pass

                    st.write(f"✨ **PIPELINE HEALED** after {attempt} attempt(s)!")
                    status.update(label=f"✨ Healed after {attempt} attempt(s)!", state="complete")
                    break

                except Exception as retry_error:
                    telemetry.record_failure(type(retry_error).__name__)
                    last_error = {
                        "error_type": type(retry_error).__name__,
                        "error_message": str(retry_error),
                        "stack_trace": traceback.format_exc(),
                    }
                    current_df = fixed_df
                    add_step_log("Pipeline Retry", "failure",
                                f"Still failing: {type(retry_error).__name__}")
                    st.write(f"⚠️ Pipeline still failing: `{type(retry_error).__name__}`")

            else:
                result["status"] = "failed"
                status.update(label="❌ Max retries exceeded", state="error")
                add_step_log("Max Retries", "failure",
                            f"Exhausted {max_retries} remediation attempts")

            if result["status"] == "unknown":
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "orchestrator_error"
            status.update(label=f"💥 Error: {type(e).__name__}", state="error")
            add_step_log("Orchestrator Error", "failure", str(e))

        finally:
            telemetry.stop_timer()
            result["telemetry"] = telemetry.get_snapshot()
            st.session_state["current_run"] = result

    return result


def main():
    """Main application controller."""

    render_header()

    # ── Sidebar ──
    sidebar_config = render_sidebar_config()

    # ── Initialize components ──
    agent, agent_error = init_agent()
    executor = init_executor()

    if agent:
        st.session_state["agent_initialized"] = True
        st.session_state["agent"] = agent
        st.session_state["executor"] = executor
    elif agent_error:
        st.error(
            f"⚠️ **Agent initialization failed:** {agent_error}\n\n"
            f"Set your API key: `export {GROQ_API_KEY_ENV_VAR}=gsk_your_key`"
        )

    # ── Tab Layout ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview",
        "🚀 Auto-Heal Demo",
        "🔍 Diagnostics",
        "📊 Telemetry",
        "🧪 Sandbox Lab",
    ])

    with tab1:


        st.markdown("### How It Works")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                #### 1️⃣ Detect
                The data pipeline runs and encounters a failure
                (schema drift, missing column, wrong datatype).
                The observability layer captures the full error context.
                """
            )

        with col2:
            st.markdown(
                """
                #### 2️⃣ Diagnose
                The GenAI agent queries RAG memory for similar past
                errors, constructs a prompt with error + context,
                and asks Llama-3 to generate a targeted Python patch.
                """
            )

        with col3:
            st.markdown(
                """
                #### 3️⃣ Heal
                The patch is validated via AST analysis, executed in
                a sandbox, and the pipeline is re-run. Successful
                fixes are stored in RAG memory for future use.
                """
            )

        st.markdown("---")
        st.markdown("### Tech Stack")
        cols = st.columns(6)
        stack_items = [
            ("🐼", "Pandas", "ETL Pipeline"),
            ("🧠", "Llama-3", "LLM via Groq"),
            ("🔗", "LangChain", "Agent Framework"),
            ("📦", "ChromaDB", "Vector Store"),
            ("🛡️", "AST Sandbox", "Safe Execution"),
            ("📊", "Streamlit", "Dashboard"),
        ]
        for col, (icon, name, desc) in zip(cols, stack_items):
            col.markdown(f"**{icon} {name}**\n\n{desc}")

    with tab2:
        st.markdown("### 🚀 Run Auto-Heal Scenario")

        if not agent:
            st.warning("⚠️ Set your GROQ_API_KEY to run the demo.")
        else:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                selected = sidebar_config["scenario"]
                st.markdown(f"**Selected Scenario:** {selected}")

            with col2:
                st.markdown(f"**Max Retries:** {sidebar_config['max_retries']}")

            with col3:
                run_button = st.button(
                    "▶️ Run Auto-Heal",
                    type="primary",
                    use_container_width=True,
                )

            if run_button:
                failure_type = SCENARIO_MAP[selected]
                result = run_heal_scenario(
                    scenario_name=selected,
                    failure_type=failure_type,
                    agent=agent,
                    executor=executor,
                    max_retries=sidebar_config["max_retries"],
                )

                # Update global telemetry
                st.session_state["heal_history"].append(result)
                st.session_state["total_scenarios_run"] += 1
                if result["healed"]:
                    st.session_state["total_healed"] += 1
                else:
                    st.session_state["total_failed"] += 1

            # ── Data Comparison ──
            st.markdown("---")
            st.markdown("### 📊 Data Comparison")

            render_dataframe_comparison(
                "🔵 Original (Clean)",
                st.session_state.get("raw_df"),
                "🔴 Corrupted (After Failure Injection)",
                st.session_state.get("corrupted_df"),
            )

            if st.session_state.get("fixed_df") is not None:
                st.markdown("")
                render_dataframe_comparison(
                    "🟡 After LLM Patch",
                    st.session_state.get("fixed_df"),
                    "🟢 Final Output",
                    st.session_state.get("final_df"),
                )

            st.markdown("---")
            st.markdown("### 📋 Step-by-Step Trace")
            render_step_trace(st.session_state.get("step_logs", []))

    with tab3:
        st.markdown("### 🔍 Agent Diagnostics")

        diag_subtab1, diag_subtab2 = st.tabs([
            "🤖 Last Agent Run",
            "📚 RAG Knowledge Base",
        ])

        with diag_subtab1:
            render_agent_diagnostics(
                error_context=st.session_state.get("last_error_context"),
                agent_result=st.session_state.get("last_agent_result"),
                exec_result=st.session_state.get("last_exec_result"),
                rag_results=st.session_state.get("last_rag_results"),
            )

        with diag_subtab2:
            try:
                rag_store = init_rag_store()
                render_rag_explorer(rag_store)
            except Exception as e:
                st.error(f"RAG store error: {e}")

    with tab4:
        st.markdown("### 📊 Telemetry Dashboard")
        render_telemetry_dashboard(
            heal_history=st.session_state.get("heal_history", []),
            latency_history=st.session_state.get("latency_history", []),
        )

    with tab5:
        st.markdown("### 🧪 Safe Execution Sandbox Lab")
        st.markdown(
            "Test the code safety scanner interactively. "
            "Try submitting safe and dangerous code to see how "
            "the 3-layer defense system works."
        )
        st.markdown("---")
        render_safety_demo(executor)

if __name__ == "__main__":
    main()