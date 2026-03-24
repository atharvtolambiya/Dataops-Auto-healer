import streamlit as st
from dataclasses import dataclass, field
from typing import Optional


def initialize_session_state() -> None:
    """
    Initialize all session state variables with defaults.

    Called once at app startup. Uses setdefault() which is
    idempotent — safe to call on every rerun without resetting
    user-modified values.
    """
    defaults = {
        # ── System Components ──
        "agent_initialized": False,
        "agent": None,
        "executor": None,

        # ── Pipeline State ──
        "selected_scenario": "Schema Drift",
        "pipeline_running": False,
        "current_step": 0,
        "max_retries": 3,

        # ── Results History ──
        "heal_history": [],          # List of all scenario run results
        "current_run": None,         # Active run result dict

        # ── Step-by-Step Trace ──
        "step_logs": [],             # Ordered list of step dicts
        "raw_df": None,              # Original extracted DataFrame
        "corrupted_df": None,        # After failure injection
        "fixed_df": None,            # After LLM patch applied
        "final_df": None,            # After full pipeline

        # ── Agent Diagnostics ──
        "last_error_context": None,
        "last_agent_result": None,
        "last_patch_code": None,
        "last_exec_result": None,
        "last_rag_results": None,

        # ── Telemetry ──
        "total_scenarios_run": 0,
        "total_healed": 0,
        "total_failed": 0,
        "latency_history": [],       # List of LLM latency values
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def add_step_log(
    step_name: str,
    status: str,
    detail: str,
    data: Optional[dict] = None,
) -> None:
    """
    Append a structured step log to the current trace.

    Args:
        step_name: Human-readable step label (e.g., "Extract Data").
        status:    One of "success", "failure", "info", "warning".
        detail:    Description of what happened.
        data:      Optional extra data to display.
    """
    st.session_state["step_logs"].append({
        "step": step_name,
        "status": status,
        "detail": detail,
        "data": data,
    })


def reset_current_run() -> None:
    """Reset state for a new scenario run without clearing history."""
    keys_to_reset = [
        "current_run", "step_logs", "current_step",
        "raw_df", "corrupted_df", "fixed_df", "final_df",
        "last_error_context", "last_agent_result",
        "last_patch_code", "last_exec_result", "last_rag_results",
        "pipeline_running",
    ]
    for key in keys_to_reset:
        if key == "step_logs":
            st.session_state[key] = []
        elif key == "current_step":
            st.session_state[key] = 0
        elif key == "pipeline_running":
            st.session_state[key] = False
        else:
            st.session_state[key] = None