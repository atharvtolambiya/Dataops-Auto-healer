import os
import re
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import (
    GROQ_MODEL_NAME,
    GROQ_API_KEY_ENV_VAR,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RAG_TOP_K,
)
from rag.vector_db import query_similar_errors, initialize_vector_store, seed_memory_if_empty
from observability.observability import StructuredLogger

load_dotenv()
logger = StructuredLogger.get_logger("agent")


SYSTEM_PROMPT = """You are an expert Python Data Engineer specializing in Pandas DataFrame remediation.

YOUR ROLE:
You receive a pipeline error and a set of similar past fixes retrieved from a knowledge base.
You must generate a Python code patch that fixes the error.

ABSOLUTE RULES — VIOLATION OF ANY RULE IS FORBIDDEN:
1. Output ONLY executable Python code. Nothing else.
2. Do NOT include any explanations, comments, or markdown formatting.
3. Do NOT use triple backticks, code fences, or language tags.
4. Do NOT include import statements of any kind.
5. Do NOT use eval(), exec(), compile(), or __import__().
6. Do NOT access the file system (open, read, write, os.path, pathlib).
7. Do NOT use os, sys, subprocess, shutil, or any system-level module.
8. Do NOT use network calls (requests, urllib, socket).
9. Do NOT define functions or classes.
10. The DataFrame variable is named exactly: df
11. ONLY perform column-level transformations on df.
12. Allowed operations: rename, fillna, astype, drop, assign, replace, str accessor, to_datetime, pd.to_numeric, pd.cut.
13. Your output will be directly executed via exec() in a sandboxed environment.
14. If you are unsure, output a safe no-op: pass

EXAMPLES OF VALID OUTPUT:
df = df.rename(columns={'cust_id': 'customer_id'})
df['email'] = 'unknown@placeholder.com'
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

EXAMPLES OF INVALID OUTPUT (NEVER DO THIS):
```python           ← NO markdown
import pandas as pd ← NO imports
def fix(df):        ← NO function definitions
# This fixes...     ← NO comments
```"""


FORBIDDEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bimport\s+\w+',        re.IGNORECASE), "Import statement detected"),
    (re.compile(r'\bfrom\s+\w+\s+import', re.IGNORECASE), "From-import statement detected"),
    (re.compile(r'\beval\s*\(',           re.IGNORECASE), "eval() call detected"),
    (re.compile(r'\bexec\s*\(',           re.IGNORECASE), "exec() call detected"),
    (re.compile(r'\bcompile\s*\(',        re.IGNORECASE), "compile() call detected"),
    (re.compile(r'\b__import__\s*\(',     re.IGNORECASE), "__import__() call detected"),
    (re.compile(r'\bopen\s*\(',           re.IGNORECASE), "File open() call detected"),
    (re.compile(r'\bos\.\w+',             re.IGNORECASE), "os module access detected"),
    (re.compile(r'\bsys\.\w+',            re.IGNORECASE), "sys module access detected"),
    (re.compile(r'\bsubprocess\.',        re.IGNORECASE), "subprocess module detected"),
    (re.compile(r'\bshutil\.',            re.IGNORECASE), "shutil module detected"),
    (re.compile(r'\brequests\.',          re.IGNORECASE), "requests module detected"),
    (re.compile(r'\burllib\.',            re.IGNORECASE), "urllib module detected"),
    (re.compile(r'\bsocket\.',            re.IGNORECASE), "socket module detected"),
    (re.compile(r'\b__builtins__',        re.IGNORECASE), "__builtins__ access detected"),
    (re.compile(r'\bglobals\s*\(',        re.IGNORECASE), "globals() call detected"),
    (re.compile(r'\blocals\s*\(',         re.IGNORECASE), "locals() call detected"),
    (re.compile(r'\bgetattr\s*\(',        re.IGNORECASE), "getattr() call detected"),
    (re.compile(r'\bsetattr\s*\(',        re.IGNORECASE), "setattr() call detected"),
    (re.compile(r'\bdelattr\s*\(',        re.IGNORECASE), "delattr() call detected"),
    (re.compile(r'\bbreakpoint\s*\(',     re.IGNORECASE), "breakpoint() call detected"),
]


class AutoHealerAgent:
    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        rag_top_k: int = RAG_TOP_K,
        api_key: Optional[str] = None,
    ):

        resolved_key = api_key or os.environ.get(GROQ_API_KEY_ENV_VAR)

        if not resolved_key:
            raise ValueError(
                f"Groq API key not found. Either:\n"
                f"  1. Set environment variable: export {GROQ_API_KEY_ENV_VAR}=gsk_...\n"
                f"  2. Create a .env file with: {GROQ_API_KEY_ENV_VAR}=gsk_...\n"
                f"  3. Pass api_key= to AutoHealerAgent()\n"
                f"NEVER hardcode API keys in source code."
            )

        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._rag_top_k = rag_top_k

        self._llm = ChatGroq(
            model=self._model_name,
            api_key=resolved_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        initialize_vector_store()
        seed_memory_if_empty()

        logger.info(
            "AutoHealerAgent initialized",
            extra={
                "model": self._model_name,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "rag_top_k": self._rag_top_k,
            },
        )


    def _format_error_for_query(self, error_context: dict) -> str:
        parts = [
            f"Error Type: {error_context.get('error_type', 'Unknown')}",
            f"Error Message: {error_context.get('error_message', 'No message')}",
        ]

        stack_trace = error_context.get("stack_trace")
        if stack_trace:

            trace_str = stack_trace if isinstance(stack_trace, str) else "".join(stack_trace)
            parts.append(f"Stack Trace:\n{trace_str[-300:]}")

        return "\n".join(parts)


    def _retrieve_similar_fixes(self, error_query: str) -> list[dict]:
        """
        Query ChromaDB for similar historical errors and their fixes.

        This is the RETRIEVAL step of Retrieval-Augmented Generation.
        The retrieved fixes provide concrete, proven examples that:
          • Ground the LLM's response in real solutions (reduces hallucination).
          • Provide pattern templates the LLM can adapt to the current error.
          • Act as few-shot examples within the prompt.

        Args:
            error_query: Formatted error string from _format_error_for_query().

        Returns:
            List of similar fix records from ChromaDB, sorted by similarity.
        """
        try:
            results = query_similar_errors(
                error_log=error_query,
                top_k=self._rag_top_k,
            )

            logger.info(
                "RAG retrieval completed",
                extra={
                    "query_preview": error_query[:100],
                    "results_count": len(results),
                    "top_similarity": results[0]["similarity_score"] if results else 0.0,
                },
            )

            return results

        except Exception as e:

            logger.warning(
                "RAG retrieval failed — proceeding without context",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            return []

    def _build_prompt(
        self,
        error_context: dict,
        similar_fixes: list[dict],
    ) -> list:


        human_parts = [
            "=== CURRENT PIPELINE ERROR ===",
            f"Error Type: {error_context.get('error_type', 'Unknown')}",
            f"Error Message: {error_context.get('error_message', 'No message')}",
        ]

        stack_trace = error_context.get("stack_trace")
        if stack_trace:
            trace_str = stack_trace if isinstance(stack_trace, str) else "".join(stack_trace)
            human_parts.append(f"Stack Trace (last 500 chars):\n{trace_str[-500:]}")

        if similar_fixes:
            human_parts.append("\n=== SIMILAR PAST ERRORS AND THEIR PROVEN FIXES ===")
            human_parts.append(
                "(These are real fixes that worked for similar errors. "
                "Use them as reference patterns.)"
            )

            for fix in similar_fixes:
                human_parts.extend([
                    f"\n--- Past Fix (Similarity: {fix['similarity_score']:.2f}) ---",
                    f"Error Type: {fix['error_type']}",
                    f"Fix Description: {fix['fix_description']}",
                    f"Fix Code:\n{fix['fix_code']}",
                ])
        else:
            human_parts.append(
                "\n(No similar past fixes found in knowledge base. "
                "Generate a fix based on the error details alone.)"
            )

        human_parts.extend([
            "\n=== YOUR TASK ===",
            "Generate a Python code patch that fixes the CURRENT error above.",
            "Use the past fixes as reference patterns if they are relevant.",
            "Output ONLY the executable Python code. No explanations.",
            "The DataFrame variable is named: df",
            "Remember: no imports, no markdown, no comments, no function definitions.",
        ])

        human_message_text = "\n".join(human_parts)

        logger.debug(
            "Prompt constructed",
            extra={
                "system_prompt_length": len(SYSTEM_PROMPT),
                "human_prompt_length": len(human_message_text),
                "rag_fixes_included": len(similar_fixes),
            },
        )

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_message_text),
        ]


    def _invoke_llm(self, messages: list) -> tuple[str, float]:
        start_time = time.perf_counter()

        try:
            response = self._llm.invoke(messages)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            raw_text = response.content

            logger.info(
                "LLM invocation completed",
                extra={
                    "model": self._model_name,
                    "latency_ms": elapsed_ms,
                    "response_length": len(raw_text),
                    "finish_reason": getattr(response, "response_metadata", {}).get(
                        "finish_reason", "unknown"
                    ),
                },
            )

            return raw_text, elapsed_ms

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "LLM invocation failed",
                extra={
                    "model": self._model_name,
                    "latency_ms": elapsed_ms,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise


    def _sanitize_response(self, raw_response: str) -> dict:

        sanitization_applied = False
        blocked_reasons = []

        cleaned = raw_response.strip()

        code_fence_pattern = re.compile(
            r'```(?:python|py)?\s*\n?(.*?)```',
            re.DOTALL | re.IGNORECASE,
        )
        fence_match = code_fence_pattern.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
            sanitization_applied = True

        if cleaned.startswith('`') and cleaned.endswith('`'):
            cleaned = cleaned.strip('`').strip()
            sanitization_applied = True

        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                sanitization_applied = True
                logger.warning(
                    "Stripped import line from LLM response",
                    extra={"removed_line": stripped_line},
                )
                continue
            if stripped_line.startswith('#'):
                sanitization_applied = True
                continue
            filtered_lines.append(line)

        cleaned = '\n'.join(filtered_lines).strip()

        for pattern, reason in FORBIDDEN_PATTERNS:
            if pattern.search(cleaned):
                blocked_reasons.append(reason)
                logger.error(
                    "Forbidden pattern detected in LLM output",
                    extra={
                        "reason": reason,
                        "pattern": pattern.pattern,
                        "response_preview": cleaned[:200],
                    },
                )

        is_safe = len(blocked_reasons) == 0

        # ── Step 4: Fallback if unsafe or empty ──
        if not is_safe:
            cleaned = "pass  # Blocked by safety scanner"
            logger.warning(
                "LLM patch blocked — falling back to no-op",
                extra={"blocked_reasons": blocked_reasons},
            )
        elif not cleaned or cleaned.isspace():
            cleaned = "pass  # Empty LLM response"
            sanitization_applied = True
            logger.warning("LLM returned empty response — falling back to no-op")

        if sanitization_applied:
            logger.info(
                "Response sanitization applied",
                extra={
                    "original_length": len(raw_response),
                    "cleaned_length": len(cleaned),
                },
            )

        return {
            "patch": cleaned,
            "was_sanitized": sanitization_applied,
            "blocked_reasons": blocked_reasons,
            "is_safe": is_safe,
        }

    def diagnose_and_fix(self, error_context: dict) -> dict:

        logger.info(
            "Agent diagnosis started",
            extra={
                "error_type": error_context.get("error_type"),
                "error_message_preview": str(error_context.get("error_message", ""))[:100],
            },
        )

        try:
            # ── Step 1: Format error for RAG query ──
            error_query = self._format_error_for_query(error_context)

            # ── Step 2: Retrieve similar past fixes ──
            similar_fixes = self._retrieve_similar_fixes(error_query)

            # ── Step 3: Build the prompt ──
            messages = self._build_prompt(error_context, similar_fixes)

            # ── Step 4: Call the LLM ──
            raw_response, latency_ms = self._invoke_llm(messages)

            # ── Step 5: Sanitize the response ──
            sanitization_result = self._sanitize_response(raw_response)

            result = {
                "generated_patch": sanitization_result["patch"],
                "llm_latency_ms": latency_ms,
                "retrieved_context_count": len(similar_fixes),
                "model_used": self._model_name,
                "rag_results": similar_fixes,
                "is_safe": sanitization_result["is_safe"],
                "was_sanitized": sanitization_result["was_sanitized"],
                "blocked_reasons": sanitization_result["blocked_reasons"],
                "status": "success",
                "error_details": None,
            }

            logger.info(
                "Agent diagnosis completed",
                extra={
                    "patch_length": len(result["generated_patch"]),
                    "latency_ms": latency_ms,
                    "rag_context_count": len(similar_fixes),
                    "is_safe": result["is_safe"],
                    "was_sanitized": result["was_sanitized"],
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Agent diagnosis failed",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

            return {
                "generated_patch": "pass  # Agent error — no patch generated",
                "llm_latency_ms": 0.0,
                "retrieved_context_count": 0,
                "model_used": self._model_name,
                "rag_results": [],
                "is_safe": False,
                "was_sanitized": False,
                "blocked_reasons": [],
                "status": "error",
                "error_details": f"{type(e).__name__}: {str(e)}",
            }


    def get_agent_info(self) -> dict:

        return {
            "model_name": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "rag_top_k": self._rag_top_k,
            "system_prompt_length": len(SYSTEM_PROMPT),
            "forbidden_patterns_count": len(FORBIDDEN_PATTERNS),
        }




if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  DataOps Auto-Healer — GenAI Agent Smoke Test")
    print("=" * 70)

    # ── Pre-flight check: API key ──
    api_key = os.environ.get(GROQ_API_KEY_ENV_VAR)
    if not api_key:
        print(
            f"\n  ⚠ ERROR: {GROQ_API_KEY_ENV_VAR} not set.\n"
            f"  Set it with: export {GROQ_API_KEY_ENV_VAR}=gsk_your_key\n"
            f"  Or create a .env file with: {GROQ_API_KEY_ENV_VAR}=gsk_your_key\n"
        )
        exit(1)

    # ── Initialize agent ──
    print("\n Initializing AutoHealerAgent...")
    agent = AutoHealerAgent()
    print(f"   Agent ready: {json.dumps(agent.get_agent_info(), indent=4)}")

    # ── Test 1: Schema Drift Error ──
    print("\n" + "-" * 70)
    print(" Test 1: Schema Drift — 'customer_id' renamed to 'cust_id'")
    print("-" * 70)

    schema_drift_error = {
        "error_type": "ValueError",
        "error_message": (
            "[SCHEMA DRIFT DETECTED] Unexpected columns found: ['cust_id']. "
            "Expected exactly: ['amount', 'customer_id', 'email', 'name', "
            "'transaction_date']."
        ),
        "stack_trace": (
            "Traceback (most recent call last):\n"
            "  File 'pipeline/data_pipeline.py', line 142, in run_pipeline\n"
            "    validate_schema(raw_df)\n"
            "ValueError: [SCHEMA DRIFT DETECTED] Unexpected columns found: ['cust_id']"
        ),
    }

    result_1 = agent.diagnose_and_fix(schema_drift_error)
    print(f"\n  Status: {result_1['status']}")
    print(f"  Safe: {result_1['is_safe']}")
    print(f"  Latency: {result_1['llm_latency_ms']}ms")
    print(f"  RAG Context: {result_1['retrieved_context_count']} fixes retrieved")
    print(f"  Generated Patch:\n    {result_1['generated_patch']}")

    # ── Test 2: Missing Column Error ──
    print("\n" + "-" * 70)
    print(" Test 2: Missing Column — 'email' dropped from source")
    print("-" * 70)

    missing_col_error = {
        "error_type": "KeyError",
        "error_message": (
            "[SCHEMA VALIDATION FAILED] Missing required columns: ['email']. "
            "Available columns: ['amount', 'customer_id', 'name', 'transaction_date']."
        ),
        "stack_trace": (
            "Traceback (most recent call last):\n"
            "  File 'pipeline/data_pipeline.py', line 142, in run_pipeline\n"
            "    validate_schema(raw_df)\n"
            "KeyError: [SCHEMA VALIDATION FAILED] Missing required columns: ['email']"
        ),
    }

    result_2 = agent.diagnose_and_fix(missing_col_error)
    print(f"\n  Status: {result_2['status']}")
    print(f"  Safe: {result_2['is_safe']}")
    print(f"  Latency: {result_2['llm_latency_ms']}ms")
    print(f"  RAG Context: {result_2['retrieved_context_count']} fixes retrieved")
    print(f"  Generated Patch:\n    {result_2['generated_patch']}")

    # ── Test 3: Dtype Mismatch Error ──
    print("\n" + "-" * 70)
    print("Test 3: Dtype Mismatch — 'amount' has string values")
    print("-" * 70)

    dtype_error = {
        "error_type": "TypeError",
        "error_message": (
            "[DTYPE MISMATCH] Column 'amount' has dtype 'object', "
            "expected 'float64'. Sample values: ['150.0', 'INVALID', '89.99']"
        ),
        "stack_trace": (
            "Traceback (most recent call last):\n"
            "  File 'pipeline/data_pipeline.py', line 142, in run_pipeline\n"
            "    validate_schema(raw_df)\n"
            "TypeError: [DTYPE MISMATCH] Column 'amount' has dtype 'object'"
        ),
    }

    result_3 = agent.diagnose_and_fix(dtype_error)
    print(f"\n  Status: {result_3['status']}")
    print(f"  Safe: {result_3['is_safe']}")
    print(f"  Latency: {result_3['llm_latency_ms']}ms")
    print(f"  RAG Context: {result_3['retrieved_context_count']} fixes retrieved")
    print(f"  Generated Patch:\n    {result_3['generated_patch']}")

    print("\n" + "=" * 70)
    print("  Agent Smoke Test Summary")
    print("=" * 70)
    for i, result in enumerate([result_1, result_2, result_3], 1):
        status_icon = "✓" if result["status"] == "success" and result["is_safe"] else "✗"
        print(
            f"  Test {i}: {status_icon} | "
            f"Status={result['status']} | "
            f"Safe={result['is_safe']} | "
            f"Latency={result['llm_latency_ms']}ms | "
            f"RAG={result['retrieved_context_count']}"
        )
    print("=" * 70)