# executor/__init__.py
# Exposes the public interface of the safe execution module.

from .safe_executor import SafeExecutor, safe_execute_patch

__all__ = ["SafeExecutor", "safe_execute_patch"]