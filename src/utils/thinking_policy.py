"""Per-call thinking policy, explicitly inherited by delivery worker tasks."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps


class ThinkingPolicy:
    def __init__(self):
        self._critical = ContextVar('thinking_latency_critical', default=False)

    @property
    def critical(self):
        return self._critical.get()

    @contextmanager
    def latency_critical(self, enabled=True):
        token = self._critical.set(bool(enabled))
        try:
            yield
        finally:
            self._critical.reset(token)

    def bind(self, function):
        """Capture this call chain; unrelated executor work keeps its own policy."""
        context = copy_context()
        @wraps(function)
        def run(*args, **kwargs):
            return context.copy().run(function, *args, **kwargs)
        return run


THINKING = ThinkingPolicy()
