"""Optional, context-local observation hooks for the existing A1 driver."""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

observer: ContextVar[Callable[..., None] | None] = ContextVar(
    "stage_observer", default=None
)


def observe(event: str, **values: Any) -> None:
    callback = observer.get()
    if callback is not None:
        callback(event, **values)
