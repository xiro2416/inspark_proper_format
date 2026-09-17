"""Mutable model compatibility fields are request-local, never weight-global."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

@dataclass
class RequestState:
    request_id: str
    values: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
CURRENT_REQUEST: ContextVar[RequestState | None] = ContextVar('acc_infer_request', default=None)

@contextmanager
def request_scope(request_id, values=None):
    if CURRENT_REQUEST.get() is not None:
        raise RuntimeError('nested synthesis requests are not supported')
    state = RequestState(str(request_id), dict(values or {}))
    token = CURRENT_REQUEST.set(state)
    try:
        yield state
    finally:
        CURRENT_REQUEST.reset(token)
        state.values.clear()

class RequestField:
    """Bridge upstream attribute syntax to explicit per-request state.

    Outside synthesis, the initialization value is used (normally None). Fields
    must contain tensors/strings, not trainable Parameters or registered buffers.
    """

    def __init__(self, key):
        self.key = key

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        state = CURRENT_REQUEST.get()
        if state is not None:
            return state.values.get(self.key)
        return instance.__dict__.get('_initial_' + self.key)

    def __set__(self, instance, value):
        state = CURRENT_REQUEST.get()
        if state is not None:
            state.values[self.key] = value
        else:
            instance.__dict__['_initial_' + self.key] = value

