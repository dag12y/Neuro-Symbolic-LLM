"""Keep a module's TorchAX global dispatch out of subsequent native PyTorch tests."""

import sys

import pytest


@pytest.fixture(scope="module", autouse=True)
def restore_torchax_mode():
    name = "frozenllm.substrate.torchax_backend"
    backend = sys.modules.get(name)
    started_enabled = bool(backend and backend.is_torchax_enabled())
    yield
    backend = sys.modules.get(name)
    if backend and not started_enabled and backend.is_torchax_enabled():
        backend.torchax.disable_globally()
        backend._torchax_enabled = False


@pytest.fixture(scope="module")
def torchax_mode(restore_torchax_mode):
    from frozenllm.substrate.torchax_backend import enable_torchax

    enable_torchax()
