"""
Offline-runtime contract tests.

The Docker image sets HF_HUB_OFFLINE=1 and pre-bakes the embedder so the
container never reaches huggingface.co at runtime. This module pins that
contract: with the model cached locally, a full upsert + search cycle must
complete with no outbound TCP connection. A regression that introduces a
new HuggingFace fetch (or any other phone-home) at runtime fails here.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest

from rag.store import DEFAULT_MODEL_NAME, VectorStore

if TYPE_CHECKING:
    from pathlib import Path

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(scope="module")
def cached_default_model() -> str:
    """
    Ensure the default embedder is cached so the offline test can load it.

    Constructing a SentenceTransformer with the default model name pulls
    the weights into the local HuggingFace cache if they are not already
    present. Subsequent runs find them cached and pay no network cost.
    """

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    SentenceTransformer(DEFAULT_MODEL_NAME)
    return DEFAULT_MODEL_NAME


@pytest.fixture
def block_outbound_tcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make every non-loopback TCP connect raise a RuntimeError.

    Monkeypatches socket.socket.connect at the class level so any code
    path that opens a real socket (urllib3, requests, raw sockets) hits
    the guard. Stacks on top of HF_HUB_OFFLINE=1: if huggingface_hub
    respects the env var, the socket is never created. If a future code
    path bypasses huggingface_hub, this guard catches it instead of
    silently making a network call.
    """

    real_connect = socket.socket.connect

    def _checked_connect(
        self: socket.socket, address: object, *args: object, **kwargs: object
    ) -> None:
        """Allow connects to loopback only, raise on anything else."""

        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK_HOSTS:
            raise RuntimeError(f"Outbound network blocked in offline test: {address!r}")
        return real_connect(self, address, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", _checked_connect)


def test_index_and_search_under_hub_offline(
    cached_default_model: str,
    tmp_path: Path,
    block_outbound_tcp: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Full upsert + search cycle works with HF_HUB_OFFLINE=1 and TCP blocked.

    The Docker image runs with HF_HUB_OFFLINE=1 and a pre-cached embedder.
    Any new code path that performs a runtime HuggingFace fetch would either
    trip huggingface_hub's offline guard or the socket guard installed by
    block_outbound_tcp, so this test catches both regression paths.
    """

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with VectorStore(tmp_path / "data", model_name=cached_default_model) as store:
        store.upsert_file(
            "/tmp/offline.txt",
            [
                {
                    "text": "the quick brown fox jumps over the lazy dog",
                    "section": "intro",
                    "chunk_index": 0,
                }
            ],
            "h1",
        )
        results = store.search("fox", top_k=1)

    assert len(results) == 1
    assert "fox" in results[0].excerpt
