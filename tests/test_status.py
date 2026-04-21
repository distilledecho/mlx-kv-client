"""Tests for MlxKvClient and AsyncMlxKvClient — all six methods."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
from collections.abc import Callable
from typing import Any

import pytest

from mlx_kv_client import (
    AsyncMlxKvClient,
    CheckpointResult,
    EvictResult,
    KVServerStatus,
    MlxKvClient,
    MlxKvConnectionError,
    PrefillResult,
    RollbackResult,
)

# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------

ResponseFn = Callable[[dict[str, Any]], list[dict[str, Any]]]


class _FakeServer:
    """Minimal fake Unix socket server for testing client behaviour.

    Call :meth:`push` with one or more dicts to enqueue them as the response
    to the next incoming request.  A single call may produce multiple frames
    (e.g. the streaming ``generate`` response).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.requests: list[dict[str, Any]] = []
        self._responses: list[list[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(10)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def push(self, *frames: dict[str, Any]) -> None:
        """Enqueue *frames* as the response to the next request."""
        with self._lock:
            self._responses.append(list(frames))

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        buf = bytearray()
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                while b"\n" in buf:
                    idx = buf.index(b"\n")
                    raw = bytes(buf[:idx])
                    del buf[: idx + 1]
                    req: dict[str, Any] = json.loads(raw)
                    with self._lock:
                        self.requests.append(req)
                        frames = self._responses.pop(0) if self._responses else []
                    for frame in frames:
                        conn.sendall((json.dumps(frame) + "\n").encode())
        finally:
            conn.close()

    def close(self) -> None:
        self._sock.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


@pytest.fixture()
def server(tmp_path: Any) -> Any:
    s = _FakeServer(str(tmp_path / "mlx_kv.sock"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_STATUS_RESULT = {
    "cache_used_tokens": 1842,
    "cache_capacity_tokens": 8192,
    "cache_used_fraction": 0.225,
    "checkpoint_present": True,
    "checkpoint_tokens": 1204,
    "last_operation": "checkpoint",
    "last_operation_at": "2026-04-06T21:44:01Z",
    "model": "mlx-community/some-model",
    "uptime_seconds": 3124,
}

_STATUS_RESULT_EMPTY = {
    "cache_used_tokens": 0,
    "cache_capacity_tokens": 8192,
    "cache_used_fraction": 0.0,
    "checkpoint_present": False,
    "checkpoint_tokens": None,
    "last_operation": None,
    "last_operation_at": None,
    "model": "mlx-community/some-model",
    "uptime_seconds": 42,
}


# ===========================================================================
# Sync client — all six methods
# ===========================================================================


def test_prefill_returns_typed_result(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"handle": "cache-abc"}})
    with MlxKvClient(server.path) as client:
        result = client.prefill([1, 2, 3], "cache-abc")

    assert isinstance(result, PrefillResult)
    assert result.handle == "cache-abc"


def test_prefill_sends_correct_request(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"handle": "c1"}})
    with MlxKvClient(server.path) as client:
        client.prefill([10, 20], "c1")

    req = server.requests[0]
    assert req["method"] == "prefill"
    assert req["params"]["tokens"] == [10, 20]
    assert req["params"]["cache_id"] == "c1"


def test_generate_yields_tokens(server: _FakeServer) -> None:
    rid = 1
    server.push(
        {"id": rid, "token": 100},
        {"id": rid, "token": 200},
        {"id": rid, "token": 300},
        {"id": rid, "done": True},
    )
    with MlxKvClient(server.path) as client:
        tokens = list(client.generate([1, 2], "cache-xyz"))

    assert tokens == [100, 200, 300]


def test_generate_sends_correct_request(server: _FakeServer) -> None:
    rid = 1
    server.push({"id": rid, "done": True})
    with MlxKvClient(server.path) as client:
        list(client.generate([5, 6, 7], "c2"))

    req = server.requests[0]
    assert req["method"] == "generate"
    assert req["params"]["tokens"] == [5, 6, 7]
    assert req["params"]["cache_id"] == "c2"


def test_checkpoint_returns_typed_result(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"position": 512}})
    with MlxKvClient(server.path) as client:
        result = client.checkpoint("c1")

    assert isinstance(result, CheckpointResult)
    assert result.position == 512


def test_checkpoint_sends_correct_request(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"position": 0}})
    with MlxKvClient(server.path) as client:
        client.checkpoint("my-cache")

    req = server.requests[0]
    assert req["method"] == "checkpoint"
    assert req["params"]["cache_id"] == "my-cache"


def test_rollback_returns_typed_result(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"position": 256}})
    with MlxKvClient(server.path) as client:
        result = client.rollback("c1", 256)

    assert isinstance(result, RollbackResult)
    assert result.position == 256


def test_rollback_sends_correct_request(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"position": 128}})
    with MlxKvClient(server.path) as client:
        client.rollback("my-cache", 128)

    req = server.requests[0]
    assert req["method"] == "rollback"
    assert req["params"]["cache_id"] == "my-cache"
    assert req["params"]["position"] == 128


def test_evict_returns_typed_result(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"freed": "cache-xyz"}})
    with MlxKvClient(server.path) as client:
        result = client.evict("cache-xyz")

    assert isinstance(result, EvictResult)
    assert result.freed == "cache-xyz"


def test_evict_sends_correct_request(server: _FakeServer) -> None:
    server.push({"id": 1, "result": {"freed": "c3"}})
    with MlxKvClient(server.path) as client:
        client.evict("c3")

    req = server.requests[0]
    assert req["method"] == "evict"
    assert req["params"]["cache_id"] == "c3"


def test_status_returns_typed_dataclass(server: _FakeServer) -> None:
    server.push({"id": 1, "result": _STATUS_RESULT})
    with MlxKvClient(server.path) as client:
        result = client.status()

    assert isinstance(result, KVServerStatus)
    assert result.cache_used_tokens == 1842
    assert result.cache_capacity_tokens == 8192
    assert result.cache_used_fraction == 0.225
    assert result.checkpoint_present is True
    assert result.checkpoint_tokens == 1204
    assert result.last_operation == "checkpoint"
    assert result.last_operation_at == "2026-04-06T21:44:01Z"
    assert result.model == "mlx-community/some-model"
    assert result.uptime_seconds == 3124


def test_status_no_checkpoint_fields_are_none(server: _FakeServer) -> None:
    server.push({"id": 1, "result": _STATUS_RESULT_EMPTY})
    with MlxKvClient(server.path) as client:
        result = client.status()

    assert result.checkpoint_present is False
    assert result.checkpoint_tokens is None
    assert result.last_operation is None
    assert result.last_operation_at is None


def test_status_is_frozen(server: _FakeServer) -> None:
    server.push({"id": 1, "result": _STATUS_RESULT})
    with MlxKvClient(server.path) as client:
        result = client.status()

    with pytest.raises(AttributeError):
        result.cache_used_tokens = 0  # type: ignore[misc]


def test_connection_error_raises_typed_exception(tmp_path: Any) -> None:
    missing = str(tmp_path / "no_such.sock")
    client = MlxKvClient(missing)
    with pytest.raises(MlxKvConnectionError) as exc_info:
        client.status()
    assert missing in str(exc_info.value)
    assert exc_info.value.socket_path == missing
    assert isinstance(exc_info.value.cause, OSError)


# ===========================================================================
# Async client — all six methods
# ===========================================================================


def test_async_prefill_returns_typed_result(server: _FakeServer) -> None:
    async def _run() -> PrefillResult:
        server.push({"id": 1, "result": {"handle": "cache-abc"}})
        async with AsyncMlxKvClient(server.path) as client:
            return await client.prefill([1, 2, 3], "cache-abc")

    result = asyncio.run(_run())
    assert isinstance(result, PrefillResult)
    assert result.handle == "cache-abc"


def test_async_generate_yields_tokens(server: _FakeServer) -> None:
    async def _run() -> list[int]:
        rid = 1
        server.push(
            {"id": rid, "token": 100},
            {"id": rid, "token": 200},
            {"id": rid, "done": True},
        )
        async with AsyncMlxKvClient(server.path) as client:
            return [t async for t in client.generate([1, 2], "c1")]

    assert asyncio.run(_run()) == [100, 200]


def test_async_checkpoint_returns_typed_result(server: _FakeServer) -> None:
    async def _run() -> CheckpointResult:
        server.push({"id": 1, "result": {"position": 512}})
        async with AsyncMlxKvClient(server.path) as client:
            return await client.checkpoint("c1")

    result = asyncio.run(_run())
    assert isinstance(result, CheckpointResult)
    assert result.position == 512


def test_async_rollback_returns_typed_result(server: _FakeServer) -> None:
    async def _run() -> RollbackResult:
        server.push({"id": 1, "result": {"position": 256}})
        async with AsyncMlxKvClient(server.path) as client:
            return await client.rollback("c1", 256)

    result = asyncio.run(_run())
    assert isinstance(result, RollbackResult)
    assert result.position == 256


def test_async_evict_returns_typed_result(server: _FakeServer) -> None:
    async def _run() -> EvictResult:
        server.push({"id": 1, "result": {"freed": "cache-xyz"}})
        async with AsyncMlxKvClient(server.path) as client:
            return await client.evict("cache-xyz")

    result = asyncio.run(_run())
    assert isinstance(result, EvictResult)
    assert result.freed == "cache-xyz"


def test_async_status_returns_typed_dataclass(server: _FakeServer) -> None:
    async def _run() -> KVServerStatus:
        server.push({"id": 1, "result": _STATUS_RESULT})
        async with AsyncMlxKvClient(server.path) as client:
            return await client.status()

    result = asyncio.run(_run())
    assert isinstance(result, KVServerStatus)
    assert result.cache_used_tokens == 1842
    assert result.checkpoint_tokens == 1204
    assert result.model == "mlx-community/some-model"


def test_async_connection_error_raises_typed_exception(tmp_path: Any) -> None:
    async def _run() -> None:
        missing = str(tmp_path / "no_such.sock")
        async with AsyncMlxKvClient(missing) as client:
            with pytest.raises(MlxKvConnectionError) as exc_info:
                await client.status()
        assert missing in str(exc_info.value)
        assert exc_info.value.socket_path == missing
        assert isinstance(exc_info.value.cause, OSError)

    asyncio.run(_run())
