"""Unix socket NDJSON client for mlx-kv-server.

Provides synchronous (:class:`MlxKvClient`) and asynchronous
(:class:`AsyncMlxKvClient`) variants for all six server methods:
``prefill``, ``generate``, ``checkpoint``, ``rollback``, ``evict``,
and ``status``.
"""

from __future__ import annotations

import asyncio
import json
import socket as _socket
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from itertools import count
from types import TracebackType
from typing import Any

__all__ = [
    "AsyncMlxKvClient",
    "CheckpointResult",
    "EvictResult",
    "KVServerStatus",
    "MlxKvClient",
    "MlxKvConnectionError",
    "MlxKvServerError",
    "PrefillResult",
    "RollbackResult",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MlxKvConnectionError(Exception):
    """Raised when the client cannot reach mlx-kv-server.

    Attributes
    ----------
    socket_path:
        The Unix socket path that was attempted.
    cause:
        The underlying exception.
    """

    def __init__(self, socket_path: str, cause: Exception) -> None:
        super().__init__(f"Cannot reach mlx-kv-server at {socket_path!r}: {cause}")
        self.socket_path = socket_path
        self.cause = cause


class MlxKvServerError(Exception):
    """Raised when mlx-kv-server returns an error frame.

    Attributes
    ----------
    socket_path:
        The Unix socket path of the server that returned the error.
    message:
        The error message string returned by the server.
    """

    def __init__(self, socket_path: str, message: str) -> None:
        super().__init__(f"server error at {socket_path!r}: {message}")
        self.socket_path = socket_path
        self.message = message


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVServerStatus:
    """Read-only snapshot of mlx-kv-server state.

    Returned by :meth:`MlxKvClient.status` and
    :meth:`AsyncMlxKvClient.status`.

    Attributes
    ----------
    cache_used_tokens:
        Number of tokens currently occupying the KV cache.
    cache_capacity_tokens:
        Total KV cache capacity in tokens.
    cache_used_fraction:
        Fraction of cache in use (0.0–1.0).
    checkpoint_present:
        Whether a checkpoint is currently saved.
    checkpoint_tokens:
        Token count at the checkpoint, or ``None`` if no checkpoint exists.
    last_operation:
        Name of the last primitive called (e.g. ``"prefill"``), or
        ``None`` if no operation has been performed since server start.
    last_operation_at:
        ISO 8601 timestamp of the last operation, or ``None``.
    model:
        Identifier of the loaded model.
    uptime_seconds:
        Seconds elapsed since mlx-kv-server started.
    """

    cache_used_tokens: int
    cache_capacity_tokens: int
    cache_used_fraction: float
    checkpoint_present: bool
    checkpoint_tokens: int | None
    last_operation: str | None
    last_operation_at: str | None
    model: str
    uptime_seconds: int


@dataclass(frozen=True)
class PrefillResult:
    """Result of a ``prefill`` call.

    Attributes
    ----------
    handle:
        Opaque cache handle — the same value as the ``cache_id`` passed in.
    """

    handle: str


@dataclass(frozen=True)
class CheckpointResult:
    """Result of a ``checkpoint`` call.

    Attributes
    ----------
    position:
        Token count at the time the checkpoint was taken.
    """

    position: int


@dataclass(frozen=True)
class RollbackResult:
    """Result of a ``rollback`` call.

    Attributes
    ----------
    position:
        Token position the cache was restored to.
    """

    position: int


@dataclass(frozen=True)
class EvictResult:
    """Result of an ``evict`` call.

    Attributes
    ----------
    freed:
        The ``cache_id`` that was freed.
    """

    freed: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _SocketReader:
    """Buffered line reader over a blocking :class:`socket.socket`."""

    def __init__(self, sock: _socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def readline(self) -> bytes:
        """Read bytes up to the next newline; return without the newline."""
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                if self._buf:
                    raise OSError("connection closed mid-line")
                return b""
            self._buf.extend(chunk)
        idx = self._buf.index(b"\n")
        line = bytes(self._buf[:idx])
        del self._buf[: idx + 1]
        return line


def _check_error(response: dict[str, Any], socket_path: str) -> None:
    if "error" in response:
        raise MlxKvServerError(socket_path, str(response["error"]))


def _parse_status(result: dict[str, Any]) -> KVServerStatus:
    cp = result["checkpoint_tokens"]
    last_op = result["last_operation"]
    last_op_at = result["last_operation_at"]
    return KVServerStatus(
        cache_used_tokens=int(result["cache_used_tokens"]),
        cache_capacity_tokens=int(result["cache_capacity_tokens"]),
        cache_used_fraction=float(result["cache_used_fraction"]),
        checkpoint_present=bool(result["checkpoint_present"]),
        checkpoint_tokens=int(cp) if cp is not None else None,
        last_operation=str(last_op) if last_op is not None else None,
        last_operation_at=str(last_op_at) if last_op_at is not None else None,
        model=str(result["model"]),
        uptime_seconds=int(result["uptime_seconds"]),
    )


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class MlxKvClient:
    """Synchronous Unix socket client for mlx-kv-server.

    Each method opens a fresh connection, sends one request, reads the
    response, then closes the connection.

    Parameters
    ----------
    socket_path:
        Filesystem path to the Unix domain socket of the running
        mlx-kv-server instance.
    """

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._ids: Iterator[int] = count(1)

    def close(self) -> None:
        """No-op. Connections are per-call and close themselves."""

    def __enter__(self) -> MlxKvClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _connect(self) -> tuple[_socket.socket, _SocketReader]:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        try:
            sock.connect(self._socket_path)
        except OSError as exc:
            sock.close()
            raise MlxKvConnectionError(self._socket_path, exc) from exc
        return sock, _SocketReader(sock)

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = next(self._ids)
        sock, reader = self._connect()
        try:
            payload = (
                json.dumps({"id": req_id, "method": method, "params": params}) + "\n"
            )
            sock.sendall(payload.encode())
            raw = reader.readline()
            if not raw:
                raise OSError("connection closed before response was received")
            response: dict[str, Any] = json.loads(raw)
            _check_error(response, self._socket_path)
            result: dict[str, Any] = response["result"]
            return result
        except OSError as exc:
            raise MlxKvConnectionError(self._socket_path, exc) from exc
        finally:
            sock.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill(self, tokens: list[int], cache_id: str) -> PrefillResult:
        """Feed tokens into the KV cache.

        Parameters
        ----------
        tokens:
            Token IDs to prefill.
        cache_id:
            Opaque cache handle. A new cache is created if not yet known.

        Returns
        -------
        PrefillResult
            Contains ``handle`` — the opaque cache reference.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = self._rpc("prefill", {"tokens": tokens, "cache_id": cache_id})
        return PrefillResult(handle=str(result["handle"]))

    def generate(self, tokens: list[int], cache_id: str) -> Iterator[int]:
        """Stream generated tokens from the KV cache.

        Parameters
        ----------
        tokens:
            Prompt tokens for this generation turn.
        cache_id:
            Opaque cache handle (must already exist from a prior prefill).

        Yields
        ------
        int
            Generated token IDs, one at a time, until the server sends
            ``done``.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached or drops the connection mid-stream.
        """
        req_id = next(self._ids)
        sock, reader = self._connect()
        try:
            payload = (
                json.dumps(
                    {
                        "id": req_id,
                        "method": "generate",
                        "params": {"tokens": tokens, "cache_id": cache_id},
                    }
                )
                + "\n"
            )
            sock.sendall(payload.encode())
            while True:
                raw = reader.readline()
                if not raw:
                    raise OSError("connection closed before generate completed")
                response: dict[str, Any] = json.loads(raw)
                _check_error(response, self._socket_path)
                if response.get("done"):
                    break
                yield int(response["token"])
        except OSError as exc:
            raise MlxKvConnectionError(self._socket_path, exc) from exc
        finally:
            sock.close()

    def checkpoint(self, cache_id: str) -> CheckpointResult:
        """Snapshot the current KV cache state.

        Parameters
        ----------
        cache_id:
            Opaque cache handle.

        Returns
        -------
        CheckpointResult
            Contains ``position`` — token count at checkpoint time.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = self._rpc("checkpoint", {"cache_id": cache_id})
        return CheckpointResult(position=int(result["position"]))

    def rollback(self, cache_id: str, position: int) -> RollbackResult:
        """Restore the KV cache to a prior position.

        Parameters
        ----------
        cache_id:
            Opaque cache handle.
        position:
            Token position to restore to (from a prior checkpoint).

        Returns
        -------
        RollbackResult
            Contains ``position`` — the restored token position.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = self._rpc("rollback", {"cache_id": cache_id, "position": position})
        return RollbackResult(position=int(result["position"]))

    def evict(self, cache_id: str) -> EvictResult:
        """Free GPU memory for a cache slot.

        Parameters
        ----------
        cache_id:
            Opaque cache handle to remove.

        Returns
        -------
        EvictResult
            Contains ``freed`` — the cache ID that was freed.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = self._rpc("evict", {"cache_id": cache_id})
        return EvictResult(freed=str(result["freed"]))

    def status(self) -> KVServerStatus:
        """Return the current KV cache status.

        Returns
        -------
        KVServerStatus
            Frozen dataclass snapshot of server state.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = self._rpc("status", {})
        return _parse_status(result)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncMlxKvClient:
    """Asynchronous Unix socket client for mlx-kv-server.

    Each method opens a fresh connection, sends one request, reads the
    response, then closes the connection.

    Parameters
    ----------
    socket_path:
        Filesystem path to the Unix domain socket of the running
        mlx-kv-server instance.
    """

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._ids: Iterator[int] = count(1)

    async def aclose(self) -> None:
        """No-op. Connections are per-call and close themselves."""

    async def __aenter__(self) -> AsyncMlxKvClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            return await asyncio.open_unix_connection(self._socket_path)
        except OSError as exc:
            raise MlxKvConnectionError(self._socket_path, exc) from exc

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = next(self._ids)
        reader, writer = await self._connect()
        try:
            payload = (
                json.dumps({"id": req_id, "method": method, "params": params}) + "\n"
            )
            writer.write(payload.encode())
            await writer.drain()
            raw = await reader.readline()
            if not raw:
                raise OSError("connection closed before response was received")
            response: dict[str, Any] = json.loads(raw)
            _check_error(response, self._socket_path)
            result: dict[str, Any] = response["result"]
            return result
        except OSError as exc:
            raise MlxKvConnectionError(self._socket_path, exc) from exc
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def prefill(self, tokens: list[int], cache_id: str) -> PrefillResult:
        """Feed tokens into the KV cache.

        Parameters
        ----------
        tokens:
            Token IDs to prefill.
        cache_id:
            Opaque cache handle. A new cache is created if not yet known.

        Returns
        -------
        PrefillResult
            Contains ``handle`` — the opaque cache reference.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = await self._rpc("prefill", {"tokens": tokens, "cache_id": cache_id})
        return PrefillResult(handle=str(result["handle"]))

    async def generate(
        self, tokens: list[int], cache_id: str
    ) -> AsyncGenerator[int, None]:
        """Stream generated tokens from the KV cache.

        Parameters
        ----------
        tokens:
            Prompt tokens for this generation turn.
        cache_id:
            Opaque cache handle (must already exist from a prior prefill).

        Yields
        ------
        int
            Generated token IDs, one at a time, until the server sends
            ``done``.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached or drops the connection mid-stream.
        """
        req_id = next(self._ids)
        reader, writer = await self._connect()
        try:
            payload = (
                json.dumps(
                    {
                        "id": req_id,
                        "method": "generate",
                        "params": {"tokens": tokens, "cache_id": cache_id},
                    }
                )
                + "\n"
            )
            writer.write(payload.encode())
            await writer.drain()
            while True:
                raw = await reader.readline()
                if not raw:
                    raise MlxKvConnectionError(
                        self._socket_path,
                        OSError("connection closed before generate completed"),
                    )
                response: dict[str, Any] = json.loads(raw)
                _check_error(response, self._socket_path)
                if response.get("done"):
                    break
                yield int(response["token"])
        except OSError as exc:
            raise MlxKvConnectionError(self._socket_path, exc) from exc
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def checkpoint(self, cache_id: str) -> CheckpointResult:
        """Snapshot the current KV cache state.

        Parameters
        ----------
        cache_id:
            Opaque cache handle.

        Returns
        -------
        CheckpointResult
            Contains ``position`` — token count at checkpoint time.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = await self._rpc("checkpoint", {"cache_id": cache_id})
        return CheckpointResult(position=int(result["position"]))

    async def rollback(self, cache_id: str, position: int) -> RollbackResult:
        """Restore the KV cache to a prior position.

        Parameters
        ----------
        cache_id:
            Opaque cache handle.
        position:
            Token position to restore to (from a prior checkpoint).

        Returns
        -------
        RollbackResult
            Contains ``position`` — the restored token position.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = await self._rpc(
            "rollback", {"cache_id": cache_id, "position": position}
        )
        return RollbackResult(position=int(result["position"]))

    async def evict(self, cache_id: str) -> EvictResult:
        """Free GPU memory for a cache slot.

        Parameters
        ----------
        cache_id:
            Opaque cache handle to remove.

        Returns
        -------
        EvictResult
            Contains ``freed`` — the cache ID that was freed.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = await self._rpc("evict", {"cache_id": cache_id})
        return EvictResult(freed=str(result["freed"]))

    async def status(self) -> KVServerStatus:
        """Return the current KV cache status.

        Returns
        -------
        KVServerStatus
            Frozen dataclass snapshot of server state.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        result = await self._rpc("status", {})
        return _parse_status(result)
