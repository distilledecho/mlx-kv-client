"""HTTP client for mlx-kv-server.

Provides synchronous (:class:`MlxKvClient`) and asynchronous
(:class:`AsyncMlxKvClient`) variants for all six server endpoints:
``prefill``, ``generate``, ``checkpoint``, ``rollback``, ``evict``,
and ``status``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import TypedDict

import httpx

__all__ = [
    "AsyncMlxKvClient",
    "KVServerStatus",
    "MlxKvClient",
    "MlxKvConnectionError",
]


class MlxKvConnectionError(Exception):
    """Raised when the client cannot reach mlx-kv-server.

    Attributes
    ----------
    url:
        The base URL that was attempted.
    cause:
        The underlying httpx exception.
    """

    def __init__(self, url: str, cause: Exception) -> None:
        super().__init__(f"Cannot reach mlx-kv-server at {url!r}: {cause}")
        self.url = url
        self.cause = cause


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


class _StatusResponse(TypedDict):
    """Shape of the JSON body returned by ``GET /status``."""

    cache_used_tokens: int
    cache_capacity_tokens: int
    cache_used_fraction: float
    checkpoint_present: bool
    checkpoint_tokens: int | None
    last_operation: str | None
    last_operation_at: str | None
    model: str
    uptime_seconds: int


def _parse_status(data: _StatusResponse) -> KVServerStatus:
    return KVServerStatus(
        cache_used_tokens=data["cache_used_tokens"],
        cache_capacity_tokens=data["cache_capacity_tokens"],
        cache_used_fraction=data["cache_used_fraction"],
        checkpoint_present=data["checkpoint_present"],
        checkpoint_tokens=data["checkpoint_tokens"],
        last_operation=data["last_operation"],
        last_operation_at=data["last_operation_at"],
        model=data["model"],
        uptime_seconds=data["uptime_seconds"],
    )


class MlxKvClient:
    """Synchronous HTTP client for mlx-kv-server.

    Parameters
    ----------
    base_url:
        Base URL of the running mlx-kv-server instance,
        e.g. ``"http://localhost:8080"``.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> MlxKvClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def status(self) -> KVServerStatus:
        """Return the current KV cache status from ``GET /status``.

        Returns
        -------
        KVServerStatus
            Frozen dataclass snapshot of server state.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        try:
            response = self._http.get("/status")
        except httpx.ConnectError as exc:
            raise MlxKvConnectionError(self._base_url, exc) from exc
        response.raise_for_status()
        return _parse_status(response.json())


class AsyncMlxKvClient:
    """Asynchronous HTTP client for mlx-kv-server.

    Parameters
    ----------
    base_url:
        Base URL of the running mlx-kv-server instance,
        e.g. ``"http://localhost:8080"``.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url)

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncMlxKvClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def status(self) -> KVServerStatus:
        """Return the current KV cache status from ``GET /status``.

        Returns
        -------
        KVServerStatus
            Frozen dataclass snapshot of server state.

        Raises
        ------
        MlxKvConnectionError
            If the server cannot be reached.
        """
        try:
            response = await self._http.get("/status")
        except httpx.ConnectError as exc:
            raise MlxKvConnectionError(self._base_url, exc) from exc
        response.raise_for_status()
        return _parse_status(response.json())
