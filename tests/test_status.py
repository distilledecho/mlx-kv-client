"""Tests for MlxKvClient.status() and AsyncMlxKvClient.status()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlx_kv_client import (
    AsyncMlxKvClient,
    KVServerStatus,
    MlxKvClient,
    MlxKvConnectionError,
)

_SAMPLE_RESPONSE = {
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

_SAMPLE_RESPONSE_NO_CHECKPOINT = {
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


def _mock_response(body: object) -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.json.return_value = body
    r.raise_for_status.return_value = None
    return r


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


def test_status_returns_typed_dataclass() -> None:
    with patch("httpx.Client.get", return_value=_mock_response(_SAMPLE_RESPONSE)):
        with MlxKvClient("http://localhost:8080") as client:
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


def test_status_no_checkpoint_fields_are_none() -> None:
    with patch(
        "httpx.Client.get", return_value=_mock_response(_SAMPLE_RESPONSE_NO_CHECKPOINT)
    ):
        with MlxKvClient("http://localhost:8080") as client:
            result = client.status()

    assert result.checkpoint_present is False
    assert result.checkpoint_tokens is None
    assert result.last_operation is None
    assert result.last_operation_at is None


def test_status_is_frozen() -> None:
    with patch("httpx.Client.get", return_value=_mock_response(_SAMPLE_RESPONSE)):
        with MlxKvClient("http://localhost:8080") as client:
            result = client.status()

    with pytest.raises(AttributeError):
        result.cache_used_tokens = 0  # type: ignore[misc]


def test_status_connection_error_raises_typed_exception() -> None:
    with patch(
        "httpx.Client.get", side_effect=httpx.ConnectError("connection refused")
    ):
        client = MlxKvClient("http://localhost:8080")
        with pytest.raises(MlxKvConnectionError) as exc_info:
            client.status()
        client.close()

    assert "http://localhost:8080" in str(exc_info.value)
    assert isinstance(exc_info.value.cause, httpx.ConnectError)
    assert exc_info.value.url == "http://localhost:8080"


def test_status_timeout_raises_typed_exception() -> None:
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("timed out")):
        client = MlxKvClient("http://localhost:8080")
        with pytest.raises(MlxKvConnectionError) as exc_info:
            client.status()
        client.close()

    assert isinstance(exc_info.value.cause, httpx.TimeoutException)


def test_status_http_error_not_wrapped() -> None:
    """4xx/5xx from raise_for_status() propagates as httpx.HTTPStatusError."""
    mock_resp = _mock_response(_SAMPLE_RESPONSE)
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )
    with patch("httpx.Client.get", return_value=mock_resp):
        client = MlxKvClient("http://localhost:8080")
        with pytest.raises(httpx.HTTPStatusError):
            client.status()
        client.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


def test_async_status_returns_typed_dataclass() -> None:
    async def _run() -> KVServerStatus:
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            return_value=_mock_response(_SAMPLE_RESPONSE),
        ):
            async with AsyncMlxKvClient("http://localhost:8080") as client:
                return await client.status()

    result = asyncio.run(_run())

    assert isinstance(result, KVServerStatus)
    assert result.cache_used_tokens == 1842
    assert result.checkpoint_tokens == 1204
    assert result.model == "mlx-community/some-model"


def test_async_status_connection_error_raises_typed_exception() -> None:
    async def _run() -> None:
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            client = AsyncMlxKvClient("http://localhost:8080")
            with pytest.raises(MlxKvConnectionError) as exc_info:
                await client.status()
            await client.aclose()

        assert "http://localhost:8080" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, httpx.ConnectError)

    asyncio.run(_run())


def test_async_status_timeout_raises_typed_exception() -> None:
    async def _run() -> None:
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            client = AsyncMlxKvClient("http://localhost:8080")
            with pytest.raises(MlxKvConnectionError) as exc_info:
                await client.status()
            await client.aclose()

        assert isinstance(exc_info.value.cause, httpx.TimeoutException)

    asyncio.run(_run())
