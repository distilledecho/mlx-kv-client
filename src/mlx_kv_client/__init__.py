"""Top level API.

.. data:: __version__
    :type: str

    Version number as calculated by https://github.com/pypa/setuptools_scm
"""

from ._client import (
    AsyncMlxKvClient,
    CheckpointResult,
    EvictResult,
    KVServerStatus,
    MlxKvClient,
    MlxKvConnectionError,
    MlxKvServerError,
    PrefillResult,
    RollbackResult,
)
from ._version import __version__

__all__ = [
    "__version__",
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
