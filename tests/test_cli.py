import subprocess
import sys

from mlx_kv_client import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "mlx_kv_client", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
