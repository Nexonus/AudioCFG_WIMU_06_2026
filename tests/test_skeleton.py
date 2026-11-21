import importlib

from typer.testing import CliRunner

import audiocfg
from audiocfg.cli import app


def test_subpackages_import():
    for name in audiocfg.SUBPACKAGES:
        importlib.import_module(f"audiocfg.{name}")


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == audiocfg.__version__
