"""Command-line entry point: one command per pipeline stage (added step by step)."""

import typer

from audiocfg import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="AudioCFG pipeline.")


@app.callback()
def main() -> None:
    """AudioCFG: learned global CFG null for a frozen Stable Audio Open."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
