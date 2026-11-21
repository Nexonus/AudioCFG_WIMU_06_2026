"""Command-line entry point: one command per pipeline stage (added step by step)."""

from pathlib import Path

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


@app.command()
def generate(
    prompt: str,
    out: Path = Path("outputs/single/clip.wav"),
    guidance: float = 7.0,
    null: str = "zero",
    seed: int = 0,
    steps: int = 50,
    seconds: float = 10.0,
    dtype: str = "auto",
) -> None:
    """Generate one clip with a given null (zero, t5_empty or a .safetensors file)."""
    import time

    import soundfile as sf

    from audiocfg.models.sao import StableAudio
    from audiocfg.nulls.variants import resolve_null

    sao = StableAudio.load(dtype=dtype)
    null_text = resolve_null(null, sao)
    t0 = time.perf_counter()
    result = sao.sample([prompt], [seed], guidance, null_text, num_steps=steps, seconds=seconds)
    elapsed = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, result.audio[0].float().cpu().numpy().T, result.sample_rate)
    typer.echo(f"{out}  ({elapsed:.1f} s, dtype={sao.dtype}, device={sao.device})")


if __name__ == "__main__":
    app()
