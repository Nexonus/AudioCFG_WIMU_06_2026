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


@app.command("prepare-data")
def prepare_data(config: Path = Path("configs/data.yaml")) -> None:
    import pandas as pd

    from audiocfg.data.fma_subset import split_by_artist
    from audiocfg.data.prompts import eval_prompts, sanity_tracks
    from audiocfg.repro.config import DataConfig
    from audiocfg.repro.manifest import build_manifest, write_manifest

    cfg = DataConfig.from_yaml(config)
    out_dir = Path(cfg.output_dir)
    song_describer = pd.read_csv(cfg.song_describer_csv)
    prompts = eval_prompts(
        song_describer,
        pd.read_csv(cfg.clotho_captions_csv),
        cfg.prompts_per_domain,
        cfg.master_seed,
    )
    tracks = split_by_artist(
        pd.read_csv(cfg.sao_fma_tracks, encoding="utf-8-sig"),
        cfg.null_train_tracks,
        cfg.null_val_tracks,
        cfg.master_seed,
    )
    sanity = sanity_tracks(
        song_describer,
        prompts.loc[prompts.domain == "music", "source_id"],
        cfg.sanity_tracks,
        cfg.master_seed,
    )
    prompts_path, tracks_path = out_dir / "eval_prompts.csv", out_dir / "null_tracks.csv"
    sanity_path = out_dir / "sanity_tracks.csv"
    sanity.to_csv(sanity_path, index=False, lineterminator="\n")
    prompts.to_csv(prompts_path, index=False, lineterminator="\n")
    tracks.to_csv(tracks_path, index=False, lineterminator="\n")
    manifest = build_manifest(
        cfg,
        inputs={
            "sao_fma_tracks": cfg.sao_fma_tracks,
            "song_describer_csv": cfg.song_describer_csv,
            "clotho_captions_csv": cfg.clotho_captions_csv,
        },
        outputs={
            "eval_prompts": prompts_path,
            "null_tracks": tracks_path,
            "sanity_tracks": sanity_path,
        },
    )
    write_manifest(out_dir / "data_manifest.json", manifest)
    typer.echo(prompts.groupby("domain").size().to_string())
    typer.echo(tracks.groupby("split").agg(tracks=("id", "size"), artists=("author", "nunique")))


@app.command("train-null")
def train_null_cmd(
    fma_root: Path,
    song_describer_audio: Path,
    config: Path = Path("configs/null_music_v1.yaml"),
    model: str = "stabilityai/stable-audio-open-1.0",
    out: Path | None = None,
) -> None:
    from audiocfg.models.sao import StableAudio
    from audiocfg.repro.config import NullTrainConfig
    from audiocfg.train.run import run_null_experiment

    cfg = NullTrainConfig.from_yaml(config)
    sao = StableAudio.load(model, dtype="auto", offload="off")
    run_null_experiment(sao, cfg, fma_root, song_describer_audio, out or cfg.run_dir())


if __name__ == "__main__":
    app()
