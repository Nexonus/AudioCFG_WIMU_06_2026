import subprocess
import sys
from pathlib import Path

COMMIT = "__COMMIT__"
CONFIG = "configs/null_music_v1.yaml"
REPO = "https://github.com/Nexonus/AudioCFG_WIMU_06_2026"
WORK = Path("/kaggle/working")
SRC = WORK / "AudioCFG"


def sh(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


sh("git", "clone", "-q", REPO, SRC)
sh("git", "checkout", "-q", COMMIT, cwd=SRC)
sh(sys.executable, "-m", "pip", "install", "-q", "-e", f"{SRC}[models]")

inputs = Path("/kaggle/input")
mounts = [p for d in ("*", "*/*", "*/*/*") for p in inputs.glob(d) if p.is_dir()]


def mount(name):
    return next(p for p in mounts if name in p.name)


fma = mount("fma-large")
song_describer = next(mount("song-describer").rglob("audio"))
sao = next(mount("stable-audio-open").rglob("model_index.json")).parent

sh(
    sys.executable,
    "-c",
    f"from audiocfg.models.weights import verify_dir; verify_dir(r'{sao}', 'configs/sao_sha256.json')",
    cwd=SRC,
)
sh(
    "audiocfg", "train-null", fma, song_describer,
    "--config", CONFIG, "--model", sao, "--out", WORK / "null_music_v1",
    cwd=SRC,
)
sh("rm", "-rf", SRC)
