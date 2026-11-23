import subprocess
import sys
from pathlib import Path

COMMIT = "__COMMIT__"
CONFIG = "configs/score_pilot_B.yaml"
REPO = "https://github.com/Nexonus/AudioCFG_WIMU_06_2026"
WORK = Path("/kaggle/working")
SRC = WORK / "AudioCFG"


def sh(*cmd, cwd=SRC):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


sh("git", "clone", "-q", REPO, SRC, cwd=WORK)
sh("git", "checkout", "-q", COMMIT)
sh(sys.executable, "-m", "pip", "install", "-q", "-e", f"{SRC}[models,metrics]")

inputs = Path("/kaggle/input")
sweep = next(inputs.rglob("clips.parquet")).parent
reference = next(p for p in inputs.rglob("audio") if "song" in str(p).lower() and p.is_dir())
sh(
    "audiocfg",
    "score",
    sweep,
    "--reference",
    reference,
    "--config",
    CONFIG,
    "--out",
    WORK / "score_pilot_B",
)
sh("rm", "-rf", SRC, cwd=WORK)
