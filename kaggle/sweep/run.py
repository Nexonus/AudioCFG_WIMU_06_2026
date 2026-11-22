import os
import subprocess
import sys
from pathlib import Path

COMMIT = "__COMMIT__"
CONFIG = "configs/sweep_pilot_A.yaml"
REPO = "https://github.com/Nexonus/AudioCFG_WIMU_06_2026"
WORK = Path("/kaggle/working")
SRC = WORK / "AudioCFG"
OUT = WORK / "sweep_pilot_A"


def sh(*cmd, cwd=SRC):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


sh("git", "clone", "-q", REPO, SRC, cwd=WORK)
sh("git", "checkout", "-q", COMMIT)
sh(sys.executable, "-m", "pip", "install", "-q", "-e", f"{SRC}[models]")

sao = next(Path("/kaggle/input").rglob("model_index.json")).parent
verify = "from audiocfg.models.weights import verify_dir; verify_dir(r'{}', '{}')"
sh(sys.executable, "-c", verify.format(sao, "configs/sao_sha256.json"))

gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).count("GPU ")
cmd = ["audiocfg", "sweep", "--config", CONFIG, "--model", str(sao), "--out", str(OUT)]
procs = [
    subprocess.Popen(
        [*cmd, "--shard", str(i), "--shards", str(gpus)],
        cwd=SRC,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": str(i)},
    )
    for i in range(gpus)
]
codes = [p.wait() for p in procs]
print("shard exit codes:", codes, flush=True)
sh("audiocfg", "collect-sweep", "--config", CONFIG, "--out", OUT)
sh("rm", "-rf", SRC, cwd=WORK)
if any(codes):
    sys.exit(1)
