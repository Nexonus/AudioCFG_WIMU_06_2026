import json
import os
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "__COMMIT__"
REPO = "https://github.com/Nexonus/AudioCFG_WIMU_06_2026"
OUT = Path("/kaggle/working")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", f"git+{REPO}@{COMMIT}#egg=audiocfg[models]"],
    check=True,
)

from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

import torch

from audiocfg.models.sao import StableAudio
from audiocfg.repro.manifest import device_info, package_versions

report = {"commit": COMMIT, "device": device_info(), "packages": package_versions()}

inputs = Path("/kaggle/input")
report["inputs"] = sorted(p.name for p in inputs.iterdir())
fma = next(inputs.glob("*fma-large*"), None)
sd = next(inputs.glob("*song-describer*"), None)
report["fma_example"] = str(next((fma / "000").glob("*.mp3"))) if fma else None
report["song_describer_example"] = str(next(sd.rglob("*.2min.mp3"))) if sd else None

sao = StableAudio.load(dtype="auto", offload="off")
report["dtype"] = str(sao.dtype)
report["text_shape"] = sao.text_shape
report["max_seconds"] = sao.max_seconds

prompts = ["rain on a tin roof, distant thunder", "calm solo piano, slow tempo"]
timings = {}
for batch in (1, 4, 8):
    ps = (prompts * batch)[:batch]
    seeds = list(range(batch))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = sao.sample(ps, seeds, 7.0, None, num_steps=50, seconds=10.0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    audio = out.audio.float()
    timings[batch] = {
        "seconds": round(dt, 2),
        "per_clip": round(dt / batch, 2),
        "finite": bool(torch.isfinite(audio).all()),
        "rms": round(float(audio.pow(2).mean().sqrt()), 4),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(batch, timings[batch], flush=True)
report["timings_50_steps_10s"] = timings

(OUT / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
