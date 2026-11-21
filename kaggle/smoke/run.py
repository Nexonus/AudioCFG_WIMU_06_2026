import json
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "__COMMIT__"
REPO = "https://github.com/Nexonus/AudioCFG_WIMU_06_2026"
OUT = Path("/kaggle/working")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", f"audiocfg[models] @ git+{REPO}@{COMMIT}"],
    check=True,
)

import urllib.request

import torch

from audiocfg.models.sao import StableAudio
from audiocfg.models.weights import verify_dir
from audiocfg.repro.manifest import device_info, package_versions

report = {"commit": COMMIT, "device": device_info(), "packages": package_versions()}

inputs = Path("/kaggle/input")
mounts = [p for depth in ("*", "*/*", "*/*/*") for p in inputs.glob(depth) if p.is_dir()]
report["mounts"] = [str(p) for p in mounts]


def mount(name: str) -> Path:
    return next(p for p in mounts if name in p.name)


fma, sd, sao_dir = mount("fma-large"), mount("song-describer"), mount("stable-audio-open")
report["fma_example"] = str(next(fma.rglob("000002.mp3")))
report["song_describer_example"] = str(next(sd.rglob("*.2min.mp3")))

# Mirror of the gated HF repo; must match the official checksums byte for byte.
reference = OUT / "sao_sha256.json"
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/Nexonus/AudioCFG_WIMU_06_2026/{COMMIT}/configs/sao_sha256.json",
    reference,
)
sao_dir = next(p.parent for p in sao_dir.rglob("model_index.json"))
verify_dir(sao_dir, reference)
report["sao_weights"] = f"{sao_dir} (sha256 verified)"

sao = StableAudio.load(str(sao_dir), dtype="auto", offload="off")
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
