from __future__ import annotations

import json
from pathlib import Path

from audiocfg.repro.manifest import sha256_file

SAO_COMPONENTS = (
    "projection_model",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)


def checksum_dir(root: str | Path) -> dict[str, str]:
    root = Path(root)
    files = [root / "model_index.json"]
    for component in SAO_COMPONENTS:
        files += sorted(p for p in (root / component).iterdir() if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def verify_dir(root: str | Path, reference: str | Path) -> None:
    """Raise if any file of a local SAO copy differs from the official checksums."""
    expected = json.loads(Path(reference).read_text(encoding="utf-8"))["sha256"]
    actual = checksum_dir(root)
    bad = sorted(k for k in expected if actual.get(k) != expected[k])
    if bad:
        raise RuntimeError(f"SAO weights differ from the official release: {bad}")
