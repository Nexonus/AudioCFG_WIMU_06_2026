from __future__ import annotations

from pathlib import Path

import torch

NULL_KEY = "null_text"


def zero_null(text_shape: tuple[int, int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(text_shape, dtype=dtype)


def t5_empty_null(sao) -> torch.Tensor:
    return sao.encode_text([""])[0].float().cpu()


def save_null(path: str | Path, null_text: torch.Tensor, metadata: dict[str, str] | None = None):
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = null_text.detach().float().cpu().contiguous()
    save_file(
        {NULL_KEY: tensor}, str(path), metadata={k: str(v) for k, v in (metadata or {}).items()}
    )
    return path


def load_null(path: str | Path) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(str(path))[NULL_KEY]


def resolve_null(spec: str, sao=None) -> torch.Tensor | None:
    if spec == "zero":
        return None
    if spec == "t5_empty":
        if sao is None:
            raise ValueError("t5_empty needs the loaded model")
        return t5_empty_null(sao)
    path = Path(spec)
    if path.suffix == ".safetensors" and path.exists():
        return load_null(path)
    raise ValueError(f"unknown null: {spec!r} (use zero, t5_empty or a .safetensors file)")
