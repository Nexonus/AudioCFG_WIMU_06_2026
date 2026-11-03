"""Audio (and text) embedders behind one protocol, with lazy model loading (§5.1, §5.2, §6).

| name         | used for                                   | native sr |
|--------------|--------------------------------------------|-----------|
| laion_clap   | CCDS (audio), CLAP_norm (audio+text), E2   | 48 kHz    |
| ms_clap      | CLAP_norm robustness check (§6.1)          | 44.1 kHz  |
| panns        | CCDS aux for SFX, E2 NN search             | 32 kHz    |
| mert         | CCDS aux for music                         | 24 kHz    |
| openl3       | FD_openl3 (§6.2)                           | 48 kHz    |
| vggish       | auxiliary FAD (§6.2)                       | 16 kHz    |
| passt        | label probabilities for KL_passt (§6.2)    | 32 kHz    |
| random       | deterministic numpy stand-in for tests/dry runs | any  |

Every model-backed class loads its weights on first use only; all of them are TODO(M1):
written against the libraries' documented APIs but not yet run here (no torch in the dev venv).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from audiocfg.audio_io import resample
from audiocfg.metrics.dsp import log_mel, to_mono


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        """One row per clip: ``(n, d)``."""
        ...


@runtime_checkable
class TextEmbedder(Embedder, Protocol):
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """One row per text: ``(n, d)``, in the same space as ``embed``."""
        ...


@runtime_checkable
class LabelModel(Protocol):
    name: str

    def label_probs(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        """Per-clip distribution over classes ``(n, n_classes)``, rows sum to 1."""
        ...


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _Lazy:
    """Base: downmix + resample to the model's rate, batch, and load the model on first call."""

    name = "base"
    target_sr = 16000
    batch_size = 16

    def __init__(self) -> None:
        self._model = None

    def _load(self):  # pragma: no cover - model-backed
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _prep(self, audios: list[np.ndarray], sr: int) -> list[np.ndarray]:
        return [resample(to_mono(a).astype(np.float32), sr, self.target_sr) for a in audios]

    def _embed_batch(self, batch: list[np.ndarray]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        xs = self._prep(audios, sr)
        out = [self._embed_batch(xs[i:i + self.batch_size]) for i in range(0, len(xs), self.batch_size)]
        return np.concatenate(out, axis=0) if out else np.zeros((0, 0))


def _pad_stack(batch: list[np.ndarray]) -> np.ndarray:
    n = max(len(x) for x in batch)
    return np.stack([np.pad(x, (0, n - len(x))) for x in batch]).astype(np.float32)


class LaionClap(_Lazy):
    """LAION-CLAP (HTSAT). TODO(M1): pick checkpoint per domain (general vs music_audioset_epoch_15)."""

    name = "laion_clap"
    target_sr = 48000

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__()
        self.checkpoint = checkpoint

    def _load(self):  # pragma: no cover
        import laion_clap

        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        m.load_ckpt(self.checkpoint) if self.checkpoint else m.load_ckpt()
        return m

    def _embed_batch(self, batch):  # pragma: no cover
        return np.asarray(self.model.get_audio_embedding_from_data(x=_pad_stack(batch), use_tensor=False))

    def embed_text(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        return np.asarray(self.model.get_text_embedding(list(texts), use_tensor=False))


class MsClap(_Lazy):
    """Microsoft CLAP 2023. TODO(M1): msclap reads files, so clips are written to a temp dir."""

    name = "ms_clap"
    target_sr = 44100

    def _load(self):  # pragma: no cover
        from msclap import CLAP

        return CLAP(version="2023", use_cuda=_device() == "cuda")

    def _embed_batch(self, batch):  # pragma: no cover
        import tempfile
        from pathlib import Path

        import soundfile as sf

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, x in enumerate(batch):
                p = Path(d) / f"{i}.wav"
                sf.write(str(p), x, self.target_sr)
                paths.append(str(p))
            return self.model.get_audio_embeddings(paths).detach().cpu().numpy()

    def embed_text(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        return self.model.get_text_embeddings(list(texts)).detach().cpu().numpy()


class Panns(_Lazy):
    """PANNs CNN14 2048-d clip embedding (panns_inference). TODO(M1)."""

    name = "panns"
    target_sr = 32000

    def _load(self):  # pragma: no cover
        from panns_inference import AudioTagging

        return AudioTagging(checkpoint_path=None, device=_device())

    def _embed_batch(self, batch):  # pragma: no cover
        _, emb = self.model.inference(_pad_stack(batch))
        return np.asarray(emb)


class Mert(_Lazy):
    """MERT-v1-95M, time-averaged mean of all hidden layers. TODO(M1): choose layer(s)."""

    name = "mert"
    target_sr = 24000
    hf_id = "m-a-p/MERT-v1-95M"

    def _load(self):  # pragma: no cover
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        model = AutoModel.from_pretrained(self.hf_id, trust_remote_code=True).to(_device()).eval()
        proc = Wav2Vec2FeatureExtractor.from_pretrained(self.hf_id, trust_remote_code=True)
        return model, proc

    def _embed_batch(self, batch):  # pragma: no cover
        import torch

        model, proc = self.model
        out = []
        for x in batch:
            inp = proc(x, sampling_rate=self.target_sr, return_tensors="pt").to(model.device)
            with torch.no_grad():
                hs = model(**inp, output_hidden_states=True).hidden_states
            out.append(torch.stack(hs).mean(dim=(0, 2)).squeeze(0).cpu().numpy())
        return np.stack(out)


class OpenL3(_Lazy):
    """OpenL3 mel256/512-d, mean over frames. TODO(M1): match stable-audio-metrics (frame-level, stereo)."""

    name = "openl3"
    target_sr = 48000

    def __init__(self, content_type: str = "env") -> None:
        super().__init__()
        self.content_type = content_type

    def _load(self):  # pragma: no cover
        import openl3

        return openl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type=self.content_type, embedding_size=512)

    def _embed_batch(self, batch):  # pragma: no cover
        import openl3

        embs, _ = openl3.get_audio_embedding(list(batch), self.target_sr, model=self.model,
                                             hop_size=0.5, verbose=False)
        return np.stack([e.mean(axis=0) for e in embs])


class Vggish(_Lazy):
    """VGGish 128-d, mean over 0.96 s frames (torch.hub harritaylor/torchvggish). TODO(M1)."""

    name = "vggish"
    target_sr = 16000

    def _load(self):  # pragma: no cover
        import torch

        m = torch.hub.load("harritaylor/torchvggish", "vggish")
        m.postprocess = False
        return m.eval()

    def _embed_batch(self, batch):  # pragma: no cover
        import torch

        with torch.no_grad():
            return np.stack([self.model.forward(x, self.target_sr).mean(0).cpu().numpy() for x in batch])


class Passt(_Lazy):
    """PaSST AudioSet tagger -> 527 sigmoid probs normalised to a distribution (KL_passt, §6.2).

    TODO(M1): align normalisation/windowing with stable-audio-metrics' passt KL.
    """

    name = "passt"
    target_sr = 32000

    def _load(self):  # pragma: no cover
        from hear21passt.base import get_basic_model

        return get_basic_model(mode="logits").to(_device()).eval()

    def label_probs(self, audios: list[np.ndarray], sr: int) -> np.ndarray:  # pragma: no cover
        import torch

        xs = self._prep(audios, sr)
        out = []
        for x in xs:
            t = torch.from_numpy(x[None]).to(next(self.model.parameters()).device)
            with torch.no_grad():
                p = torch.sigmoid(self.model(t)).squeeze(0).cpu().numpy()
            out.append(p / p.sum())
        return np.stack(out)

    def embed(self, audios, sr):  # pragma: no cover
        return self.label_probs(audios, sr)


class RandomProjectionEmbedder:
    """Deterministic numpy embedder for tests and ``--embedder-override random`` dry runs.

    Audio: log-mel frame mean+std, fixed Gaussian projection to ``dim``. Same clip -> same
    vector; different content -> different vector. Text: seeded vector from a hash of the text.
    Also provides ``label_probs`` (softmax of a projection) so it can stand in for PaSST.
    """

    def __init__(self, dim: int = 64, seed: int = 0, n_mels: int = 32, n_classes: int = 32,
                 name: str = "random") -> None:
        self.name = name
        self.dim = dim
        self.seed = seed
        self.n_mels = n_mels
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((2 * n_mels, dim)) / np.sqrt(2 * n_mels)
        self._cls = rng.standard_normal((2 * n_mels, n_classes)) / np.sqrt(2 * n_mels)

    def _features(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        feats = []
        for a in audios:
            lm = log_mel(a, sr, n_fft=1024, hop=512, n_mels=self.n_mels)
            feats.append(np.concatenate([lm.mean(0), lm.std(0)]))
        f = np.asarray(feats)
        return f - f.mean(axis=1, keepdims=True)

    def embed(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        return self._features(audios, sr) @ self._proj

    def embed_text(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            h = int.from_bytes(hashlib.sha256(f"{self.seed}:{t}".encode()).digest()[:8], "little")
            out.append(np.random.default_rng(h).standard_normal(self.dim))
        return np.asarray(out)

    def label_probs(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        z = self._features(audios, sr) @ self._cls
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


_REGISTRY = {
    "laion_clap": LaionClap,
    "clap": LaionClap,
    "ms_clap": MsClap,
    "panns": Panns,
    "mert": Mert,
    "openl3": OpenL3,
    "vggish": Vggish,
    "passt": Passt,
    "random": RandomProjectionEmbedder,
}


def get_embedder(name: str, override: str | None = None, **kwargs):
    """Embedder by name. ``override="random"`` returns a RandomProjectionEmbedder named after
    ``name`` (seeded per name, so different embedders give different spaces)."""
    if override == "random":
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        return RandomProjectionEmbedder(seed=seed, name=name)
    if override:
        raise ValueError(f"unknown embedder override {override!r}")
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError as e:
        raise ValueError(f"unknown embedder {name!r}; known: {sorted(_REGISTRY)}") from e
