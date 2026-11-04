"""Shared data contracts: one generated clip, its path, and the metadata table.

Every module reads and writes these, so the experiments stay joinable:
generation writes ``meta.parquet`` with ``CLIP_COLUMNS``; metrics add columns
keyed by ``clip_id``; analysis groups by (model, domain, w, null).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

DOMAINS = ("sfx", "music")

# Columns of data/prompts.csv
PROMPT_COLUMNS = ("prompt_id", "domain", "text", "source", "source_id", "reference_audio")

# Columns of outputs/<model>/meta.parquet
CLIP_COLUMNS = (
    "clip_id",
    "model",
    "prompt_id",
    "domain",
    "text",
    "w",
    "null",
    "seed",
    "sampler",
    "steps",
    "sample_rate",
    "seconds",
    "path",
    "gen_time_s",
)


# Result files (relative to results_dir) and their columns. Producers write exactly these;
# analysis reads only these. One row per unit noted in the comment.
RESULT_FILES = {
    # E1, one row per (model, domain, null, w, prompt_id)
    "e1_per_prompt.parquet": ("model", "domain", "null", "w", "prompt_id",
                              "ccds_clap", "ccds_aux", "clap_norm", "kl_passt"),
    # E1, one row per (model, domain, null, w, embedder); FD is a set-level metric
    "e1_fd.parquet": ("model", "domain", "null", "w", "embedder", "n", "fd", "ci_low", "ci_high"),
    # E1 calibration, one row per (domain, clap_model)
    "clap_calibration.csv": ("domain", "clap_model", "c_real", "c_mis", "n"),
    # E2, one row per generated clip
    "e2_copies.parquet": ("clip_id", "model", "null", "w", "prompt_id", "seed", "domain", "embedder",
                          "nn_train_id", "s_gen", "is_candidate", "match_score", "is_copy"),
    # E2 control, one row per held-out clip
    "e2_held.parquet": ("held_id", "domain", "embedder", "nn_train_id", "s_held", "match_score"),
    # E3, one row per logged step
    "e3_null_training.csv": ("init", "step", "train_loss", "val_loss", "val_loss_zero_null"),
    # E4 judge, one row per (pair, question, order)
    "e4_judge.parquet": ("judge", "pair_id", "prompt_id", "domain", "seed", "w_a", "w_b",
                         "question", "order", "choice"),
    # E4 listening plan, one row per (listener, presented pair)
    "e4_listening_plan.csv": ("listener_id", "position", "pair_id", "prompt_id", "domain", "seed",
                              "w_left", "w_right", "path_left", "path_right",
                              "is_attention_check"),
    # E4 listener answers, one row per (listener, pair, question)
    "e4_listeners.csv": ("listener_id", "pair_id", "prompt_id", "domain", "w_a", "w_b",
                         "question", "choice", "is_attention_check"),
}
# Judge presentation order: "ab" shows w_a as A, "ba" shows w_b as A.
ORDERS = ("ab", "ba")
# Choices are always expressed relative to (w_a, w_b) with w_a < w_b: "a", "b" or "tie".
CHOICES = ("a", "b", "tie")
QUESTIONS = ("match", "realism", "overall")


@dataclass(frozen=True)
class ClipSpec:
    """Everything that determines one generation. Only ``w`` and ``null`` vary within a sweep."""

    model: str
    prompt_id: str
    domain: str
    text: str
    w: float
    null: str
    seed: int
    sampler: str
    steps: int
    seconds: float

    @property
    def clip_id(self) -> str:
        # Path-safe and unique: the id encodes every setting, so a sweep is resumable.
        return f"{self.model}__{self.null}__w{self.w:g}__{self.prompt_id}__s{self.seed}"

    def relpath(self) -> Path:
        return Path(self.model) / self.null / f"w{self.w:g}" / self.prompt_id / f"s{self.seed}.wav"

    def to_row(self, path: Path, sample_rate: int, gen_time_s: float) -> dict:
        row = asdict(self)
        row.update(clip_id=self.clip_id, path=str(path), sample_rate=sample_rate,
                   gen_time_s=gen_time_s)
        return {c: row[c] for c in CLIP_COLUMNS}
