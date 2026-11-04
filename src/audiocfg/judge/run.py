"""CLI ``judge``: pairwise judge on SAO clips, every pair in both A/B orders (§5.4).

Modes
    default            full E1 set: all 10 scale pairs for every prompt, one seed (``--seed``).
    --pairs-from-plan  exactly the listeners' pairs from ``e4_listening_plan.csv``.
Both write/merge ``results/e4_judge.parquet`` (one row per judge, pair, question, order; choice
relative to (w_a, w_b)). Rows already present for the same key are skipped (resumable).
Analysis selects the listener subset by ``pair_id`` from the plan.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from audiocfg.schema import RESULT_FILES

OUT_FILE = "e4_judge.parquet"
OUT_COLUMNS = RESULT_FILES[OUT_FILE]
KEY = ["judge", "pair_id", "question", "order"]


class ClipLocator:
    """(prompt_id, seed, w) -> wav path for one model/null, from ``meta.parquet`` if present.

    Falls back to the ``ClipSpec.relpath`` layout ``<model>/<null>/w<w>/<prompt_id>/s<seed>.wav``
    under ``outputs_dir``.
    """

    def __init__(self, outputs_dir: str | Path, model: str = "sao", null: str = "default",
                 meta: pd.DataFrame | None = None) -> None:
        self.outputs_dir, self.model, self.null = Path(outputs_dir), model, null
        self._map: dict[tuple[str, int, float], Path] = {}
        if meta is None:
            mp = self.outputs_dir / model / "meta.parquet"
            meta = pd.read_parquet(mp) if mp.exists() else None
        if meta is not None:
            m = meta[(meta["model"] == model) & (meta["null"] == null)]
            for r in m.itertuples(index=False):
                p = Path(r.path)
                self._map[(r.prompt_id, int(r.seed), float(r.w))] = (
                    p if p.is_absolute() else self.outputs_dir / p)

    def __call__(self, prompt_id: str, seed: int, w: float) -> Path:
        key = (prompt_id, int(seed), float(w))
        if key in self._map:
            return self._map[key]
        return (self.outputs_dir / self.model / self.null / f"w{float(w):g}" / prompt_id
                / f"s{int(seed)}.wav")


def pairs_from_plan(plan: pd.DataFrame, prompts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Unique non-attention pairs of the listening plan, as (w_a < w_b) pair rows."""
    p = plan[~plan["is_attention_check"].astype(bool)].copy()
    p["w_a"] = p[["w_left", "w_right"]].min(axis=1).astype(float)
    p["w_b"] = p[["w_left", "w_right"]].max(axis=1).astype(float)
    p = p.drop_duplicates("pair_id")[["pair_id", "prompt_id", "domain", "seed", "w_a", "w_b"]]
    if prompts is not None and "text" in prompts:
        p = p.merge(prompts[["prompt_id", "text"]], on="prompt_id", how="left")
    p["text"] = p.get("text", pd.Series("", index=p.index)).fillna("")
    return p.reset_index(drop=True)


def judge_pairs(judge, pairs: pd.DataFrame, locate, questions: Sequence[str] | None = None,
                done: Iterable[tuple] = ()) -> pd.DataFrame:
    """Run ``judge`` on every pair in both orders; returns rows with ``OUT_COLUMNS``."""
    from .pairs import expand_orders, to_relative

    qs = [q for q in (questions or judge.questions) if q in judge.questions]
    done = set(done)
    rows = []
    for r in expand_orders(pairs).itertuples(index=False):
        path_A = locate(r.prompt_id, r.seed, r.w_A)
        path_B = locate(r.prompt_id, r.seed, r.w_B)
        for q in qs:
            if (judge.name, r.pair_id, q, r.order) in done:
                continue
            answer = judge.compare(path_A, path_B, r.text, q)
            rows.append({"judge": judge.name, "pair_id": r.pair_id, "prompt_id": r.prompt_id,
                         "domain": r.domain, "seed": int(r.seed), "w_a": float(r.w_a),
                         "w_b": float(r.w_b), "question": q, "order": r.order,
                         "choice": to_relative(answer, r.order)})
    return pd.DataFrame(rows, columns=list(OUT_COLUMNS))


def merge_results(path: str | Path, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows into ``e4_judge.parquet`` (newer wins on the same key) and save."""
    path = Path(path)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=list(OUT_COLUMNS))
    both = pd.concat([old, new], ignore_index=True) if len(old) else new
    both = both.drop_duplicates(KEY, keep="last")[list(OUT_COLUMNS)].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    both.to_parquet(path, index=False)
    return both


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge", default="qwen", choices=["qwen", "audiobox"])
    p.add_argument("--pairs-from-plan", type=Path, default=None,
                   help="e4_listening_plan.csv: judge exactly the listeners' pairs")
    p.add_argument("--seed", type=int, default=None, help="seed for the full E1 set "
                   "(default: first seed in config)")
    p.add_argument("--model", default="sao")
    p.add_argument("--null", default="default")
    p.add_argument("--questions", nargs="+", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="only the first N pairs (smoke test)")


def run(args: argparse.Namespace) -> None:
    from audiocfg.config import DEFAULT_CONFIG, load_config

    from .pairs import build_pairs
    from .qwen_judge import make_judge

    cfg = load_config(args.config or DEFAULT_CONFIG)
    prompts = pd.read_csv(cfg.prompts_csv)
    if args.pairs_from_plan is not None:
        pairs = pairs_from_plan(pd.read_csv(args.pairs_from_plan), prompts)
    else:
        seed = cfg.seeds[0] if args.seed is None else args.seed
        pairs = build_pairs(prompts, [seed], cfg.guidance_scales)
    if args.limit:
        pairs = pairs.head(args.limit)

    kw = {"load_in_4bit": args.load_in_4bit} if args.judge == "qwen" else {}
    if args.judge == "qwen":
        kw["loudness_lufs"] = cfg.loudness_lufs
        kw["model_id"] = cfg.section("e4_judge")["judge_model"]
    judge = make_judge(args.judge, **kw)
    out = cfg.results_dir / OUT_FILE
    done = []
    if out.exists():
        done = [tuple(r) for r in pd.read_parquet(out)[KEY].itertuples(index=False)]
    locate = ClipLocator(cfg.outputs_dir, args.model, args.null)
    # chunked so a crash loses at most one chunk
    for start in range(0, len(pairs), 20):
        new = judge_pairs(judge, pairs.iloc[start:start + 20], locate, args.questions, done)
        merge_results(out, new)
        print(f"[judge] {judge.name}: {min(start + 20, len(pairs))}/{len(pairs)} pairs")
