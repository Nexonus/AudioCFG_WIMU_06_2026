"""E3: learn one null conditioning tensor ``n`` for a frozen SAO (PROJECT_DESIGN.md §5.3).

What this is
    ``n`` (shape ``(L, d)`` of SAO's cross-attention conditioning) replaces the zero null in the
    unconditional CFG branch. It is trained with the model's own v-prediction diffusion loss on
    real clips with the condition set to ``n``, so that ``v(x_t, t, n)`` models ``p(x)``:
    ``L(n) = E ||v_theta(x_t, t, n) - v_target||^2``. Gradients flow only into ``n``.

What this is NOT (§1.2)
    Not a negative prompt: ``n`` is not the embedding of any text, and the prompt text of the
    conditional branch is never touched.

Deviation from arXiv:2609.25411
    There the null is learned jointly with the whole TTS model while the condition is dropped
    with p = 0.1. Here SAO is frozen and only ``n`` is optimised (post-hoc). Reported as a
    deliberate deviation.

Outputs: ``results/e3_null_training.csv`` (``RESULT_FILES``) and
``outputs/null/sao_learned_<init>.pt`` = ``{"null", "shape", "val_loss", "step", "init"}``.
Top-level imports are light: ``cli.py`` imports this module at startup.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from audiocfg.schema import RESULT_FILES

LOG_FILE = "e3_null_training.csv"
LOG_COLUMNS = RESULT_FILES[LOG_FILE]
INITS = ("zeros", "normal")


# ----------------------------------------------------------------------------- pure python
@dataclass(frozen=True)
class LogRow:
    init: str
    step: int
    train_loss: float
    val_loss: float
    val_loss_zero_null: float


def write_log(path: str | Path, rows: Iterable[LogRow], append: bool = False) -> None:
    """Write rows with exactly ``RESULT_FILES['e3_null_training.csv']`` columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.exists())
    with path.open("a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LOG_COLUMNS))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: asdict(r)[c] for c in LOG_COLUMNS})


def replace_init_in_log(path: str | Path, init: str, rows: Sequence[LogRow]) -> None:
    """Rewrite the log keeping other inits' rows and replacing this init's rows."""
    path = Path(path)
    kept: list[LogRow] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["init"] != init:
                    kept.append(LogRow(r["init"], int(r["step"]), float(r["train_loss"]),
                                       float(r["val_loss"]), float(r["val_loss_zero_null"])))
    write_log(path, [*kept, *rows])


def select_best(rows: Sequence[LogRow]) -> LogRow:
    """Checkpoint rule (§5.3): lowest validation loss; ties -> earliest step."""
    if not rows:
        raise ValueError("no logged rows")
    return min(rows, key=lambda r: (r.val_loss, r.step))


def null_output_path(outputs_dir: str | Path, init: str) -> Path:
    """Where generate/ loads the learned null from."""
    return Path(outputs_dir) / "null" / f"sao_learned_{init}.pt"


# ----------------------------------------------------------------------------- torch (lazy)
def assert_only_null_trainable(frozen: Any, null: Any) -> None:
    """All model params frozen, only ``null`` trainable, and no model param holds a gradient."""
    for name, p in frozen.named_parameters():
        assert not p.requires_grad, f"model parameter {name} is trainable"
        assert p.grad is None, f"model parameter {name} received a gradient"
    assert null.requires_grad


def init_null(shape: Sequence[int], init: str, seed: int):
    import torch

    g = torch.Generator().manual_seed(seed)
    if init == "zeros":  # starts from SAO's default null
        return torch.zeros(*shape)
    if init == "normal":  # N(0, 1), as in arXiv:2609.25411
        return torch.randn(*shape, generator=g)
    raise ValueError(f"unknown init {init!r}; expected one of {INITS}")


def evaluate(loss_fn: Callable, cond: Any, val_batches: Sequence[Any], seed: int) -> float:
    """Mean val loss with fixed (t, eps) per batch: learned vs zero null see identical noise."""
    import torch

    total = 0.0
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for i, batch in enumerate(val_batches):
            torch.manual_seed(seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + i)
            total += float(loss_fn(batch, cond))
    return total / max(len(val_batches), 1)


def train_null(loss_fn: Callable[[Any, Any], Any], frozen: Any, shape: Sequence[int], init: str,
               train_batches: Iterator[Any], val_batches: Sequence[Any], *, lr: float,
               max_steps: int, eval_every: int, seed: int = 0, weight_decay: float = 0.0,
               device: str = "cpu", log_cb: Callable[[LogRow], None] | None = None) -> dict:
    """Optimise only ``n`` with AdamW; log every ``eval_every`` steps; keep the best by val loss.

    ``loss_fn(batch, cond)`` must be the model's own diffusion loss with conditioning ``cond``
    (see ``sao.make_v_loss_fn``). ``frozen`` is the model (``nn.Module``), already frozen here.
    """
    import torch

    frozen.requires_grad_(False)
    torch.manual_seed(seed)
    null = torch.nn.Parameter(init_null(shape, init, seed).to(device))
    zero = torch.zeros_like(null.data)
    opt = torch.optim.AdamW([null], lr=lr, weight_decay=weight_decay)
    assert_only_null_trainable(frozen, null)

    val_zero = evaluate(loss_fn, zero, val_batches, seed=10_000)
    rows: list[LogRow] = []
    best: dict = {"val_loss": float("inf"), "step": -1, "null": null.detach().clone()}

    def log(step: int, train_loss: float) -> None:
        v = evaluate(loss_fn, null.detach(), val_batches, seed=10_000)
        row = LogRow(init, step, train_loss, v, val_zero)
        rows.append(row)
        if log_cb:
            log_cb(row)
        if v < best["val_loss"]:
            best.update(val_loss=v, step=step, null=null.detach().clone())

    log(0, float("nan"))
    running, n_running = 0.0, 0
    for step in range(1, max_steps + 1):
        batch = next(train_batches)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(batch, null)
        loss.backward()
        if step == 1:
            assert_only_null_trainable(frozen, null)
        opt.step()
        running += float(loss.detach())
        n_running += 1
        if step % eval_every == 0 or step == max_steps:
            log(step, running / n_running)
            running, n_running = 0.0, 0

    chosen = select_best(rows)
    assert chosen.step == best["step"]
    return {"null": best["null"].cpu(), "shape": tuple(shape), "val_loss": best["val_loss"],
            "step": best["step"], "init": init, "rows": rows}


def save_null(path: str | Path, result: dict) -> Path:
    """Save the selected null as the dict generate/ expects."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: result[k] for k in ("null", "shape", "val_loss", "step", "init")}, path)
    return path


def load_null(path: str | Path):
    """Return the learned null tensor (helper for generate/)."""
    import torch

    return torch.load(Path(path), map_location="cpu")["null"]


def _latent_batches(paths: Sequence[Path], batch_size: int, seed: int, cycle: bool):
    import torch

    g = torch.Generator().manual_seed(seed)
    while True:
        order = (torch.randperm(len(paths), generator=g).tolist() if cycle
                 else list(range(len(paths))))
        for i in range(0, len(paths), batch_size):
            idx = order[i:i + batch_size]
            yield torch.stack([torch.load(paths[j]) for j in idx])
        if not cycle:
            return


# ----------------------------------------------------------------------------- CLI
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--init", choices=[*INITS, "all"], default="all")
    p.add_argument("--attribution-list", type=Path, help="SAO attribution ids (csv/txt)")
    p.add_argument("--exclude-ids", type=Path, help="E2 training-neighbour set T ids to exclude")
    p.add_argument("--latents-dir", type=Path, default=Path("outputs/null/latents"))
    p.add_argument("--audio-dir", type=Path, help="downloaded clips <id>.wav (for encoding)")
    p.add_argument("--split-file", type=Path, default=Path("outputs/null/split.csv"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=100, help="K: log every K steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")


def run(args: argparse.Namespace) -> None:
    from audiocfg.config import DEFAULT_CONFIG, REPO_ROOT, load_config

    from . import data, sao

    cfg = load_config(args.config or DEFAULT_CONFIG)
    e3 = cfg.section("e3_null")
    split_file = args.split_file if args.split_file.is_absolute() else REPO_ROOT / args.split_file
    if split_file.exists():
        train_ids, val_ids = data.read_split(split_file)
    else:
        if args.attribution_list is None or args.exclude_ids is None:
            raise SystemExit("need --attribution-list and --exclude-ids (E2 set T) for a new split")
        ids = data.select_clips(data.read_id_list(args.attribution_list),
                                data.read_id_list(args.exclude_ids),
                                int(e3["n_train_clips"]), seed=args.seed)
        train_ids, val_ids = data.train_val_split(ids, float(e3["val_fraction"]), seed=args.seed)
        data.write_split(split_file, train_ids, val_ids)

    model, model_config = sao.load_sao(args.device)
    latents_dir = args.latents_dir if args.latents_dir.is_absolute() else REPO_ROOT / args.latents_dir
    if args.audio_dir is not None:
        data.encode_latents({i: args.audio_dir / f"{i}.wav" for i in train_ids + val_ids},
                            latents_dir, seconds=cfg.clip_seconds, device=args.device)
    train_paths = [latents_dir / f"{i}.pt" for i in train_ids]
    val_paths = [latents_dir / f"{i}.pt" for i in val_ids]

    import torch

    val_batches = [b.to(args.device) for b in
                   _latent_batches(val_paths, args.batch_size, args.seed, cycle=False)]
    shape = sao.null_shape(model, model_config, cfg.clip_seconds)
    loss_fn = sao.make_v_loss_fn(model, cfg.clip_seconds)
    inits = INITS if args.init == "all" else (args.init,)
    for init in inits:
        batches = (b.to(args.device) for b in
                   _latent_batches(train_paths, args.batch_size, args.seed, cycle=True))
        res = train_null(loss_fn, model, shape, init, batches, val_batches,
                         lr=args.lr or float(e3["lr"]),
                         max_steps=args.max_steps or int(e3["max_steps"]),
                         eval_every=args.eval_every, seed=args.seed, device=args.device,
                         log_cb=lambda r: print(r, flush=True))
        replace_init_in_log(cfg.results_dir / LOG_FILE, init, res["rows"])
        out = save_null(null_output_path(cfg.outputs_dir, init), res)
        print(f"[train-null] {init}: best step {res['step']} val {res['val_loss']:.5f} -> {out}")
        del res
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
