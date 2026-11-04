"""CLI ``listening-plan``: balanced pair plan for the E4 listening test (PROJECT_DESIGN.md §5.4).

- 24 prompts (12 sfx + 12 music), stratified by CLAP tercile at w = 7 (SAO, default null).
- one random seed per prompt; all 10 pairs of the 5 scales -> 240 pairs.
- 12 listeners x 60 pairs; each pair rated exactly 3 times (balanced incomplete design).
- each listener: 30 sfx + 30 music pairs, every scale pair 3x per domain, plus 4 attention-check
  pairs (same clip on both sides); random order and random left/right side.

Design per domain (n_prompts = n_listeners = 12): a Latin-square split into 12 chunks of 10 pairs
where chunk c holds scale-pair k of prompt (c + k) mod 12, so a chunk covers all 10 scale pairs
from 10 different prompts. Listener l gets chunks (l + o) mod 12 for 3 distinct offsets o (different
offsets per domain, so co-rater groups differ between domains); each chunk thus goes to exactly
3 listeners.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.schema import RESULT_FILES

OUT_FILE = "e4_listening_plan.csv"
OUT_COLUMNS = RESULT_FILES[OUT_FILE]
DOMAIN_OFFSETS = {"sfx": (0, 1, 2), "music": (0, 3, 7)}


def prompt_clap_at_w(table: pd.DataFrame, w: float, model: str = "sao",
                     null: str = "default", prompts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-prompt CLAP_norm at scale ``w`` from an e1_per_prompt-like or per-clip table."""
    t = table.copy()
    if "model" in t:
        t = t[t["model"] == model]
    if "null" in t:
        t = t[t["null"] == null]
    t = t[np.isclose(t["w"].astype(float), float(w))]
    if "domain" not in t:
        if prompts is None:
            raise ValueError("CLAP table has no 'domain' column; pass prompts")
        t = t.merge(prompts[["prompt_id", "domain"]], on="prompt_id")
    return t.groupby(["prompt_id", "domain"], as_index=False)["clap_norm"].mean()


def stratified_prompts(clap: pd.DataFrame, n_per_domain: int, rng: np.random.Generator,
                       n_strata: int = 3) -> pd.DataFrame:
    """Within each domain, split prompts into CLAP terciles and sample evenly from each."""
    chosen = []
    for domain in sorted(clap["domain"].unique()):
        d = clap[clap["domain"] == domain].sort_values(["clap_norm", "prompt_id"]).reset_index(
            drop=True)
        if len(d) < n_per_domain:
            raise ValueError(f"only {len(d)} prompts with CLAP in domain {domain}")
        d["tercile"] = (np.arange(len(d)) * n_strata) // len(d)
        quota = [n_per_domain // n_strata + (i < n_per_domain % n_strata)
                 for i in range(n_strata)]
        for s, q in enumerate(quota):
            pool = d[d["tercile"] == s]
            chosen.append(pool.iloc[np.sort(rng.choice(len(pool), size=q, replace=False))])
    return pd.concat(chosen, ignore_index=True)


def _domain_chunks(pairs: pd.DataFrame, n_listeners: int, rng: np.random.Generator
                   ) -> list[pd.DataFrame]:
    """Split one domain's pairs into ``n_listeners`` disjoint, balanced chunks."""
    prompt_ids = list(pairs["prompt_id"].unique())
    rng.shuffle(prompt_ids)
    types = sorted({(a, b) for a, b in zip(pairs["w_a"], pairs["w_b"])})
    if len(pairs) % n_listeners:
        raise ValueError("pairs per domain must be divisible by the number of listeners")
    if len(prompt_ids) == n_listeners and len(types) <= n_listeners:
        idx = pairs.set_index(["prompt_id", "w_a", "w_b"])
        return [pd.concat([idx.loc[[(prompt_ids[(c + k) % n_listeners], *types[k])]]
                           for k in range(len(types))]).reset_index()
                for c in range(n_listeners)]
    perm = pairs.iloc[rng.permutation(len(pairs))].reset_index(drop=True)
    size = len(pairs) // n_listeners
    return [perm.iloc[c * size:(c + 1) * size] for c in range(n_listeners)]


def build_plan(prompts: pd.DataFrame, seeds: Sequence[int], scales: Sequence[float], locate,
               *, n_listeners: int = 12, pairs_per_listener: int = 60,
               attention_checks: int = 4, seed: int = 2026,
               domain_offsets: dict[str, Sequence[int]] | None = None) -> pd.DataFrame:
    """Build the full plan from the already-selected prompts (``prompt_id, domain``)."""
    from .pairs import build_pairs

    rng = np.random.default_rng(seed)
    offsets = domain_offsets or DOMAIN_OFFSETS
    prompt_seed = {pid: int(rng.choice(list(seeds))) for pid in prompts["prompt_id"]}
    pairs = build_pairs(prompts, prompt_seed, scales)
    domains = sorted(pairs["domain"].unique())
    n_ratings = n_listeners * pairs_per_listener // len(pairs)
    if n_ratings * len(pairs) != n_listeners * pairs_per_listener:
        raise ValueError("listeners x pairs_per_listener must be a multiple of the pair count")

    assigned: list[list[pd.DataFrame]] = [[] for _ in range(n_listeners)]
    for domain in domains:
        chunks = _domain_chunks(pairs[pairs["domain"] == domain], n_listeners, rng)
        offs = list(offsets.get(domain, range(n_ratings)))[:n_ratings]
        if len(set(offs)) != n_ratings:
            raise ValueError(f"need {n_ratings} distinct offsets for domain {domain}")
        for listener in range(n_listeners):
            for o in offs:
                assigned[listener].append(chunks[(listener + o) % n_listeners])

    rows = []
    for listener in range(n_listeners):
        lid = f"L{listener + 1:02d}"
        mine = pd.concat(assigned[listener], ignore_index=True)
        items = []
        for r in mine.itertuples(index=False):
            left_is_a = bool(rng.integers(2))
            w_l, w_r = (r.w_a, r.w_b) if left_is_a else (r.w_b, r.w_a)
            items.append(dict(pair_id=r.pair_id, prompt_id=r.prompt_id, domain=r.domain,
                              seed=r.seed, w_left=w_l, w_right=w_r, is_attention_check=False))
        for k in range(attention_checks):
            dom = domains[k % len(domains)]
            pool = mine[mine["domain"] == dom]
            r = pool.iloc[int(rng.integers(len(pool)))]
            w = float(r["w_a"] if rng.integers(2) else r["w_b"])
            items.append(dict(pair_id=f"attn__{lid}__{k + 1}__{r['prompt_id']}__s{r['seed']}"
                                      f"__w{w:g}",
                              prompt_id=r["prompt_id"], domain=dom, seed=int(r["seed"]),
                              w_left=w, w_right=w, is_attention_check=True))
        for pos, i in enumerate(rng.permutation(len(items))):
            it = items[i]
            rows.append({"listener_id": lid, "position": pos + 1, **it,
                         "path_left": str(locate(it["prompt_id"], it["seed"], it["w_left"])),
                         "path_right": str(locate(it["prompt_id"], it["seed"], it["w_right"]))})
    return pd.DataFrame(rows)[list(OUT_COLUMNS)]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--clap-table", type=Path, default=None,
                   help="e1_per_prompt.parquet (default) or a per-clip CLAP file (csv/parquet)")
    p.add_argument("--w", type=float, default=7.0, help="scale used for CLAP terciles")
    p.add_argument("--seed", type=int, default=None, help="plan RNG (default: prompt seed)")
    p.add_argument("--out", type=Path, default=None)


def run(args: argparse.Namespace) -> None:
    from audiocfg.config import DEFAULT_CONFIG, load_config

    from .run import ClipLocator

    cfg = load_config(args.config or DEFAULT_CONFIG)
    lcfg = cfg.section("e4_judge")["listening"]
    path = args.clap_table or cfg.results_dir / "e1_per_prompt.parquet"
    table = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    prompts = pd.read_csv(cfg.prompts_csv)
    seed = args.seed if args.seed is not None else int(cfg.section("prompt_sampling")["random_seed"])
    rng = np.random.default_rng(seed)
    clap = prompt_clap_at_w(table, args.w, prompts=prompts)
    chosen = stratified_prompts(clap, int(lcfg["n_prompts_per_domain"]), rng)
    plan = build_plan(chosen, cfg.seeds, cfg.guidance_scales, ClipLocator(cfg.outputs_dir),
                      n_listeners=int(lcfg["n_listeners"]),
                      pairs_per_listener=int(lcfg["pairs_per_listener"]),
                      attention_checks=int(lcfg["attention_checks"]), seed=seed)
    out = args.out or cfg.results_dir / OUT_FILE
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out, index=False)
    print(f"[listening-plan] {plan['pair_id'].nunique()} ids, {len(plan)} rows -> {out}")
