"""Scale pairs, A/B order handling and tie rule for E4 (PROJECT_DESIGN.md §5.4).

Conventions (schema.py): a pair is (w_a, w_b) with w_a < w_b; stored choices are relative,
one of ``CHOICES = ("a", "b", "tie")``. A presentation has an ``order``:
``"ab"`` shows w_a as A (left/first) and w_b as B; ``"ba"`` shows w_b as A and w_a as B.
The judge answers positionally (``"A"|"B"|"tie"``); ``to_relative`` maps that back.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import pandas as pd

from audiocfg.schema import CHOICES

ORDERS = ("ab", "ba")
POSITIONAL = ("A", "B", "tie")


def scale_pairs(scales: Iterable[float]) -> list[tuple[float, float]]:
    """All C(k, 2) pairs (w_a, w_b), w_a < w_b. For 5 scales: 10 pairs."""
    ws = sorted({float(w) for w in scales})
    return list(combinations(ws, 2))


def pair_id(prompt_id: str, seed: int, w_a: float, w_b: float) -> str:
    """Stable id shared by judge rows, the listening plan and listener answers."""
    lo, hi = sorted((float(w_a), float(w_b)))
    return f"{prompt_id}__s{int(seed)}__w{lo:g}-w{hi:g}"


def build_pairs(prompts: pd.DataFrame, seeds: Sequence[int] | dict[str, int],
                scales: Iterable[float]) -> pd.DataFrame:
    """One row per (prompt, seed, scale pair).

    ``prompts`` needs ``prompt_id, domain`` (``text`` kept if present). ``seeds`` is either a list
    applied to every prompt or a ``{prompt_id: seed}`` map (listening plan: one seed per prompt).
    """
    rows = []
    for p in prompts.itertuples(index=False):
        p_seeds = [seeds[p.prompt_id]] if isinstance(seeds, dict) else list(seeds)
        for s in p_seeds:
            for w_a, w_b in scale_pairs(scales):
                rows.append({"pair_id": pair_id(p.prompt_id, s, w_a, w_b),
                             "prompt_id": p.prompt_id, "domain": p.domain,
                             "text": getattr(p, "text", ""), "seed": int(s),
                             "w_a": w_a, "w_b": w_b})
    return pd.DataFrame(rows, columns=["pair_id", "prompt_id", "domain", "text", "seed",
                                       "w_a", "w_b"])


def expand_orders(pairs: pd.DataFrame) -> pd.DataFrame:
    """Each pair twice: ``order="ab"`` (A = w_a) and ``order="ba"`` (A = w_b)."""
    out = []
    for order in ORDERS:
        d = pairs.copy()
        d["order"] = order
        d["w_A"] = d["w_a"] if order == "ab" else d["w_b"]
        d["w_B"] = d["w_b"] if order == "ab" else d["w_a"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def to_relative(answer: str, order: str) -> str:
    """Positional answer ("A"/"B"/"tie") under ``order`` -> relative choice ("a"/"b"/"tie")."""
    if answer == "tie":
        return "tie"
    if answer not in ("A", "B") or order not in ORDERS:
        raise ValueError(f"bad answer/order: {answer!r}, {order!r}")
    first_is_a = order == "ab"
    return "a" if (answer == "A") == first_is_a else "b"


def side_to_relative(answer: str, w_left: float, w_right: float) -> str:
    """Listener answer "left"/"right"/"tie" (or "A"/"B") -> relative choice w.r.t. w_a < w_b."""
    a = {"left": "A", "right": "B"}.get(answer, answer)
    if w_left == w_right:  # attention check: no a/b, keep positional answer as tie or side
        return "tie" if a == "tie" else a.lower()
    return to_relative(a, "ab" if w_left < w_right else "ba")


def resolve_orders(choice_ab: str, choice_ba: str) -> str:
    """Combine the two presentations of one pair (§5.4): consistent -> that choice, else tie.

    ``choice_ab`` is the positional answer when A = w_a; ``choice_ba`` the positional answer
    when A = w_b (so "A" there means w_b). Relative answers ("a"/"b"/"tie") are accepted too.
    """
    rel_ab = choice_ab if choice_ab in CHOICES else to_relative(choice_ab, "ab")
    rel_ba = choice_ba if choice_ba in CHOICES else to_relative(choice_ba, "ba")
    return rel_ab if rel_ab == rel_ba else "tie"


def resolve_table(judge_rows: pd.DataFrame) -> pd.DataFrame:
    """e4_judge rows (relative ``choice`` per order) -> one resolved choice per pair & question.

    Adds ``flipped`` (relative choices differ between orders) and ``strict_flip`` (one order says
    a, the other b, i.e. pure position preference).
    """
    keys = ["judge", "pair_id", "prompt_id", "domain", "seed", "w_a", "w_b", "question"]
    wide = judge_rows.pivot_table(index=keys, columns="order", values="choice",
                                  aggfunc="first").reset_index()
    wide = wide.dropna(subset=list(ORDERS))
    wide["choice"] = [resolve_orders(a, b) for a, b in zip(wide["ab"], wide["ba"])]
    wide["flipped"] = wide["ab"] != wide["ba"]
    wide["strict_flip"] = ((wide["ab"] == "a") & (wide["ba"] == "b")) | (
        (wide["ab"] == "b") & (wide["ba"] == "a"))
    wide.columns.name = None
    return wide[[*keys, "ab", "ba", "choice", "flipped", "strict_flip"]]


def flip_rate(judge_rows: pd.DataFrame, by: Sequence[str] = ("judge", "question"),
              strict: bool = False) -> pd.DataFrame:
    """Share of pairs whose decision changes when A/B are swapped (positional bias, §5.4)."""
    r = resolve_table(judge_rows)
    col = "strict_flip" if strict else "flipped"
    return (r.groupby(list(by))[col].agg(["mean", "size"])
            .rename(columns={"mean": "flip_rate", "size": "n_pairs"}).reset_index())
