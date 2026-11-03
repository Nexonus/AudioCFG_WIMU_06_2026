"""Pure logic of the listening-test form (PROJECT_DESIGN.md §5.4); no Gradio here, so it is testable.

The listener sees a pair as left/right ("A" = left, "B" = right). Results are stored relative to
``(w_a, w_b)`` with ``w_a < w_b`` (``schema.CHOICES``), so left/right is converted back here.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from audiocfg.schema import CHOICES, QUESTIONS, RESULT_FILES

PLAN_FILE = "e4_listening_plan.csv"
ANSWERS_FILE = "e4_listeners.csv"
LISTENER_COLUMNS = RESULT_FILES[ANSWERS_FILE]
SIDES = ("left", "right", "tie")  # what the form returns for A (left), B (right), no difference


def lr_to_ab(w_left: float, w_right: float, answer: str) -> tuple[str, float, float]:
    """Convert a left/right/tie answer into ``(choice, w_a, w_b)`` with ``w_a <= w_b``.

    For an attention check (same clip both sides, ``w_left == w_right``) left maps to "a".
    """
    if answer not in SIDES:
        raise ValueError(f"answer must be one of {SIDES}, got {answer!r}")
    w_left, w_right = float(w_left), float(w_right)
    w_a, w_b = min(w_left, w_right), max(w_left, w_right)
    if answer == "tie":
        return "tie", w_a, w_b
    left_is_a = w_left <= w_right
    choice = "a" if (answer == "left") == left_is_a else "b"
    assert choice in CHOICES
    return choice, w_a, w_b


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def answers_to_rows(plan_row: Mapping, answers: Mapping[str, str]) -> list[dict]:
    """One ``e4_listeners.csv`` row per question from a plan row and ``{question: side}``."""
    missing = [q for q in QUESTIONS if q not in answers]
    if missing:
        raise ValueError(f"missing answers for {missing}")
    rows = []
    for q in QUESTIONS:
        choice, w_a, w_b = lr_to_ab(plan_row["w_left"], plan_row["w_right"], answers[q])
        rows.append({"listener_id": str(plan_row["listener_id"]), "pair_id": plan_row["pair_id"],
                     "prompt_id": plan_row["prompt_id"], "domain": plan_row["domain"],
                     "w_a": w_a, "w_b": w_b, "question": q, "choice": choice,
                     "is_attention_check": _as_bool(plan_row["is_attention_check"])})
    return rows


def read_csv(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_plan(plan_csv: Path, listener_id: str) -> list[dict]:
    """The listener's plan rows ordered by ``position``."""
    rows = [r for r in read_csv(plan_csv) if r["listener_id"] == str(listener_id)]
    return sorted(rows, key=lambda r: int(float(r["position"])))


def answered_pairs(answers_csv: Path, listener_id: str) -> set[str]:
    """Pairs for which the listener already answered every question (to resume a session)."""
    seen: dict[str, set[str]] = {}
    for r in read_csv(answers_csv):
        if r["listener_id"] == str(listener_id):
            seen.setdefault(r["pair_id"], set()).add(r["question"])
    return {p for p, qs in seen.items() if set(QUESTIONS) <= qs}


def remaining(plan: list[dict], answers_csv: Path, listener_id: str) -> list[dict]:
    done = answered_pairs(answers_csv, listener_id)
    return [r for r in plan if r["pair_id"] not in done]


def save_rows(answers_csv: Path, rows: list[dict]) -> None:
    """Upsert rows into ``e4_listeners.csv`` keyed by (listener_id, pair_id, question)."""
    answers_csv = Path(answers_csv)
    key = lambda r: (str(r["listener_id"]), str(r["pair_id"]), str(r["question"]))  # noqa: E731
    new = {key(r): r for r in rows}
    kept = [r for r in read_csv(answers_csv) if key(r) not in new]
    answers_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = answers_csv.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LISTENER_COLUMNS))
        w.writeheader()
        for r in kept + list(new.values()):
            w.writerow({c: r[c] for c in LISTENER_COLUMNS})
    tmp.replace(answers_csv)


def record(answers_csv: Path, plan_row: Mapping, answers: Mapping[str, str]) -> list[dict]:
    """Convert and save one pair's answers; returns the saved rows."""
    rows = answers_to_rows(plan_row, answers)
    save_rows(answers_csv, rows)
    return rows
