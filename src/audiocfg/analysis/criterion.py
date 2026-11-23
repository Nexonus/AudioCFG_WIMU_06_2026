from __future__ import annotations

import numpy as np
import pandas as pd

from audiocfg.repro.config import CriterionConfig


def fd_at(fd: pd.DataFrame, null: str, g: float) -> float:
    curve = fd[fd["null"] == null].sort_values("guidance")
    return float(np.interp(g, curve["guidance"], curve["fd_music"]))


def verdict(matched: pd.DataFrame, cfg: CriterionConfig, fd: pd.DataFrame | None = None):
    rows = []
    for domain, part in matched.groupby("domain"):
        margin = cfg.ccds_margin[domain]
        for g in cfg.guidance:
            pq = part[(part["metric"] == "PQ") & (part["g_learned"] == g)].iloc[0]
            cc = part[(part["metric"] == "ccds") & (part["g_learned"] == g)].iloc[0]
            row = {"domain": domain, "g_learned": g, "g_star_zero": pq["g_star_zero"]}
            row["pq_better"] = bool(pq["lo"] > 0)
            row["ccds_not_worse"] = bool(cc["lo"] > -margin)
            row["fd_agrees"] = None
            if fd is not None and domain == "music" and not np.isnan(pq["g_star_zero"]):
                row["fd_agrees"] = fd_at(fd, "learned", g) < fd_at(fd, "zero", pq["g_star_zero"])
            checks = [row["pq_better"], row["ccds_not_worse"], row["fd_agrees"]]
            row["advantage"] = all(c for c in checks if c is not None)
            rows.append(row)
    return pd.DataFrame(rows)
