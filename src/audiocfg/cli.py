"""Single entry point: ``audiocfg <command>``. Each command maps to one step of an experiment."""

from __future__ import annotations

import argparse
import importlib

# command -> (module, help). Each module exposes ``add_args(parser)`` and ``run(args)``.
COMMANDS = {
    "prompts": ("audiocfg.generate.prompts", "E0: sample the 200 fixed prompts into data/prompts.csv"),
    "sweep": ("audiocfg.generate.sweep", "E1/E3: resumable CFG sweep (model x prompts x seeds x w x null)"),
    "metrics": ("audiocfg.metrics.run", "E1: CCDS, CLAP_norm, FD_openl3, KL_passt per clip/setting"),
    "copies": ("audiocfg.metrics.copying", "E2: nearest-neighbour search + signal-level verification"),
    "train-null": ("audiocfg.null_embedding.train", "E3: learn the null conditioning tensor (frozen SAO)"),
    "judge": ("audiocfg.judge.run", "E4: pairwise judge in both orders"),
    "listening-plan": ("audiocfg.judge.listening_plan", "E4: build the balanced pair plan for listeners"),
    "analyze": ("audiocfg.analysis.report", "Tables T1-T4 and figures F1-F6 from results/"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="audiocfg")
    parser.add_argument("--config", default=None, help="path to experiment.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (module, help_) in COMMANDS.items():
        p = sub.add_parser(name, help=help_)
        p.set_defaults(_module=module)
        importlib.import_module(module).add_args(p)
    args = parser.parse_args(argv)
    importlib.import_module(args._module).run(args)


if __name__ == "__main__":
    main()
