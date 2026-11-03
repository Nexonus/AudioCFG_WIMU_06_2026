"""Auxiliary Gradio app (PROJECT_DESIGN.md §8): clip preview + listening-test form.

NOT a research result. Two tabs:
  * Preview: (model, prompt, w, null variant, seed) -> one clip from a generation backend
    (``--backend fake`` by default; ``--backend model`` uses the real backend of the chosen model).
    There is deliberately no negative-prompt field (§1.2).
  * Listening test: a listener's pairs from ``e4_listening_plan.csv``; answers are written to
    ``e4_listeners.csv`` relative to (w_a, w_b), see ``app/listening.py``.

Run from the repo root: ``python -m app.app [--backend fake|model] [--port 7860]``.
Gradio and the generation backends are imported lazily so this module imports without them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script: python app/app.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import listening  # type: ignore[no-redef]
else:
    from . import listening

from audiocfg.config import DEFAULT_CONFIG, load_config
from audiocfg.schema import ClipSpec

NULL_VARIANTS = ("default", "learned", "t5_empty", "mean_caption")
SIDE_LABELS = {"A (left)": "left", "B (right)": "right", "no difference": "tie"}


def load_prompts(prompts_csv: Path) -> list[dict]:
    if not Path(prompts_csv).exists():
        return []
    with Path(prompts_csv).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Previewer:
    """Caches one backend per model; ``get_backend`` is imported on first use."""

    def __init__(self, cfg, backend: str = "fake", null_path: str | None = None) -> None:
        self.cfg, self.backend, self.null_path = cfg, backend, null_path
        self._cache: dict[str, object] = {}

    def _backend(self, model: str):
        if model not in self._cache:
            try:
                from audiocfg.generate.backends import get_backend
            except ImportError as e:  # generation module missing or its deps not installed
                raise RuntimeError(f"generation backend unavailable: {e}") from e
            name = model if self.backend == "model" else self.backend
            self._cache[model] = get_backend(name, self.cfg.models[model])
        return self._cache[model]

    def generate(self, model: str, prompt_id: str, domain: str, text: str, w: float, null: str,
                 seed: int, captions: list[str] | None = None):
        m = self.cfg.models[model]
        be = self._backend(model)
        null_cond = be.null_condition(null, path=self.null_path, captions=captions)
        spec = ClipSpec(model, prompt_id, domain, text, float(w), null, int(seed), m.sampler,
                        m.steps, self.cfg.clip_seconds)
        return be.generate(spec, null_cond)


def build_app(cfg, backend: str = "fake", null_path: str | None = None):
    import gradio as gr

    prompts = load_prompts(cfg.prompts_csv)
    by_label = {f"{p['prompt_id']}: {p['text']}": p for p in prompts}
    scales = sorted(cfg.guidance_scales)
    previewer = Previewer(cfg, backend, null_path)
    results = cfg.results_dir
    questions = cfg.raw.get("e4_judge", {}).get("questions", {})
    text_of = {p["prompt_id"]: p["text"] for p in prompts}

    def preview(model, prompt_label, free_text, w, null, seed):
        if free_text and free_text.strip():
            pid, domain, text = "free_text", "sfx", free_text.strip()
        elif prompt_label in by_label:
            p = by_label[prompt_label]
            pid, domain, text = p["prompt_id"], p["domain"], p["text"]
        else:
            raise gr.Error("Choose a prompt or type one.")
        caps = [p["text"] for p in prompts if p["domain"] == domain] or [text]
        try:
            audio, sr = previewer.generate(model, pid, domain, text, w, null, seed, caps)
        except Exception as e:
            raise gr.Error(str(e)) from e
        return sr, audio.T if audio.ndim == 2 else audio

    def resolve(path: str) -> str:
        p = Path(path)
        for cand in (p, cfg.outputs_dir / p, DEFAULT_CONFIG.parents[1] / p):
            if cand.exists():
                return str(cand)
        return str(p)

    def show(state):
        rows, i = state["rows"], state["i"]
        if i >= len(rows):
            return ("All pairs done — thank you!", None, None)
        r = rows[i]
        txt = text_of.get(r["prompt_id"], r["prompt_id"])
        return (f"Pair {i + 1}/{len(rows)} — description: “{txt}”",
                resolve(r["path_left"]), resolve(r["path_right"]))

    def load(listener_id):
        lid = (listener_id or "").strip()
        plan = listening.load_plan(results / listening.PLAN_FILE, lid)
        if not plan:
            raise gr.Error(f"No pairs for listener {lid!r} in {listening.PLAN_FILE}.")
        state = {"lid": lid, "rows": listening.remaining(plan, results / listening.ANSWERS_FILE,
                                                          lid), "i": 0}
        return (state, *show(state))

    def submit(state, a1, a2, a3):
        if not state or state["i"] >= len(state["rows"]):
            raise gr.Error("Load a listener first.")
        if None in (a1, a2, a3):
            raise gr.Error("Please answer all three questions.")
        ans = dict(zip(listening.QUESTIONS, (SIDE_LABELS[a] for a in (a1, a2, a3))))
        listening.record(results / listening.ANSWERS_FILE, state["rows"][state["i"]], ans)
        state = {**state, "i": state["i"] + 1}
        return (state, *show(state), None, None, None)

    with gr.Blocks(title="AudioCFG (auxiliary tool)") as demo:
        gr.Markdown("**AudioCFG — auxiliary tool** (preview and listening test; not a research "
                    "result).")
        with gr.Tab("Preview"):
            model = gr.Dropdown(list(cfg.models), value=next(iter(cfg.models)), label="model")
            prompt = gr.Dropdown(list(by_label), label="prompt (data/prompts.csv)")
            free = gr.Textbox(label="or free text (overrides the list)")
            w = gr.Slider(scales[0], scales[-1], value=cfg.models[next(iter(cfg.models))].default_w,
                          step=0.5, label="guidance scale w")
            null = gr.Dropdown(list(NULL_VARIANTS), value="default", label="null condition")
            seed = gr.Number(value=0, precision=0, label="seed")
            btn = gr.Button("Generate")
            out = gr.Audio(label="clip", type="numpy")
            btn.click(preview, [model, prompt, free, w, null, seed], out)
        with gr.Tab("Listening test"):
            state = gr.State({"rows": [], "i": 0})
            lid = gr.Textbox(label="listener id")
            load_btn = gr.Button("Load my pairs")
            info = gr.Markdown()
            with gr.Row():
                left = gr.Audio(label="A (left)", type="filepath")
                right = gr.Audio(label="B (right)", type="filepath")
            radios = [gr.Radio(list(SIDE_LABELS), label=questions.get(q, q))
                      for q in listening.QUESTIONS]
            save = gr.Button("Save and next")
            load_btn.click(load, lid, [state, info, left, right])
            save.click(submit, [state, *radios], [state, info, left, right, *radios])
    return demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--backend", default="fake",
                   help="'fake' (default) or 'model' (real backend named after the chosen model)")
    p.add_argument("--null-path", default=None, help="tensor file for the 'learned' null (E3)")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args(argv)
    build_app(load_config(a.config), a.backend, a.null_path).launch(server_port=a.port)


if __name__ == "__main__":
    main()
