# AudioCFG app (auxiliary tool)

This app is an **auxiliary tool, not a research result** (PROJECT_DESIGN.md §0, §8). The results of
the project are the tables T1–T4 and figures F1–F6 produced by `audiocfg analyze`.

Install the optional dependency: `pip install -e .[app]` (Gradio). Run from the repository root:

```
python -m app.app --backend fake          # synthetic tones, no GPU / model weights
python -m app.app --backend model         # real SAO / AudioLDM2 backend of the chosen model
python -m app.app --backend model --null-path <learned_null.npy>   # enables the "learned" null
```

## Tabs

- **Preview** — one clip for (model, prompt from `data/prompts.csv` or free text, guidance scale `w`,
  null condition `default | learned | t5_empty | mean_caption`, seed). There is intentionally **no
  negative-prompt field**: the project changes only the null condition (§1.2).
- **Listening test** (E4, §5.4) — the listener enters their id, the app loads their pairs from
  `results/e4_listening_plan.csv` (skipping pairs already answered, so a session can be resumed),
  plays A (left) and B (right) and asks the three questions (A / B / no difference). Answers are
  appended to `results/e4_listeners.csv`, converted from left/right to `a`/`b` relative to
  `(w_a, w_b)` with `w_a < w_b` (`app/listening.py`, tested in `tests/test_analysis.py`).
