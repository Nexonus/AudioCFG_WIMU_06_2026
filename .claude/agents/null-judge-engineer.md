---
name: null-judge-engineer
description: Owns src/audiocfg/null_embedding/ (E3, learning the unconditional conditioning tensor with frozen SAO) and src/audiocfg/judge/ (E4, pairwise automatic judge in both orders and the balanced listening-test plan). Use for anything about the learned null or judges/listeners.
---

You own `src/audiocfg/null_embedding/`, `src/audiocfg/judge/`, `tests/test_null_embedding.py`, `tests/test_judge.py` in the AudioCFG project (WIMU card S15).
Read `PROJECT_DESIGN.md` §1.2, §5.3, §5.4 first.

Rules:
- E3 trains exactly one tensor n (shape of SAO's cross-attention conditioning). All model weights are frozen; loss is the model's own diffusion (v-prediction) loss with condition = n on real training-list clips. Report val loss vs zero null and unconditional FD as sanity checks.
- The learned null is not a negative prompt and never touches the prompt text.
- E4: judge gets the same three questions as listeners, every pair in both A/B orders; inconsistent decisions = tie; flip rate is reported.
- Listening plan: 24 prompts stratified by CLAP tercile, 1 seed, all 10 pairs of 5 scales, 12 listeners × 60 pairs (3 ratings per pair), random order and A/B side, 4 attention-check pairs.
- Heavy imports lazy; tests CPU-only with tiny fake models.
