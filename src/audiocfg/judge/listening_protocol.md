# E4 listening test protocol (PROJECT_DESIGN.md §5.4)

## Before the test
- **Ethics.** The test starts only after the ethics committee has approved it (application in
  November, test planned for January). Each participant reads an information sheet and signs a
  consent form: voluntary participation, right to stop at any time without giving a reason,
  answers stored pseudonymously (`L01`..`L12`), no personal data beyond optional age range and
  listening experience, data used only for this course project and deleted after grading.
- **Equipment.** Closed or open over-ear headphones (no laptop speakers), a quiet room.
  The participant sets a comfortable volume on a calibration clip and does not change it later.
- **Stimuli.** All clips are SAO generations (default null), 10 s, normalised to −23 LUFS
  (ITU-R BS.1770), the same files the automatic judge receives.

## Instructions read to the participant (PL)
> Usłyszysz pary nagrań wygenerowanych na podstawie opisu tekstowego. Opis zobaczysz na ekranie.
> Każde nagranie możesz odtworzyć dowolnie wiele razy. Dla każdej pary odpowiedz na trzy pytania,
> wybierając A, B albo „bez różnicy”. Nie ma dobrych ani złych odpowiedzi — interesuje nas Twoje
> wrażenie. Test trwa około 25–30 minut; w połowie możesz zrobić przerwę.

## The three questions (each: A / B / bez różnicy)
1. `match` — „Które nagranie lepiej odpowiada opisowi?”
2. `realism` — „Które nagranie brzmi bardziej naturalnie, jak prawdziwe nagranie?”
3. `overall` — „Które nagranie ogólnie wolisz?” (used to estimate the preferred scale)

The automatic judge receives exactly the same questions (`templates/pairwise.txt`).

## Plan (`results/e4_listening_plan.csv`, built by `audiocfg listening-plan`)
- 24 prompts (12 SFX + 12 music) stratified by CLAP tercile at w = 7; one random seed per prompt;
  all 10 pairs of w ∈ {1, 3, 5, 7, 9} → 240 pairs.
- 12 listeners × 60 pairs (30 SFX + 30 music); each pair is rated by exactly 3 listeners.
- Pair order and left/right (A/B) side are randomised per listener; the guidance scale is never
  shown.
- 4 attention-check pairs per listener (the same clip as A and B). Expected answer: „bez różnicy”.
  A listener who prefers one side in ≥ 3 of the 4 checks on the `overall` question is reported
  and analysed with and without exclusion (rule fixed before the test).

## Recording answers
Answers are saved to `results/e4_listeners.csv` (one row per listener, pair, question) with
choices converted from left/right to the relative form `a`/`b`/`tie` w.r.t. (w_a < w_b).
