# data/

- `prompts.csv` — 200 promptów (100 SFX z AudioCaps-test, 100 muzycznych z niewokalnego podzbioru
  Song Describer), losowanych stałym seedem (`configs/experiment.yaml`) poleceniem `audiocfg prompts`.
  Plik commitujemy **przed** pierwszym uruchomieniem E1 i później go nie zmieniamy.
- `raw/` (nie w git) — pobrane zbiory: AudioCaps, Song Describer, listy atrybucji SAO,
  zbiór treningowy T i kontrolny H dla E2, klipy do uczenia nulla w E3.

Kolumny `prompts.csv`: `prompt_id, domain, text, source, source_id, reference_audio`
(`reference_audio` = ścieżka do prawdziwego nagrania dla FD i kalibracji CLAP).
