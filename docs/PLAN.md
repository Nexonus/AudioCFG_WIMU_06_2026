# AudioCFG — plan działania i mapa kodu

Ten dokument łączy każde pytanie badawcze z `PROJECT_DESIGN.md` z konkretnym porównaniem,
danymi, poleceniem w kodzie, plikiem wyników oraz tabelą lub wykresem. Rezultatem projektu są
tabele T1–T4 i wykresy F1–F6; aplikacja Gradio (`app/`) jest wyłącznie narzędziem pomocniczym.

## 1. PB → porównanie → dane → kod → wynik

| PB | Porównanie | Dane | Polecenie (`audiocfg …`) | Pliki w `results/` | Wynik |
|---|---|---|---|---|---|
| **PB1a** różnorodność vs `w` | CCDS dla `w ∈ {1,3,5,7,9}`, te same seedy (test parowy vs `w=1`) | 200 promptów × 10 seedów × 5 skal × 2 modele | `prompts` → `sweep` → `metrics` | `e1_per_prompt.parquet` (`ccds_clap`, `ccds_aux`) | **F1**, **F2**, **T1** |
| **PB1b** zgodność i jakość vs `w`, zalecana skala | CLAP_norm, FD_openl3, KL_passt dla każdego `w`; kryterium `w*_auto` (§5.1) | jw. + zbiory odniesienia AudioCaps-test / Song Describer | `metrics` (+ kalibracja CLAP) → `analyze` | `e1_per_prompt.parquet`, `e1_fd.parquet`, `clap_calibration.csv` | **F3**, **T1** |
| **PB2** kopiowanie vs `w` | odsetek kandydatów / potwierdzonych kopii dla każdego `w` vs rozkład kontrolny H→T | generacje SAO z E1; T = nagrania z list atrybucji SAO; H = nagrania spoza list | `copies` | `e2_copies.parquet`, `e2_held.parquet` | **F4**, **T2** |
| **PB3** wyuczony null | wyuczony vs domyślny null, ten sam sweep i seedy, różnice parowe | SAO; ~5000 klipów z list atrybucji (rozłącznych z T) | `train-null` → `sweep --null learned` → `metrics`, `copies` | `e3_null_training.csv` + pliki E1/E2 z `null=learned` | **F5**, **T3** |
| **PB4** sędzia vs słuchacze | Bradley–Terry po skalach u sędziego i u słuchaczy na tych samych 240 parach; różnica preferowanych skal z CI; zgodność parami, kappa, pułap między słuchaczami | 24 prompty × 10 par; 12 słuchaczy × 60 par | `listening-plan` → `judge` (+ `app/` test odsłuchowy) | `e4_listening_plan.csv`, `e4_judge.parquet`, `e4_listeners.csv` | **F6**, **T4** |

Kontrakt kolumn każdego pliku jest w `src/audiocfg/schema.py` (`RESULT_FILES`). Producenci
zapisują dokładnie te kolumny, a `analysis/` czyta tylko te pliki.

## 2. Struktura repozytorium

```
configs/experiment.yaml     stałe ustawienia porównania (§4.3): skale, seedy, sampler, kroki, LUFS
data/prompts.csv            200 promptów ustalonych przed E1 (commitowane przed pierwszym sweepem)
src/audiocfg/
  config.py schema.py audio_io.py cli.py   wspólny rdzeń i kontrakty danych
  generate/        E0/E1/E3  próbkowanie promptów, backendy SAO/AudioLDM2/Fake, jawne CFG, sweep wznawialny
  metrics/         E1/E2     embeddery, CCDS, CLAP_norm, FD (+CI, równe N), KL_passt, detekcja kopii
  null_embedding/  E3        uczenie tensora null przy zamrożonym SAO, testy poprawności
  judge/           E4        sędzia parami w obu kolejnościach, plan testu odsłuchowego, protokół
  analysis/        T1–T4, F1–F6  bootstrap, Holm, Bradley–Terry, kryterium w*_auto
app/                         Gradio: podgląd generacji (bez negatywnego promptu) + formularz odsłuchu
outputs/                     klipy i meta.parquet (nie w git)
results/  figures/           tabele i wykresy raportu (w git, każda liczba z pliku)
agent_logs/                  prompty i sesje agentów (wymóg handoutu)
.claude/agents/              definicje ról agentów, którzy budowali szkielet
tests/                       testy CPU bez pobierania modeli (`pytest`); testy GPU: `-m gpu`
```

## 3. Agenci i podział odpowiedzialności

| Agent (`.claude/agents/`) | Odpowiada za | Eksperymenty |
|---|---|---|
| `generation-engineer` | `src/audiocfg/generate/`, `tests/test_generate.py` | E0, E1, E3 (generacja) |
| `metrics-engineer` | `src/audiocfg/metrics/`, `tests/test_metrics.py` | E1, E2 |
| `null-judge-engineer` | `src/audiocfg/null_embedding/`, `src/audiocfg/judge/`, testy | E3, E4 |
| `analysis-engineer` | `src/audiocfg/analysis/`, `app/`, `tests/test_analysis.py` | T1–T4, F1–F6, narzędzie pomocnicze |

Rdzeń (`config.py`, `schema.py`, `audio_io.py`, `cli.py`) jest wspólny i zmieniany tylko
świadomie, bo definiuje kontrakty między modułami.

## 4. Plan działania (etapy z §9 projektu)

### M1 Prototyp (do 04.11)
1. `pip install -e ".[gen,metrics,dev]"` na stacji z GPU; pomiar czasu generacji 10-s klipu.
2. **Potwierdzić w kodzie `stable-audio-tools`**, jak budowana jest gałąź bezwarunkowa i jak
   podstawić własny tensor null (wszystkie miejsca oznaczone `TODO(M1)`). Sprawdzić, że
   `null=default` daje bit-identyczne wyniki z oficjalnym pipeline'em dla tego samego seeda.
3. Zweryfikować dostępność list atrybucji SAO (Freesound / FMA) na Hugging Face.
4. `audiocfg prompts …` → zatwierdzić i zacommitować `data/prompts.csv` **przed** sweepem.
5. Mini-pilotaż E1: `audiocfg sweep --model sao --limit-prompts 20`, `audiocfg metrics`,
   `audiocfg analyze` → pierwsze F1 i F3 (CCDS i CLAP_norm vs `w`).

### M2 E1 dla SAO (05.11–24.11)
Pełny sweep SAO, kalibracja CLAP, FD/KL, **decyzja według kryterium obalenia** (§2).
Wniosek do komisji etycznej (E4).

### M3 E1 dla AudioLDM2 + E2 (26.11–16.12)
Sweep AudioLDM2; pobranie T i H; `audiocfg copies`; T2, F4.

### M4 E3 + sędzia (17.12–10.01)
`audiocfg train-null` (oba inicjalizacje), testy poprawności, `sweep --null learned`;
`audiocfg judge` na pełnym zbiorze; T3, F5.

### M5 Test odsłuchowy (04.01–12.01)
`audiocfg listening-plan`, uruchomienie `app/` dla 12 słuchaczy, `audiocfg judge --pairs-from-plan`;
T4, F6.

### M6 Raport (do 15.01)
`audiocfg analyze` generuje wszystkie tabele i wykresy z `results/`; odpowiedzi na PB1–PB4.

## 5. Szybki start (bez GPU)

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/audiocfg sweep --model sao --backend fake --limit-prompts 2
.venv/Scripts/audiocfg analyze --synthetic
```
