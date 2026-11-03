# AudioCFG
WIMU Project — Warsaw University of Technology, Winter 2026, Team 06

**Topic (card S15):** “What guidance does to generated audio: the guidance scale, the null embedding, diversity, copying and what judges prefer”.

Projekt badawczy: mierzymy, jak skala classifier-free guidance i wybór null embeddingu wpływają na różnorodność, kopiowanie danych treningowych, zgodność z promptem i jakość generacji w dwóch modelach text-to-audio (Stable Audio Open, AudioLDM2), oraz czy automatyczny sędzia preferuje tę samą skalę co słuchacze.

**Aktualny projekt wstępny:** [PROJECT_DESIGN.md](PROJECT_DESIGN.md) (wersja 2, po uwagach prowadzącego).
**Plan działania i mapa PB → eksperyment → kod → tabela/wykres:** [docs/PLAN.md](docs/PLAN.md).

## Szybki start (CPU, bez pobierania modeli)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/audiocfg analyze --synthetic        # suchy przebieg T1–T4, F1–F6
```

Na stacji z GPU: `pip install -e ".[gen,metrics,judge,app]"`, następnie polecenia
`audiocfg prompts | sweep | metrics | copies | train-null | listening-plan | judge | analyze`
(opis w `docs/PLAN.md`). Miejsca wymagające weryfikacji na prawdziwych wagach są oznaczone `TODO(M1)`.

Zespół: Jan Konarski, Jakub Szewczyk, Vladyslav Serhieiev
