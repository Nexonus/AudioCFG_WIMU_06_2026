# AudioCFG

WIMU 2026Z, Politechnika Warszawska, zespół 06: Jan Konarski, Jakub Szewczyk, Vladyslav Serhieiev.

**Hipoteza.** Jeden globalny null CFG, wyuczony błędem odszumiania przy zamrożonym Stable Audio Open 1.0, poprawia jakość generacji przy porównywalnej zgodności z opisem i różnorodności, ponad to, co daje samo dostrojenie skali CFG. Ten sam embedding działa dla nowych promptów bez ponownego treningu.

**Wszystko w jednym miejscu: [notebooks/prototyp_audiocfg.ipynb](notebooks/prototyp_audiocfg.ipynb).** Notatnik zawiera:
- hipotezę i specyfikację: uczony tensor, cel treningowy, dane, metryki, kryterium przewagi,
- porównanie z pracami z listy,
- polecenia do odtworzenia każdego kroku,
- wyniki: trening nulla, przemiatanie CFG, ocenę, odsłuch A/B i powtórkę na nowych promptach.

## Szybki start

```bash
pip install -e ".[models,metrics,dev]"
pytest
audiocfg --help
```

Wyniki w `results/` są zacommitowane razem z manifestami (commit, hash configu, wersje pakietów, sha256 wejść i wyjść). Tabele i wykresy odtwarza się na CPU z tych plików, bez GPU. Obliczenia GPU działają na Kaggle przez joby w `kaggle/`.

## Struktura

| Katalog | Zawartość |
|---|---|
| `src/audiocfg/` | kod: `models` (SAO i własna pętla CFG), `nulls`, `data`, `train`, `generate`, `metrics`, `analysis`, `listen`, `repro` |
| `configs/` | jeden plik YAML na każdy etap |
| `data/` | zamrożone listy utworów i promptów |
| `results/` | wyniki z manifestami |
| `kaggle/` | joby GPU, uruchamiane przez `kaggle/push.py` z konkretnego commita |
| `notebooks/` | notatnik z propozycją, instrukcją odtworzenia i wynikami |
