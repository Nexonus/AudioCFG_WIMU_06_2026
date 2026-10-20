# AudioCFG — Projekt wstępny (Project Design)

**WIMU, semestr zimowy 2026 · Zespół 06 · Politechnika Warszawska**
Karta z handoutu: **S15 — „What guidance does to generated audio: the guidance scale, the null embedding, diversity, copying and what judges prefer”**
(Co guidance robi z generowanym dźwiękiem: skala guidance, null embedding, różnorodność, kopiowanie i co preferują sędziowie)

| | |
|---|---|
| Zespół | Jan Konarski, Jakub Szewczyk, Vladyslav Serhieiev |
| Repozytorium | https://github.com/Nexonus/AudioCFG_WIMU_06_2026.git |
| Moc obliczeniowa | jedna stacja robocza z konsumenckim GPU |
| Docelowe konferencje (wg karty) | ISMIR 2027, ICASSP 2028 |
| Termin oddania design proposal | **21.10.2026 (środa), 23:59** |
| Status | projekt wstępny, 2026-10-20 |

---

## 1. Opis problemu

Dyfuzyjne modele text-to-audio podążają za promptem dzięki **classifier-free guidance (CFG)**. W każdym kroku odszumiania model wykonuje dwie predykcje: jedną warunkowaną promptem `c` i jedną warunkowaną warunkiem *pustym* (null) `∅`. Następnie łączy je:

```
ε̂ = ε(x_t, ∅) + w · (ε(x_t, c) − ε(x_t, ∅))
```

`w` to **skala guidance** (guidance scale). Dla `w = 1` mamy zwykłe próbkowanie warunkowe. Większe `w` mocniej przesuwa próbkę w stronę promptu. W generowaniu obrazów wyższe `w` daje lepszą zgodność z promptem, ale też **mniejszą różnorodność** i **częstsze kopiowanie danych treningowych**. Praca o przyspieszaniu Stable Audio Open (arXiv:2505.08175) wprost nazywa CFG „mieczem obosiecznym”: daje jakość i zgodność z promptem kosztem niskiej różnorodności i przesycenia (over-saturation). Gałąź null `∅` to zwykle stały wektor: same zera albo embedding pustego napisu. Praca z 2026 roku dotycząca mowy (arXiv:2609.25411) zastępuje go **wyuczonym null embeddingiem**. Raportuje, że jakość przy stałym nullu gwałtownie spada powyżej pewnej skali, a przy wyuczonym nullu osiąga stabilne plateau.

> **Uwaga o konwencji skali.** W pracy 2609.25411 używany jest wariant „conditional baseline”: `ε̂ = ε(c) + w'·(ε(c) − ε(∅))`. Tam `w' = 0` oznacza brak guidance, co odpowiada `w = 1 + w'` w zapisie powyżej. W całym projekcie używamy zapisu standardowego (jak w Stable Audio Open i `diffusers`) i przy porównaniach z literaturą przeliczamy skale.

W audio `w` jest hiperparametrem dobieranym „na ucho”, a wartości domyślne się różnią. Stable Audio Open używa 7 (tak też w 2603.04366), a AudioLDM2, Tango i MusicGen około 3–3,5. Nie znaleźliśmy pracy, która dla audio raportowałaby, jak różnorodność, kopiowanie i preferencje sędziego zmieniają się z `w`. Nie znaleźliśmy też pracy sprawdzającej, czy automatyczny sędzia preferuje tę samą skalę co słuchacze.

### Pytania badawcze

- **PB1:** Jak *różnorodność w obrębie promptu*, *różnorodność między promptami*, *copy rate* (odsetek kopii), *zgodność z promptem* i *jakość* zmieniają się wraz ze skalą guidance? Badamy to na dwóch otwartych modelach text-to-audio, osobno dla muzyki i efektów dźwiękowych.
- **PB2:** Czy wybór null embeddingu (zerowy, pusty napis, średnia z opisów, **wyuczony**) zmienia te krzywe, szczególnie przy wysokich skalach?
- **PB3:** Czy sędzia audio-językowy (audio-language judge) preferuje **inną skalę niż ludzcy słuchacze**? (luka sędzia–słuchacz)

### Hipotezy

- **H1:** Różnorodność maleje monotonicznie z `w`. Zgodność z promptem (CLAP) rośnie, a potem się nasyca. Copy rate rośnie z `w`.
- **H2:** Wyuczony null utrzymuje wyższą różnorodność i jakość przy dużym `w`, czyli spłaszcza degradującą część krzywych. W mowie zaobserwowano plateau jakości zamiast spadku (2609.25411).
- **H3:** Skala preferowana przez sędziego jest wyższa niż skala preferowana przez słuchaczy. Automatyczne strojenie `w` daje więc za wysoką wartość.

### Co obaliłoby hipotezę (zapisane przed jakimkolwiek uruchomieniem na dużą skalę)

> Na **obu** modelach różnorodność, copy rate i preferencja sędziego **nie** zmieniają się ze skalą guidance bardziej niż rozrzut między seedami (95% przedziały bootstrapowe dla `w=1` i `w=9` nakładają się dla każdej metryki), **oraz** słuchacze i sędzia preferują tę samą skalę.

Jeśli już pilotaż (etap M2) pokaże płaskie krzywe na Stable Audio Open, raportujemy to jako wynik („guidance w audio jest tańsze, niż sugeruje literatura obrazowa”) i nie skalujemy eksperymentu.

---

## 2. Rezultaty projektu

1. **Zbiór danych z przemiatania (sweep):** wygenerowane klipy i metadane (model, id promptu, skala, typ nulla, seed) dla każdego warunku, wraz ze wszystkimi metrykami w plikach Parquet/CSV. Każda liczba w raporcie pochodzi z jednego z tych plików.
2. **Krzywe:** różnorodność (CCDS), copy rate, CLAP, FD/KL i ocena sędziego w funkcji skali guidance. Osobno dla każdego modelu, typu nulla i domeny (muzyka vs. efekty dźwiękowe), z przedziałami bootstrapowymi.
3. **Wyuczony null embedding** dla Stable Audio Open (wagi i skrypt treningowy).
4. **Test odsłuchowy:** preferencje słuchaczy vs. preferencje sędziego w zależności od skali.
5. **Dashboard w Gradio (prototyp na 04.11):** prompt tekstowy na wejściu i przebieg audio na wyjściu. Zawiera **suwak CFG**, **wybór null embeddingu**, wybór seeda i liczby próbek oraz zakładkę do przeglądania wyników przemiatania i krzywych.
6. **Raport w formie artykułu:** krzywe, zalecana skala dla każdego zastosowania (muzyka / efekty) i luka sędzia–słuchacz.
7. **Log agentów:** prompty i sesje agentów kodujących, które wytworzyły kod, w katalogu `agent_logs/` (zasada z handoutu).

---

## 3. Modele i dane

### 3.1 Generatory (dwa otwarte modele text-to-audio)

| Model | Uzasadnienie | Warunkowanie / null | Dane treningowe (do sprawdzania kopii) |
|---|---|---|---|
| **Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`, `stable-audio-tools` / `diffusers`) | Model główny. Latentny DiT, enkoder tekstu T5, 44,1 kHz stereo, do 47,55 s; VAE daje latent 64 kanały × 21,5 Hz | Sekwencja T5 + warunkowanie czasem (`seconds_start`, `seconds_total`); gałąź bezwarunkowa = wyzerowane / puste warunkowanie | Podzbiory **Freesound + Free Music Archive (FMA)** na licencjach CC. Część muzyczna to 13 874 nagrań FMA (970 h, wg 2603.04366). Częściowo publiczne, więc nadają się do sprawdzania kopii |
| **AudioLDM2** (`cvssp/audioldm2`, `diffusers`) *(alternatywnie: Tango 2)* | Drugi model o innej architekturze (latentny UNet, warunkowanie CLAP + GPT-2/T5) | Embeddingi pustego / negatywnego promptu | AudioCaps, AudioSet, WavCaps itd. Część publiczna: **AudioCaps train** |
| *Punkt odniesienia:* **Stable Audio Open Small** (model po ARC post-training, 2505.08175) | Ta sama rodzina co SAO, ale **działa bez CFG**, czyli daje punkt „bez guidance” z innego mechanizmu | brak gałęzi null | jak SAO |

**Wyuczony null embedding** (PB2) trenujemy **tylko dla Stable Audio Open**. Tego wymaga karta, a to mieści się w budżecie obliczeniowym. SAO Small jest opcjonalnym punktem odniesienia, a nie trzecim pełnym przemiataniem: generujemy dla niego te same prompty i seedy raz.

### 3.2 Prompty (200)

- **100 promptów efektów dźwiękowych** z opisów zbioru testowego AudioCaps, np. „birds chirping while wind blows through trees”. Ten sam zbiór służy do oceny SAO w 2505.08175, gdzie CCDS liczono na 203 promptach AudioCaps.
- **100 promptów muzycznych** z **niewokalnego podzbioru Song Describer Dataset**, używanego do ewaluacji SAO w 2603.04366 i w pracy o SAO.
- Do oceny różnorodności dodajemy kilka **szerokich, niejednoznacznych promptów** (np. „fire crackling”, „water”), tak jak w teście odsłuchowym 2505.08175. Przy takich promptach spadek różnorodności jest najlepiej słyszalny.
- Lista promptów jest ustalona i wersjonowana (`data/prompts.csv`) przed pierwszym uruchomieniem. Obie domeny raportujemy osobno.

### 3.3 Korpusy referencyjne

- **Referencja dla FD/KL:** prawdziwe nagrania dla tych samych opisów (AudioCaps test, Song Describer).
- **Sprawdzanie kopii:** podzbiór *treningowy* (publiczna część Freesound/FMA dla SAO, AudioCaps-train dla AudioLDM2) oraz zbiór **odłożonych** (held-out) prawdziwych nagrań, o których wiadomo, że *nie* było ich w treningu. Ten drugi zbiór daje rozkład zerowy (sekcja 4.3).

---

## 4. Metodologia

### 4.1 Plan przemiatania (sweep)

| Czynnik | Poziomy |
|---|---|
| Skala guidance `w` | 1, 3, 5, 7, 9 (pilotaż); w pełnym przemiataniu dodajemy 2 i 12, jeśli krzywe się zaginają |
| Null embedding (SAO) | zerowy (domyślny), pusty napis T5, średni embedding opisów, **wyuczony** |
| Null embedding (AudioLDM2) | domyślny pusty prompt |
| Seedy na prompt | 10, wspólne dla wszystkich skal (ta sama lista seedów, czyli ten sam szum początkowy `x_T`, dla każdego `w`) |
| Długość klipu | 10 s (mniejszy koszt obliczeń; ta sama dla każdego warunku) |
| Sampler / kroki | stałe dla danego modelu: SAO 100 kroków (jak w 2505.08175 i 2603.04366); zmniejszamy do 50 tylko jednakowo dla wszystkich warunków. 2505.08175 pokazuje, że mniej kroków pogarsza FD, więc zmiana kroków jest osobnym czynnikiem, a nie skrótem |

**Te same seedy dla każdej skali** dają porównania parami: jedyną rzeczą, która zmienia się między dwoma klipami, jest `w`. DITTO (2401.12179) pokazuje, że początkowy szum `x_T` w dużej mierze wyznacza wynik i można nim nawet sterować. Dlatego:
- zapisujemy `x_T` (seed) dla każdego klipu;
- zapisujemy, czy sampler jest **deterministyczny** (DDIM, η = 0: różnorodność pochodzi tylko z `x_T`) czy **stochastyczny** (dodatkowy szum w każdym kroku). Różnorodność porównujemy tylko przy tym samym samplerze;
- zapisujemy dokładną konfigurację samplera i harmonogramu szumu. 2603.04366 zauważa, że SAO zachowuje się stabilniej w formie variance-preserving niż w variance-exploding z publicznego kodu.

Rozmiar: 200 promptów × 10 seedów × 5 skal = **10 000 klipów na parę (model, null)**. To około 6 konfiguracji, czyli łącznie około 60 tys. klipów. Szacunek czasu obliczeń jest w sekcji 7.

### 4.2 Różnorodność

- **Różnorodność w obrębie promptu: CCDS** (*CLAP Conditional Diversity Score*, zaproponowany w 2505.08175). To średnia kosinusowa odległość CLAP między parami generacji dla tego samego promptu, uśredniona po promptach. Autorzy pokazują, że CCDS zgadza się z oceną różnorodności w teście odsłuchowym, więc przyjmujemy go jako metrykę główną. Wynik SAO (100 kroków, CFG 7) z tej pracy służy nam jako punkt kontrolny odtworzenia.
- Drugi ekstraktor (**PANNs** dla efektów, **MERT** dla muzyki) sprawdza, czy trend nie zależy od jednego embeddingu.
- **Różnorodność między promptami:** **recall i coverage w przestrzeni PaSST** (`R_passt`, `C_passt`, jak w 2505.08175) oraz odległość między średnimi embeddingami promptów.
- **Vendi score** jako pomocnicza miara różnorodności, uwzględniająca liczebność próby.
- **Zastrzeżenie interpretacyjne:** 2505.08175 zauważa, że różnorodność *rośnie*, gdy model przestaje słuchać promptu, bo staje się wtedy w praktyce generatorem bezwarunkowym. Różnorodności przy niskim `w` nie interpretujemy więc w oderwaniu od CLAP: raportujemy obie krzywe razem (wykres różnorodność vs. CLAP z `w` jako parametrem).

### 4.3 Copy rate (z właściwym rozkładem zerowym zamiast sztywnego progu)

1. Dla każdego wygenerowanego klipu znajdujemy najbliższego sąsiada w podzbiorze treningowym (embedding CLAP / PANNs, indeks FAISS) i zapisujemy podobieństwo `s_gen`.
2. Dla każdego **odłożonego prawdziwego nagrania** liczymy to samo podobieństwo do najbliższego sąsiada w zbiorze treningowym (`s_held`). Tak wygląda rozkład podobieństw dla nowego audio, które nie zostało skopiowane.
3. **Copy rate** = odsetek wygenerowanych klipów, dla których `s_gen` przekracza 99. percentyl `s_held`. Dla każdej skali raportujemy też pełny rozkład `s_gen` na tle `s_held`.
4. Najbardziej podejrzane przypadki weryfikujemy odsłuchem i dopasowaniem chromy / fingerprintu. Potwierdzony odsetek raportujemy obok odsetka z embeddingów.
5. Zbiór treningowy jest tylko częściowo publiczny, więc copy rate raportujemy jako **ograniczenie dolne**. Karta wskazuje to jako główne ryzyko.

### 4.4 Zgodność z promptem i jakość

Używamy **tego samego zestawu metryk co prace o Stable Audio Open** (2505.08175, 2603.04366), żeby nasze wyniki dało się porównać z literaturą:
- **FD_openl3** (odległość Frécheta w przestrzeni OpenL3) zamiast samego FAD-VGGish. Jako metrykę uzupełniającą liczymy też FAD-VGGish (`frechet-audio-distance`). FD liczymy na całym warunku (2000 klipów na skalę), a jest on obciążony liczebnością próby, więc utrzymujemy jednakowe N we wszystkich warunkach.
- **KL_passt:** rozbieżność KL rozkładów etykiet klasyfikatora PaSST między generacją a referencją.
- **CLAP score:** podobieństwo kosinusowe embeddingów tekstu i audio (LAION-CLAP; MS-CLAP jako drugi model, żeby nie oceniać tym samym CLAP-em, którego używamy do różnorodności).
- **Audiobox Aesthetics** (wyuczony predyktor jakości) jako dodatkowy sygnał jakości.

**Interpretacja CLAP:** porównujemy wartości **względne** między skalami, a nie sztywne progi. Dla odniesienia: oryginalny SAO (100 kroków, CFG 7) osiąga na AudioCaps CLAP ≈ 0,29 (2505.08175, tab. 1), a na muzyce z Song Describer ≈ 0,41 (2603.04366, tab. 1). Ten sam model daje więc różne wartości bezwzględne w zależności od domeny i checkpointu CLAP, dlatego progi typu „> 0,4 = bardzo dobrze” nie byłyby wiarygodne.

### 4.5 Wyuczony null embedding (PB2)

**Co robi praca źródłowa (2609.25411).** Model TTS jest trenowany od zera. Każdy warunek (mówca, tekst) jest niezależnie zastępowany z prawdopodobieństwem 0,1 przez **osobny wyuczalny embedding** (inicjalizacja N(0, 1)), optymalizowany **razem z modelem**. Stały null zerowy ma dwie wady: leży poza rozkładem wejść modelu i nie rozróżnia typów warunkowania. Wynik: plateau jakości przy dużych skalach zamiast spadku, lepsze podobieństwo do mówcy i preferencja słuchaczy w CMOS.

**Nasza adaptacja do SAO.** Nie możemy trenować SAO od zera, więc testujemy dwa warianty o rosnącym koszcie:
1. **Null post-hoc (zamrożony model):** cały SAO jest zamrożony, a warunek null to trenowalny tensor o kształcie warunkowania T5 (inicjalizacja N(0, 1) jak w pracy albo od wektora zerowego). Optymalizujemy go stratą dyfuzyjną (v-prediction) w gałęzi bezwarunkowej na publicznej części FMA/Freesound. Trenujemy tylko tensor, więc mieści się to na jednym konsumenckim GPU. To odstępstwo od pracy (tam model uczy się razem z nullem) i tak je opisujemy w raporcie.
2. **Krótki wspólny fine-tuning (bliżej pracy, jeśli pozwoli budżet):** LoRA na DiT + wyuczalny null, z dropoutem warunku p = 0,1, jak w pracy. Trenujemy na tym samym podzbiorze FMA CC, krótko i z porównaniem do fine-tuningu z nullem zerowym przy identycznej liczbie kroków. Bez tej kontroli nie wiedzielibyśmy, czy zmiana pochodzi z nulla, czy z samego fine-tuningu.

Tanie warianty kontrolne, bez treningu: **zerowy** (domyślny), **kodowanie pustego napisu przez T5**, **średnia embeddingów opisów**.

**Rozszerzenie opcjonalne, rozdzielony CFG:** SAO ma dwa warunki, tekst i czas trwania (`seconds_total`). Analogicznie do równania (5) z 2609.25411 można użyć osobnych nulli i osobnych skal `w_text`, `w_timing`. Robimy to tylko, jeśli zostanie czas po PB1–PB3.

Ten sam plan przemiatania i metryki co w sekcji 4.1, więc krzywe można bezpośrednio nałożyć na siebie.

### 4.6 Automatyczny sędzia

- Otwarty sędzia audio-językowy: **Qwen2-Audio-7B-Instruct**, skwantyzowany do 4/8 bitów, jeśli wymaga tego VRAM.
- **Ocena punktowa:** ocena 1–10 ogólnej jakości i zgodności z promptem dla każdego klipu. Szablon promptu sędziego jest stały i trzymany w repozytorium.
- **Ocena parami:** ten sam prompt i seed przy dwóch skalach; sędzia wybiera lepszy klip. Każdą parę pokazujemy **w obu kolejnościach**, żeby kontrolować skłonność do wybierania pozycji (position bias). Dopasowanie modelu Bradleya–Terry'ego po skalach daje **skalę preferowaną przez sędziego**.
- Opcjonalnie drugi sędzia (wyuczony scorer, np. Audiobox Aesthetics), żeby wynik nie zależał od jednego sędziego.

### 4.7 Test odsłuchowy (PB3)

- **12 słuchaczy** (wymóg karty). Dla porównania: 2603.04366 miała n = 15. Zrównoważona próbka, np. 20 promptów (10 muzycznych + 10 efektów) × 5 skal, ten sam seed.
- Zadanie w formacie **MUSHRA-podobnym** (jak webMUSHRA w 2505.08175): dla każdego promptu słuchacz odsłuchuje 5 wersji w losowej kolejności i ocenia każdą w skali 1–5 w trzech osiach: **jakość dźwięku**, **zgodność z promptem** i, dla zestawów kilku seedów, **różnorodność**. Te same trzy osie stosują obie prace o SAO, więc wyniki będą porównywalne. Z ocen wyprowadzamy porównania parami do modelu Bradleya–Terry'ego.
- Wariant alternatywny: porównania parami **CMOS** (skala −2…+2, jak w 2609.25411), jeśli pilotaż pokaże, że ocena 5 wersji naraz jest dla słuchaczy zbyt trudna.
- Słuchacze dostają te same instrukcje co sędzia, więc oba wyniki dotyczą tego samego zadania.
- Interfejs: strona Gradio w tej samej aplikacji (zakładka `listening_test`). Odpowiedzi są anonimizowane i zapisywane do CSV.
- Karta wymaga **zgody komisji etycznej przed zbieraniem danych** oraz zaliczenia dla każdego uczestnika. Wniosek składamy do **04.11.2026**, żeby zgoda nie blokowała testu w styczniu.

### 4.8 Statystyka

- 95% **przedziały bootstrapowe** dla każdej metryki: losowanie promptów (i seedów w obrębie promptów) ze zwracaniem.
- „Zmienia się ze skalą” oznacza, że przedział dla `w=1` nie nakłada się z przedziałem dla skali o skrajnej wartości. To operacyjna postać kryterium obalenia.
- Testy parami między skalami (te same seedy), z poprawką Holma na wiele metryk.
- Luka sędzia–słuchacz: różnica między dwiema preferowanymi skalami (argmax z modelu Bradleya–Terry'ego), z przedziałem bootstrapowym.

---

## 5. Architektura systemu

```
                ┌──────────── configs/*.yaml (model, skale, null, seedy, sampler) ───────────┐
                ▼                                                                             │
data/prompts.csv ─► src/generate/ ─► outputs/clips/{model}/{null}/{w}/{prompt}_{seed}.wav     │
                     (wrappery SAO,        │   + outputs/meta.parquet                         │
                      AudioLDM2 z hookiem  ▼                                                  │
                      CFG + null)       src/metrics/  ─► results/metrics.parquet              │
                                         ├ różnorodność (CCDS, PANNs/MERT, R/C_passt, Vendi)  │
                                         ├ kopie (FAISS NN vs train + rozkład held-out)       │
                                         ├ clap_score, fd_openl3, kl_passt, fad, aesthetics   │
                                         └ sędzia (Qwen2-Audio punktowo + parami)             │
                                                   ▼                                          │
                                        src/analysis/ ─► figures/, tables/ (bootstrap)        │
                                                                                              │
src/null_embedding/ ─► checkpoints/null_sao.pt ───────────────────────────────────────────────┘
app/ (Gradio): zakładki Generowanie (prompt, suwak CFG, wybór nulla, seed, N) · Wyniki · Test odsłuchowy
```

### Struktura repozytorium

```
AudioCFG/
├── app/                  # dashboard Gradio (generowanie / wyniki / test odsłuchowy)
├── articles/             # przeczytane artykuły (PDF) z listy „read first”
├── configs/              # konfiguracje YAML przemiatania, po jednej na eksperyment
├── data/                 # prompts.csv, listy referencyjne i held-out (bez audio w gicie)
├── src/
│   ├── models/           # wspólny wrapper: generate(prompt, w, null, seed) dla SAO i AudioLDM2
│   ├── null_embedding/   # trening wyuczonego nulla + warianty bazowe
│   ├── metrics/          # różnorodność, kopie, clap, fd/kl, sędzia
│   └── analysis/         # bootstrap, krzywe, Bradley–Terry
├── scripts/              # punkty wejścia CLI (run_sweep.py, compute_metrics.py, ...)
├── results/              # tabele metryk (w gicie), wykresy
├── agent_logs/           # prompty i sesje agentów kodujących (zasada z handoutu)
├── PROJECT_DESIGN.md
└── README.md
```

Kluczowa zasada: generowanie jest **wznawialne i idempotentne**. Ścieżka każdego klipu koduje jego warunek, więc przerwane wielogodzinne przemiatanie wznawia się w miejscu przerwania i nigdy nie generuje gotowych klipów ponownie.

### Stos technologiczny

Python 3.11, PyTorch, `stable-audio-tools` / `diffusers`, `transformers` (Qwen2-Audio, T5), `stable-audio-metrics` (FD_openl3, KL_passt, CLAP, jak w pracach o SAO), `laion-clap`, `msclap`, `frechet-audio-distance`, `faiss`, `peft` (LoRA, wariant 2 z 4.5), `torchaudio`/`librosa`, `pandas`/`pyarrow`, `matplotlib`/`seaborn`, **Gradio** (dashboard i test odsłuchowy), konfiguracje `hydra`/YAML, `pytest`. Pliki audio są przechowywane poza gitem (dysk lokalny / dataset na Hugging Face). Wersjonujemy tylko metadane i metryki.

---

## 6. Harmonogram

### 6.1 Terminy przedmiotu (rozkład jazdy WIMU 26Z)

| Data | Wydarzenie | Znaczenie dla projektu |
|---|---|---|
| 07.10.2026 | Publikacja tematów | — |
| 14.10.2026 | Przypisanie zespołów | — |
| **21.10.2026 (śr.) 23:59** | **Design proposal deadline** | ten dokument |
| **04.11.2026 (śr.) 23:59** | **Zgłoszenie gotowości prototypu** | działający Gradio + SAO + suwak CFG + mini-pilotaż |
| **10.11.2026 (wt.) 23:59** | **Prototyp: deadline na spotkania** | prezentacja prototypu prowadzącemu |
| 25.11.2026 | Kolokwium 1 | tydzień o mniejszym obciążeniu projektem |
| _TBA_ | Standup 1, Standup 2 | statusy po M3 i po M4 |
| 13.01.2027 | Wykład: diffusion (gen. audio pt. 4) | temat bezpośrednio związany z projektem |
| **15.01.2027 (pt.) 23:59** | **Projekty: termin zwolnienia** | **cel zespołu:** zamknięte wyniki i raport |
| 20.01.2027 | Kolokwium 2 | — |
| **22.01.2027 (pt.) 23:59** | **Projekty: deadline finalny** | bufor na poprawki |
| **27.01.2027** | **Projekty demo** | prezentacja dashboardu i wyników |

### 6.2 Plan prac

| Etap | Termin | Zakres | Kryterium zakończenia |
|---|---|---|---|
| **M0** Design | 14.10 – **21.10** | Ten dokument; lektura 4 prac z `articles/`; zamrożenie listy promptów; **zapisanie w repo kryterium obalenia** | Design proposal oddany |
| **M1** Prototyp | 22.10 – **04.11** | Środowisko; wrapper SAO z hookiem CFG + null; dashboard Gradio (zakładka „Generowanie”: prompt, suwak CFG, wybór nulla, seed, N); **mini-pilotaż**: 20 promptów × 10 seedów × `w ∈ {1,3,5,7,9}` = 1000 klipów, CCDS + CLAP; odtworzenie wartości SAO z literatury; **złożenie wniosku etycznego** | Zgłoszenie gotowości prototypu 04.11; spotkanie do 10.11 |
| **M2** Pilotaż pełny | 05.11 – 24.11 | Pierwsze kroki z karty: SAO, 200 promptów × 10 seedów × 5 skal; pipeline FD_openl3 / KL_passt / CLAP / R,C_passt; pipeline copy rate z rozkładem held-out; preferowana skala jednego sędziego | Pierwsze krzywe z przedziałami; **decyzja: kontynuujemy / nie** (kryterium obalenia) |
| **M3** Drugi model + nulle | 26.11 – 16.12 | Przemiatanie AudioLDM2; SAO Small jako punkt bez CFG; bazowe nulle; trening wyuczonego nulla (wariant post-hoc, potem LoRA, jeśli pozwoli czas) | Wszystkie krzywe PB1; wyuczony null gotowy; **standup 1** |
| **M4** Nulle + sędzia | 17.12 – 10.01 | Przemiatanie SAO z wyuczonym nullem; ocena sędziego punktowa i parami z zamianą kolejności; model Bradleya–Terry'ego; przygotowanie próbki do testu odsłuchowego (przerwa świąteczna: długie przebiegi GPU w tle) | Krzywe PB2; preferowana skala sędziego; **standup 2** |
| **M5** Test odsłuchowy | 04.01 – 12.01 | 12 słuchaczy przez zakładkę Gradio (po uzyskaniu zgody etycznej) | Preferowana skala słuchaczy; luka sędzia–słuchacz |
| **M6** Raport | 11.01 – **15.01** | Krzywe, zalecenia per zastosowanie, luka; zakładka „Wyniki” w dashboardzie | Raport i odtwarzalne repo na **termin zwolnienia 15.01** |
| **M7** Bufor i demo | 16.01 – 27.01 | Poprawki po uwagach; ewentualnie rozdzielony CFG (4.5); przygotowanie demo | Deadline finalny 22.01; **demo 27.01** |

---

## 7. Szacunek kosztów obliczeniowych

- Punkt odniesienia z literatury: SAO przy 100 krokach i CFG 7 generuje 47 s audio w **11,3 s na H100** i zajmuje **5,5 GB VRAM** (2603.04366, tab. 1). Model mieści się więc na konsumenckim GPU, a czas na klip zależy od karty.
- Przy klipach 10 s i konsumenckim GPU zakładamy roboczo około 5–10 s na klip (z CFG, czyli dwoma przejściami w przód na krok). Dokładną wartość mierzymy w M1. Przy 5 s na klip 10 000 klipów zajmuje około **14 h** na konfigurację.
- Około 6 konfiguracji (SAO × 4 nulle + AudioLDM2 × 1, plus dodatkowe skale) to około **3–4 dni pracy GPU** na samo generowanie. Embeddingi, FD/KL i sędzia dokładają około 1–2 dni. Wariant LoRA (4.5) jest opcjonalny i dochodzi do tego tylko, jeśli zmieści się w czasie.
- Środki zaradcze: dla dodatkowych nulli pełne 10 seedów tylko przy `w ∈ {1,5,9}`, a mniej seedów przy pozostałych skalach. Nocne zadania wsadowe z możliwością wznowienia. Najdłuższe przebiegi planujemy na przerwę świąteczną.

---

## 8. Ryzyka i środki zaradcze

| Ryzyko | Środek zaradczy |
|---|---|
| Dane treningowe tylko częściowo publiczne, więc copy rate jest niedoszacowany | Raportujemy copy rate jako **ograniczenie dolne** i podajemy, którą część korpusu sprawdzono (rozwiązanie zaproponowane w karcie) |
| Obliczenia przekraczają możliwości jednego GPU | Klipy 10 s, mniej seedów dla dodatkowych nulli, decyzja po pilotażu |
| Wyuczony null post-hoc nie zbiega się lub daje to samo co null zerowy | To nadal wynik (negatywna odpowiedź na PB2). Wariant LoRA jest bliższy pracy źródłowej. Warianty bez treningu zachowują sens porównania |
| Efekt wyuczonego nulla myli się z efektem samego fine-tuningu | Kontrola: fine-tuning z nullem zerowym przy identycznych krokach (4.5) |
| Sędzia ma skłonność do wybierania pozycji lub sugeruje się promptem | Obie kolejności dla każdej pary; drugi sędzia; raportujemy odsetek zmian decyzji (flip rate) |
| Wysoka różnorodność przy niskim `w` wynika z ignorowania promptu | Różnorodność raportujemy zawsze razem z CLAP (4.2) |
| Wybór embeddingu decyduje o wyniku różnorodności / kopii | Dwa ekstraktory embeddingów na domenę; trendy muszą być zgodne |
| Opóźniona zgoda komisji etycznej | Wniosek do 04.11; test odsłuchowy dopiero w styczniu |
| Kumulacja terminów w styczniu (kolokwium 2, zwolnienie, deadline) | Cel: wyniki na 15.01; okres 16.01–22.01 to tylko bufor |
| Bezwzględne wartości CLAP / FD nie są porównywalne między modelami | Porównujemy **w obrębie** modelu między skalami; raportujemy zmiany względne |

---

## 9. Oczekiwane wyniki („result either way”, z karty)

- **Krzywe spadają i rosną zgodnie z oczekiwaniami:** pierwszy zmierzony kompromis guidance dla audio oraz zalecane `w` dla każdego zastosowania (muzyka vs. efekty dźwiękowe).
- **Płaskie krzywe:** guidance w audio jest tańsze, niż sugeruje literatura obrazowa. To również cenna informacja dla twórców modeli.
- **Luka sędzia–słuchacz:** automatyczne strojenie skali daje za wysoką wartość, więc `w` nie powinno być dobierane wyłącznie przez sędziego audio-językowego.

---

## 10. Literatura i wnioski dla projektu

### 10.1 Prace z karty (PDF w `articles/`)

| Praca | Co z niej bierzemy |
|---|---|
| **Learnable CFG Null Embeddings for Enhanced Controllable Speech Synthesis**, arXiv:2609.25411 | Główna inspiracja PB2: wyuczony null (dropout warunku 0,1, inicjalizacja N(0, 1), jeden null na typ warunku); hipoteza plateau jakości przy dużym `w`; rozdzielony CFG dla wielu warunków (rozszerzenie dla tekstu i czasu trwania w SAO); test CMOS jako wariant testu odsłuchowego; uwaga o innej konwencji skali `w` |
| **Low-Resource Guidance for Controllable Latent Audio Diffusion**, arXiv:2603.04366 | Szczegóły SAO (latent 64 × 21,5 Hz, 100 kroków DDIM, CFG 7, forma variance-preserving); koszt SAO (11,3 s, 5,5 GB na H100); dane FMA CC użyte do treningu SAO (do sprawdzania kopii); metryki FD_openl3 / KL_passt / CLAP; niewokalny Song Describer jako zbiór promptów muzycznych; MOS 1–5 dla jakości i zgodności z promptem |
| **Fast Text-to-Audio Generation with Adversarial Post-Training (ARC)**, arXiv:2505.08175 | Metryka **CCDS** (różnorodność w obrębie promptu) i R/C_passt; wartości bazowe SAO na AudioCaps (punkt kontrolny odtworzenia); obserwacja, że wysoka różnorodność może oznaczać ignorowanie promptu; webMUSHRA z osiami różnorodność / jakość / zgodność; SAO Small jako model bez CFG |
| **DITTO: Diffusion Inference-Time T-Optimization for Music Generation**, arXiv:2401.12179 | Szum początkowy `x_T` silnie wyznacza wynik, co uzasadnia te same seedy dla wszystkich skal i osobne raportowanie samplera deterministycznego / stochastycznego; CFG wymaga 2× aktywacji, co ma znaczenie dla budżetu VRAM |

### 10.2 Literatura uzupełniająca

Ho i Salimans, *Classifier-Free Diffusion Guidance* (arXiv:2207.12598); Evans i in., *Stable Audio Open* (arXiv:2407.14358); Kilgour i in., *Fréchet Audio Distance* (arXiv:1812.08466); Wu i in., *LAION-CLAP* (arXiv:2211.06687); Chung i in., *CFG++: Manifold-constrained classifier free guidance* (arXiv:2406.08070, cytowane w ARC jako źródło obserwacji o różnorodności i przesyceniu).

### Zasady projektu (z handoutu)

- Każda liczba w raporcie pochodzi z pliku wytworzonego przez zespół.
- Każda cytowana praca została przeczytana przez członka zespołu.
- Prompty i sesje agentów, które wytworzyły kod, są przechowywane w repozytorium (`agent_logs/`).
