# AudioCFG — Projekt wstępny (Project Design), wersja 2

**WIMU, semestr zimowy 2026 · Zespół 06 · Politechnika Warszawska**
Karta z handoutu: **S15 — „What guidance does to generated audio: the guidance scale, the null embedding, diversity, copying and what judges prefer”**
(Co guidance robi z generowanym dźwiękiem: skala guidance, null embedding, różnorodność, kopiowanie i co preferują sędziowie)

| | |
|---|---|
| Zespół | Jan Konarski, Jakub Szewczyk, Vladyslav Serhieiev |
| Repozytorium | https://github.com/Nexonus/AudioCFG_WIMU_06_2026.git |
| Moc obliczeniowa | jedna stacja robocza z konsumenckim GPU |
| Docelowe konferencje (wg karty) | ISMIR 2027, ICASSP 2028 |
| Status | wersja 2 po uwagach prowadzącego, 2026-10-28 |

---

## 0. Zmiany względem pierwszej wersji (odpowiedź na uwagi)

| Uwaga prowadzącego | Co zmieniliśmy | Gdzie |
|---|---|---|
| Projekt sprowadzony do panelu z suwakiem; interfejs opisany dokładniej niż eksperymenty | Projekt jest teraz opisany jako **badanie**: 4 eksperymenty, każdy z porównaniem, danymi i planowaną tabelą lub wykresem. Gradio jest tylko narzędziem pomocniczym (podgląd i test odsłuchowy) i nie jest rezultatem badawczym | §2, §3, §8 |
| Uczenie null embeddingu utożsamione z negatywnym promptem | **Negatywny prompt nie jest częścią projektu** i znika z interfejsu. Opisujemy, co dokładnie jest uczone (tensor warunkowania), na jakich danych, z jaką funkcją celu i jak porównujemy go z nullem domyślnym | §1.2, §5.3 |
| Związek „unikania zakłóceń” z kopiowaniem | Nie twierdzimy, że negatywny prompt ani wyuczony null ograniczają kopiowanie. Kopiowanie mierzymy osobno (E2), a kopia może brzmieć zupełnie poprawnie | §5.2 |
| 1. Porównanie skal CFG na dwóch modelach | Modele, wartości `w`, prompty, liczba generacji i ustawienia stałe są wypisane; kryterium wyboru „zalecanej skali” jest zdefiniowane przed eksperymentem | §4, §5.1 |
| 2. Pomiar różnorodności | Różnorodność liczymy z **odległości między nagraniami tego samego promptu** (CCDS + drugi ekstraktor), a nie z CLAP tekst–audio | §5.1 |
| 3. Sprawdzanie kopiowania | Zbiór porównawczy = nagrania z **opublikowanych list atrybucji** danych treningowych SAO. Kontrola = nagrania spoza treningu. Dwuetapowe odróżnienie kopii od podobnego dźwięku. Jasno opisane granice wniosków | §5.2 |
| 4. Sędzia i test odsłuchowy | Sędzia: Qwen2-Audio, porównania parami w obu kolejnościach, model Bradleya–Terry'ego. Test odsłuchowy: dobór próbek, dokładne pytania, plan porównania z sędzią | §5.4 |
| 5. Uzasadnienie miar | Progi CLAP usunięte i zastąpione kalibracją na prawdziwych nagraniach. Dla FD wskazane zbiory odniesienia i sposób porównywania ustawień | §6 |
| „Wysokie CFG czyni generator deterministycznym” | Poprawione: CFG **nie** zmienia determinizmu, tylko może zmniejszać różnorodność między seedami. To właśnie mierzymy | §1.1 |

---

## 1. Opis problemu

### 1.1 Classifier-free guidance

Dyfuzyjne modele text-to-audio podążają za promptem dzięki **classifier-free guidance (CFG)**. W każdym kroku odszumiania model wykonuje dwie predykcje: jedną warunkowaną promptem `c` i jedną warunkowaną warunkiem pustym `∅` (null). Następnie łączy je:

```
ε̂ = ε(x_t, ∅) + w · (ε(x_t, c) − ε(x_t, ∅))
```

`w` to **skala guidance**. Dla `w = 1` mamy zwykłe próbkowanie warunkowe, a większe `w` mocniej przesuwa próbkę w stronę promptu.

**Korekta względem pierwszej wersji:** wysokie `w` **nie czyni generatora deterministycznym**.
- Determinizm zależy od samplera: przy tym samym seedzie i deterministycznym samplerze (np. DDIM, η = 0) wynik jest identyczny dla *każdej* wartości `w`.
- `w` może natomiast zmniejszać **różnorodność**, czyli zmniejszać różnice między generacjami z *różnych* seedów dla tego samego promptu.

Czy i jak bardzo tak się dzieje w audio, jest jednym z pytań projektu, a nie jego założeniem. W obrazach wyższe `w` zmniejsza różnorodność i zwiększa kopiowanie danych treningowych. W pracy o Stable Audio Open (arXiv:2505.08175) CFG jest opisane jako źródło niskiej różnorodności i przesycenia, ale nikt nie zmierzył tego w funkcji `w` dla audio.

### 1.2 Null embedding a negatywny prompt

**Null embedding** to reprezentacja warunku, którą model dostaje w gałęzi bezwarunkowej `ε(x_t, ∅)`. W Stable Audio Open jest to stała (zerowa) sekwencja warunkowania. Model nauczył się ją rozumieć jako „brak opisu”, bo podczas treningu warunek był losowo zastępowany tą wartością. Praca arXiv:2609.25411 (synteza mowy) zastępuje stały null **wyuczonym** i raportuje stabilniejsze wyniki przy dużych skalach.

**Negatywny prompt** to inna operacja: w miejsce `∅` wstawia się embedding *konkretnego tekstu*, np. „loud buzzing”. Kierunek guidance zmienia się wtedy z „bardziej jak prompt niż brak opisu” na „bardziej jak prompt niż ten tekst”. **Negatywny prompt nie jest częścią projektu.** W eksperymencie E3 zmieniamy wyłącznie reprezentację braku warunku, a tekst promptu pozostaje ten sam.

> **Konwencja skali.** Praca 2609.25411 używa zapisu `ε̂ = ε(c) + w'·(ε(c) − ε(∅))`, w którym `w' = 0` oznacza brak guidance, czyli `w = 1 + w'`. W projekcie używamy zapisu standardowego (jak w Stable Audio Open i `diffusers`).

### 1.3 Luka w literaturze

W audio `w` dobiera się „na ucho”, a wartości domyślne się różnią: Stable Audio Open używa 7, AudioLDM2, Tango i MusicGen około 3–3,5. Nie znaleźliśmy pracy, która dla audio raportuje krzywe różnorodności, kopiowania i preferencji sędziego w funkcji `w`. Nie znaleźliśmy też pracy sprawdzającej, czy automatyczny sędzia preferuje tę samą skalę co słuchacze.

---

## 2. Pytania badawcze, eksperymenty i planowane wyniki

Każde pytanie badawcze ma przypisany eksperyment, dane, porównanie i konkretną tabelę lub wykres. Rezultatem projektu są te tabele i wykresy, a nie interfejs.

| PB | Pytanie | Eksperyment | Porównanie | Dane | Planowany wynik |
|---|---|---|---|---|---|
| **PB1a** | Czy generacje tego samego promptu stają się do siebie bardziej podobne, gdy rośnie `w`? | **E1** sweep skal (§5.1) | różnorodność dla `w ∈ {1,3,5,7,9}`, te same seedy; 2 modele × 2 domeny | 200 promptów × 10 seedów × 5 skal × 2 modele | **Wykres F1:** CCDS vs `w` (linie: model × domena, 95% CI). **Wykres F2:** CLAP vs CCDS z `w` jako parametrem (krzywa kompromisu). **Tabela T1** |
| **PB1b** | Jak zgodność z promptem i jakość zmieniają się z `w`? Jaka jest zalecana skala? | **E1** | CLAP (skalibrowany), FD_openl3, KL_passt dla każdego `w` | jak wyżej + zbiory odniesienia (§6) | **Wykres F3:** CLAP_norm, FD, KL vs `w`. **Tabela T1** z zalecaną skalą wg kryterium z §5.1 |
| **PB2** | Czy częstość kopiowania danych treningowych rośnie z `w`? | **E2** detekcja kopii (§5.2) | odsetek kopii dla każdego `w` vs rozkład kontrolny z nagrań spoza treningu | generacje z E1 dla SAO; nagrania z list atrybucji SAO; nagrania kontrolne | **Wykres F4:** rozkłady podobieństwa do najbliższego sąsiada (generacje dla każdego `w` vs kontrola). **Tabela T2:** odsetek kandydatów i potwierdzonych kopii dla każdego `w` (ograniczenie dolne) |
| **PB3** | Czy wyuczony null embedding zmienia krzywe z PB1 i PB2, szczególnie przy dużym `w`? | **E3** wyuczony null (§5.3) | wyuczony null vs null domyślny (zerowy), ten sam sweep, te same seedy | SAO; dane do uczenia nulla z list atrybucji | **Wykres F5:** krzywe z F1/F3 nałożone (null domyślny vs wyuczony). **Tabela T3:** Δ(wyuczony − domyślny) dla każdego `w` i metryki, z parowym CI |
| **PB4** | Którą skalę preferuje sędzia automatyczny, a którą słuchacze? Czy się różnią? | **E4** sędzia + test odsłuchowy (§5.4) | ranking skal (Bradley–Terry) u sędziego vs u słuchaczy, na tych samych parach | 24 prompty × 10 par skal; 12 słuchaczy | **Wykres F6:** wyniki Bradleya–Terry'ego vs `w` (sędzia i słuchacze na jednym wykresie). **Tabela T4:** preferowana skala obu stron, różnica z CI, zgodność decyzji parami |

### Hipotezy

- **H1:** CCDS maleje z `w`, a CLAP rośnie i się nasyca, więc między nimi istnieje kompromis.
- **H2:** Odsetek kandydatów na kopie rośnie z `w`.
- **H3:** Wyuczony null zmniejsza spadek jakości (FD) przy dużym `w`. W mowie zaobserwowano plateau jakości zamiast spadku.
- **H4:** Sędzia preferuje wyższą skalę niż słuchacze.

### Co obaliłoby hipotezę (zapisane przed uruchomieniem na dużą skalę)

> Na **obu** modelach różnorodność, odsetek kopii i preferencja sędziego **nie** zmieniają się z `w` bardziej niż rozrzut między seedami (95% CI dla `w=1` i `w=9` nakładają się dla każdej z tych metryk), **oraz** słuchacze i sędzia preferują tę samą skalę (CI różnicy zawiera 0).

Jeśli pilotaż (§9, etap M2) pokaże płaskie krzywe, raportujemy to jako wynik („guidance w audio jest tańsze, niż sugeruje literatura obrazowa”) i nie rozszerzamy eksperymentu.

---

## 3. Rezultaty projektu

1. **Tabele T1–T4 i wykresy F1–F6** z §2, każdy z 95% przedziałami ufności. Każda liczba pochodzi z pliku w `results/`.
2. **Zbiór generacji** z metadanymi (model, prompt, `w`, null, seed, sampler) oraz tabela metryk (Parquet).
3. **Wyuczony null embedding** dla SAO: tensor, skrypt uczący, krzywa straty.
4. **Wyniki testu odsłuchowego** (zanonimizowane CSV) i protokół testu.
5. **Raport w formie artykułu:** odpowiedzi na PB1–PB4 i zalecana skala dla każdego modelu i domeny.
6. **Narzędzie pomocnicze (Gradio):** podgląd generacji dla zadanego `w` i nulla oraz interfejs testu odsłuchowego. Bez pola na negatywny prompt. To narzędzie nie jest rezultatem badawczym.
7. **Log agentów** (`agent_logs/`), zgodnie z zasadami handoutu.

---

## 4. Modele, prompty i stałe ustawienia

### 4.1 Dwa modele

| | **Stable Audio Open 1.0** (SAO) | **AudioLDM2** (`cvssp/audioldm2`) |
|---|---|---|
| Architektura | latentny DiT, enkoder tekstu T5, 44,1 kHz stereo | latentny UNet, warunkowanie CLAP + GPT-2/T5, 16 kHz |
| Null domyślny | zerowa sekwencja warunkowania (do potwierdzenia w kodzie `stable-audio-tools` w M1) | embedding pustego napisu `""` (bez negatywnego promptu) |
| Domyślne `w` | 7 | 3,5 |
| Dane treningowe | Freesound + FMA na licencjach CC; **opublikowane listy atrybucji** (§5.2) | AudioCaps, AudioSet, WavCaps i in.; tylko część publiczna |
| Rola | E1, E2, E3, E4 | E1 (drugi model wymagany przez kartę) |

Przy porównaniu dwóch modeli nie zestawiamy bezwzględnych wartości metryk między nimi, bo różnią się pasmem, długością i danymi. Porównujemy **kształt krzywych w funkcji `w`** w obrębie każdego modelu.

### 4.2 Prompty (200, ustalone przed eksperymentem)

- **100 promptów SFX** losowanych ze zbioru testowego AudioCaps (stały seed losowania).
- **100 promptów muzycznych** losowanych z niewokalnego podzbioru Song Describer Dataset.
- Lista jest zapisana w `data/prompts.csv` i commitowana przed pierwszym uruchomieniem E1.
- Prompty są tekstami z tych zbiorów, więc do każdego promptu istnieje prawdziwe nagranie. Daje to zbiór odniesienia dla FD i kalibracji CLAP (§6).

### 4.3 Ustawienia stałe podczas porównania

| Ustawienie | Wartość |
|---|---|
| Badane skale `w` | **1, 3, 5, 7, 9**, wszystkie dla obu modeli (zawierają domyślne 3,5 / 7 w przybliżeniu). W pełnym przebiegu dodajemy `w = 2` i `w = 12`, jeśli krzywe zaginają się na krańcach |
| Generacje na (prompt, `w`) | **10**, seedy 0–9, **identyczne dla każdego `w`**: ten sam szum początkowy, więc między klipami zmienia się tylko `w` |
| Długość klipu | 10 s |
| Sampler i liczba kroków | stały w obrębie modelu: SAO DPM++ 3M SDE, 100 kroków; AudioLDM2 DDIM, 200 kroków (domyślne). Zapisywany w metadanych |
| Null | domyślny (E1, E2); w E3 zmieniany jest **tylko** null |
| Prompt | bez zmian, bez negatywnego promptu |
| Głośność | klipy normalizowane do −23 LUFS przed metrykami i odsłuchem |

Skala: 200 promptów × 10 seedów × 5 skal = **10 000 klipów na model**. E3 dokłada 10 000 klipów SAO z wyuczonym nullem. Razem około 30 000 klipów.

---

## 5. Eksperymenty

### 5.1 E1 — Sweep skal CFG: różnorodność, zgodność, jakość (PB1)

**Różnorodność w obrębie promptu: porównanie nagrań między sobą.** CLAP tekst–audio mierzy zgodność z opisem, a nie podobieństwo generacji do siebie, więc różnorodności z niego nie liczymy. Dla każdego promptu `p` i skali `w` mamy 10 klipów:

1. Liczymy embedding audio każdego klipu: **LAION-CLAP (audio)** oraz drugi ekstraktor, **PANNs** dla SFX i **MERT** dla muzyki.
2. **CCDS(p, w)** = średnia odległość kosinusowa po wszystkich 45 parach klipów. To metryka *CLAP Conditional Diversity Score* z arXiv:2505.08175, gdzie autorzy pokazali jej zgodność z oceną różnorodności przez słuchaczy.
3. Wynik dla `w` to średnia CCDS(p, w) po promptach. 95% CI liczymy bootstrapem po promptach.
4. Seedy są te same dla każdego `w`, więc stosujemy test parowy: różnica CCDS(p, `w`) − CCDS(p, 1) dla każdego promptu, test Wilcoxona z poprawką Holma.
5. **Sprawdzenie odporności:** trend musi być taki sam dla obu ekstraktorów.
6. **Uwaga interpretacyjna:** wysoka różnorodność przy niskim `w` może oznaczać, że model ignoruje prompt (obserwacja z 2505.08175). Dlatego CCDS pokazujemy zawsze razem z CLAP (wykres F2).

**Zgodność z promptem i jakość:** CLAP_norm, FD_openl3, KL_passt dla każdego `w`. Definicje, zbiory odniesienia i uzasadnienie są w §6.

**Kryterium „zalecanej skali”** (zdefiniowane przed eksperymentem, zamiast ogólnego „szukania optymalnych parametrów”):
- **Skala automatyczna `w*_auto`:** najmniejsza `w`, dla której CLAP_norm mieści się w 95% CI swojego maksimum po wszystkich `w`, a FD nie jest istotnie gorsze niż przy najlepszym `w`. Przy tej skali raportujemy, ile różnorodności (CCDS) traci się względem `w = 1`.
- **Skala ludzka `w*_ludzie`:** argmax wyniku Bradleya–Terry'ego ze słuchaczy (E4).
- Zalecenie końcowe podajemy osobno dla każdego modelu i domeny i wskazujemy, czy obie skale się zgadzają.

**Tabela T1** (dla każdego modelu i domeny):

| `w` | CCDS (CLAP) ± CI | CCDS (PANNs/MERT) ± CI | CLAP_norm ± CI | FD_openl3 ± CI | KL_passt ± CI |
|---|---|---|---|---|---|
| 1 … 9 | | | | | |

### 5.2 E2 — Kopiowanie danych treningowych (PB2)

**Z czym porównujemy i skąd wiemy, że to dane treningowe.** Stability AI opublikowała razem z SAO **listy atrybucji** nagrań użytych w treningu: identyfikatory Freesound i utwory FMA. W M1 weryfikujemy dostępność plików w repozytorium modelu na Hugging Face. Z tych list pobieramy nagrania, które są nadal publicznie dostępne. To jest **zbiór treningowy T**: wiemy z listy, że każde z nich było w treningu. Zakładamy podzbiór kilkudziesięciu tysięcy klipów, dobrany do kategorii promptów. Dokładny rozmiar zależy od tego, ile da się pobrać.

**Kontrola: nagrania spoza treningu (zbiór H).** Nagrania z Freesound i FMA, których **nie ma** na listach atrybucji, w szczególności dodane po dacie zebrania danych SAO. Dobieramy je tak, żeby miały ten sam rozkład kategorii i długości co T. Przy kopiowaniu znaczenie ma pytanie, czy generacja jest bliżej treningu, niż przypadkiem byłoby nowe nagranie tego samego typu. Zbiór H daje właśnie ten poziom odniesienia.

**Jak odróżniamy kopię od podobnego rodzaju dźwięku (dwa etapy):**

1. **Wyszukiwanie kandydatów (podobieństwo semantyczne).** Dla każdej generacji szukamy najbliższego sąsiada w T po embeddingu (CLAP, PANNs; indeks FAISS) i zapisujemy podobieństwo `s_gen`. To samo liczymy dla każdego nagrania z H: jego najbliższy sąsiad w T daje `s_held`. **Kandydat** = generacja z `s_gen` powyżej 99. percentyla rozkładu `s_held`. Etap 1 nie wystarcza: dwa różne nagrania deszczu mają podobne embeddingi.
2. **Weryfikacja kopii (podobieństwo sygnałowe).** Dla każdej pary (kandydat, sąsiad z T) liczymy **maksimum znormalizowanej korelacji wzajemnej log-mel spektrogramów** po przesunięciach w czasie, a dla muzyki dodatkowo dopasowanie chromy i fingerprint (Chromaprint). Próg weryfikacji ustalamy na parach z H (nowe nagranie vs najbliższe nagranie z T, czyli „ten sam typ, inne nagranie”) jako 99,9. percentyl. **Kopia potwierdzona** = przekroczenie tego progu, czyli fragment tego samego nagrania, a nie ten sam rodzaj dźwięku. Najwyżej ocenione przypadki dodatkowo odsłuchujemy w zespole.

Wynik dla każdego `w`: odsetek kandydatów i odsetek potwierdzonych kopii, z CI (bootstrap po promptach). Do tego test, czy odsetek rośnie z `w` (regresja logistyczna: kopia ~ `w`, efekty losowe promptu).

**Jakie wnioski będą możliwe przy ograniczonym dostępie.**
- Część nagrań z list nie jest już dostępna, więc T jest *podzbiorem* treningu. Wykryty odsetek kopii jest więc **ograniczeniem dolnym**: kopia nagrania spoza naszego podzbioru T nie zostanie wykryta.
- **Trend** odsetka w funkcji `w` jest jednak wiarygodny, bo ten sam podzbiór T dotyczy wszystkich skal. Główny wniosek E2 dotyczy więc *zmiany z `w`*, a nie bezwzględnej częstości.
- Dla AudioLDM2 część danych (AudioSet/WavCaps) jest niedostępna, a część z YouTube trudna do pobrania. E2 wykonujemy więc **tylko dla SAO**. Dla AudioLDM2 najwyżej w trybie eksploracyjnym na AudioCaps-train, z wyraźnym zastrzeżeniem.

**Wykres F4** i **Tabela T2:**

| `w` | N generacji | kandydaci (%) ± CI | potwierdzone kopie (%) ± CI | mediana `s_gen` | mediana `s_held` |
|---|---|---|---|---|---|

### 5.3 E3 — Wyuczony null embedding (PB3)

**Co jest uczone.** Jeden tensor `n ∈ R^{L×d}` o kształcie sekwencji warunkowania cross-attention SAO (tokeny T5 i tokeny czasu trwania; dokładne `L` i `d` odczytamy z konfiguracji modelu w M1). Ten tensor zastępuje zerowy null w gałęzi bezwarunkowej CFG. **Wszystkie wagi SAO są zamrożone**: zmienia się wyłącznie reprezentacja „braku warunku”.

**Na jakich danych.** Nagrania z list atrybucji treningu SAO (z §5.2), rozłączne z nagraniami użytymi później jako sąsiedzi w E2. Zakładamy około 5000 klipów po 10 s, zakodowanych wcześniej do latentów VAE. 10% odkładamy jako zbiór walidacyjny.

**Funkcja celu.** Null ma reprezentować rozkład bezwarunkowy `p(x)`. Minimalizujemy więc standardową stratę dyfuzyjną modelu (v-prediction) **przy warunku równym `n`**:

```
L(n) = E_{x ~ dane, t, ε} ‖ v_θ(x_t, t, n) − v_target(x, t, ε) ‖²
```

Gradient płynie tylko do `n` (AdamW, inicjalizacja zerami, czyli od nulla domyślnego, oraz wariant z N(0, 1) jak w 2609.25411). Wybór checkpointu: najniższa strata walidacyjna. To adaptacja pracy 2609.25411, gdzie null uczy się razem z całym modelem przy losowym (p = 0,1) pomijaniu warunku. W naszym przypadku model jest zamrożony. W raporcie opisujemy to jako świadome odstępstwo.

**Sprawdzenie, czy uczenie zadziałało (przed sweepem):**
- strata walidacyjna z `n` vs strata z nullem zerowym (musi być niższa);
- generacje **bezwarunkowe** (tylko gałąź null) z obu nulli: FD_openl3 względem zbioru walidacyjnego. Wyuczony null powinien dawać bardziej realistyczne próbki bezwarunkowe.

**Porównanie z nullem domyślnym.** Pełny sweep E1 dla SAO (te same prompty, seedy, skale, sampler) z wyuczonym nullem i te same metryki (CCDS, CLAP_norm, FD, KL, odsetek kopii z E2). Seedy są wspólne, więc dla każdej pary (prompt, seed, `w`) liczymy różnicę parową. **Tabela T3:** Δ = wyuczony − domyślny dla każdego `w` i każdej metryki, z parowym bootstrapowym CI. **Wykres F5:** nałożone krzywe. H3 potwierdza się, jeśli Δ FD < 0 istotnie przy `w ≥ 7`.

**Kontrole tanie (opcjonalne):** null = embedding pustego napisu T5 oraz średnia embeddingów opisów. Te warianty pokazują, ile daje samo przesunięcie nulla bez uczenia.

### 5.4 E4 — Sędzia automatyczny vs słuchacze (PB4)

**Sędzia automatyczny.** Qwen2-Audio-7B-Instruct, otwarte wagi, stała wersja (skwantyzowany, jeśli wymaga tego VRAM). Oceny drugiego, wyuczonego oceniacza (Audiobox Aesthetics) raportujemy obok, żeby wynik nie zależał od jednego sędziego.

**Jak ustalamy, którą skalę preferuje sędzia.**
- Porównania **parami**: ten sam prompt i ten sam seed, dwie różne skale. Pytanie do sędziego jest identyczne jak pytanie do słuchaczy (poniżej). Szablon jest zapisany w repo.
- Każdą parę pokazujemy **w obu kolejnościach** (A/B i B/A). Raportujemy odsetek zmian decyzji po zamianie kolejności jako miarę stronniczości pozycyjnej. Decyzja niespójna między kolejnościami liczy się jako remis.
- Z wszystkich porównań dopasowujemy **model Bradleya–Terry'ego** z parametrem siły dla każdej skali. **Skala preferowana przez sędziego** = argmax siły, z CI z bootstrapu po promptach.
- Sędzia ocenia pełny zbiór E1 (wszystkie 10 par skal dla każdego promptu i jednego seeda) oraz osobno **dokładnie te same pary co słuchacze**.

**Dobór próbek do testu odsłuchowego.**
- **24 prompty** (12 SFX + 12 muzyka), losowanie warstwowe z 200 promptów E1 według tercyli CLAP przy `w = 7`, tak żeby objąć prompty łatwe i trudne.
- Dla każdego promptu jeden losowy seed i 5 skal, czyli **10 par** (wszystkie kombinacje): łącznie 240 par.
- **12 słuchaczy**. Każdy ocenia 60 par (zrównoważony plan niekompletny), więc każda para ma 3 oceny. Kolejność par i pozycja A/B są losowe. Około 25–30 minut na osobę, słuchawki, wszystkie klipy znormalizowane do −23 LUFS.
- 4 pary kontrolne (ten sam klip jako A i B) sprawdzają uważność słuchacza.

**Pytania do słuchaczy** (dla każdej pary, wybór A / B / bez różnicy):
1. „Które nagranie lepiej odpowiada opisowi?” *(zgodność z promptem)*
2. „Które nagranie brzmi bardziej naturalnie, jak prawdziwe nagranie?” *(realizm / jakość)*
3. „Które nagranie ogólnie wolisz?” *(preferencja ogólna; na jej podstawie wyznaczamy preferowaną skalę)*

Sędzia dostaje te same trzy pytania. Dzięki temu porównanie dotyczy tego samego zadania, a nie ocen realizmu zebranych w oderwaniu od skal.

**Porównanie ocen słuchaczy i sędziego.**
- **Preferowana skala:** argmax Bradleya–Terry'ego dla słuchaczy vs dla sędziego, na tych samych 240 parach. Raportujemy **różnicę preferowanych skal z 95% CI** (bootstrap po promptach i słuchaczach). H4 potwierdza się, jeśli CI różnicy (sędzia − słuchacze) leży powyżej 0.
- **Zgodność decyzji parami:** odsetek par, w których sędzia wybiera to samo co większość słuchaczy, oraz współczynnik kappa Cohena.
- **Pułap odniesienia:** zgodność między samymi słuchaczami (każdy słuchacz vs większość pozostałych). Zgodność sędziego oceniamy na tle tej wartości.
- Osobno dla trzech pytań i dla dwóch domen.

**Zgoda etyczna.** Zgodnie z kartą test wymaga zgody komisji etycznej przed zebraniem danych. Wniosek składamy w listopadzie, a test jest zaplanowany na styczeń.

**Tabela T4:**

| | preferowana skala (BT) ± CI | zgodność z większością słuchaczy | kappa | flip rate (A/B) |
|---|---|---|---|---|
| słuchacze | | (pułap: zgodność między słuchaczami) | | — |
| Qwen2-Audio | | | | |
| Audiobox Aesthetics | | | | — |

---

## 6. Uzasadnienie miar

### 6.1 CLAP: zgodność z promptem, bez progów jakościowych

Pierwsza wersja podawała progi („> 0,4 bardzo dobrze”, „< 0,15 bardzo źle”) bez źródła. **Usuwamy je.** Wartości bezwzględne CLAP zależą od checkpointu i domeny. Dla tego samego modelu SAO literatura podaje CLAP ≈ 0,29 na AudioCaps (2505.08175) i ≈ 0,41 na muzyce z Song Describer (2603.04366), więc jeden próg nie może znaczyć „dobrze” w obu domenach.

Zamiast progów stosujemy **kalibrację na prawdziwych nagraniach** z tych samych zbiorów:
- **pułap** `C_real` = średni CLAP prawdziwego nagrania z jego własnym opisem;
- **podłoga** `C_mis` = średni CLAP prawdziwego nagrania z losowym, niepasującym opisem z tej samej domeny;
- **CLAP_norm** = (CLAP_gen − C_mis) / (C_real − C_mis). Wartość 0 oznacza „nie lepiej niż losowy opis”, a 1 oznacza „tak zgodne z opisem jak prawdziwe nagranie”.

Liczymy to dla LAION-CLAP oraz, jako sprawdzenie, dla MS-CLAP: wnioski o trendzie muszą się zgadzać dla obu modeli CLAP. Porównujemy CLAP_norm **między skalami w obrębie modelu i domeny**.

### 6.2 FD (Fréchet Distance): realizm rozkładu

- **Co mierzy:** odległość między rozkładem embeddingów generacji a rozkładem embeddingów prawdziwych nagrań. FD nie ocenia pojedynczych klipów, tylko cały zbiór.
- **Embedding:** OpenL3 (FD_openl3, jak w pracach o SAO, więc wyniki są porównywalne z literaturą), a pomocniczo VGGish (klasyczne FAD, biblioteka `frechet-audio-distance`).
- **Zbiory odniesienia:** SFX = nagrania ze zbioru testowego AudioCaps, z których pochodzą prompty, plus dalsze nagrania z AudioCaps-test; muzyka = nagrania z niewokalnego podzbioru Song Describer. Odniesienie jest **to samo dla wszystkich ustawień** danego modelu i domeny.
- **Porównywanie ustawień:** FD jest obciążone liczebnością próby, więc każde ustawienie (model, domena, `w`, null) ma **identyczne N = 1000** klipów (100 promptów × 10 seedów). CI liczymy bootstrapem po promptach. Różnicę między dwoma ustawieniami uznajemy za istotną, gdy CI różnicy nie zawiera 0.
- **Poziom odniesienia:** FD między dwiema rozłącznymi połówkami prawdziwego zbioru odniesienia, czyli najlepszy osiągalny wynik.
- **KL_passt** (rozbieżność rozkładów etykiet klasyfikatora PaSST między generacją a jej prawdziwym odpowiednikiem) uzupełnia FD na poziomie pojedynczego promptu.

### 6.3 Różnorodność i kopie

Uzasadnienie jest w §5.1 (CCDS liczony z odległości między generacjami, zgodny z ocenami słuchaczy w 2505.08175) i w §5.2 (próg z rozkładu kontrolnego zamiast arbitralnego).

### 6.4 Statystyka wspólna

- 95% CI z bootstrapu po promptach (oraz po słuchaczach w E4), 10 000 powtórzeń.
- Porównania między skalami są parowe (wspólne seedy), z poprawką Holma na liczbę metryk.
- „Zmienia się z `w`” = CI dla `w = 1` i dla skali o skrajnej wartości nie nakładają się. To operacyjna postać kryterium obalenia z §2.

---

## 7. Moc obliczeniowa

- SAO przy 100 krokach i CFG generuje 47 s audio w 11,3 s na H100 i zajmuje około 5,5 GB VRAM (2603.04366), więc mieści się na konsumenckim GPU. Rzeczywisty czas dla 10-sekundowego klipu na naszej karcie mierzymy w M1. Roboczo zakładamy 5–10 s na klip.
- E1 + E3 to około 30 000 klipów, czyli roboczo **2–3 doby pracy GPU**. Uczenie nulla to kilka godzin, bo uczymy tylko jeden tensor. Embeddingi, FD, wyszukiwanie kopii i sędzia dokładają około 1–2 doby.
- Jeśli czas okaże się za długi: 50 kroków samplera (jednakowo dla wszystkich ustawień) albo 5 zamiast 10 seedów w E3 przy `w ∈ {3, 5}`.

---

## 8. Architektura i narzędzie pomocnicze

```
data/prompts.csv ─► src/generate/ (SAO, AudioLDM2; parametry: w, null, seed) ─► outputs/clips + meta.parquet
                                                    │
src/null_embedding/ (E3: uczenie tensora null) ─────┘
                                                    ▼
src/metrics/  ─ ccds, clap_norm, fd_openl3, kl_passt (E1)
              ─ copy_search (FAISS), copy_verify (korelacja log-mel, chroma) (E2)
              ─ judge (Qwen2-Audio, pary w obu kolejnościach) (E4)
                                                    ▼
src/analysis/ ─ bootstrap, Bradley–Terry, testy parowe ─► results/ (T1–T4), figures/ (F1–F6)

app/ (Gradio, pomocniczo): podgląd generacji dla zadanego w i nulla · interfejs testu odsłuchowego
```

Generowanie jest wznawialne: ścieżka każdego klipu koduje jego ustawienia, więc przerwany przebieg nie generuje gotowych klipów ponownie.

**Stos:** Python, PyTorch, `stable-audio-tools`, `diffusers`, `transformers` (Qwen2-Audio), `laion-clap`, `msclap`, `stable-audio-metrics` / `frechet-audio-distance`, `faiss`, `librosa`, `pandas`, `matplotlib`, Gradio.

---

## 9. Harmonogram

| Etap | Termin | Zakres | Wynik |
|---|---|---|---|
| **M0** Design v2 | do 28.10 | Ta wersja; lista promptów zapisana w repo; kryterium obalenia zapisane | dokument |
| **M1** Prototyp | 29.10 – **04.11** | Wrapper SAO z parametrami `w`, null, seed; potwierdzenie postaci nulla w kodzie; weryfikacja list atrybucji; **mini-pilotaż E1**: 20 promptów × 10 seedów × 5 skal → pierwszy wykres CCDS i CLAP_norm vs `w`; podgląd w Gradio | zgłoszenie prototypu 04.11, spotkanie do 10.11 |
| **M2** E1 dla SAO | 05.11 – 24.11 | Pełny sweep SAO; kalibracja CLAP; FD/KL; **decyzja wg kryterium obalenia**; wniosek do komisji etycznej | T1 i F1–F3 dla SAO |
| **M3** E1 dla AudioLDM2 + E2 | 26.11 – 16.12 | Sweep AudioLDM2; pobranie T i H; wyszukiwanie i weryfikacja kopii | T1 dla obu modeli, T2, F4; standup 1 |
| **M4** E3 + sędzia | 17.12 – 10.01 | Uczenie nulla, sprawdzenie, sweep SAO z wyuczonym nullem; sędzia na pełnym zbiorze | T3, F5, część T4; standup 2 |
| **M5** Test odsłuchowy | 04.01 – 12.01 | 12 słuchaczy (po uzyskaniu zgody etycznej); sędzia na tych samych parach | T4, F6 |
| **M6** Raport | do **15.01** | Odpowiedzi na PB1–PB4, zalecane skale | raport (termin zwolnienia) |
| **M7** Bufor | do 22.01; demo 27.01 | Poprawki, demo | — |

---

## 10. Ryzyka

| Ryzyko | Postępowanie |
|---|---|
| Część nagrań z list atrybucji niedostępna | Odsetek kopii jako ograniczenie dolne; główny wniosek dotyczy trendu z `w` (§5.2) |
| Wyuczony null nie obniża straty względem zerowego | Negatywna odpowiedź na PB3 jest wynikiem; raportujemy krzywą straty |
| Model zamrożony ogranicza efekt nulla (inaczej niż w 2609.25411) | Opisujemy jako odstępstwo; jeśli zostanie czas, krótki fine-tuning LoRA razem z nullem, z kontrolą „LoRA + null zerowy” |
| Sędzia stronniczy pozycyjnie | Obie kolejności, raport flip rate, drugi sędzia |
| Opóźniona zgoda etyczna | Wniosek w listopadzie; E1–E3 nie zależą od testu odsłuchowego |
| Za mało czasu GPU | Mniej kroków lub seedów, jednakowo dla porównywanych ustawień (§7) |

---

## 11. Literatura

Prace z karty (PDF w `articles/`):
1. *Learnable Classifier-Free Guidance Null Embeddings for Enhanced Controllable Speech Synthesis*, arXiv:2609.25411. Wyuczony null, plateau jakości przy dużym `w`, konwencja skali (E3).
2. *Low-Resource Guidance for Controllable Latent Audio Diffusion*, arXiv:2603.04366. Szczegóły i koszt SAO, dane FMA, metryki FD_openl3 / KL_passt / CLAP, Song Describer.
3. *Fast Text-to-Audio Generation with Adversarial Post-Training*, arXiv:2505.08175. Metryka CCDS, wartości bazowe SAO, związek różnorodności z ignorowaniem promptu (E1).
4. *DITTO: Diffusion Inference-Time T-Optimization for Music Generation*, arXiv:2401.12179. Rola szumu początkowego, czyli uzasadnienie wspólnych seedów.

Uzupełniające: Ho i Salimans, *Classifier-Free Diffusion Guidance* (arXiv:2207.12598); Evans i in., *Stable Audio Open* (arXiv:2407.14358); Kilgour i in., *Fréchet Audio Distance* (arXiv:1812.08466); Wu i in., *LAION-CLAP* (arXiv:2211.06687).

**Zasady z handoutu:** każda liczba w raporcie pochodzi z pliku wytworzonego przez zespół; każdą cytowaną pracę przeczytał członek zespołu; prompty i sesje agentów są w `agent_logs/`.
