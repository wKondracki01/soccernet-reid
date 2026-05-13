# Plan eksperymentów — SoccerNet Re-ID (praca magisterska)

**Temat**: Re-identyfikacja zawodników piłki nożnej na podstawie wycinka obrazu (bounding box) z wykorzystaniem technik uczenia metryki odległości. Studium porównawcze różnych backbone'ów, funkcji straty, strategii samplowania i augmentacji.

**Dataset**: SoccerNet Re-Identification 2023 (340 993 miniatur, 400 meczów, 6 lig). Lokalnie w `dataSoccerNet/reid-2023/`.

**Charakter pracy**: systematyczne studium porównawcze, **nie próba bicia SOTA** (leaderboard 2023 ≈ 91–93 mAP).

---

## 1. Sformułowanie zadania i kluczowe ograniczenia datasetu

**Zadanie retrievalowe**: dla zapytania (`query` bbox) zwrócić ranking obrazów `gallery` posortowany malejąco wg podobieństwa do tej samej osoby.

**Ograniczenie nr 1 (krytyczne)**: w SoccerNet ReID etykieta tożsamości jest ważna **wyłącznie w obrębie jednej akcji** (`action_idx`). Oficjalny ewaluator liczy mAP/Rank-k tylko po galerii z tej samej akcji co query. Konsekwencje:

- Etykieta treningowa = para `(action_idx, person_uid)`, **nie globalne `person_uid`**.
- Sampler musi rozumieć granicę akcji.
- Walidacja = pętla po akcjach → per-action mAP/Rank-k → uśrednienie po wszystkich query.

**Ograniczenie nr 2**: oficjalny podział `query/` ↔ `gallery/` w `valid/` i `test/` jest częścią datasetu — **nie konstruujemy go sami**, używamy zastanego.

**Źródło metadanych**: `train/train_bbox_info.json`, `valid/bbox_info.json`, `test/bbox_info.json` — komplet pól (`bbox_idx, action_idx, person_uid, frame_idx, clazz, id, UAI, relative_path, height, width`). Parser nazwy pliku tylko jako sanity check.

**Klasy osób (zweryfikowane na rzeczywistych plikach, nie z dokumentacji)**: 7 klas — `Player_team_{left,right}`, `Goalkeeper_team_{left,right}`, `Main_referee`, `Side_referee`, `Staff_members`. Dokumentacja SoccerNet wspominała o klasach „unknown" (10 łącznie), ale w datasecie ich nie ma.

**Rozkład próbek (zweryfikowany)**: train 248 234, valid 11 638 query + 34 355 gallery, test 11 777 query + 34 989 gallery, challenge 9 021 query + 26 082 gallery (anonimowy). Dystrybucja `(action, uid)` w train jest **skrajnie płaska**: 54.8% par to singletony (1 próbka), 33.5% ma 2 próbki, max to 8–10. Tylko 3% par ma ≥4 próbki. To dataset-specyficzny rozkład — kluczowy dla doboru P×K (patrz §3).

**Metryki raportowane**: mAP (główna), Rank-1, Rank-5, Rank-10 (krzywa CMC).

---

## 2. Cztery osie eksperymentalne

Pełny iloczyn kartezjański (5 backbone'ów × 6 strat × 5 samplerów × 3 augmentacje = 450 przebiegów) jest niewykonalny. Stosujemy **podejście „krzyżowe"**: ustalamy *baseline* na każdej osi, zmieniamy jedną oś naraz, a najlepsze kombinacje testujemy w fazie końcowej.

### Oś A — backbone (ekstraktor cech)
Wszystkie pretrenowane na ImageNet, wymieniona głowa → embedding `D = 512` (po BNNeck + L2-norm).

| Kod | Architektura | ~Parametry | Uwaga |
|-----|--------------|-----------:|-------|
| `R18` | ResNet-18 | 11 M | mały punkt odniesienia |
| `R34` | ResNet-34 | 21 M | środek skali |
| `EB1` | EfficientNet-B1 | 7 M | wydajny |
| `EB2` | EfficientNet-B2 | 9 M | trochę większy EB |
| `VGG16-BN` | VGG-16 z BatchNorm | 138 M | „starszy" baseline architektoniczny |

(Opcjonalnie `VGG11-BN` dla pełniejszego pokrycia rodziny VGG.)

### Oś B — funkcja straty
Domyślnie embedding po L2-norm dla strat metric (`CONT`, `TRI`, `MS`, `CIRCLE`) — kompatybilne z cosine similarity podczas retrievalu. `ARC` wymaga L2-norm z definicji (cosine margin). `CE` operuje na logitach z klasyfikatora — L2-norm embeddingu **nie jest wymagana** w treningu, ale jest stosowana w inferencji dla spójności metryki dystansu.

| Kod | Strata | Hiper-parametry startowe |
|-----|--------|--------------------------|
| `CE` | Cross-entropy nad klasami `(action,uid)` (sanity, klasyfikacyjny baseline) | label smoothing 0.1 |
| `CONT` | Contrastive (parowa, *siamese*) | margin = 0.5 |
| `TRI` | Triplet loss z hard mining | margin = 0.3 |
| `MS` | MultiSimilarityLoss | α=2, β=50, λ=1 |
| `CIRCLE` | CircleLoss | m=0.25, γ=64 |
| `ARC` | ArcFace — klasyfikator z cosine margin, po treningu odcinany | m=0.5, s=30 |

> **Uwaga terminologiczna**: w temacie pracy „kontrastywna" i „syjamska" to praktycznie ta sama rodzina (sieć bliźniacza + strata kontrastywna). W tabelach traktujemy je jako jeden wpis `CONT` i ewentualnie różnicujemy konfiguracje (parowa vs. trójkowa) w opisie.

### Oś C — strategia samplowania informatywnych przykładów

Oś C to **pakiety strategii**, łączące dwie ortogonalne decyzje:
- **Sampler** (co trafia do batcha): `RANDOM`, `PK` (P klas × K próbek), `PK-SA` (PK ograniczone do jednej akcji).
- **Miner** (co z batcha trafia do straty): `ALL`, `BATCH-HARD`, `SEMI-HARD`.

Testujemy 5 pakietów (a nie pełen iloczyn 3×3) — w pracy zaznaczamy, że nie odróżniamy wkładu samplera od minera w obrębie pakietu, tylko porównujemy strategie jako całości:

| Kod | Sampler | Miner | Dodatek | Komentarz |
|-----|---------|-------|---------|-----------|
| `RAND` | RANDOM | ALL | — | naiwny baseline |
| `PK-BH` | PK | BATCH-HARD | — | klasyka triplet/MS |
| `PK-SH` | PK | SEMI-HARD | — | FaceNet-style |
| `PK-SA-BH` | PK-per-action | BATCH-HARD | — | zgodne z protokołem ewaluacji |
| `PK-BH-XBM` | PK | BATCH-HARD | Cross-Batch Memory | bank cech, dla MS / CircleLoss |

### Oś D — augmentacje (3 zestawy z tematu)
Wejście: bbox o zmiennym H×W → resize do **256×128** (standard person re-id), normalizacja ImageNet.

| Zestaw | Skład |
|--------|-------|
| `AUG-MIN` | resize, horizontal flip, normalizacja |
| `AUG-MED` | AUG-MIN + ColorJitter (0.2/0.2/0.2/0.05), RandomCrop z paddingiem, Random Erasing (p=0.5) |
| `AUG-STRONG` | AUG-MED + RandAugment (n=2, m=9), Gaussian blur, RandomPerspective (p=0.3), AutoAugment policy „imagenet", mocniejsze RE (p=0.7, większy zakres scale/ratio) |

Uzasadnienie: ReID szczególnie korzysta z **Random Erasing** (Zhong et al.). Świadomie nie stosujemy **MixUp/CutMix** — te augmentacje mieszają etykiety, co działa tylko w klasyfikacji (CE/ARC); w stratach metric learning (CONT/TRI/MS/CIRCLE) nie istnieje „częściowo pozytywna para", więc miksowanie obrazów psułoby mining. AUG-STRONG musi działać z każdą stratą z Osi B, dlatego ograniczamy się do augmentacji obrazo-tylko.

---

## 3. Macierz eksperymentów — podejście etapowe

**Konfiguracja referencyjna** (start każdej osi):
`R18 + TRI + PK-BH + AUG-MIN`, embedding D=512, 60 epok, Adam(lr=3.5e-4, wd=5e-4), cosine LR z warmup 5 epok, batch **P=16/K=2 (=32)** dla samplerów cross-action; **P=8/K=2 (=16)** dla samplerów per-action (PK-SA).

> **Uzasadnienie P×K = 16×2 zamiast 16×4**: w SoccerNet ReID rozkład próbek per (action, uid) jest skrajnie *płaski* — 54.8% par jest singletonami, 33.5% ma dokładnie 2 próbki, maksimum to 8–10. **Tylko 4 z 9 181 akcji** ma 16 ID z ≥4 próbkami każde (=plan z K=4 dla PK-SA wycina 99.96% akcji); 39.5% akcji ma 8 ID z ≥2 próbkami (=PK-SA z P=8/K=2 jest wykonalny). K=4 wycina globalnie 90% datasetu, K=3 wycina 75%, K=2 zachowuje 66% próbek. Konwencje literatury z Market-1501 (K=4 standard, bo ID mają 15–30 zdjęć) **nie przenoszą się 1:1** na ten dataset — to dataset-specyficzny fakt udokumentowany w pracy.

### Faza 0 — sanity check i punkty odniesienia
- **F0a**: konfiguracja referencyjna do końca, zapisany checkpoint.
- **F0b**: **Wariant K (classifier baseline)** — `R18 + CE + losowy sampler` na wszystkich klasach `(action,uid)` z train po filtrze klas zawodniczych = **138 861 klas / 225 652 próbki** (singletony zostawione — klasyfikator z natury nie potrzebuje par, podobnie jak ArcFace na MS-Celeb-1M). Po treningu ucinamy głowę FC i używamy embeddingu. To drugi punkt odniesienia (klasyfikacja vs. metric learning, użyty potem w ablacji §7.3). Klasyfikator FC: 512 × 138 861 ≈ **71 M parametrów** samej głowy; logity per batch 32 w fp32 ≈ 17.6 MB.
- Walidacja narzędzia: nasz evaluator musi dać identyczny wynik co `tools/evaluate_soccernetv3_reid.py` z repo `sn-reid` na losowych embeddingach z `R18-ImageNet` (smoke test, do 4 miejsc po przecinku).

### Faza 1 — oś C (sampler+miner)
`R18 + TRI`, pakiet ∈ {`RAND, PK-BH, PK-SH, PK-SA-BH, PK-BH-XBM`}. **5 przebiegów.** → wybieramy `S*`.

### Faza 2 — oś B (strata)
`R18 + S*`, strata ∈ {`CE, CONT, TRI, MS, CIRCLE, ARC`}. **6 przebiegów.** → wybieramy `L*`.

> **Doprecyzowanie**: z pakietu `S*` przenosimy do Fazy 2 tylko **sampler**, **miner dobieramy do straty** zgodnie z literaturą:
> - `CONT` → all-pairs (bez minera),
> - `TRI` → BATCH-HARD (z `S*`, jeśli ma) lub SEMI-HARD,
> - `MS` → `MultiSimilarityMiner` (część definicji straty),
> - `CIRCLE` → BATCH-HARD lub własny pair miner z `pytorch-metric-learning`,
> - `CE`, `ARC` → **losowy sampler** niezależnie od `S*` (PK-SA daje w batchu klasy tylko z 1 akcji → softmax na dziesiątkach tysięcy klas degeneruje).
>
> XBM z pakietu `S*` dziedziczymy jeśli był i jeśli strata jest parowa.

### Faza 3 — oś A (backbone)
`{R18, R34, EB1, EB2, VGG16-BN} + S* + L*`. **5 przebiegów.** → wybieramy `B*`.

### Faza 4 — oś D (augmentacje)
`B* + S* + L* + {AUG-MIN, AUG-MED, AUG-STRONG}`. **3 przebiegi.** → wykres „augmentacja vs. mAP".

### Faza 5 — interakcje
2–3 najciekawsze kombinacje wybrane na podstawie poprzednich faz (np. czy mocne augmentacje pomagają tylko większym backbone'om; czy MS+`PK-BH-XBM` bije CircleLoss+`PK-BH` na każdym backbone). **6–9 przebiegów.**

**Łącznie Fazy 0–5**: ~25–30 pełnych przebiegów + sanity checks.
**Plus ablacje §7**: ~12–15 dodatkowych **treningów** (§7.1: 3 warianty głowy = 3, distance to wybór inferencji bez kosztu; §7.2 wymiar D: 5; §7.3 hybrydowy wariant H: 1 dodatkowy; §7.4 pretraining: 1; §7.5 pooling: 1; §7.6/§7.7 darmowe — post-hoc / z istniejących checkpointów; §7.8 efekt K: 2). Wariant K i Wariant M w §7.3 są już w F0b i Fazie 5 — nie liczymy podwójnie.
**Razem**: **~38–43 przebiegów**.

**Realny czas**: dla 225k próbek przy batch 32 (5000 iter/epoka, def. §5) i 60 epokach jeden przebieg R18 to ok. **2–4 h** na GPU klasy 3080/A6000 (mniejszy batch + krótsza forward niż w Market-1501 setup). Dla EB2/VGG16-BN ok. **5–10 h**. Łączny budżet GPU: **5–12 dni ciągłej pracy** (1 GPU), realnie 2–3 tyg. z przerwami. Akceleratory: AMP, skrócenie do 40 epok w Fazach 1–3, checkpoint co N epok + early-stop przy braku poprawy mAP przez 10 epok.

**Uwaga o porównywalności samplerów (Faza 1)**: PK-SA ma efektywny batch 16 vs. 32 dla pozostałych — utrzymujemy **tę samą liczbę iteracji (=update'ów wagowych)** dla wszystkich, akceptując że PK-SA widzi w sumie połowę próbek. Alternatywa „same próbki widziane" wymagałaby 2× więcej iteracji dla PK-SA i mieszałaby budżet z efektem samplera. Decyzja udokumentowana w pracy.

### Konwencja nazewnicza eksperymentów
`<faza>_<backbone>_<loss>_<sampler>_<aug>_<seed>` — np. `F3_EB2_MS_PK-SA_AUG-MED_s42`. Każdy przebieg → katalog z configiem (Hydra/OmegaConf), logami CSV/TensorBoard i checkpointem best-mAP.

---

## 4. Pipeline danych

1. **Loader `bbox_info.json`** → DataFrame z kolumnami `path, split, role` (query/gallery dla valid/test, brak dla train), `championship, season, game, action_idx, person_uid, clazz, frame_idx, h, w`.
2. **Sanity check parser nazwy pliku** vs. `bbox_info.json` — nazwa pliku musi być spójna z metadanymi (assert na losowych próbkach).
3. **Filtr klas — tylko w treningu**: decyzja do udokumentowania w pracy — czy w treningu uwzględniamy `Staff`, `Side referee`, `Main referee` (osoby z innym strojem, inna semantyka). Domyślnie: trening tylko na klasach „zawodniczych" (`Player_team_*`, `Goalkeeper_*`), sędziowie i staff odrzuceni. **Ewaluacja NIE filtruje klas** — zawsze pełny zbiór query/gallery z oficjalnego podziału, inaczej wynik byłby nieporównywalny z leaderboardem. Konsekwencja: model w teście musi sensownie embedować również klasy, których nie widział w treningu (test out-of-distribution dla `Staff`/sędziów). To samo w sobie ciekawa rzecz do dyskusji w pracy. Wariant alternatywny (też trening na pełnym zbiorze) można dodać jako mini-ablację.
4. **Singletony — bez explicit'nego filtra na katalogu**. Para `(action, uid)` z 1 próbką nie generuje pozytywnej pary, więc dla strat metric jest „bezużyteczna jako anchor". Ale **PK-style samplery (PK, PK-SA, SEMI, XBM) wybierają tylko klasy z ≥K próbek — singletony są naturalnie pomijane na poziomie batcha** bez ruszania katalogu. Dla strat klasyfikacyjnych (`CE`, `ArcFace`) singletony są w pełni użyteczne (każda osoba dostaje jeden gradient na FC; tak działa rozpoznawanie twarzy na MS-Celeb-1M / ArcFace). Wniosek: trzymamy pełen katalog (po filtrze klas), każdy sampler/strata używa go zgodnie ze swoją naturą. Liczby do raportu: 138 861 par `(action, uid)` po filtrze klas; z tego 76 147 (54.8%) singletonów (=trafia tylko do losowego samplera) i 62 714 par ≥2-próbkowych (=trafia też do PK-samplerów). To dataset-specyficzny rozkład udokumentowany w pracy (kontrast z Market-1501, gdzie ID mają 15–30 zdjęć).
5. **Sampler `PKPerActionBatchSampler`**: w każdym batchu wybiera 1 akcję, z niej P tożsamości × K próbek (próg odcięcia: ID musi mieć ≥K próbek w tej akcji). Wariant `PK` wybiera ID cross-action z tym samym progiem.
6. **Resize do 256×128** z paddingiem zachowującym aspekt (mini-ablacja: czy zachowanie aspektu pomaga).
7. **Augmentacje** — moduł z 3 presetami przełączanymi z configu (Albumentations lub torchvision v2).

---

## 5. Protokół treningowy (spójny dla wszystkich przebiegów)

- **Wejście**: 256 × 128, normalizacja ImageNet.
- **Głowa (`projection head`)**: `GAP → BN → FC(D) → BN → L2-norm`. Wymienialna przez config (parametr `head: {projection, bnneck, plain, classifier_cut}`):
  - `projection` — domyślna jak wyżej, dla strat metric,
  - `bnneck` — klasyczna wersja Luo et al. (BoT-ReID): triplet na cechach **przed** BN, klasyfikator na cechach **po** BN+FC; używana dla wariantu hybrydowego §7.3,
  - `plain` — bez końcowej L2, opcjonalnie bez końcowego BN (do ablacji §7.1),
  - `classifier_cut` — głowa klasyfikacyjna na czas treningu, odcinana w inferencji (Wariant K §7.3, F0b).
- **Optymalizator**: Adam(lr=3.5e-4, wd=5e-4), cosine schedule z warmup 5 epok.
- **Definicja epoki**: przy samplerach PK-style jeden batch nie odpowiada „przeglądowi datasetu". Przyjmujemy **epoka = 5000 iteracji** (≈ jeden przegląd 225 k próbek dla batcha 32; PK-SA z batch 16 widzi w sumie połowę próbek na epokę — patrz uwaga w §3 o porównywalności samplerów).
- **Epoki**: 60 (plateau na podobnych re-id setupach ok. 40–50). W Fazach 1–3 można skrócić do 40 epok i tylko najlepsze konfiguracje przedłużyć do 60.
- **Batch**: domyślnie **P=16/K=2 = 32** (samplery cross-action: PK, RAND, SEMI, XBM); **P=8/K=2 = 16** dla PK-SA (constraint datasetu: tylko 5% akcji ma 16 ID z ≥2 próbkami; 39% akcji ma 8 ID z ≥2 próbkami). Dla VGG-16 z 8 GB VRAM batch może wymagać dalszej redukcji do P=8/K=2 = 16 — odnotowane, ale finalne ciężkie przebiegi VGG idą i tak na cloud (A6000/A100).
- **Mixed precision (AMP)** — przyspiesza ~2×.
- **Seedy**: 3 seedy per kluczowy przebieg w Fazach 3/4/5 → raportujemy średnią ± odch. std. Faza 1/2: 1 seed.
- **Logowanie**: TensorBoard + CSV (loss, lr, valid mAP/R-1 co N epok); checkpoint best-mAP; pełna konfiguracja (Hydra) zapisana w katalogu eksperymentu.
- **Stack**: PyTorch + `pytorch-metric-learning` (gotowe MS/Triplet/Circle/ArcFace + miners + XBM) + `timm` (backbone'y) + Hydra/OmegaConf.

---

## 6. Protokół ewaluacji

1. Wyciągnij cechy dla wszystkich obrazów w `valid/query` i `valid/gallery` (i analogicznie dla `test/`).
2. Dla każdego query:
   - zawęź gallery do tej samej akcji (`action_idx`),
   - policz cosine similarity (lub euclidean — patrz ablacja §7.1),
   - wyznacz AP i pozycję pierwszego trafienia.
3. Uśrednij mAP, R-1, R-5, R-10 po wszystkich query.
4. **Walidacja narzędzia**: nasz evaluator musi dać identyczny wynik co `tools/evaluate_soccernetv3_reid.py` z repo `sn-reid` (smoke test w Fazie 0).
5. **`test/`** — używamy raz, na finalnych konfiguracjach z Faz 4/5. Nie używamy testu do tuningu.
6. **`challenge/`** — opcjonalnie jeden submission na koniec pracy (ground-truth ukryte, wynik tylko z leaderboardu). Nie używamy challenge do żadnej walidacji w trakcie pracy.

---

## 7. Ablacje uzupełniające (do dyskusji w pracy)

1. **L2-normalizacja embeddingu**: porównanie 3 wariantów głowy × 2 metryki dystansu = **6 konfiguracji** (na 1 najlepszym backbonie + stracie):
   - **Warianty głowy**: (a) `FC → BN → L2` [pełna], (b) `FC → BN` [bez L2], (c) `FC` [bez BN, bez L2].
   - **Metryki retrieval**: cosine, euclidean.

   Cosine z nieznormalizowanymi cechami efektywnie normalizuje na inferencji, ale strata podczas treningu widzi inne gradienty (Triplet/Contrastive z marginesem euklidesowym zachowuje się inaczej niż na sferze). Tabelka 3×2 z mAP i R-1.

2. **Wymiar embeddingu** D ∈ {128, 256, 512, 1024, 2048} — krzywa mAP(D) i czas inferencji. Hipoteza: plateau w okolicy 512; D=128 może być wystarczające do zastosowań produkcyjnych.

3. **Podejście klasyfikacyjne vs. metryczne** — *najważniejsza ablacja koncepcyjna pracy*:
   - **Wariant K (classification-then-cut)**: trening z głową klasyfikacyjną CE+label smoothing 0.1 + losowym samplerem nad **138 861 klasami** (`(action, uid)` po filtrze klas zawodniczych, singletony WŁĄCZNIE — klasyfikator nie potrzebuje par). Po treningu odcinamy FC i używamy embeddingu.
   - **Wariant M (metric learning)**: nasza najlepsza konfiguracja z Fazy 5 (PK-style sampler — singletony naturalnie pomijane na poziomie batcha, więc efektywnie 62 714 klas / 149 505 próbek).
   - **Wariant H (hybrid)**: CE + Triplet/MS jednocześnie (klasyczny przepis BoT-ReID, *Luo et al.*). Implementacja: dwie głowy — klasyfikacyjna nad pełnymi 138k klasami (jak K), metric nad cechami z PK samplera (jak M). W jednym batchu obie straty są liczone na rozłącznych podzbiorach (singletony tylko do CE, multi-próbkowe do obu).
   - Wszystkie trzy na tym samym backbonie / D / augmentacji. Każdy wariant używa **danych zgodnych z naturą swojej straty** (klasyfikator korzysta z singletonów, metric je naturalnie omija via sampler). To NIE jest „nieuczciwe porównanie" — to porównanie jak każdy paradygmat radzi sobie z naturalnym rozkładem datasetu, dokładnie jak robi to literatura ReID i face recognition.
   - Daje rozdział w pracy: „Czy klasyfikacja z odciętą głową konkuruje z deep metric learning na zbiorach z dużą liczbą małolicznych klas?"

4. **Pretraining**: ImageNet vs. od zera (1 backbone) — pokazuje wartość transferu.

5. **Pooling**: GAP vs. GeM — często +0.5–1.0 mAP w ReID.

6. **Re-ranking (k-reciprocal)** post-hoc — „darmowe" ulepszenie metryki na inferencji.

7. **Krzywa zbieżności**: mAP/Rank-1 vs. liczba epok dla top-3 konfiguracji z Fazy 5. Pokazuje, czy 60 epok wystarcza, gdzie jest plateau i czy któraś strata uczy się istotnie szybciej. *Darmowa* ablacja — wystarczy zapisywać metryki walidacyjne co N epok zamiast tylko best.

8. **Efekt K w samplerze (dataset-specyficzna ablacja)**: porównanie K=2 vs. K=3 dla najlepszej strategii Fazy 5 (PK lub PK-SA z najlepszą stratą metric). K=2 zachowuje 66% próbek (62 714 klas), K=3 zachowuje 25% (16 158 klas) — duży kompromis. MS / CircleLoss potencjalnie korzystają z większego K (więcej pozytywnych par per anchor), ale ceną jest 75% datasetu. Sprawdza czy ten kompromis się opłaca w tym konkretnie datasecie. **2 przebiegi** (K=2 vs K=3 na 1 najlepszej konfiguracji). Ciekawe, bo specyficzne dla SoccerNet ReID — kontrast z konwencją Market-1501 (K=4 standard).

> Uwaga implementacyjna: ablacje #1 i #3 wymagają wymienialnego modułu `head` (`bnneck` / `plain` / `classifier_cut`). Trzeba to założyć w pętli treningowej od początku — inaczej będziemy mieli 3 osobne pętle.

---

## 8. Co dostanie się do pracy magisterskiej (struktura wyników)

1. **Tabela A** — wpływ samplera (Faza 1).
2. **Tabela B** — wpływ funkcji straty (Faza 2).
3. **Tabela C** — wpływ backbone'u (Faza 3), z liczbą parametrów i czasem treningu/inferencji.
4. **Tabela D** — wpływ augmentacji (Faza 4) + krzywa uczenia.
5. **Tabela E** — interakcje (Faza 5).
6. **Krzywe CMC** dla top-3 konfiguracji.
7. **Wizualizacje**: t-SNE / UMAP embeddingów dla 1 akcji; przykłady top-k trafień i porażek (failure analysis: ten sam strój, podobna sylwetka, occlusion, zawodnik częściowo poza kadrem).
8. **Ablacje** z §7 w jednej sekcji.
9. **Tabela porównawcza z leaderboardem 2023** — uczciwe pozycjonowanie pracy względem SOTA.

---

## 9. Ryzyka i mitigacje

| Ryzyko | Mitigacja |
|--------|-----------|
| Zbyt duża macierz przebiegów na 1 GPU | Etapowa redukcja (§3); krótsze przebiegi na osi A/B (40 epok), pełne 60 tylko najlepsze |
| Mylenie globalnego `person_uid` z `(action,uid)` | Etykieta treningowa = `(action,uid)`, sampler `PK-SA` to wymusza, evaluator zawęża do akcji |
| Klasy „dziwne" (`Staff`, sędziowie) zaszumiają trening | Filtr klas **tylko w treningu** (§4.3); ewaluacja zawsze na pełnym zbiorze dla zgodności z leaderboardem |
| Niezgodność filtra treningowego z pełną ewaluacją | Ewaluacja jest „out-of-distribution" dla wykluczonych klas — udokumentowane, ewentualna mini-ablacja z pełnym treningiem |
| Niewłaściwa konfiguracja P×K dla tego datasetu | Liczby zweryfikowane na realnych danych: P=16/K=2 dla cross-action, P=8/K=2 dla PK-SA. K=4 wycina 90% datasetu — NIE używać. |
| Mylenie „filtra singletonów" z naturalnym pomijaniem ich przez PK sampler | NIE filtrujemy katalogu. PK samplery same omijają singletony przez wymóg ≥K próbek per ID. Klasyfikatory (CE/ArcFace) używają singletonów produktywnie. Zgodne z literaturą ReID i face recognition. |
| Niereprodukowalność | Seedy, deterministyczne cuDNN, konfigi Hydra zapisane w katalogu eksperymentu |
| Niezgodność z oficjalnym evalem | Smoke test (§6.4) przed Fazą 1 |
| Nadmierne dopasowanie do test-setu | `test/` używamy tylko raz, na finalnych konfiguracjach z Faz 4/5 |
| Konstruowanie własnego query/gallery | NIE — używamy oficjalnego podziału z `valid/{query,gallery}` i `test/{query,gallery}` |

---

## 10. Następne kroki implementacyjne

1. **Loader `bbox_info.json` + DataFrame z indeksem** + sanity check parser nazwy pliku (½ dnia).
2. **Evaluator zgodny z oficjalnym** + smoke test (1 dzień).
3. **Sampler PK-per-action + Dataset + augmentacje** (1 dzień).
4. **Pętla treningowa z konfiguracją Hydra**, integracja `pytorch-metric-learning` i `timm`, wymienialna głowa (1–2 dni).
5. **Faza 0** — sanity check + Wariant K.
6. Dalsze fazy zgodnie z §3.
