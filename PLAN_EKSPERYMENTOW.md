# Plan eksperymentów — SoccerNet Re-ID (praca magisterska)

**Temat**: Re-identyfikacja zawodników piłki nożnej na podstawie wycinka obrazu (bounding box) z wykorzystaniem technik uczenia metryki odległości. Studium porównawcze różnych backbone'ów, funkcji straty, strategii samplowania i augmentacji.

**Dataset**: SoccerNet Re-Identification 2023 (340 993 miniatur z adnotacjami, 399 meczów, 6 lig). Lokalnie w `dataSoccerNet/reid-2023/`. Liczba meczów policzona na podziałach z adnotacjami: 290 (train) + 55 (valid) + 54 (test), bez wspólnych meczów między podziałami; oficjalny opis zbioru podaje 400.

**Charakter pracy**: systematyczne studium porównawcze, **nie próba bicia SOTA** (leaderboard 2023 ≈ 91–93 mAP).

---

## 0. Seria G (październik 2026) — zmiany względem serii F z maja 2026

Przebiegi z maja 2026 (nazwy `F0`–`F5`) są od października 2026 powtarzane jako seria `G` z poniższymi zmianami. Reszta ustawień (lr, harmonogram, definicja epoki, ziarno, głowa modelu) jest bez zmian. **Sekcje §1–§10 opisują stan obowiązujący dla serii G**; fragmenty dotyczące wyłącznie serii F są oznaczone jako zapis historyczny.

**W tekście pracy opisujemy wyłącznie serię G.** O serii F, o wykrytych w niej błędach i o ich diagnozie (w tym o parze przebiegów `DIAG_BOT_*`) w pracy nie piszemy — decyzja z 7.10.2026. Zapisy o serii F w tym dokumencie są notatką roboczą.

| Zmiana | Seria F (maj) | Seria G | Powód |
|---|---|---|---|
| Weight decay | 5e-4 (L2 wbudowane w `torch.optim.Adam`) | 0 | L2 w Adamie jest dzielone przez skalę gradientu, więc przy stratach o małym gradiencie (triplet, MS, contrastive) ściągało wagi do zera: na końcu treningu 97–99% wag konwolucji było zerami, a przebiegi AUG-STRONG / AUG-BOT kończyły z zerowymi embeddingami. Ten sam przebieg AUG-BOT z wd=0 daje mAP 0,785 (`DIAG_BOT_WD5E4` vs `DIAG_BOT_WD0`). |
| Sampler PK / PK-SA | te same 5000 batchy w każdej epoce (błąd) | nowe batche w każdej epoce | generator liczb losowych był tworzony od nowa co epokę |
| Margines ArcFace | 0,5° (błąd jednostek) | 28,6° = 0,5 rad | biblioteka przyjmuje margines w stopniach |
| Precyzja obliczeń | AMP w części przebiegów, FP32 w pozostałych | FP32 wszędzie | AMP powodował błędy CUDA i NaN z EfficientNet; mieszane ustawienia utrudniały porównania |
| Strata Circle | z minerem batch-hard (1 pozytyw i 1 negatyw na kotwicę) | wszystkie pary w batchu, bez minera | tak zdefiniowano ją w pracy źródłowej (Sun i in., 2020); miner usuwał ważenie par, które jest istotą tej straty |
| Random Erasing | AUG-MED: 2–33% pola; AUG-STRONG / AUG-BOT: 2–40% | 2–40% we wszystkich | zestawy mają się różnić tylko wymienionymi operacjami |
| Ewaluacja na valid | co 10 epok | co 5 epok | gęstsze krzywe uczenia do wykresów |
| Rozmycie w AUG-STRONG | każdy obraz (p = 1) | losowo, p = 0,5 | pozostałe operacje zestawu są losowe; przy p = 1 trening widział wyłącznie rozmyte wycinki, a ewaluacja żadnych |
| Kolejność osi | 1 dobór przykładów → 2 strata → 3 backbone → 4 augmentacje | 1 dobór przykładów → 2 strata → **3 augmentacje (na R18) → 4 backbone (z wybraną augmentacją)** | porównanie sieci przy samym odbiciu poziomym sprzyja małym sieciom; w serii F ranking backbone'ów zmienił się po przejściu na AUG-MED |
| Oś 1 — warianty | RAND, PK-BH, PK-SH, PK-SA-BH, PK-BH-XBM | te same + PK-SA-SH i PK-SA-BH-XBM | semi-hard i XBM były sprawdzone tylko z samplerem PK |
| Oś 5 | 6 kombinacji (§3, zapis historyczny) | skład do ustalenia po przeglądzie osi 1–4 | zwycięzcy osi mogą być inni niż w serii F |
| Przejście między osiami | kolejna oś uruchamiana po odczytaniu zwycięzcy | po każdej osi przegląd wyników (czy przebiegi przeszły poprawnie) i dopiero decyzja o konfiguracji następnej | kontrola przed dalszymi kosztami |
| Zbiór treningowy | 225 643 wycinki / 138 852 klasy — 9 plików pominiętych (nazwy zniekształcone przy rozpakowaniu na Windowsie) | 225 652 / 138 861 (komplet) | na maszynie z Linuksem nazwy plików przywrócono z adnotacji; samplery PK tych plików i tak nie losują (klasy jednoelementowe) |
| Sprzęt | laptop z RTX 3080 (16 GB), jeden przebieg naraz | wynajęta maszyna z 2 × RTX 4090, do 9 przebiegów równolegle | czas; pojedynczy przebieg trwa podobnie, bo ogranicza go procesor, nie karta |

Wnioski serii F o collapse'ie AUG-STRONG / AUG-BOT (niżej w tym dokumencie) dotyczą starego optymalizatora i nie opisują własności augmentacji. Opisy poniżej zostają jako zapis stanu z maja 2026.

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

**Rozkład próbek (zweryfikowany)**: train 248 234, valid 11 638 query + 34 355 gallery, test 11 777 query + 34 989 gallery, challenge 9 021 query + 26 082 gallery (anonimowy). Dystrybucja `(action, uid)` w train (po filtrze klas zawodniczych, 138 861 par) jest **skrajnie płaska**: 54.8% par to singletony (1 próbka), 33.5% ma 2 próbki, max to 9. Tylko 3.6% par ma ≥4 próbki (przed filtrem klas byłoby to 3.1%). To dataset-specyficzny rozkład — kluczowy dla doboru P×K (patrz §3).

**Metryki raportowane**: mAP (główna), Rank-1, Rank-5, Rank-10 (krzywa CMC).

---

## 2. Cztery osie eksperymentalne

Pełny iloczyn kartezjański (6 backbone'ów × 6 strat × 7 pakietów doboru przykładów × 4 augmentacje = 1008 przebiegów) jest niewykonalny. Stosujemy **podejście „krzyżowe"**: ustalamy *baseline* na każdej osi, zmieniamy jedną oś naraz, a najlepsze kombinacje testujemy w fazie końcowej.

### Oś A — backbone (ekstraktor cech)
Wszystkie pretrenowane na ImageNet, wymieniona głowa → embedding `D = 512` (głowa `projection`: BN → FC → BN → L2-norm, §5).

| Kod | Architektura | ~Parametry | Uwaga |
|-----|--------------|-----------:|-------|
| `R18` | ResNet-18 | 11,4 M | mały punkt odniesienia |
| `R34` | ResNet-34 | 21,5 M | środek skali |
| `EB1` | EfficientNet-B1 | 7,2 M | wydajny EfficientNet |
| `EB2` | EfficientNet-B2 | 8,4 M | większy EfficientNet |
| `VGG11-BN` | VGG-11 z BatchNorm | 130,9 M | shallow VGG (8 conv layers) |
| `VGG16-BN` | VGG-16 z BatchNorm | 136,4 M | mid VGG (13 conv layers) |

Liczby parametrów: sieć z głową `projection`, bez klasyfikatora. W serii G wszystkie sieci są trenowane w pełnej precyzji (FP32).

> **Zapis historyczny (seria F, maj 2026) — EfficientNet-B2 i większe z AMP**: pierwotnie plan zakładał `EB2` jako drugi EfficientNet point. Trening EB2 z PK-SA-BH sampler (batch=16) + **AMP=true** crashował 3× pod rząd z `CUDA error: invalid argument` (różne miejsca: `BatchHardMiner` lub `AMP scaler`), powtarzalnie nawet po reboot'cie i update driver'a NVIDIA (591.86 → 596.49). Test z **EB3** (12 M) dał identyczny crash w `scaler.step()` AMP. Test z **EB4** (19 M) miał inny failure mode: trening nie crashował explicit'nie, ale eval mode dawał **identyczne mAP=0.2808 we wszystkich 4 epokach eval** (to dokładnie wynik modelu zwracającego ten sam wektor dla każdego obrazu — zweryfikowane: stały embedding daje na valid mAP 0.2808 / R-1 0.1235 / R-5 0.5143 / R-10 0.7762, identycznie jak EB4; dla porównania R18 z ImageNet bez żadnego uczenia daje 0.3295, a losowy ranking 0.1942) — wskazuje na FP16 underflow w BatchNorm running statistics, które propaguje przez momentum update do permanently corrupted running_mean/running_var. **Empirycznie potwierdzony root cause**: PyTorch AMP + EfficientNet B2+ (depthwise convs + dużo BN layers) + small batch (16 z PK-SA-BH) = numerical instability w FP16. **Rozwiązanie**: trening **EB2 z AMP=false (FP32)** — eliminuje source 3 różnych failure modes (eksperymentalnie zweryfikowane: F3_EB2 z AMP=false zakończyło 40 epok stabilnie z mAP=0.7054). Wybrano EB2 zamiast EB3/EB4 jako "wystarczająco większy" point ponieważ pattern z całej Fazy 3 (R18→R34 +0.32pp; EB1 7M bije R34 21M) wskazuje że na 225k próbkach skalowanie EfficientNet poza B1 przynosi diminishing returns. Asterisk dla EB2 w tabeli wynikowej: *"trained in FP32 (AMP disabled) due to documented PyTorch AMP+EfficientNet+small-batch instability; AMP-on vs AMP-off typically differs <0.5pp mAP in literature"*. **W serii G** wszystkie przebiegi idą w FP32, więc to obejście nie jest potrzebne, a wniosek o malejących korzyściach ze skalowania pochodzi z serii F (stary optymalizator) i wymaga potwierdzenia w osi 4.

VGG przedstawiony w 2 wariantach (shallow/mid) zgodnie z briefem promotora o „wybranych wariantach VGG" — pozwala wyizolować efekt głębokości od pojemności w rodzinie VGG (oba mają >130 M params dzięki dense FC layers).

### Oś B — funkcja straty
Domyślnie embedding po L2-norm dla strat metric (`CONT`, `TRI`, `MS`, `CIRCLE`) — kompatybilne z cosine similarity podczas retrievalu. `ARC` wymaga L2-norm z definicji (cosine margin). `CE` operuje na logitach z klasyfikatora — L2-norm embeddingu **nie jest wymagana** w treningu, ale jest stosowana w inferencji dla spójności metryki dystansu.

| Kod | Strata | Hiper-parametry startowe |
|-----|--------|--------------------------|
| `CE` | Cross-entropy nad klasami `(action,uid)` (klasyfikacyjny punkt odniesienia) | label smoothing 0.1; sampler losowy |
| `CONT` | Contrastive (parowa, *siamese*) | margines pozytywów 0, negatywów 0.5 (odległość euklidesowa); wszystkie pary w batchu |
| `TRI` | Triplet loss | margin = 0.3 (odległość euklidesowa); miner wg zwycięzcy osi 1 |
| `MS` | MultiSimilarityLoss | α=2, β=50, λ=1; Multi-Similarity miner (ε=0.1) |
| `CIRCLE` | CircleLoss | m=0.25, γ=64; wszystkie pary w batchu, bez minera (strata sama je waży) |
| `ARC` | ArcFace — klasyfikator z marginesem kątowym, po treningu odcinany | margines 0.5 rad (= 28.6°), s=30; sampler losowy; jeden przebieg |

Każda strata ma jeden zestaw hiperparametrów, wzięty z literatury i niestrojony pod ten zbiór. Ranking strat dotyczy więc tych konkretnych ustawień — ograniczenie do zapisania w pracy.

> **Uwaga terminologiczna**: w temacie pracy „kontrastywna" i „syjamska" to praktycznie ta sama rodzina (sieć bliźniacza + strata kontrastywna). W tabelach traktujemy je jako jeden wpis `CONT` i ewentualnie różnicujemy konfiguracje (parowa vs. trójkowa) w opisie.

### Oś C — strategia samplowania informatywnych przykładów

Oś C to **pakiety strategii**, łączące dwie ortogonalne decyzje:
- **Sampler** (co trafia do batcha): `RANDOM`, `PK` (P klas × K próbek), `PK-SA` (PK ograniczone do jednej akcji).
- **Miner** (co z batcha trafia do straty): `ALL`, `BATCH-HARD`, `SEMI-HARD`.

Testujemy 7 pakietów (a nie pełen iloczyn). Dla samplerów PK i PK-SA sprawdzamy te same trzy warianty (batch-hard, semi-hard, batch-hard + XBM), więc wpływ minera i pamięci XBM da się odczytać osobno dla każdego samplera; `RAND` jest punktem odniesienia (tylko 0,35% losowych batchy zawiera parę pozytywną):

| Kod | Sampler | Miner | Dodatek | Komentarz |
|-----|---------|-------|---------|-----------|
| `RAND` | RANDOM | ALL | — | naiwny baseline |
| `PK-BH` | PK | BATCH-HARD | — | klasyka triplet/MS |
| `PK-SH` | PK | SEMI-HARD | — | FaceNet-style |
| `PK-SA-BH` | PK-per-action | BATCH-HARD | — | zgodne z protokołem ewaluacji |
| `PK-BH-XBM` | PK | BATCH-HARD | Cross-Batch Memory | bank 1024 embeddingów z poprzednich batchy |
| `PK-SA-SH` | PK-per-action | SEMI-HARD | — | dodany w serii G |
| `PK-SA-BH-XBM` | PK-per-action | BATCH-HARD | Cross-Batch Memory | dodany w serii G; pamięć zawiera wycinki z innych akcji |

### Oś D — augmentacje (4 zestawy: 3 z tematu + 1 alternatywa ReID-świadoma)
Wejście: bbox o zmiennym H×W → resize do **256×128** (standard person re-id), normalizacja ImageNet.

| Zestaw | Skład |
|--------|-------|
| `AUG-MIN` | resize, horizontal flip, normalizacja |
| `AUG-MED` | AUG-MIN + ColorJitter (0.2/0.2/0.2/0.05), RandomCrop z paddingiem 10 px, Random Erasing |
| `AUG-STRONG` | AUG-MED + RandAugment (n=2, m=9), Gaussian blur (p=0.5), RandomPerspective (p=0.3) |
| `AUG-BOT` | wariant kolorystyczny: AUG-MED z mocniejszym ColorJitter (0.4/0.4/0.4/0.1) + **RandomGrayscale** (p=0.2). **Bez** RandAugment/Perspective/Blur. Nazwa historyczna — to **nie** jest receptura BoT-ReID (zob. sprostowanie niżej). |

Random Erasing jest identyczne we wszystkich trzech zestawach: p=0.5, 2–40% pola, proporcje 0.3–3.3 (wartości Zhong et al. i BoT-ReID). Zestawy różnią się więc tylko operacjami wymienionymi w tabeli. Collapse AUG-STRONG / AUG-BOT z serii F nie wynikał z augmentacji, tylko z weight decay (§0).

Uzasadnienie: ReID szczególnie korzysta z **Random Erasing** (Zhong et al.). Świadomie nie stosujemy **MixUp/CutMix** — te augmentacje mieszają etykiety, co działa tylko w klasyfikacji (CE/ARC); w stratach metric learning (CONT/TRI/MS/CIRCLE) nie istnieje „częściowo pozytywna para", więc miksowanie obrazów psułoby mining. AUG-STRONG musi działać z każdą stratą z Osi B, dlatego ograniczamy się do augmentacji obrazo-tylko.

**AUG-BOT — sprostowanie (8.10.2026).** Wcześniejszy opis tego zestawu przypisywał go pracom BoT-ReID (Luo et al., 2019) i MGN (Wang et al., 2018) i twierdził, że zawierają one ColorJitter i RandomGrayscale. To nieprawda: oficjalna implementacja BoT-ReID (`reid-strong-baseline`, `data/transforms/build.py`) stosuje wyłącznie odbicie poziome, dopełnienie 10 px z losowym przycięciem i Random Erasing. Receptura BoT-ReID odpowiada więc naszemu **AUG-MED bez ColorJitter**. Zestaw `AUG-BOT` jest własnym wariantem: AUG-MED z mocniejszym ColorJitter i losową konwersją do skali szarości, bez operacji geometrycznych i rozmycia z AUG-STRONG. Nazwa zostaje w kodzie i w nazwach przebiegów (`G3_AUG_BOT`), ale w tekście pracy zestawu nie należy opisywać jako receptury BoT-ReID.

---

## 3. Macierz eksperymentów — podejście etapowe

**Konfiguracja referencyjna** (start każdej osi):
`R18 + TRI + PK-BH + AUG-MIN`, embedding D=512, 60 epok, Adam(lr=3.5e-4, weight decay 0 — §0), cosine LR z warmup 5 epok, batch **P=16/K=2 (=32)** dla samplerów cross-action; **P=8/K=2 (=16)** dla samplerów per-action (PK-SA).

> **Uzasadnienie P×K = 16×2 zamiast 16×4**: w SoccerNet ReID rozkład próbek per (action, uid) jest skrajnie *płaski* — 54.8% par jest singletonami, 33.5% ma dokładnie 2 próbki, maksimum to 9. **Tylko 4 z 9 181 akcji** ma 16 ID z ≥4 próbkami każde (=plan z K=4 dla PK-SA wycina 99.96% akcji); 39.5% akcji ma 8 ID z ≥2 próbkami (=PK-SA z P=8/K=2 jest wykonalny). K=4 wycina globalnie 90% datasetu, K=3 wycina 75%, K=2 zachowuje 66% próbek. Konwencje literatury z Market-1501 (K=4 standard, bo ID mają 15–30 zdjęć) **nie przenoszą się 1:1** na ten dataset — to dataset-specyficzny fakt udokumentowany w pracy.

### Faza 0 — sanity check i punkty odniesienia
- **F0a**: konfiguracja referencyjna do końca, zapisany checkpoint.
- **F0b**: **Wariant K (classifier baseline)** — `R18 + CE + losowy sampler` na wszystkich klasach `(action,uid)` z train po filtrze klas zawodniczych = **138 861 klas / 225 652 próbki** (singletony zostawione — klasyfikator z natury nie potrzebuje par, podobnie jak ArcFace na MS-Celeb-1M). Po treningu ucinamy głowę FC i używamy embeddingu. To drugi punkt odniesienia (klasyfikacja vs. metric learning, użyty potem w ablacji §7.3). Klasyfikator FC: 512 × 138 861 ≈ **71 M parametrów** samej głowy; logity per batch 32 w fp32 ≈ 17.6 MB.
- **G0c** (seria G, 8.10.2026): Wariant K z weight decay 5·10⁻⁴ (`G0c_variant_K_WD`, 60 epok). Wersja bez weight decay (`G0b`) zapamiętuje zbiór treningowy i jej mAP spada od pierwszej ewaluacji, więc punkt odniesienia klasyfikacyjnego ma dwie wersje.
- Walidacja narzędzia: nasz evaluator daje identyczny wynik co oficjalny `SoccerNet.Evaluation.ReIdentification.evaluate` — sprawdzane testami jednostkowymi na losowych rankingach oraz skryptem `scripts/smoke_eval.py` na cechach `R18-ImageNet` (różnica 0).
- Wartości odniesienia bez treningu (valid): losowy ranking mAP 0.1942, stały wektor 0.2808, `R18-ImageNet` 0.3295.

### Faza 1 — oś C (sampler+miner)
`R18 + TRI + AUG-MIN`, 40 epok, pakiet ∈ {`RAND, PK-BH, PK-SH, PK-BH-XBM, PK-SA-BH, PK-SA-SH, PK-SA-BH-XBM`}. **7 przebiegów.** → wybieramy `S*`.

> **Wybór (7.10.2026): `S*` = PK-SA + BATCH-HARD, bez XBM.** Wariant z XBM dał wynik równorzędny (różnica mAP w granicach niepewności pomiaru na zbiorze walidacyjnym), więc dalej idzie prostszy z dwóch. XBM zostaje wynikiem osi 1 i możliwym dodatkiem do konfiguracji końcowej (oś 5).

### Faza 2 — oś B (strata)
`R18 + S* + AUG-MIN`, 40 epok, strata ∈ {`CE, CONT, TRI, MS, CIRCLE, ARC`}. **6 wpisów** (TRI to zwycięzca osi 1). Uruchomione 7.10.2026 jako **7 przebiegów**: `G2_CONT`, `G2_MS`, `G2_CIRCLE` oraz straty klasyfikacyjne w dwóch wersjach — bez weight decay (`G2_CE`, `G2_ARC`) i z weight decay 5·10⁻⁴ (`G2_CE_WD`, `G2_ARC_WD`). → wybieramy `L*`.

> **Doprecyzowanie samplera/minera**: z pakietu `S*` przenosimy do Fazy 2 tylko **sampler**, **miner dobieramy do straty** zgodnie z literaturą:
> - `CONT` → all-pairs (bez minera),
> - `TRI` → miner zwycięskiego pakietu z osi 1,
> - `MS` → `MultiSimilarityMiner` (część definicji straty),
> - `CIRCLE` → all-pairs (bez minera) — strata sama waży wszystkie pary; w serii F użyto BATCH-HARD (zob. §0),
> - `CE`, `ARC` → **losowy sampler** niezależnie od `S*` (PK-SA daje w batchu klasy tylko z 1 akcji → softmax na dziesiątkach tysięcy klas degeneruje).
>
> XBM: wybrany pakiet `S*` go nie zawiera, więc straty parowe w tej fazie też go nie używają.
>
> **Weight decay dla strat klasyfikacyjnych**: przebieg odniesienia `G0b` (CE bez weight decay) zapamiętał zbiór treningowy — strata treningowa zbliżyła się do minimum, a mAP na zbiorze walidacyjnym spadało od pierwszej ewaluacji. Dlatego `CE` i `ARC` idą w dwóch wersjach (weight decay 0 i 5·10⁻⁴); która z nich jest wierszem głównym tabeli, ustalamy po wynikach. Straty metryczne pozostają bez weight decay.
>
> **Ewaluacja**: w czterech przebiegach klasyfikacyjnych co epokę (`eval.every_n_epochs=1`), bo najlepszy wynik może wypadać przed 5. epoką; checkpoint trafia wtedy do W&B raz, na końcu (`wandb.upload_best=end`).
>
> **Uzupełnienie po przeglądzie (8.10.2026)**: przy wartościach z tabeli §2.B strat `CONT` (margines negatywów 0.5) i `MS` (λ=1) embeddingi zbioru walidacyjnego skupiają się w wąskim stożku (średni kosinus losowych par 0.85 i 0.93, wobec 0.02 dla `TRI`), a gradient `MS` zanika. Dlatego dochodzą dwa przebiegi z wartościami z oficjalnej implementacji / domyślnymi biblioteki: `G2_CONT_M1` (`loss.neg_margin=1.0`) i `G2_MS_B05` (`loss.base=0.5`). Idą równolegle z Fazą 3; jeśli któryś wyprzedzi `TRI`, Faza 3 zostanie powtórzona z nową stratą.
>
> `CIRCLE` jest jedyną stratą metryczną niepoliczoną na wartościach z artykułu dla wariantu parowego (m=0.4, γ=80), więc równolegle z Fazą 4 idzie `G2_CIRCLE_M04` (`loss.m=0.4 loss.gamma=80`).
>
> `ARC` bez weight decay nie zbiega (wszystkie wagi klas ustawiają się równolegle), więc wierszem `ARC` jest wersja z weight decay.
>
> `ARC`: właściwy margines 0.5 rad (28.6°), bez wariantu z innym marginesem; w tabeli jeden wiersz. W serii F margines wynosił przez błąd jednostek 0.5°.
>
> **Doprecyzowanie głowy modelu**: Faza 2 używa domyślnej głowy `projection` (BN→FC→BN→L2-norm) dla strat metric (`CONT`, `TRI`, `MS`, `CIRCLE`) i dla `ARC` (ArcFace ma wewnętrzny scale=30 który neutralizuje saturację logitów). Wyjątek dla `CE`: musi używać głowy **`classifier_cut`** (raw features, bez L2-norm), ponieważ L2-norma na 138k-class CE classifierze powoduje saturację — logity skalują się do ~[-0.06, 0.06], softmax wychodzi praktycznie uniform, gradient zerowy, train loss zatrzymuje się na `ln(138852)≈11.84`. Empirycznie zweryfikowane (F2_CE v1 z projection head: train_loss stale 11.84 przez 40 epok, mAP=0.363). Z `classifier_cut`: normalna konwergencja. Same head jak w Wariancie K (F0b) z §7.3, ale F2_CE używa 40 epok dla spójności tabeli Fazy 2.

### Faza 3 — oś D (augmentacje)
`R18 + S* + L* + {AUG-MIN, AUG-MED, AUG-STRONG, AUG-BOT}`, 40 epok. **4 wpisy** (AUG-MIN to zwycięzca osi 2; 3 nowe przebiegi, nazwy `G3_AUG_*`). Uruchomione 8.10.2026 z `L*` = `TRI` (najlepsza strata po Fazie 2: mAP wyższe od `CIRCLE` o 1,15 pp i jedyna krzywa, która nie spada po 10. epoce); wybór jest warunkowy do czasu zakończenia dwóch uzupełniających przebiegów Fazy 2. → wybieramy `A*`, wykres „augmentacja vs. mAP".

### Faza 4 — oś A (backbone)
`{R18, R34, EB1, EB2, VGG11-BN, VGG16-BN} + S* + L* + A*`, 40 epok. **6 wpisów** (R18 to zwycięzca osi 3; 5 nowych przebiegów, nazwy `G4_*`). → wybieramy `B*`.

> **Uruchomione 8.10.2026** z `L*` = `TRI` i `A*` = `AUG-MED`: `G4_R34`, `G4_EB1`, `G4_EB2`, `G4_VGG11_BN`, `G4_VGG16_BN`; wiersz `R18` to `G3_AUG_MED`. Checkpoint do W&B trafia raz, na końcu (`wandb.upload_best=end`) — checkpoint VGG ma ok. 1,6 GB.
>
> **Uzupełnienie (8.10.2026): VGG bez warstw w pełni połączonych.** Kody `VGG11-BN` / `VGG16-BN` to model timm z fc6 i fc7 (ok. 120 mln parametrów wspólnych dla obu sieci, cecha 4096, mapa cech rozciągana z 8×4 do 8×7), podczas gdy pozostałe backbone'y kończą się uśrednieniem ostatniej mapy cech. Dlatego dochodzą `G4_VGG11_BN_CONV` i `G4_VGG16_BN_CONV` (`backbone=vgg11_bn_conv` / `vgg16_bn_conv`): same warstwy konwolucyjne + global average pooling, cecha 512, 9,5 / 15,0 mln parametrów z głową. Reszta ustawień jak w pozostałych przebiegach Fazy 4.
>
> Wybory po Fazie 3 i uzupełnieniach Fazy 2: `AUG-MED` i `AUG-STRONG` dały wynik równorzędny (różnica mAP w granicach niepewności pomiaru), dalej idzie prostszy zestaw. `MS` z λ=0.5 zrównała się z `TRI`; `TRI` zostaje, bo uczy się szybciej i Faza 3 jest na niej policzona. Kandydaci do Fazy 5 zapisani przy przeglądach: `MS` (λ=0.5) z `AUG-MED`, XBM jako dodatek, `AUG-STRONG` przy 60 epokach.

> W serii F kolejność była odwrotna (backbone'y przy AUG-MIN, potem augmentacje na `B*`). Zmiana: porównanie sieci przy samym odbiciu poziomym sprzyja małym sieciom (§0).

### Faza 5 — interakcje
> **Seria G: skład osi 5 nie jest ustalony.** Zostanie zaprojektowany po zakończeniu i przeglądzie osi 1–4, bo zwycięzcy osi mogą być inni niż w serii F. Poniższy opis to zapis historyczny serii F; jego uzasadnienia odwołują się do wyników ze starym optymalizatorem.

Najciekawsze kombinacje wybrane na podstawie wyników Faz 1-4. Każdy run **60 epok** (zamiast 40 z Faz 1-3) — finalna konfiguracja zasługuje na pełny budżet czasowy, a krzywa AUG-MED w Fazie 4 wciąż rosła w ep 40.

**Wybór 6 runów** podzielony na 4 grupy pytań:

**A. Finalna referencja (Wariant M do §7.3)**
| ID | Config | Pytanie |
|---|---|---|
| `F5_REF` | `EB1 + TRI + PK-SA + AUG-MED, 60ep, AMP=false` | Czy 60ep przebije 0.7417 z 40ep (krzywa rosła)? To jest **Wariant M** używany potem w ablacji §7.3 vs Wariant K (CE) i Wariant H (hybryda) |

**B. Najbliżsi rywale z pełnym stackiem** (czy ranking z Faz 2/3 utrzyma się przy A* i 60ep?)
| ID | Config | Pytanie |
|---|---|---|
| `F5_CIRCLE_PKSA` | `EB1 + CIRCLE + PK-SA + AUG-MED, 60ep, AMP=false` | W Fazie 2 CIRCLE=TRI w noise (Δ<1pp). Z AUG-MED + 60ep może wygrać? Test L*=CIRCLE alternative |
| `F5_R34_AUG` | `R34 + TRI + PK-SA + AUG-MED, 60ep, AMP=false` | Plan literalnie pyta "czy AUG-MED pomaga większym backbone'om?" R34 miał Δ=1.7pp do EB1 z AUG-MIN — czy AUG-MED zamyka lukę? |
| `F5_EB2_AUG` | `EB2 + TRI + PK-SA + AUG-MED, 60ep, AMP=false` | EB2 był 2. w Fazie 3 (Δ 0.5pp do EB1). Z AUG-MED może bije EB1 jako finalne B*? |

**C. Interakcje sampler × loss**
| ID | Config | Pytanie |
|---|---|---|
| `F5_CIRCLE_XBM` | `EB1 + CIRCLE + PK-BH-XBM + AUG-MED, 60ep, AMP=false` | XBM zaszkodził TRI w Fazie 1, ale CIRCLE używa par inaczej. Plan literalnie wymienia "CircleLoss + PK-BH-XBM" jako kandydata. |
| `F5_PKBH_EB1` | `EB1 + TRI + PK-BH + AUG-MED, 60ep, AMP=false` | W Fazie 1 PK-SA pokonał PK-BH cross-action o +8pp (na R18+AUG-MIN). Z lepszym stackiem — czy gap się utrzymuje, czy PK-SA był backbone/aug-dependent? |

**Budżet Fazy 5**: 6 runów × ~6h/run (60ep) = ~36h GPU ≈ 1.5 doby.

**Mapowanie do tabeli E** (§6 raportowanie): tabela porównawcza wszystkich 6 kombinacji + wykres CMC dla top-3 + wybór Wariantu M.

**Seria G — liczba przebiegów przed osią 5**: faza 0: 2, faza 1: 7, faza 2: 5, faza 3: 3, faza 4: 5 — razem **22**. Skład osi 5 do ustalenia.

*Zapis historyczny (plan serii F):* **Łącznie Fazy 0–5**: ~27–32 pełnych przebiegów + sanity checks (6 backbone'ów: R18, R34, EB1, EB2, VGG11-BN, VGG16-BN; 4 zestawy augmentacji).
**Plus ablacje §7** (nie są częścią serii G; decyzja, które wykonać, po osi 5): ~12–15 dodatkowych **treningów** (§7.1: 3 warianty głowy = 3, distance to wybór inferencji bez kosztu; §7.2 wymiar D: 5; §7.3 hybrydowy wariant H: 1 dodatkowy; §7.4 pretraining: 1; §7.5 pooling: 1; §7.6/§7.7 darmowe — post-hoc / z istniejących checkpointów; §7.8 efekt K: 2). Wariant K i Wariant M w §7.3 są już w F0b i Fazie 5 — nie liczymy podwójnie.
**Razem**: **~41–46 przebiegów** (~27-32 z Faz 0-5 + 12-15 z ablacji §7).

**Czas (seria G)**: przebieg R18 na 40 epok trwa ok. 2–2,5 h, EfficientNet ok. 4 h, 60 epok odpowiednio dłużej. Przebiegi jednej osi idą równolegle na wynajętej maszynie, więc czas osi wyznacza jej najdłuższy przebieg. Czasów epok z serii G nie używamy do porównań szybkości (zaburza je równoległość) — szybkość każdej sieci mierzona osobno. *Seria F:* ok. 131 h pracy GPU na laptopie, przebiegi jeden po drugim.

**Uwaga o porównywalności samplerów (Faza 1)**: PK-SA ma efektywny batch 16 vs. 32 dla pozostałych — utrzymujemy **tę samą liczbę iteracji (=update'ów wagowych)** dla wszystkich, akceptując że PK-SA widzi w sumie połowę próbek. Alternatywa „same próbki widziane" wymagałaby 2× więcej iteracji dla PK-SA i mieszałaby budżet z efektem samplera. Decyzja udokumentowana w pracy.

### Konwencja nazewnicza eksperymentów
`<seria i faza>_<to, co w tej fazie się zmienia>` — np. `G1_PK_SA_BH`, `G2_CIRCLE`, `G3_AUG_MED`, `G4_EB1`. Seria F (maj 2026) ma prefiks `F`, seria G — `G`; ta sama nazwa nigdy nie jest używana dwa razy. Każdy przebieg → katalog `outputs/runs/<nazwa>/` z configiem Hydry, logiem, checkpointem best-mAP i embeddingami zbioru walidacyjnego; pełne krzywe w W&B (tag `series-g`).

---

## 4. Pipeline danych

1. **Loader `bbox_info.json`** → DataFrame z kolumnami `path, split, role` (query/gallery dla valid/test, brak dla train), `championship, season, game, action_idx, person_uid, clazz, frame_idx, h, w`.
2. **Sanity check parser nazwy pliku** vs. `bbox_info.json` — nazwa pliku musi być spójna z metadanymi (assert na losowych próbkach).
3. **Filtr klas — tylko w treningu**: decyzja do udokumentowania w pracy — czy w treningu uwzględniamy `Staff`, `Side referee`, `Main referee` (osoby z innym strojem, inna semantyka). Domyślnie: trening tylko na klasach „zawodniczych" (`Player_team_*`, `Goalkeeper_*`), sędziowie i staff odrzuceni. **Ewaluacja NIE filtruje klas** — zawsze pełny zbiór query/gallery z oficjalnego podziału, inaczej wynik byłby nieporównywalny z leaderboardem. **Zweryfikowane na danych**: oficjalne zapytania (query) to wyłącznie zawodnicy i bramkarze — 0 zapytań o sędziów i staff zarówno w valid, jak i w test (decyzja autorów zbioru). Re-identyfikacji tych klas nie mierzy więc ani nasza ewaluacja, ani leaderboard. Sędziowie i staff występują wyłącznie w galerii, jako dystraktory: w valid 3 971 wycinków (11.6% galerii), obecni w galerii 90.9% zapytań; w test 4 391 wycinków (12.5%), obecni w galerii 93.0% zapytań. Konsekwencja: model musi umieszczać w rankingu niżej niż właściwego zawodnika osoby z klas, których nie widział w treningu — to test odporności na dystraktory spoza rozkładu treningowego, a nie test re-identyfikacji tych klas. Filtr treningowy jest spójny z protokołem, bo zapytania dotyczą tylko klas, na których trenujemy. To samo w sobie ciekawa rzecz do dyskusji w pracy. Wariant alternatywny (trening na pełnym zbiorze, z sędziami i staffem) można dodać jako mini-ablację — sprawdziłby, czy widzenie tych klas w treningu pomaga odsuwać je w rankingu.
4. **Singletony — bez explicit'nego filtra na katalogu**. Para `(action, uid)` z 1 próbką nie generuje pozytywnej pary, więc dla strat metric jest „bezużyteczna jako anchor". Ale **PK-style samplery (PK, PK-SA, SEMI, XBM) wybierają tylko klasy z ≥K próbek — singletony są naturalnie pomijane na poziomie batcha** bez ruszania katalogu. Dla strat klasyfikacyjnych (`CE`, `ArcFace`) singletony są w pełni użyteczne (każda osoba dostaje jeden gradient na FC; tak działa rozpoznawanie twarzy na MS-Celeb-1M / ArcFace). Wniosek: trzymamy pełen katalog (po filtrze klas), każdy sampler/strata używa go zgodnie ze swoją naturą. Liczby do raportu: 138 861 par `(action, uid)` po filtrze klas; z tego 76 147 (54.8%) singletonów (=trafia tylko do losowego samplera) i 62 714 par ≥2-próbkowych (=trafia też do PK-samplerów). To dataset-specyficzny rozkład udokumentowany w pracy (kontrast z Market-1501, gdzie ID mają 15–30 zdjęć).
5. **Sampler `PKPerActionBatchSampler`**: w każdym batchu wybiera 1 akcję, z niej P tożsamości × K próbek (próg odcięcia: ID musi mieć ≥K próbek w tej akcji). Wariant `PK` wybiera ID cross-action z tym samym progiem.
6. **Resize do 256×128 bez zachowania proporcji** — zwykłe skalowanie (`v2.Resize((256, 128))`), identyczne w treningu i ewaluacji; standardowa praktyka w ReID (BoT-ReID, torchreid, fast-reid). Zniekształcenie proporcji jest niepomijalne: mediana 25%, ok. 21% wycinków > 50%, ok. 6% > 100% (rozkład praktycznie identyczny w train i valid). Wariant z paddingiem zachowującym proporcje (letterbox) **nie jest zaimplementowany** — kandydat na mini-ablację. Kompromis: padding usuwa zniekształcenie, ale część i tak małej rozdzielczości (mediana wycinka 123×59 px) zajmują puste pasy.
7. **Augmentacje** — moduł z 4 presetami przełączanymi z configu (torchvision v2, `src/soccernet_reid/transforms.py`).

---

## 5. Protokół treningowy (spójny dla wszystkich przebiegów)

- **Wejście**: 256 × 128, normalizacja ImageNet.
- **Głowa (`projection head`)**: `GAP → BN → FC(D) → BN → L2-norm`. Wymienialna przez config (parametr `head: {projection, bnneck, plain, classifier_cut}`):
  - `projection` — domyślna jak wyżej, dla strat metric,
  - `bnneck` — klasyczna wersja Luo et al. (BoT-ReID): triplet na cechach **przed** BN, klasyfikator na cechach **po** BN+FC; używana dla wariantu hybrydowego §7.3,
  - `plain` — bez końcowej L2, opcjonalnie bez końcowego BN (do ablacji §7.1),
  - `classifier_cut` — głowa klasyfikacyjna na czas treningu, odcinana w inferencji (Wariant K §7.3, F0b).
- **Optymalizator**: Adam(lr=3.5e-4), cosine schedule z warmup 5 epok. Weight decay: 0 w serii G, 5e-4 w serii F (zob. §0).
- **Definicja epoki**: przy samplerach PK-style jeden batch nie odpowiada „przeglądowi datasetu". Przyjmujemy **epoka = 5000 iteracji** (≈ jeden przegląd 225 k próbek dla batcha 32; PK-SA z batch 16 widzi w sumie połowę próbek na epokę — patrz uwaga w §3 o porównywalności samplerów).
- **Epoki**: 40 w Fazach 1–4, 60 w Fazie 0 i w Fazie 5.
- **Batch**: domyślnie **P=16/K=2 = 32** (samplery cross-action: PK, RAND, SEMI, XBM); **P=8/K=2 = 16** dla PK-SA (constraint datasetu: tylko 5% akcji ma 16 ID z ≥2 próbkami; 39% akcji ma 8 ID z ≥2 próbkami). Przebieg R18 lub EfficientNet zajmuje do ok. 2 GB pamięci karty (zmierzone); dla VGG nie mierzono.
- **Precyzja obliczeń**: pełna (FP32) we wszystkich przebiegach; mixed precision (AMP) wyłączone (§0).
- **Ziarna**: jedno ziarno (0) dla każdej konfiguracji. Różnice poniżej ok. 1 pp mAP nie są więc rozstrzygające — ograniczenie do zapisania w pracy.
- **Ewaluacja w trakcie treningu**: na zbiorze walidacyjnym co 5 epok; zapisywany jest checkpoint o najwyższym mAP.
- **Logowanie** (W&B + katalog przebiegu): strata i lr w każdym kroku; co 250 kroków stan sieci (odsetek niezerowych wag, rozrzut embeddingów w batchu, udział trudnych trójek, norma gradientu straty); mAP / Rank-1/5/10 co 5 epok; checkpoint best-mAP; embeddingi zbioru walidacyjnego dla najlepszego checkpointu; pełna konfiguracja Hydry i commit kodu.
- **Stack**: PyTorch + `pytorch-metric-learning` (gotowe MS/Triplet/Circle/ArcFace + miners + XBM) + `timm` (backbone'y) + Hydra/OmegaConf.

---

## 6. Protokół ewaluacji

1. Wyciągnij cechy dla wszystkich obrazów w `valid/query` i `valid/gallery` (i analogicznie dla `test/`).
2. Dla każdego query:
   - zawęź gallery do tej samej akcji (`action_idx`),
   - policz cosine similarity (lub euclidean — patrz ablacja §7.1),
   - wyznacz AP i pozycję pierwszego trafienia.
3. Uśrednij mAP, R-1, R-5, R-10 po wszystkich query.
4. **Walidacja narzędzia**: nasz evaluator daje identyczny wynik co oficjalny `SoccerNet.Evaluation.ReIdentification.evaluate` (testy jednostkowe + `scripts/smoke_eval.py`, Faza 0).
5. **`test/`** — używamy raz, na końcu serii G, dla konfiguracji końcowych. Nie używamy testu do żadnego wyboru.
6. **`challenge/`** — nie używamy. Zbiór nie ma etykiet, a serwer konkursowy (EvalAI) został zamknięty; organizatorzy potwierdzili, że do prac naukowych służy zbiór `test/`. Porównanie z leaderboardem 2023 jest więc orientacyjne (inny zbiór).
7. **Re-ranking** — na końcu serii G, dla najlepszych konfiguracji: k-reciprocal, normalizacja po zapytaniach akcji oraz ich połączenie (`scripts/eval_rerank.py`). Parametry dobierane wyłącznie na `valid/`, potem zamrożone i zastosowane raz na `test/`.
8. **Finalne liczby** liczymy na jednej maszynie: skalowanie obrazu daje na różnych procesorach wynik różniący się o jeden poziom jasności w części pikseli, co zmienia mAP o ok. 0.0003.

---

## 7. Ablacje uzupełniające (do dyskusji w pracy)

> **Stan względem serii G.** Na część pytań odpowiedzą dane zbierane w serii G; reszta wymaga osobnych treningów, o których zdecydujemy po osi 5.
>
> | # | Ablacja | Pokrycie przez serię G | Co wymagałoby osobnych przebiegów |
> |---|---|---|---|
> | 1 | Normalizacja L2 | brak | wszystko; punkt wymaga przeprojektowania (biblioteka normalizuje embeddingi wewnątrz strat, a dla znormalizowanych wektorów ranking cosinusowy i euklidesowy są identyczne) |
> | 2 | Wymiar embeddingu | częściowe: z zapisanych embeddingów — efektywna liczba używanych wymiarów i mAP po obcięciu (PCA) do mniejszego wymiaru; to analiza przybliżona | trening z innym `D` |
> | 3 | Klasyfikacja vs. metryka | częściowe: Wariant K vs. uczenie metryki na R18 (faza 0: `G0a` / `G0b`; faza 2: CE i ArcFace obok strat metrycznych) | Wariant K na konfiguracji końcowej; Wariant H (hybryda) — także zmiana kodu |
> | 4 | Pretraining | brak (jest tylko punkt odniesienia `R18-ImageNet` bez treningu) | przebieg od zera |
> | 5 | Pooling GAP vs. GeM | brak | zmiana kodu i przebieg |
> | 6 | Re-ranking | pełne — wykonywany na końcu serii (§6.7) | — |
> | 7 | Krzywa zbieżności | pełne — mAP co 5 epok w każdym przebiegu; `G0a` (60 epok) i `G1_PK_BH` (40 epok) to ta sama konfiguracja | — |
> | 8 | K w samplerze | brak | 2 przebiegi |
>
> Poza listą: wpływ weight decay na stan sieci zbadano parą przebiegów `DIAG_BOT_WD5E4` / `DIAG_BOT_WD0` (§0) — to notatka robocza, w pracy nieopisywana.

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
3. **Tabela C** — wpływ augmentacji (Faza 3) + krzywa uczenia.
4. **Tabela D** — wpływ backbone'u (Faza 4), z liczbą parametrów i czasem treningu/inferencji (mierzonym osobno, §3).
5. **Tabela E** — interakcje (Faza 5).
6. **Krzywe CMC** dla top-3 konfiguracji.
7. **Wizualizacje**: t-SNE / UMAP embeddingów dla 1 akcji; wyniki wyszukiwania dla najlepszych konfiguracji — zapytanie i najbliższe wycinki z galerii z oznaczeniem trafień, błędów i odległości (`scripts/visualize_retrieval.py`; failure analysis: ten sam strój, podobna sylwetka, occlusion, zawodnik częściowo poza kadrem).
8. **Ablacje** z §7 w jednej sekcji.
9. **Tabela porównawcza z leaderboardem 2023** — uczciwe pozycjonowanie pracy względem SOTA, z zastrzeżeniem, że leaderboard liczono na zbiorze `challenge/`, a nasze wyniki na `valid/` i `test/`.
10. **Re-ranking** dla najlepszych konfiguracji na `valid/` i `test/` (§6.7).

Do pracy nie wchodzi nic z serii F ani diagnoza jej błędów (§0).

---

## 9. Ryzyka i mitigacje

| Ryzyko | Mitigacja |
|--------|-----------|
| Zbyt duża macierz przebiegów na 1 GPU | Etapowa redukcja (§3); krótsze przebiegi na osi A/B (40 epok), pełne 60 tylko najlepsze |
| Mylenie globalnego `person_uid` z `(action,uid)` | Etykieta treningowa = `(action,uid)`, sampler `PK-SA` to wymusza, evaluator zawęża do akcji |
| Klasy „dziwne" (`Staff`, sędziowie) zaszumiają trening | Filtr klas **tylko w treningu** (§4.3); ewaluacja zawsze na pełnym zbiorze dla zgodności z leaderboardem |
| Niezgodność filtra treningowego z pełną ewaluacją | Niezgodność dotyczy tylko galerii: zapytania to wyłącznie zawodnicy i bramkarze (zweryfikowane, §4.3), a sędziowie i staff występują jedynie jako dystraktory w galerii (klasy niewidziane w treningu) — udokumentowane, ewentualna mini-ablacja z pełnym treningiem |
| Niewłaściwa konfiguracja P×K dla tego datasetu | Liczby zweryfikowane na realnych danych: P=16/K=2 dla cross-action, P=8/K=2 dla PK-SA. K=4 wycina 90% datasetu — NIE używać. |
| Mylenie „filtra singletonów" z naturalnym pomijaniem ich przez PK sampler | NIE filtrujemy katalogu. PK samplery same omijają singletony przez wymóg ≥K próbek per ID. Klasyfikatory (CE/ArcFace) używają singletonów produktywnie. Zgodne z literaturą ReID i face recognition. |
| Niereprodukowalność | Ziarno, konfiguracja Hydry i commit kodu zapisane dla każdego przebiegu. cuDNN działa w trybie niedeterministycznym, więc powtórzenie przebiegu nie jest identyczne co do bitu |
| Ciche „umieranie" sieci (wagi ściągane do zera, zerowe embeddingi) | Weight decay 0 (§0); odsetek niezerowych wag i rozrzut embeddingów logowane w każdym przebiegu i sprawdzane przy przeglądzie każdej osi |
| Wnioski z jednego ziarna | Różnic poniżej ok. 1 pp nie interpretujemy jako przewagi |
| Niezgodność z oficjalnym evalem | Smoke test (§6.4) przed Fazą 1 |
| Nadmierne dopasowanie do test-setu | `test/` używamy tylko raz, na końcu serii G; parametry re-rankingu dobierane wyłącznie na `valid/` |
| Konstruowanie własnego query/gallery | NIE — używamy oficjalnego podziału z `valid/{query,gallery}` i `test/{query,gallery}` |

---

## 10. Następne kroki

Implementacja (loader, evaluator zgodny z oficjalnym, samplery, augmentacje, pętla treningowa, re-ranking, wizualizacja wyszukiwania) jest gotowa. Pozostało, w tej kolejności:

1. **Seria G, fazy 0–4** (§3). Po każdej fazie przegląd wyników i decyzja o konfiguracji następnej.
2. **Faza 5** — zaprojektowanie składu po fazach 1–4, potem przebiegi (60 epok).
3. **Ocena konfiguracji końcowych na `test/`** — raz.
4. **Re-ranking** dla najlepszych konfiguracji na `valid/` i `test/`.
5. **Materiały do pracy**: tabele A–E, krzywe uczenia i CMC, wizualizacje wyszukiwania, pomiar szybkości sieci.
6. **Ablacje §7** — decyzja, które wykonać.
