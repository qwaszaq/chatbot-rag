# Plan dla AI: Ulepszenie Generowania Grafów Wiedzy (PRIORYTET)

**Cel:** Zmodyfikować proces ekstrakcji grafu w `app.py`, aby generować bardziej połączone, informacyjne i strukturalne grafy wiedzy na podstawie odpowiedzi tekstowej LLM. Głównym narzędziem będzie ulepszenie promptu instruktażowego dla LLM oraz dostosowanie kodu parsującego odpowiedź.

**Kroki Implementacyjne:**

**Krok 3.1: Lokalizacja i Kopia Zapasowa**

*   Zlokalizuj metodę `_extract_graph_from_response` w pliku `app.py`.
*   **WAŻNE:** Przed wprowadzeniem zmian, utwórz kopię zapasową obecnej wersji pliku `app.py`.

**Krok 3.2: Zastąpienie Promptu Ekstrakcji**

*   W metodzie `_extract_graph_from_response` znajdź zmienną przechowującą obecny prompt ekstrakcji (prawdopodobnie nazywa się `extraction_prompt`).
*   **Zastąp całą zawartość** tej zmiennej nowym, ulepszonym promptem (zgodnie z treścią podaną w Twoim przykładzie, zaczynającym się od `ROLE: Jesteś zaawansowanym analitykiem...`).
    *   **Kluczowe elementy nowego promptu:** Zdefiniowane typy bytów (Person, Organization, Location, Method, Project, FinancialAsset, Document, Concept), zdefiniowane typy znaczących relacji (ZARZĄDZA, KONTROLUJE, FINANSUJE,TRANSFERUJE_ŚRODKI_DO, KOMUNIKUJE_Z itp.), instrukcja łączenia kontekstowego (tworzenie łańcuchów relacji dla procesów), wymagany format wyjściowy JSON z listami `"nodes"` i `"edges"` zawierającymi określone atrybuty (`id`, `label`, `type` dla węzłów; `source`, `target`, `label` dla krawędzi).

**Krok 3.3: Modyfikacja Kodu Parsowania Odpowiedzi JSON**

*   Zlokalizuj fragment kodu *po* otrzymaniu odpowiedzi od LLM (`extraction_response`) i oczyszczeniu jej (`response_text_cleaned`), a *przed* zwróceniem grafu (`return G` lub `return None`). Ten fragment jest odpowiedzialny za parsowanie JSON-a i budowanie obiektu grafu NetworkX.
*   **Zastąp obecną logikę parsowania** (prawdopodobnie iterującą po `extracted_data["triples"]`) **nową logiką obsługującą format `{ "nodes": [...], "edges": [...] }`**. Użyj kodu z sekcji `--- Fragment do modyfikacji w PARSOWANIU JSON ---` podanego w Twoim przykładzie.
    *   **Kluczowe elementy nowej logiki:**
        *   Sprawdzenie, czy JSON zawiera klucze `"nodes"` i `"edges"` będące listami.
        *   Iteracja po liście `"nodes"` z JSON-a:
            *   Dodawanie węzłów do grafu `G` (`G.add_node(node_id, ...)`).
            *   Wykonywanie NER na atrybucie `label` węzła.
            *   Zapisywanie `id`, `label`, typu zasugerowanego przez LLM (`llm_type`) oraz typu z NER (`ner_type`) jako **atrybutów węzła NetworkX**.
            *   Użycie zbioru `nodes_added_ner_checked` do unikania wielokrotnego NER.
        *   Iteracja po liście `"edges"` z JSON-a:
            *   Sprawdzenie, czy węzły `source` i `target` istnieją w grafie `G`.
            *   Dodawanie krawędzi do grafu `G` (`G.add_edge(source_id, target_id, ...)`).
            *   Zapisywanie etykiety relacji (`label`) jako **atrybutu krawędzi NetworkX**.
        *   Obsługa błędów i przypadków, gdy LLM zwraca puste listy.

**Krok 3.4: Weryfikacja Wywołania LLM i Parsowania**

*   **Wywołanie:** Sprawdź, czy parametry wywołania LLM w `_extract_graph_from_response` są nadal odpowiednie (zwłaszcza `max_tokens=10000` i niska `temperature`, np. 0.2).
*   **Parsowanie:** Upewnij się, że Twój kod, który później pobiera dane grafu z `_extract_graph_from_response` i konwertuje je do formatu `node-link` dla AGraph (`nx.node_link_data(graph)`), nadal działa poprawnie po zmianach w strukturze obiektu `G` (który teraz ma więcej atrybutów węzłów).

**Krok 3.5: Testowanie Funkcjonalne**

*   Uruchom aplikację Streamlit.
*   Wybierz tryb "Analityk Grafów (RAG)".
*   Zadaj **złożone pytanie testowe** (np. to dotyczące powiązań metod Agnieszek ze strategią i finansami).
*   **Obserwuj:**
    *   **Logi:** Czy ekstrakcja grafu przebiega bez błędów? Czy logowane są nowe typy węzłów i krawędzi? Ile krawędzi/węzłów zostało dodanych?
    *   **Graf w UI:**
        *   Czy graf jest teraz **bardziej połączony**? Czy widać łańcuchy relacji lub bardziej złożone podsieci?
        *   Czy pojawiają się **nowe etykiety krawędzi** (np. `KONTROLUJE`, `TRANSFERUJE_ŚRODKI_DO`)?
        *   Czy kolorystyka węzłów (oparta na `ner_type`) jest spójna? Czy pojawiają się węzły z typem `Concept`, `Method` itp. (domyślnie będą szare, jeśli nie dodasz dla nich kolorów do `NER_COLORS`)?
*   **Iteruj:** Jeśli grafy nadal są niezadowalające, wróć do **Kroku 3.2** i **dostrajaj ulepszony prompt ekstrakcji**, precyzując instrukcje, dodając przykłady, modyfikując typy relacji. Testuj ponownie.

**Opcjonalnie (Później):**

*   **Kolorowanie Nowych Typów:** Rozważ dodanie kolorów do słownika `NER_COLORS` w `ui_streamlit.py` dla nowych typów bytów zasugerowanych przez LLM (`llm_type`, np. `Method`, `Project`), aby wizualnie je wyróżnić.
*   **Wykorzystanie `llm_type`:** Zastanów się, jak wykorzystać `llm_type` w analizie lub wizualizacji obok `ner_type` ze spaCy.

Powyższy plan dostarcza precyzyjnych kroków do implementacji ulepszeń w generowaniu grafów.

---
## KOD do wykorzystania

Oto propozycja nowego, bardziej szczegółowego promptu, który powinieneś wstawić do metody `_extract_graph_from_response` w `app.py`, zastępując obecny `extraction_prompt`:

```python
# Fragment pliku app.py, wewnątrz metody _extract_graph_from_response

        # NOWY, ULEPSZONY PROMPT EKSTRAKCJI:
        extraction_prompt = f"""
ROLE:
Jesteś zaawansowanym analitykiem specjalizującym się w budowaniu Grafów Wiedzy (Knowledge Graphs) na podstawie analizy tekstu śledczego. Twoim zadaniem jest zidentyfikowanie kluczowych bytów (Encji) oraz **znaczących, wieloetapowych relacji** między nimi w poniższym tekście dotyczącym sprawy Zorganizowanej Grupy Przestępczej "Agnieszka". Celem jest stworzenie struktury grafu, która odzwierciedla **procesy, hierarchie, przepływy (finansowe, informacyjne) i kluczowe powiązania** opisane w tekście.

ZADANIA:
1.  **Zidentyfikuj Kluczowe Byty (Węzły):** Rozpoznaj następujące typy encji i użyj ich jako węzłów grafu:
    *   **Person:** Osoby (np. "Agnieszka Malinowska", "Janusz Wąs", "dr Klaus Steiner"). Staraj się używać pełnych nazwisk, rozpoznawaj i grupuj aliasy/pseudonimy (np. "Czas", "Kosmiczna Rachunkowość" -> "Agnieszka Malinowska").
    *   **Organization:** Organizacje, firmy (realne i fikcyjne), instytucje, grupy (np. "ZGP 'Agnieszka'", "BeigeLife Innovations", "NCBR", "CBŚP", "NeutralTech Solutions Ltd", "Fundacja Niebieskie Perspektywy").
    *   **Location:** Miejsca geograficzne (kraje, miasta, regiony), konkretne adresy (np. "Cypr", "Szwajcaria", "Pcim Dolny", "ul. Cienista 10").
    *   **Method:** Specyficzne metody działania grupy (np. "kluczodiagnostyka", "prognozy chmur", "beżowa dieta", "wyścigi ślimaków", "analiza cienia").
    *   **Project:** Wewnętrzne nazwy projektów lub operacji grupy (np. "'Projekt Beż'", "'Projekt Echo'", "'Strategia Okienna...'").
    *   **FinancialAsset:** Aktywa finansowe, waluty (np. "środki 4.8M PLN", "kryptowaluta XMR", "sztabki platyny", "dotacja", "łapówka").
    *   **Document:** Dokumenty, raporty, pliki, certyfikaty (np. "fałszywy certyfikat", "operat szacunkowy", "Raport PIA", "plik .xlsx", "notatka służbowa").
    *   **Concept:** Inne istotne pojęcia lub role (np. "centralny hub finansowy", "mechanizm legalizacji", "kamuflaż", "strategia grupy").
2.  **Zidentyfikuj Znaczące Relacje (Krawędzie):** Połącz zidentyfikowane byty za pomocą **konkretnych, opisowych relacji**. Unikaj ogólnych relacji jak "jest", "ma". Skup się na:
    *   **Struktura/Kontrola:** `ZARZĄDZA`, `KONTROLUJE`, `NALEŻY_DO`, `JEST_CZĘŚCIĄ`, `MA_SIEDZIBĘ_W`.
    *   **Działania/Procesy:** `WYKONUJE` (metodę, zadanie), `STOSUJE_METODĘ`, `PROWADZI` (projekt, księgowość), `GENERUJE` (dokument, zyski), `SPRZEDAJE`, `KUPUJE`, `FAŁSZUJE` (dokument).
    *   **Przepływy:** `OTRZYMUJE_ŚRODKI_Z`, `TRANSFERUJE_ŚRODKI_DO`, `LEGALIZUJE_PRZEZ`, `PŁACI_ZA`, `DOSTARCZA_INFORMACJE_DO`.
    *   **Komunikacja/Powiązania:** `KOMUNIKUJE_Z`, `WSPÓŁPRACUJE_Z`, `POWIĄZANY_Z`, `WSKAZUJE_NA`, `ZAWIERA_INFORMACJE_O`, `MENTIONED_IN`.
    *   **Cel/Motywacja:** `W_CELU` (np. `[Oszustwo] W_CELU [Korzyść Majątkowa]`).
3.  **Łącz Kontekstowo:** **NAJWAŻNIEJSZE:** Jeśli tekst opisuje **proces lub sekwencję zdarzeń** angażującą wiele bytów (np. przepływ pieniędzy: NCBR -> BeigeLife -> NeutralTech -> Malinowska -> Raj Podatkowy), **stwórz łańcuch połączonych krawędzi**, aby odzwierciedlić ten przepływ (np. `(NCBR, FINANSUJE, BeigeLife)`, `(BeigeLife, TRANSFERUJE_ŚRODKI_DO, NeutralTech)`, `(NeutralTech, JEST_KANAŁEM_DLA?, Malinowska?)`, `(Malinowska, TRANSFERUJE_ŚRODKI_DO, 'Raj podatkowy Seszele')`). Nie twórz tylko izolowanych par. Połącz byty wspomniane w tym samym akapicie lub zdaniu, jeśli opisują wspólną akcję lub powiązanie.
4.  **Format Wyjściowy:** Zwróć wynik **WYŁĄCZNIE** jako pojedynczy obiekt JSON, zawierający dwa klucze: "nodes" i "edges".
    *   `"nodes"`: Lista obiektów, gdzie każdy obiekt reprezentuje unikalny węzeł i ma klucze:
        *   `"id"`: (String) Unikalny identyfikator węzła (użyj nazwy bytu, np. "Agnieszka Malinowska"). Staraj się ujednolicać identyfikatory dla tej samej encji.
        *   `"label"`: (String) Pełna etykieta do wyświetlenia (może zawierać alias, np. "Agnieszka Malinowska ('Czas')").
        *   `"type"`: (String) Typ encji z listy powyżej (np. "Person", "Organization", "Method"). Jeśli typ jest niejasny, użyj "Concept" lub pomiń.
    *   `"edges"`: Lista obiektów, gdzie każdy obiekt reprezentuje relację (krawędź) i ma klucze:
        *   `"source"`: (String) ID węzła źródłowego (musi pasować do ID w liście "nodes").
        *   `"target"`: (String) ID węzła docelowego (musi pasować do ID w liście "nodes").
        *   `"label"`: (String) Etykieta relacji z listy powyżej (np. "KONTROLUJE", "TRANSFERUJE_ŚRODKI_DO").

PRZYKŁAD FORMATU WYJŚCIOWEGO:
```json
{{
  "nodes": [
    {{"id": "Agnieszka Malinowska", "label": "Agnieszka Malinowska ('Czas')", "type": "Person"}},
    {{"id": "Centralny Hub Finansowy", "label": "Centralny Hub Finansowy", "type": "Concept"}},
    {{"id": "Projekt Echo", "label": "Projekt Echo - sample audio", "type": "Project"}},
    {{"id": "ZGP 'Agnieszka'", "label": "ZGP 'Agnieszka'", "type": "Organization"}}
  ],
  "edges": [
    {{"source": "Agnieszka Malinowska", "target": "Centralny Hub Finansowy", "label": "KONTROLUJE"}},
    {{"source": "Agnieszka Malinowska", "target": "Projekt Echo", "label": "KSIĘGUJE_DLA"}},
    {{"source": "Projekt Echo", "target": "ZGP 'Agnieszka'", "label": "JEST_CZĘŚCIĄ"}}
  ]
}}
```

Tekst do analizy:
{text}
JSON z wynikiem:
"""
# Koniec NOWEGO PROMPTU

# --- Fragment do modyfikacji w PARSOWANIU JSON ---
try:
        # ... (reszta kodu metody _extract_graph_from_response bez zmian,
        #      zakładając, że parsuje teraz ten nowy format JSON z kluczami "nodes" i "edges") ...

        extracted_data = json.loads(response_text_cleaned)

        if "nodes" in extracted_data and "edges" in extracted_data and \
           isinstance(extracted_data["nodes"], list) and isinstance(extracted_data["edges"], list):

            nodes_from_llm = extracted_data["nodes"]
            edges_from_llm = extracted_data["edges"]

            if not nodes_from_llm and not edges_from_llm:
                logger.info("   LLM nie znalazł żadnych węzłów ani relacji.")
                return None

            G = nx.DiGraph()
            nodes_added_ner_checked = set()
            edges_added_count = 0

            # 1. Dodaj węzły i wykonaj NER
            for node_data in nodes_from_llm:
                node_id = node_data.get("id", "").strip()
                node_label = node_data.get("label", node_id).strip()
                node_type_llm = node_data.get("type", "Unknown").strip() # Typ zasugerowany przez LLM

                if not node_id:
                    logger.warning(f"   Pominięto węzeł bez ID: {node_data}")
                    continue

                if node_id not in nodes_added_ner_checked:
                    ner_type = None # Domyślnie brak typu spaCy
                    if self.nlp:
                        try:
                            doc_node = self.nlp(node_label) # Użyj label do NER
                            if doc_node.ents:
                                ner_type = doc_node.ents[0].label_
                                # logger.info(f"   NER dla węzła '{node_label}' (ID: {node_id}): Rozpoznano typ '{ner_type}'") # Zmniejszono gadatliwość logów
                        except Exception as ner_exc:
                            logger.error(f"   Błąd NER dla '{node_label}': {ner_exc}")
                            ner_type = "ERROR"

                    # Dodaj węzeł do grafu NetworkX z atrybutami
                    G.add_node(node_id, label=node_label, llm_type=node_type_llm, ner_type=ner_type)
                    nodes_added_ner_checked.add(node_id)

            # 2. Dodaj krawędzie
            for edge_data in edges_from_llm:
                source_id = edge_data.get("source", "").strip()
                target_id = edge_data.get("target", "").strip()
                relation_label = edge_data.get("label", "").strip()

                # Sprawdź czy węzły istnieją i relacja nie jest pusta
                if source_id in G and target_id in G and relation_label:
                    G.add_edge(source_id, target_id, label=relation_label)
                    edges_added_count += 1
                else:
                     logger.warning(f"   Pominięto krawędź z brakującymi elementami lub węzłami: {edge_data}")

            if G.number_of_nodes() > 0 or G.number_of_edges() > 0:
                logger.info(f"   Dodano {G.number_of_nodes()} węzłów i {G.number_of_edges()} krawędzi do grafu.")
                return G
            else:
                logger.info("   Nie dodano żadnych węzłów ani krawędzi do grafu.")
                return None
        # --- Koniec fragmentu do modyfikacji w PARSOWANIU JSON ---

        else:
            logger.error(f"❌ Odpowiedź ekstrakcji LLM (po czyszczeniu) nie zawiera kluczy 'nodes'/'edges' lub wartości nie są listami: {response_text_cleaned}")
            return None
except Exception as e: # Zmieniono obsługę błędu JSON na bardziej ogólną
    logger.error(f"❌ Błąd podczas parsowania JSON lub budowania grafu z odpowiedzi LLM: {e}")
    logger.debug(f"   Odpowiedź (po czyszczeniu) powodująca błąd: {response_text_cleaned}")
    return None

# Koniec fragmentu app.py
```

**Kluczowe Zmiany i Ulepszenia w Tym Nowym Prompcie:**

1.  **Szczegółowa Rola i Cel:** Wyraźnie określa zadanie budowy *połączonego* grafu wiedzy, koncentrującego się na *procesach* i *strukturach*.
2.  **Zdefiniowane Typy Bytów:** Wprowadza bardziej szczegółowe typy encji (`Method`, `Project`, `FinancialAsset`, `Document`, `Concept`) istotne dla sprawy, oprócz standardowych (Person, Org, Loc). To pomoże w kategoryzacji węzłów.
3.  **Zdefiniowane Typy Relacji:** Proponuje **konkretny zestaw etykiet krawędzi**, które opisują działania, przepływy i hierarchie. To *najważniejsza zmiana*, która powinna "zmusić" LLM do identyfikowania bardziej znaczących połączeń niż tylko ogólne "jest powiązany z".
4.  **Instrukcja Łączenia Kontekstowego:** Explicite prosi o tworzenie **łańcuchów relacji** dla procesów sekwencyjnych (jak przepływy finansowe).
5.  **Nowy Format Wyjściowy (JSON z `nodes` i `edges`):** Proponuje bardziej ustrukturyzowany format JSON, który:
    *   Ułatwia parsowanie w Pythonie.
    *   Naturalnie obsługuje atrybuty węzłów (id, label, type).
    *   Lepiej reprezentuje strukturę grafu niż prosta lista trójek.
6.  **Obsługa Aliasów:** Wskazuje na konieczność rozpoznawania pseudonimów.

**Kroki Implementacji:**

1.  **Zamień Prompt:** Wklej nowy `extraction_prompt` do metody `_extract_graph_from_response` w `app.py`.
2.  **Zmodyfikuj Kod Parsowania JSON:** Zaktualizuj fragment kodu w `_extract_graph_from_response`, który parsuje odpowiedź LLM, aby obsługiwał nowy format JSON z kluczami `nodes` i `edges`. Powyższy przykład zawiera już sugerowaną modyfikację tej części. Kluczowe jest teraz iterowanie po liście `nodes` i `edges` z JSON-a i budowanie grafu NetworkX na tej podstawie. Zwróć uwagę na dodanie atrybutów `label`, `llm_type` i `ner_type` do węzłów NetworkX.
3.  **Dostosuj Logikę NER (Opcjonalne):** W zmodyfikowanym kodzie parsowania, NER (spaCy) jest nadal używany do przypisania standardowego typu (`ner_type`). Możesz teraz wykorzystać oba typy w wizualizacji (`llm_type` zasugerowany przez LLM i `ner_type` ze spaCy).
4.  **Przetestuj:** Uruchom zapytania (szczególnie to złożone, które testowaliśmy) i zobacz, czy generowane grafy są teraz bardziej połączone, strukturalne i czy używają nowych, bardziej opisowych etykiet relacji.

To duża zmiana w logice ekstrakcji, która ma potencjał znacząco poprawić jakość wizualizacji grafów.
# Plan Wdrożenia Analizy Klastrowania w Chatbocie RAG (Sprawa ZGP "Agnieszka")

## 1. Cel

Celem jest dodanie do chatbota funkcjonalności automatycznego grupowania (klastrowania) dokumentów źródłowych (lub ich fragmentów/chunków) na podstawie podobieństwa ich treści (reprezentowanej przez wektory embeddings). Pozwoli to na:

*   Odkrywanie ukrytych struktur tematycznych w korpusie dokumentów.
*   Identyfikację grup dokumentów dotyczących podobnych osób, metod, wydarzeń lub wątków śledztwa.
*   Wykrywanie nietypowych lub odosobnionych dokumentów (outlierów).
*   Usprawnienie nawigacji i analizy dużego zbioru danych tekstowych.
*   Stworzenie nowych funkcji interaktywnych dla użytkownika (np. przeglądanie klastrów, zadawanie pytań w kontekście klastra).

## 2. Wymagania Wstępne

*   Istniejąca baza wektorowa (Qdrant) zawierająca wektory embeddings dla dokumentów źródłowych lub ich fragmentów (chunków).
*   Dostęp do tych wektorów embeddings z poziomu aplikacji chatbota (prawdopodobnie przez rozszerzenie funkcjonalności `QdrantConnector`).
*   Zainstalowane niezbędne biblioteki Python: `scikit-learn`, `pandas`, `numpy` (oprócz już istniejących).

## 3. Proponowane Etapy Wdrożenia

### Krok 1: Pobieranie Danych do Klastrowania

*   **Zadanie:** Zaimplementować w `app.py` lub nowym module (np. `clustering_module.py`) funkcję, która pobierze z bazy Qdrant:
    *   Wszystkie (lub reprezentatywną próbkę) wektory embeddings dla dokumentów/chunków.
    *   Odpowiadające im identyfikatory (np. ID punktu w Qdrant) i metadane (np. nazwa pliku źródłowego, numer chunka, potencjalnie sam tekst chunka).
*   **Implementacja:**
    *   Rozszerzenie klasy `QdrantConnector` (lub stworzenie nowej funkcji w `app.py`) o metodę `get_all_embeddings_with_metadata(limit=None, sample_size=None)`.
    *   Metoda ta powinna iterować przez punkty w Qdrant (z potencjalnym limitem lub próbkowaniem dla wydajności) i zwracać listę słowników lub obiekt DataFrame zawierający `id`, `embedding`, `metadata`.
*   **Uwagi:** Pobranie *wszystkich* embeddings dla bardzo dużej bazy może być czasochłonne i pamięciochłonne. Rozważ strategie:
    *   Pobieranie tylko reprezentatywnej próbki (np. 10 000 losowych punktów).
    *   Przeprowadzanie klastrowania w trybie offline (jako osobny skrypt), zapisywanie wyników i tylko wczytywanie ich w aplikacji.
    *   Użycie technik redukcji wymiarowości (np. PCA, UMAP) przed klastrowaniem, jeśli wymiar embeddings jest bardzo duży.

### Krok 2: Wybór i Implementacja Algorytmu Klastrowania

*   **Zadanie:** Wybrać i zaimplementować algorytm klastrowania na pobranych wektorach embeddings.
*   **Wybór Algorytmu:**
    *   **K-Means (`sklearn.cluster.KMeans`):** Prosty i szybki. Wymaga ustalenia liczby klastrów (`k`). Można użyć "metody łokcia" (elbow method) lub analizy współczynnika sylwetki (silhouette score) do oszacowania optymalnego `k`. Dobry na początek.
    *   **DBSCAN (`sklearn.cluster.DBSCAN`):** Nie wymaga podania liczby klastrów. Potrafi znajdować klastry o nieregularnych kształtach i identyfikować punkty odstające (outliers/szum). Wymaga strojenia parametrów `eps` (maksymalny dystans między próbkami) i `min_samples` (liczba próbek w sąsiedztwie). Dobry do identyfikacji anomalii.
    *   **(Opcjonalnie) Hierarchical Clustering (`sklearn.cluster.AgglomerativeClustering`):** Buduje hierarchię klastrów (dendrogram), nie wymaga `k`, ale może być wolniejszy dla dużych zbiorów.
*   **Implementacja:**
    *   Stworzenie funkcji `perform_clustering(embeddings, algorithm='kmeans', **kwargs)` w `clustering_module.py`.
    *   Funkcja przyjmuje macierz embeddings i parametry algorytmu.
    *   Wykorzystuje wybraną implementację z `scikit-learn` do przypisania etykiety klastra do każdego wektora embedding.
    *   Zwraca listę lub tablicę etykiet klastrów odpowiadającą wejściowym embeddings. Algorytmy takie jak DBSCAN zwrócą etykietę `-1` dla punktów odstających.

### Krok 3: Przypisanie Klastrów do Dokumentów i Przechowywanie Wyników

*   **Zadanie:** Połączyć wyniki klastrowania (etykiety) z powrotem z metadanymi dokumentów/chunków i zapisać te informacje.
*   **Implementacja:**
    *   Po wykonaniu `perform_clustering`, należy zmapować zwrócone etykiety klastrów do odpowiednich `id` i `metadata` pobranych w Kroku 1.
    *   **Opcje przechowywania wyników:**
        *   **W pamięci (`st.session_state` lub atrybut klasy `RAGChatbot`):** Najprostsze na start, ale dane znikną po restarcie aplikacji. Dobre do testowania i klastrowania na żądanie dla mniejszych zbiorów. Przykład: słownik `st.session_state.cluster_assignments = {chunk_id: cluster_label, ...}`.
        *   **W pliku (np. JSON, CSV, Pickle):** Pozwala na utrwalenie wyników między sesjami. Klastrowanie może być wykonywane rzadziej (np. raz po załadowaniu nowych dokumentów) lub offline, a aplikacja tylko wczytuje gotowe przypisania.
        *   **(Zaawansowane) Aktualizacja metadanych w Qdrant:** Można dodać pole `cluster_id` do payloadu każdego punktu w Qdrant. Wymaga to modyfikacji procesu indeksowania i aktualizacji punktów. Daje to możliwość filtrowania wyników wyszukiwania w Qdrant po ID klastra.
*   **Rekomendacja:** Zacznij od przechowywania w pamięci/stanie sesji dla prototypu, a następnie przejdź do zapisu do pliku dla utrwalenia i potencjalnie lepszej wydajności.

### Krok 4: (Opcjonalnie) Generowanie Etykiet dla Klastrów

*   **Zadanie:** Nadać klastrom bardziej opisowe nazwy niż tylko numery.
*   **Implementacja:**
    *   **Metoda 1: Top N słów kluczowych (TF-IDF):** Dla każdego klastra, zbierz teksty chunków do niego należących. Oblicz TF-IDF dla tych tekstów i wybierz np. 3-5 słów o najwyższym wyniku jako reprezentację klastra.
    *   **Metoda 2: Podsumowanie przez LLM:** Dla każdego klastra, wyślij reprezentatywny zestaw tekstów (lub próbkę) do LLM z promptem "Podsumuj w kilku słowach kluczowych główny temat tej grupy dokumentów: [teksty]".
    *   Funkcja `generate_cluster_labels(cluster_data)` w `clustering_module.py`.
    *   Zapisz etykiety np. w słowniku: `st.session_state.cluster_labels = {cluster_id: "Etykieta Klastra", ...}`.

### Krok 5: Integracja z Interfejsem Użytkownika (`ui_streamlit.py`)

*   **Zadanie:** Dodać nowe elementy UI do interakcji z klastrami.
*   **Implementacja:**
    *   **Przycisk/Opcja Uruchomienia Klastrowania:** Jeśli klastrowanie ma być na żądanie, dodaj przycisk w sidebarze "Analizuj / Odśwież Klastry Dokumentów", który wywoła funkcje z `clustering_module.py` i zaktualizuje stan sesji (`st.session_state.cluster_assignments`, `st.session_state.cluster_labels`). Pokaż `st.spinner` podczas obliczeń.
    *   **Przeglądarka Klastrów:**
        *   Dodaj nową sekcję w sidebarze lub w głównej części aplikacji (np. pod `st.expander`).
        *   Wyświetl listę wykrytych klastrów (z ich etykietami, jeśli wygenerowane).
        *   Pokaż liczbę dokumentów/chunków w każdym klastrze.
        *   (Opcjonalnie) Pokaż top N słów kluczowych dla klastra.
        *   Umożliw kliknięcie na klaster, aby wyświetlić listę dokumentów/chunków do niego należących (np. w nowym oknie modalnym lub dedykowanej sekcji).
    *   **Informacja o Klastrze w Kontekście Odpowiedzi:** W pętli wyświetlającej historię czatu, gdy pokazujesz źródła (`sources`) odpowiedzi asystenta, dodaj informację, do którego klastra należy dany chunk źródłowy (jeśli dane klastrowania są dostępne). Przykład: `- `dokument_xyz.pdf` (Chunk 5 / **Klaster 3: Korupcja Przetargowa**)`.
    *   **(Zaawansowane) Pasek Filtrowania Klastrów:** Pozwól użytkownikowi wybrać jeden lub więcej klastrów, aby filtrować wyświetlaną historię czatu lub wyniki wyszukiwania (wymagałoby większych zmian w logice RAG).

### Krok 6: (Zaawansowane) Wykorzystanie Klastrów w Logice RAG

*   **Zadanie:** Opcjonalnie zmodyfikować sposób zadawania pytań lub przetwarzania kontekstu, aby uwzględniał klastry.
*   **Implementacja:**
    *   **Pytanie w kontekście klastra:** Dodaj możliwość zadania pytania w stylu "Co wiadomo o Agnieszce Malinowskiej **w kontekście Klastra 5 (Finanse Kryptowalutowe)?**". Twoja funkcja `chatbot.query()` musiałaby najpierw pobrać chunki z Klastra 5, a następnie użyć ich jako dodatkowego (lub jedynego) kontekstu dla LLM, modyfikując standardowy przepływ RAG.
    *   **Rozszerzanie Kontekstu:** Po znalezieniu najbardziej relewantnych chunków przez standardowy RAG, chatbot mógłby automatycznie dodać do kontekstu kilka innych chunków z *tych samych klastrów*, zakładając, że mogą one zawierać powiązane, wartościowe informacje.

## 4. Testowanie i Ewaluacja

*   Przetestuj działanie klastrowania na danych sprawy ZGP "Agnieszka".
*   Sprawdź, czy tworzone klastry są spójne tematycznie i sensowne z perspektywy analityka.
*   Oceń wydajność klastrowania (czas wykonania).
*   Zbierz feedback od potencjalnych użytkowników na temat przydatności nowych funkcji.
*   Dostosuj parametry algorytmów (np. `k` dla K-Means, `eps` dla DBSCAN) dla uzyskania optymalnych wyników.

## 5. Potencjalne Wyzwania

*   **Wydajność:** Klastrowanie dużych zbiorów embeddings może być zasobożerne.
*   **Dobór Parametrów:** Algorytmy jak K-Means czy DBSCAN wymagają dobrania parametrów, co może być niełatwe.
*   **Interpretowalność Klastrów:** Wygenerowane automatycznie etykiety (np. TF-IDF) mogą nie zawsze być intuicyjne.
*   **Jakość Embeddings:** Wyniki klastrowania zależą od jakości użytych wektorów embeddings.
*   **Integracja z UI:** Zaprojektowanie przejrzystego i użytecznego interfejsu do przeglądania i interakcji z klastrami.

Ten plan stanowi ramy wdrożenia. Konkretne szczegóły implementacyjne będą zależeć od wybranego algorytmu, struktury Twojego kodu i preferencji dotyczących interfejsu użytkownika.
## DODATKOWE

*   Context caching: https://www.youtube.com/watch?v=hhMXE9-JUAc