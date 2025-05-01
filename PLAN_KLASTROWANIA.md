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