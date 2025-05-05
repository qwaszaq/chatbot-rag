# Plan Implementacji: Trwałe Przechowywanie Metadanych Klastrów w Qdrant

## Opis Problemu

Obecnie metadane klastrów (etykiety, podsumowania, kluczowe byty) są zapisywane w lokalnym pliku `cluster_metadata.json`, który jest usuwany przy tworzeniu nowych czatów lub resetowaniu istniejących. Prowadzi to do utraty tych metadanych i konieczności ich ponownego generowania.

## Cel Rozwiązania

Przenieść przechowywanie metadanych klastrów do bazy danych wektorowej Qdrant, aby zapewnić ich trwałość i dostępność niezależnie od sesji czatu.

## Plan Iteracji

### Iteracja 1: Przygotowanie QdrantConnector i Modyfikacja Zapisywania Metadanych

**Cel:** Dodać do `QdrantConnector` możliwość zapisywania metadanych klastrów w Qdrant i zmodyfikować `clustering_module.py` tak, aby korzystał z tej funkcji zamiast zapisywać do pliku.

**Kroki:**

1.  **Modyfikacja `database/qdrant_connector.py`:**
    *   Dodanie metody `save_cluster_metadata_to_qdrant(self, cluster_metadata: Dict[int, Dict[str, Any]])`. Ta metoda będzie odpowiedzialna za połączenie z Qdrant i zapisanie metadanych klastrów.
    *   Rozważenie struktury przechowywania metadanych: Najprostszym podejściem jest stworzenie nowej kolekcji w Qdrant (np. `cluster_metadata_collection`) gdzie każdy punkt będzie reprezentował klaster, a jego ID będzie ID klastra (int, skonwertowane na string), a payload będzie zawierał etykietę, podsumowanie i listę encji dla tego klastra.
    *   Zaimplementowanie logiki tworzenia nowej kolekcji `cluster_metadata_collection` w metodzie `initialize`, jeśli jeszcze nie istnieje.
    *   W metodzie `save_cluster_metadata_to_qdrant`, przygotowanie danych w formacie `PointStruct` dla nowej kolekcji i użycie `client.upsert` do zapisania/aktualizacji punktów (klastrów).

2.  **Modyfikacja `clustering_module.py`:**
    *   Importowanie `QdrantConnector`.
    *   W funkcji, która generuje metadane klastrów (prawdopodobnie pod koniec logiki klastrowania w `ui_streamlit.py`, ale `clustering_module.py` może wymagać dostępu do obiektu `QdrantConnector`), wywołanie nowej metody `save_cluster_metadata_to_qdrant` z wygenerowanymi metadanymi. **Uwaga:** Obiekt `QdrantConnector` jest dostępny w `st.session_state.chatbot.qdrant_connector` w `ui_streamlit.py`. Modyfikacja będzie głównie w `ui_streamlit.py`.
    *   Usunięcie wywołania `save_cluster_metadata` (zapis do pliku JSON) w `ui_streamlit.py`.

**Kryteria Akceptacji Iteracji 1:**

*   Metoda `save_cluster_metadata_to_qdrant` istnieje w `QdrantConnector` i jest w stanie zapisać metadane klastrów (etykieta, podsumowanie, encje) do nowej kolekcji w Qdrant.
*   Kod w `ui_streamlit.py` wywołuje nową metodę zapisu do Qdrant po wygenerowaniu metadanych klastrów.
*   Lokalny plik `cluster_metadata.json` nie jest już tworzony ani aktualizowany po klastrowaniu.

### Iteracja 2: Ładowanie Metadanych Klastrów z Qdrant

**Cel:** Dodać do `QdrantConnector` możliwość ładowania metadanych klastrów z Qdrant i zmodyfikować `ui_streamlit.py` tak, aby ładował je przy starcie aplikacji.

**Kroki:**

1.  **Modyfikacja `database/qdrant_connector.py`:**
    *   Dodanie metody `load_cluster_metadata_from_qdrant(self) -> Optional[Dict[int, Dict[str, Any]]]`. Ta metoda będzie odpowiedzialna za pobranie wszystkich punktów z kolekcji `cluster_metadata_collection`.
    *   Zaimplementowanie logiki pobierania punktów z `cluster_metadata_collection` przy użyciu metody `client.scroll` lub `client.retrieve`, konwertując ID punktu z powrotem na int i strukturując wynik w słownik mapujący ID klastra na jego metadane.

2.  **Modyfikacja `ui_streamlit.py`:**
    *   Usunięcie logiki ładowania metadanych klastrów z pliku `cluster_metadata.json` przy starcie aplikacji (`load_cluster_metadata`).
    *   Po załadowaniu przypisań klastrów z Qdrant przy starcie aplikacji, wywołanie nowej metody `load_cluster_metadata_from_qdrant` z obiektu `st.session_state.chatbot.qdrant_connector`.
    *   Zapisanie załadowanych metadanych do `st.session_state.cluster_labels`, `st.session_state.cluster_summaries`, `st.session_state.cluster_entities`.
    *   Dostosowanie logiki sprawdzania spójności metadanych/przypisań, aby porównywała liczbę przypisań z Qdrant z liczbą klastrów, dla których załadowano metadane z Qdrant.

**Kryteria Akceptacji Iteracji 2:**

*   Metoda `load_cluster_metadata_from_qdrant` istnieje w `QdrantConnector` i poprawnie pobiera metadane klastrów z Qdrant.
*   Kod w `ui_streamlit.py` ładuje metadane klastrów z Qdrant przy starcie aplikacji.
*   Interfejs użytkownika poprawnie wyświetla załadowane z Qdrant etykiety, podsumowania i encje klastrów.
*   Usunięta została wszelka logika związana z plikiem `cluster_metadata.json` w `ui_streamlit.py`.

### Iteracja 3: Czyszczenie i Testowanie Końcowe

**Cel:** Zaimplementować czyszczenie metadanych klastrów w Qdrant przy czyszczeniu głównej kolekcji i przeprowadzić testy końcowe.

**Kroki:**

1.  **Modyfikacja `database/qdrant_connector.py`:**
    *   Modyfikacja metody `clear_collection`, aby oprócz usuwania głównej kolekcji danych, usuwała również kolekcję `cluster_metadata_collection`, jeśli istnieje.

2.  **Modyfikacja `ui_streamlit.py`:**
    *   Upewnienie się, że resetowanie stanu sesji w UI po czyszczeniu bazy danych lub resecie czatu poprawnie odzwierciedla fakt, że metadane klastrów również zostały usunięte z Qdrant (np. ustawiając stany metadanych na puste słowniki/None).

3.  **Testowanie Końcowe:**
    *   Przetestowanie pełnego workflow: dodanie dokumentów -> klastrowanie (sprawdzenie zapisu do Qdrant) -> restart aplikacji -> sprawdzenie ładowania metadanych klastrów -> stworzenie nowego czatu (sprawdzenie czy metadane pozostają) -> wyczyszczenie bazy (sprawdzenie czy metadane znikają z Qdrant i UI).

**Kryteria Akceptacji Iteracji 3:**

*   Czyszczenie kolekcji w Qdrant usuwa również metadane klastrów.
*   Aplikacja poprawnie obsługuje brak metadanych klastrów po czyszczeniu bazy.
*   Cały proces (zapis, odczyt, czyszczenie metadanych klastrów w Qdrant) działa poprawnie.
*   Lokalny plik `cluster_metadata.json` nie jest już używany w projekcie.

## Status

Plan stworzony. Oczekuje na zgodę na rozpoczęcie Iteracji 1.