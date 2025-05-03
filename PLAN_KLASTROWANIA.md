# Plan dla AI: Wdrożenie Trwałego Przechowywania Wyników Klastrowania w Metadanych Qdrant (Opcja B)

**Cel:** Zmodyfikować system tak, aby po przeprowadzeniu analizy klastrowania, identyfikator klastra (`cluster_id`), do którego został przypisany każdy punkt (chunk), był zapisywany bezpośrednio w metadanych (payload) tego punktu w bazie wektorowej Qdrant. Umożliwi to trwałe przechowywanie przypisań i przyszłe filtrowanie wyszukiwań RAG po ID klastra.

**Wymagania Wstępne:** Działający proces klastrowania zwracający mapowanie `point_id -> cluster_id`. Zainicjalizowany klient Qdrant w `QdrantConnector`.

**Kroki Implementacyjne:**

**Krok 1: Przygotowanie Środowiska i Kodu**

1.  **(AI)** Utwórz kopię zapasową plików, które będą modyfikowane: `clustering_module.py` (jeśli tam będzie logika aktualizacji) oraz `ui_streamlit.py` (logika wywołująca).
2.  **(AI)** Upewnij się, że w pliku `requirements.txt` jest zainstalowana odpowiednia wersja biblioteki `qdrant-client`, która wspiera operacje na payloadach, których będziemy używać (najnowsza stabilna jest zazwyczaj najlepsza).

**Krok 2: Modyfikacja Logiki Klastrowania / Zapisu Wyników**

*   **Cel:** Po wykonaniu klastrowania (np. przez `perform_clustering`), zamiast tylko zapisywać wyniki do `st.session_state`, uruchomić proces aktualizacji payloadów w Qdrant.
*   **Miejsce Modyfikacji:** Logika ta najprawdopodobniej znajduje się obecnie w pliku `ui_streamlit.py`, wewnątrz bloku `if st.button("🚀 Analizuj / Odśwież Klastry", ...):`. Trzeba będzie wywołać nową funkcję aktualizującą Qdrant po udanym `perform_clustering`.
*   **Działania:**
    1.  **(AI)** **Dodaj Nową Metodę w `QdrantConnector`:** W pliku `database/qdrant_connector.py`, do klasy `QdrantConnector`, dodaj nową metodę, np. `update_payload_with_cluster_ids`. (Kod metody został dostarczony w poprzedniej wiadomości).
    2.  **(AI)** **Zmodyfikuj Logikę Przycisku w `ui_streamlit.py`:** W bloku `if st.button("🚀 Analizuj / Odśwież Klastry", ...):`, po linii, gdzie `cluster_assignments` jest pomyślnie obliczane i zapisywane do `st.session_state`, **dodaj wywołanie nowej metody** `update_payload_with_cluster_ids`. (Fragment kodu został dostarczony w poprzedniej wiadomości).

**Krok 3: Modyfikacja Pobierania Danych i Wykorzystania w UI**

*   **Cel:** Dostosowanie istniejącego kodu, aby ładował i używał `cluster_id` z Qdrant zamiast (lub obok) danych z `st.session_state`.
*   **Plik(i):** `database/qdrant_connector.py`, `ui_streamlit.py`.
*   **Działania:**
    1.  **(AI)** **Modyfikacja `get_all_data_for_clustering`:** Upewnij się, że ta metoda **zawsze pobiera payload** (`with_payload=True`), ponieważ teraz to w payloadzie będzie zapisane `cluster_id`.
    2.  **(AI)** **Modyfikacja Ładowania na Starcie Aplikacji (Jeśli potrzebne):** Jeśli chcesz, aby wyniki klastrowania były dostępne od razu po starcie (bez klikania "Analizuj"), musisz na starcie aplikacji (`ui_streamlit.py`, w bloku inicjalizacji stanu) wywołać `get_all_data_for_clustering`, a następnie zbudować słownik `cluster_assignments` **na podstawie pola `cluster_id` z pobranego payloadu każdego punktu**. (Fragment kodu został dostarczony w poprzedniej wiadomości).
    3.  **(AI)** **Wykorzystanie `cluster_id` z Payloadu w UI:** W miejscach, gdzie odwołujesz się do `st.session_state.cluster_assignments`, upewnij się, że logika nadal działa (powinna, bo słownik ma ten sam format). Jeśli implementowałeś ładowanie na starcie (pkt 3.2), to UI od razu będzie miało dostęp do wyników.

**Krok 4: Implementacja Filtrowania RAG (Krok 6a z poprzedniego planu)**

*   **Cel:** Umożliwienie funkcji "Pytaj w Kontekście Klastra".
*   **Plik(i):** `ui_streamlit.py`, `app.py` (metoda `query`), `database/qdrant_connector.py` (metoda `search`).
*   **Działania:**
    1.  **(AI)** **UI:** Dodaj interfejs do zadawania pytania dla konkretnego klastra i przekazania `filter_cluster_id` do `app.py`.
    2.  **(AI)** **App:** Metoda `query` musi akceptować `filter_cluster_id`.
    3.  **(AI)** **DB:** Metoda `similarity_search` musi akceptować `filter_cluster_id` (jako `Optional[int]`) i budować filtr Qdrant. (Fragment kodu został dostarczony w poprzedniej wiadomości).

**Krok 5: Testowanie**

*   **(Człowiek)** Po każdej modyfikacji dokładnie testuj:
    *   Czy analiza klastrowania działa i poprawnie zapisuje `cluster_id` w Qdrant? (Sprawdź payloady w Qdrant, np. przez UI Qdranta lub pobierając kilka punktów z `with_payload=True`).
    *   Czy ładowanie wyników na starcie działa?
    *   Czy wyświetlanie informacji o klastrach w UI jest poprawne?
    *   Czy RAG działa normalnie (bez filtra)?
    *   Czy funkcja "Pytaj w Kontekście Klastra" poprawnie filtruje wyniki?
    *   Czy wydajność (zwłaszcza zapisu payloadów) jest akceptowalna?

Ten plan zapewnia przejście na bardziej trwały i elastyczny sposób zarządzania wynikami klastrowania, wykorzystując bezpośrednio możliwości bazy Qdrant i otwierając drogę do zaawansowanego filtrowania RAG.