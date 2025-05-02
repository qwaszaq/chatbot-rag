# Implementacja Analizy Klastrowania

Dodanie funkcjonalności automatycznego grupowania dokumentów (chunków) na podstawie podobieństwa ich embeddings w celu odkrywania struktur tematycznych i usprawnienia analizy w chatbocie RAG.

## Completed Tasks

- [x] **Krok 1: Pobieranie Danych:** Zaimplementowano funkcję pobierania embeddings i metadanych z Qdrant (`QdrantConnector.get_all_data_for_clustering`).
- [x] **Krok 2: Algorytm Klastrowania:** Zaimplementowano algorytmy (K-Means, DBSCAN) w `clustering_module.py`.
- [x] **Krok 3: Przypisanie Klastrów:** Wyniki klastrowania są łączone z metadanymi i przechowywane w stanie sesji Streamlit.
- [x] **Krok 4: Generowanie Etykiet Klastrów:** Zaimplementowano generowanie etykiet za pomocą LLM w `clustering_module.py`.
- [x] **Integracja Podstawowa z UI (Krok 5a):** Dodano przyciski do uruchamiania analizy, wyświetlania listy klastrów i podstawowego przeglądania chunków w `ui_streamlit.py`. Poprawiono błędy związane z kompatybilnością wsteczną historii czatu.

## In Progress Tasks

- [ ] **Ulepszenie Kroku 5 (Integracja z UI):**
    - [ ] **5b: Podsumowanie Klastra:** Dodanie funkcji generowania i wyświetlania podsumowania LLM dla wybranego klastra.
    - [ ] **5c: Kluczowe Byty:** Dodanie funkcji ekstrakcji (NER) i wyświetlania najczęstszych bytów dla wybranego klastra.

## Future Tasks

- [ ] **Krok 6: Wykorzystanie w Logice RAG:** Modyfikacja `chatbot.py` do zadawania pytań w kontekście klastra lub rozszerzania kontekstu.
- [ ] Testowanie i Ewaluacja: Weryfikacja spójności klastrów, wydajności i użyteczności funkcji.

## Implementation Plan
Plan opiera się na krokach zdefiniowanych w `PLAN_KLASTROWANIA.md`. Kroki 1-4 oraz podstawowa integracja z UI (5a) zostały zaimplementowane. Obecnie skupiamy się na ulepszeniu interfejsu użytkownika (Krok 5b i 5c) poprzez dodanie funkcji podsumowania klastra za pomocą LLM oraz ekstrakcji i wyświetlania kluczowych bytów (NER). Następnie rozważymy wykorzystanie klastrów w logice RAG (Krok 6) oraz przeprowadzimy testy. Szczegółowy opis znajduje się w `PLAN_KLASTROWANIA.md`.

### Relevant Files
*   `../chatbot010525/PLAN_KLASTROWANIA.md` (szczegółowy plan)
*   `../chatbot010525/clustering_module.py` (główny moduł implementacji - **do rozszerzenia**)
*   `../chatbot010525/database/qdrant_connector.py` (potencjalne rozszerzenie)
*   `../chatbot010525/app.py` / `../chatbot010525/chatbot.py` (potencjalne zmiany w logice)
*   `../chatbot010525/ui_streamlit.py` (zmiany w interfejsie - **do rozszerzenia**)
*   `../chatbot010525/requirements.txt` (zależności)