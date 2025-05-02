# Implementacja Analizy Klastrowania

Dodanie funkcjonalności automatycznego grupowania dokumentów (chunków) na podstawie podobieństwa ich embeddings w celu odkrywania struktur tematycznych i usprawnienia analizy w chatbocie RAG.

## Completed Tasks

- [x] **Krok 1: Pobieranie Danych:** Zaimplementowano funkcję pobierania embeddings i metadanych z Qdrant (`QdrantConnector.get_all_data_for_clustering`).
- [x] **Krok 2: Algorytm Klastrowania:** Zaimplementowano algorytmy (K-Means, DBSCAN) w `clustering_module.py`.
- [x] **Krok 3: Przypisanie Klastrów:** Wyniki klastrowania są łączone z metadanymi i przechowywane w stanie sesji Streamlit.
- [x] **Krok 4: Generowanie Etykiet Klastrów:** Zaimplementowano generowanie etykiet za pomocą LLM w `clustering_module.py`.
- [x] **Integracja Podstawowa z UI (Krok 5a):** Dodano przyciski do uruchamiania analizy, wyświetlania listy klastrów i podstawowego przeglądania chunków w `ui_streamlit.py`. Poprawiono błędy związane z kompatybilnością wsteczną historii czatu.
- [x] **Ulepszenie Kroku 5 (Integracja z UI):**
    - [x] **5b: Podsumowanie Klastra:** Dodano funkcję generowania i wyświetlania podsumowania LLM dla wybranego klastra.
    - [x] **5c: Kluczowe Byty:** Dodano funkcję ekstrakcji (NER) i wyświetlania najczęstszych bytów dla wybranego klastra.
- [x] **Krok 6: Wykorzystanie w Logice RAG (Rozszerzanie Kontekstu):** Zmodyfikowano `app.py` i `ui_streamlit.py`, aby opcjonalnie rozszerzać kontekst odpowiedzi o dodatkowe dokumenty z dominujących klastrów.

## In Progress Tasks

(brak zadań w toku dla tej funkcji)

## Future Tasks

- [ ] **Krok 7: Wizualizacja Grafu dla Klastra:** Implementacja funkcji ekstrakcji i wizualizacji grafu wiedzy ograniczonego do dokumentów/chunków w wybranym klastrze.
- [ ] Testowanie i Ewaluacja: Weryfikacja spójności klastrów, wydajności i użyteczności funkcji.

## Implementation Plan
Plan opiera się na krokach zdefiniowanych w `PLAN_KLASTROWANIA.md`. Kroki 1-5 (wraz z ulepszeniami 5b i 5c) oraz Krok 6 (rozszerzanie kontekstu na podstawie klastrów) zostały zaimplementowane. Następne kroki obejmują dodanie wizualizacji grafu specyficznego dla klastra (Krok 7) oraz przeprowadzenie testów. Szczegółowy opis znajduje się w `PLAN_KLASTROWANIA.md`.

### Relevant Files
*   `../chatbot010525/PLAN_KLASTROWANIA.md` (szczegółowy plan)
*   `../chatbot010525/clustering_module.py` (główny moduł implementacji)
*   `../chatbot010525/database/qdrant_connector.py` (potencjalne rozszerzenie)
*   `../chatbot010525/app.py` / `../chatbot010525/chatbot.py` (zmiany w logice RAG)
*   `../chatbot010525/ui_streamlit.py` (zmiany w interfejsie)
*   `../chatbot010525/requirements.txt` (zależności)