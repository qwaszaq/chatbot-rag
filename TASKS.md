# Implementacja Analizy Klastrowania

Dodanie funkcjonalności automatycznego grupowania dokumentów (chunków) na podstawie podobieństwa ich embeddings w celu odkrywania struktur tematycznych i usprawnienia analizy w chatbocie RAG.

## Completed Tasks

(brak ukończonych zadań dla tej funkcji)

## In Progress Tasks

- [ ] **Krok 1: Pobieranie Danych:** Zaimplementować funkcję pobierania embeddings i metadanych z Qdrant (np. w `clustering_module.py` lub rozszerzając `QdrantConnector`).
- [ ] **Krok 2: Algorytm Klastrowania:** Wybrać i zaimplementować algorytm (np. K-Means, DBSCAN) w `clustering_module.py`.
- [ ] **Krok 3: Przypisanie Klastrów:** Połączyć wyniki klastrowania z metadanymi i zdefiniować sposób przechowywania (pamięć, plik, Qdrant).

## Future Tasks

- [ ] **Krok 4: Generowanie Etykiet Klastrów:** Implementacja metody nadawania opisowych nazw klastrom (TF-IDF, LLM).
- [ ] **Krok 5: Integracja z UI:** Dodanie elementów w `ui_streamlit.py` do uruchamiania analizy, przeglądania klastrów i wyświetlania informacji o klastrze przy źródłach.
- [ ] **Krok 6: Wykorzystanie w Logice RAG:** Modyfikacja `chatbot.py` do zadawania pytań w kontekście klastra lub rozszerzania kontekstu.
- [ ] Testowanie i Ewaluacja: Weryfikacja spójności klastrów, wydajności i użyteczności funkcji.

## Implementation Plan
Plan opiera się na krokach zdefiniowanych w `PLAN_KLASTROWANIA.md`. Rozpoczynamy od przygotowania danych (Krok 1), następnie implementujemy algorytm (Krok 2) i zarządzanie wynikami (Krok 3). Kolejne kroki obejmują generowanie etykiet, integrację z interfejsem użytkownika oraz potencjalne wykorzystanie w logice RAG. Szczegółowy opis znajduje się w `PLAN_KLASTROWANIA.md`.

### Relevant Files
*   `../chatbot010525/PLAN_KLASTROWANIA.md` (szczegółowy plan)
*   `../chatbot010525/clustering_module.py` (główny moduł implementacji)
*   `../chatbot010525/database/qdrant_connector.py` (potencjalne rozszerzenie)
*   `../chatbot010525/app.py` / `../chatbot010525/chatbot.py` (potencjalne zmiany w logice)
*   `../chatbot010525/ui_streamlit.py` (zmiany w interfejsie)
*   `../chatbot010525/requirements.txt` (zależności)