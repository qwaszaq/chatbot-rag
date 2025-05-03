# Lista Zadań Rozwoju Chatbota

**GŁÓWNY PRIORYTET:** **Stabilizacja i Użyteczność Rdzenia Funkcjonalności**

## Ukończone Kroki (z Nowego Planu)

*   [x] **Krok 1: Weryfikacja i Poprawa Istniejącego Kodu (Priorytet NAJWYŻSZY)**
    *   [x] **1.1 Weryfikacja `QdrantConnector`:** Potwierdzono, że `get_all_data_for_clustering` pobiera potrzebne dane.
    *   [x] **1.2 Weryfikacja i Modyfikacja `clustering_module.py`:**
        *   [x] Potwierdzono implementację `perform_clustering` i `generate_cluster_labels_llm`.
        *   [x] Wdrożono metodę reprezentantów w `generate_cluster_summary`.
    *   [x] **1.3 Finalizacja Logiki Trybu Grafowego:** Zaimplementowano warunkową ekstrakcję grafu w `app.py` i `ui_streamlit.py`.

---

# Implementacja Analizy Klastrowania (Status Szczegółowy)

*   **Cel:** Dodanie funkcjonalności automatycznego grupowania dokumentów (chunków).
*   **Status Kroków z Pierwotnego Planu (zaktualizowany):**
    *   [x] Krok 1: Pobieranie Danych
    *   [x] Krok 2: Algorytm Klastrowania
    *   [ ] **Krok 3: Przypisanie i Przechowywanie:** Przypisanie działa, ale trwałe przechowywanie **jest w trakcie implementacji (Opcja B - Qdrant)**.
    *   [x] Krok 4: Generowanie Etykiet Klastrów
    *   [x] Krok 5: Integracja z UI (Podstawowa + Podsumowania/Encje)
    *   [x] Krok 6: Wykorzystanie w Logice RAG (Rozszerzanie kontekstu)

## In Progress Tasks (Wg Nowego Planu)

*   [ ] **Krok 2: Implementacja Trwałego Przechowywania Wyników Klastrowania (Opcja B - Qdrant)** (Priorytet WYSOKI)
    *   [x] Krok 2.1: Dodano metodę `update_payload_with_cluster_ids` do `QdrantConnector`.
    *   [x] Krok 2.2: Dodano wywołanie aktualizacji Qdrant w `ui_streamlit.py`.
    *   [x] Krok 3.1: Zweryfikowano `get_all_data_for_clustering`.
    *   [x] Krok 3.2: Zaimplementowano ładowanie `cluster_assignments` z Qdrant na starcie w `ui_streamlit.py`.
    *   [x] Krok 3.3: Potwierdzono, że UI wykorzystuje `cluster_assignments` ze stanu sesji.

## Future Tasks (Wg Nowego Planu)

*   [x] **Krok 4: Implementacja Filtrowania RAG ("Pytaj w Kontekście Klastra")** (Priorytet Średni)
    *   [x] UI: Dodanie interfejsu.
    *   [x] App: Modyfikacja `query`.
    *   [x] DB: Modyfikacja `similarity_search`.
*   [ ] **Krok 3 (kontynuacja): Ulepszenie UI Eksploracji Klastrów** (Priorytet Średni) - Wyświetlanie Top N bytów, podsumowań, ulepszona tabela chunków (po zakończeniu Kroku 2).
*   [ ] **Krok 4 (kontynuacja): Dalsze Ulepszanie Jakości Grafów** (Priorytet Średni/Niski) - Dostrajanie promptu, kolory, konfiguracja `agraph`.
*   [ ] **Krok 5: Eksperymenty z Algorytmem Klastrowania** (Priorytet Niski) - BERTopic/HDBSCAN.
*   [ ] Testowanie i Ewaluacja całości.
*   [ ] **Implementacja Eksportu Grafu do Excel (.xlsx)** (Priorytet Niski/Średni) - Dodanie przycisku w UI (`ui_streamlit.py`) umożliwiającego pobranie pliku Excel z dwoma arkuszami: "Nodes" (lista węzłów z atrybutami) i "Edges" (lista krawędzi z etykietami). Wymaga dodania `openpyxl` do `requirements.txt` i użycia `pandas` do generowania pliku.

## Implementation Plan
Aktualny plan rozwoju jest opisany w `PLAN_KLASTROWANIA.md`. Zakończono Krok 1 (Weryfikacja i Poprawa Kodu). Jesteśmy w trakcie implementacji Kroku 2 (Trwałe Przechowywanie w Qdrant).

### Relevant Files
*   `../chatbot010525/PLAN_KLASTROWANIA.md` (szczegółowy plan)
*   `../chatbot010525/clustering_module.py`
*   `../chatbot010525/database/qdrant_connector.py`
*   `../chatbot010525/app.py`
*   `../chatbot010525/ui_streamlit.py`
*   `../chatbot010525/requirements.txt`