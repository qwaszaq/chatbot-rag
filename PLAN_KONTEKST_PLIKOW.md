# Plan Implementacji: Dodawanie Plików jako Kontekst dla LLM (bez RAG)

## Cel

Dodanie możliwości dołączania plików TXT i PDF do czatu, tak aby ich treść była dostępna dla LLM jako dodatkowy kontekst, ale bez indeksowania w systemie RAG.

## Kroki Implementacji

1.  **Modyfikacja Interfejsu Użytkownika (`ui_streamlit.py`)**
    *   Dodanie nowego widżetu `st.file_uploader` w panelu bocznym, dedykowanego do przesyłania plików wyłącznie jako kontekst (bez indeksowania w RAG). Etykieta np. "Dodaj pliki jako kontekst dla LLM (TXT, PDF)".
    *   Implementacja logiki odczytu treści tych plików (TXT i PDF - można wykorzystać części `DocumentLoader` do samego odczytu) i przechowywania ich w `st.session_state` (np. w słowniku `st.session_state.context_files = {nazwa_pliku: tresc}`).
    *   Aktualizacja logiki wysyłania zapytania do LLM (w sekcji obsługi `st.chat_input`), aby pobierała zawartość `st.session_state.context_files` i przekazywała ją jako nowy argument do odpowiedniej metody w klasie `RAGChatbot`.

2.  **Modyfikacja Logiki Backendu (`app.py` - klasa `RAGChatbot`)**
    *   Dodanie nowego parametru, np. `context_files_content: Optional[Dict[str, str]] = None`, do metody `query` (lub stworzenie dedykowanej metody np. `query_with_additional_context`).
    *   Modyfikacja metody `_generate_answer` (dla lokalnego LLM):
        *   Dodanie parametru `context_files_content`.
        *   Wewnątrz, przed skonstruowaniem finalnego promptu (`prompt_to_send`), sprawdzenie, czy `context_files_content` został przekazany.
        *   Jeśli tak, sformatowanie treści plików w czytelny sposób (np. z nagłówkami `-- Plik: nazwa --`) i włączenie jej do `prompt_to_send`, wyraźnie oddzielając od ewentualnego kontekstu RAG i pytania użytkownika.
    *   Analogiczna modyfikacja metody `_generate_gemini_answer`, aby również akceptowała `context_files_content` i dołączała tę treść do promptu wysyłanego do Gemini API.
    *   Zapewnienie, że treść z `context_files_content` **nie** jest wykorzystywana w procesie wyszukiwania RAG (co jest domyślne, jeśli modyfikujemy tylko budowanie promptu w `_generate_answer` / `_generate_gemini_answer`).

## Diagram Przepływu Danych (Mermaid)

```mermaid
graph TD
    subgraph "Interfejs Użytkownika (ui_streamlit.py)"
        A[Użytkownik dodaje plik kontekstowy przez nowy uploader] --> B(Odczyt treści pliku i zapis w st.session_state.context_files);
        C[Użytkownik wpisuje pytanie w st.chat_input] --> D{Pobierz pytanie i context_files};
        D --> E[Wywołaj chatbot.query(pytanie, context_files_content=...)];
    end

    subgraph "Backend (app.py - RAGChatbot)"
        E --> F[Metoda query];
        F --> G{Tryb RAG włączony?};
        G -- Tak --> H[Pobierz kontekst RAG z Qdrant];
        G -- Nie --> I[Kontekst RAG pusty];
        H --> J{Przekazano context_files_content?};
        I --> J;
        J -- Tak --> K[Przygotuj sformatowaną treść plików kontekstowych];
        J -- Nie --> L[Treść plików kontekstowych pusta];
        K & L & H & I --> M[Wywołaj _generate_answer/_generate_gemini_answer(pytanie, kontekst_RAG, treść_plików_kontekstowych)];
        M --> N[Zbuduj finalny prompt dla LLM];
        N --> O[Wyślij prompt do LLM];
        O --> P[Odbierz odpowiedź];
    end

    P --> Q[UI: Wyświetl odpowiedź];

    style H fill:#f9f,stroke:#333,stroke-width:1px;
    style K fill:#ccf,stroke:#333,stroke-width:1px;