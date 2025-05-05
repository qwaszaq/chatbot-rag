# Dokumentacja Projektu Chatbot RAG

## Opis Projektu

Projekt "Chatbot RAG" to zaawansowany system Retrieval-Augmented Generation (RAG) z rozszerzoną funkcjonalnością analizy klastrowania i eksploracji grafowej (GrafRAG). Celem systemu jest umożliwienie użytkownikom przeszukiwania i analizy dużych zbiorów dokumentów tekstowych poprzez zadawanie pytań w języku naturalnym oraz wizualizację powiązań między kluczowymi bytami w formie grafu wiedzy.

## Logika Przepływu Danych

System składa się z kilku kluczowych komponentów współpracujących w celu przetworzenia dokumentów, zbudowania bazy wiedzy i udzielania odpowiedzi na zapytania użytkowników.

### Proces Ingestii Danych

1.  **Ładowanie Dokumentów:** Moduł `data_processing/document_loader.py` odpowiada za wczytywanie dokumentów z różnych źródeł (lokalne pliki PDF, DOCX, TXT) i konwersję ich zawartości do ujednoliconego formatu. Obsługiwane są różne formaty plików.
2.  **Dzielenie Tekstu:** Wczytana treść dokumentów jest dzielona na mniejsze, zarządzalne fragmenty (chunki) przez moduł `data_processing/text_splitter.py`. Używany jest algorytm `RecursiveCharacterTextSplitter` zdefiniowany z określonym rozmiarem chunka i stopniem nakładania się, co zapewnia odpowiedni kontekst dla embeddingów.
3.  **Generowanie Embeddingów:** Dla każdego chunka tekstu generowany jest wektor numeryczny (embedding) przez moduł `data_processing/embedding_generator.py`. System wykorzystuje API lokalnego serwera LM Studio do generowania embeddingów. Te wektory reprezentują semantyczną treść chunków.
4.  **Indeksowanie w Bazie Wektorowej:** Wygenerowane embeddingi wraz z oryginalnym tekstem chunków i metadanymi (np. nazwa pliku źródłowego) są indeksowane w bazie danych wektorowej Qdrant przez moduł `database/qdrant_connector.py`. Qdrant umożliwia szybkie i efektywne wyszukiwanie podobnych wektorów w dużej skali.

### Workflow Analityk RAG

Ten tryb koncentruje się na precyzyjnym odpowiadaniu na pytania użytkownika w oparciu o zgromadzone dokumenty.

1.  **Zapytanie Użytkownika:** Użytkownik wprowadza pytanie w języku naturalnym w interfejsie Streamlit.
2.  **Wyszukiwanie Podobieństwa:** Zapytanie użytkownika jest konwertowane na wektor embeddingowy przez `data_processing/embedding_generator.py`. Moduł `database/qdrant_connector.py` używa tego embeddingu do wyszukania w bazie Qdrant N najbardziej podobnych wektorowo chunków. Wyszukiwanie może być opcjonalnie filtrowane do konkretnego klastra, jeśli użytkownik go wybrał.
3.  **Reranking Wyników:** Znalezione chunki (z ich oryginalną treścią) są przekazywane do modułu `data_processing/reranker.py`. Model CrossEncoder ocenia relewantność każdego chunka względem pierwotnego zapytania i uszeregowuje je.
4.  **Selekcja Kontekstu:** Wybierane jest Top K najbardziej relewantnych chunków (zgodnie z konfiguracją rerankera). Jeśli włączono rozszerzanie kontekstu o klastry, system może dodać dodatkowe chunki z klastrów, które były dominujące w początkowych wynikach wyszukiwania.
5.  **Generowanie Odpowiedzi przez LLM:** Zselekcjonowane chunki są łączone w jeden blok tekstowy, który służy jako kontekst dla modelu językowego (LLM), zdefiniowanego w `app.py` (domyślnie lokalny model przez LM Studio). LLM otrzymuje kontekst oraz pytanie użytkownika i generuje zwięzłą, bezpośrednią odpowiedź, bazując **wyłącznie** na dostarczonych danych.
6.  **Prezentacja Odpowiedzi:** Wygenerowana odpowiedź tekstowa wraz z listą źródeł (nazwy plików źródłowych chunków użytych w kontekście) jest wyświetlana użytkownikowi w interfejsie Streamlit.

### Workflow Analityk Grafów (GrafRAG)

Ten tryb rozszerza funkcjonalność Analityk RAG o wizualizację powiązań między kluczowymi bytami.

1.  **Kroki 1-5 są IDENTYCZNE jak w Workflow Analityk RAG:** Zapytanie użytkownika jest przetwarzane w ten sam sposób, aby uzyskać relewantny kontekst z dokumentów i wygenerować odpowiedź tekstową przez LLM. **Ważne:** W tym trybie używany jest specjalny prompt systemowy dla LLM (zdefiniowany w `ui_streamlit.py`), który instruuje model, aby w generowanej odpowiedzi **jawnie identyfikował kluczowe byty i relacje między nimi**. Jest to kluczowe dla kolejnego kroku.
2.  **Ekstrakcja Grafu Wiedzy:** Po wygenerowaniu odpowiedzi przez LLM, moduł `app.py` (`_extract_graph_from_response`) analizuje tekst **wygenerowanej odpowiedzi**. Używa do tego samego modelu LLM (z innym promptem ekstrakcyjnym) oraz modelu spaCy do identyfikacji Nazwanych Encji (NER). Na podstawie tekstu odpowiedzi LLM, moduł ten identyfikuje kluczowe byty (np. osoby, organizacje, miejsca) i relacje między nimi, tworząc strukturę grafu (węzły i krawędzie).
3.  **Wizualizacja Grafu:** Skonstruowany graf wiedzy jest serializowany do formatu node-link i przekazywany do interfejsu Streamlit (`ui_streamlit.py`). Interfejs wykorzystuje bibliotekę `streamlit-agraph` do interaktywnej wizualizacji grafu. Węzły grafu są kolorowane i oznaczane na podstawie typów NER zidentyfikowanych przez spaCy.
4.  **Prezentacja Wyników:** Użytkownik otrzymuje zarówno odpowiedź tekstową (jak w trybie Analityk RAG), jak i interaktywną wizualizację grafu wiedzy, która pozwala na eksplorację powiązań między bytami wspomnianymi w odpowiedzi.

### Analiza Klastrowania (Uzupełnienie Workflow)

Analiza klastrowania jest niezależnym procesem, który użytkownik może uruchomić z panelu bocznego UI. Jej wyniki (przypisania klastrów do punktów, metadane klastrów) są następnie wykorzystywane jako uzupełnienie workflow RAG/GrafRAG (np. do filtrowania zapytań lub rozszerzania kontekstu).

1.  **Pobranie Danych:** Moduł `qdrant_connector` pobiera wszystkie punkty z Qdrant wraz z wektorami i payloadami.
2.  **Klastrowanie:** Moduł `clustering_module.py` stosuje wybrany algorytm (np. K-Means) do wektorów embeddingów, grupując semantycznie podobne chunki.
3.  **Generowanie Metadanych:** Dla każdej grupy generowane są etykiety (przez LLM), podsumowania (przez LLM) i kluczowe byty (przez spaCy).
4.  **Zapis Wyników:** Przypisania punktów do klastrów są zapisywane w payloadzie Qdrant, a metadane klastrów w lokalnym pliku JSON.
5.  **Wykorzystanie w RAG/GrafRAG:** W trakcie wyszukiwania RAG, obecność przypisań klastrów w payloadzie Qdrant umożliwia:
    *   **Filtrowanie:** Ograniczenie wyszukiwania tylko do punktów z wybranego klastra.
    *   **Rozszerzanie Kontekstu:** Dodanie do kontekstu dla LLM dodatkowych punktów z dominujących klastrów znalezionych w początkowych wynikach wyszukiwania.

## Porównanie Trybów: Analityk RAG vs Analityk Grafów

| Cecha                 | Analityk RAG                                                                 | Analityk Grafów (GrafRAG)                                                                  |
| :-------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| **Główny Cel**        | Precyzyjne odpowiadanie na konkretne pytania na podstawie dokumentów.         | Identyfikacja i wizualizacja powiązań między bytami, oprócz odpowiadania na pytania.       |
| **Wyjście**           | Odpowiedź tekstowa + lista źródeł.                                           | Odpowiedź tekstowa + lista źródeł + **interaktywny graf wiedzy**.                           |
| **Proces Zapytania**  | Wyszukiwanie w bazie wektorowej -> Reranking -> Generowanie odpowiedzi przez LLM na podstawie kontekstu. | **Identyczny proces wyszukiwania/rerankingu/generowania odpowiedzi tekstowej**, z modyfikacją promptu LLM. Następnie **dodatkowa ekstrakcja grafu z wygenerowanej odpowiedzi**. |
| **Wykorzystanie LLM** | Do generowania odpowiedzi na podstawie kontekstu i pytania.                   | Do generowania odpowiedzi (ze wskazaniem bytów/relacji) **oraz** do ekstrakcji bytów/relacji z tej odpowiedzi (z pomocą spaCy). |
| **Analiza Powiązań**  | Ograniczona do kontekstu w odpowiedzi tekstowej.                           | **Jawna wizualizacja powiązań** w formie grafu, ułatwiająca analizę struktury danych.       |
| **Zastosowanie**      | Szybkie znajdowanie faktów, odpowiedzi na konkretne pytania, podsumowania dokumentów. | Analiza relacji między osobami/organizacjami/miejscami, śledzenie przepływów, zrozumienie złożonych sieci powiązań. |
| **Złożoność**         | Standardowy RAG.                                                             | Rozszerzony RAG z dodatkowym etapem ekstrakcji i wizualizacji grafu.                      |

### Do czego najlepiej stosować poszczególne funkcjonalności:

*   **Analityk RAG:** Idealny do szybkich, faktograficznych zapytań, gdy potrzebna jest precyzyjna odpowiedź bazująca na treści dokumentów. Doskonale sprawdza się przy szukaniu konkretnych liczb, dat, nazwisk, definicji, czy podsumowywaniu fragmentów tekstu. Jest to podstawowa i często wystarczająca funkcjonalność.
*   **Analityk Grafów (GrafRAG):** Niezastąpiony, gdy celem jest zrozumienie struktury danych, powiązań między różnymi podmiotami (osobami, firmami, lokalizacjami, projektami), śledzenie przepływów (np. finansowych, informacyjnych) lub identyfikacja kluczowych graczy w sieci powiązań opisanych w dokumentach. Wizualizacja grafu znacznie ułatwia wykrywanie wzorców i zależności, które mogłyby być trudne do zauważenia w samym tekście. Jest to narzędzie do głębszej, strukturalnej analizy danych.
*   **Analiza Klastrowania:** Użyteczna jako wstępny etap eksploracji danych. Pomaga zidentyfikować główne tematy obecne w zbiorze dokumentów i pogrupować powiązane ze sobą chunki. Podsumowania i kluczowe byty dla klastrów dają szybki przegląd ich zawartości. Funkcjonalność filtrowania RAG wg klastra pozwala następnie na szczegółową analizę wybranego tematu w izolacji od reszty zbioru.

## Instrukcja Instalacji i Uruchomienia

Aby uruchomić projekt, należy spełnić następujące wymagania i wykonać poniższe kroki:

### Wymagania Systemowe i Środowiskowe

*   **System Operacyjny:** Projekt powinien działać na większości systemów operacyjnych (Windows, macOS, Linux).
*   **Python:** Wersja 3.8 lub nowsza. Zalecane jest użycie wirtualnego środowiska (venv).
*   **Biblioteki Python:** Wszystkie wymagane biblioteki są wymienione w pliku `requirements.txt`.
*   **Model Językowy (LLM) i Model Embeddingowy:**
    *   **Lokalnie (zalecane):** Do uruchomienia lokalnego LLM i modelu embeddingowego potrzebne jest oprogramowanie takie jak [LM Studio](https://lmstudio.ai/). Należy pobrać i uruchomić model LLM (kompatybilny z API OpenAI Chat Completion) oraz model embeddingowy (kompatybilny z API OpenAI Embeddings) w LM Studio, upewniając się, że są dostępne pod adresem `http://localhost:1234`. Domyślnie używany model embeddingowy to "default", a model LLM to "local-model" (nazwy konfigurowalne w `app.py`).
    *   **Google Gemini (opcjonalnie):** Aby korzystać z modelu Google Gemini, potrzebny jest klucz API Google Cloud (dostępny po włączeniu Gemini API). Klucz API należy umieścić w zmiennej środowiskowej lub bezpośrednio w kodzie (UWAGA: przechowywanie kluczy w kodzie NIE jest bezpieczne - patrz komentarz w `app.py`).
*   **Baza Danych Wektorowa:** Qdrant. Można uruchomić go lokalnie np. przy użyciu Docker Compose lub zainstalować jako samodzielną aplikację. Domyślnie aplikacja oczekuje Qdrant pod adresem `localhost:6333`.

### Kroki Instalacyjne

1.  **Sklonowanie Repozytorium:** Otwórz terminal i sklonuj repozytorium z GitHub:
    ```bash
    git clone https://github.com/qwaszaq/chatbot01.git
    cd chatbot01
    ```
    *(Uwaga: Projekt może znajdować się w podkatalogu, jeśli sklonowano go do istniejącego folderu. Dostosuj ścieżkę `cd` jeśli potrzebne).*
2.  **Utworzenie i Aktywacja Wirtualnego Środowiska (zalecane):**
    ```bash
    python -m venv .venv
    # Na Windows:
    # .venv\Scripts\activate
    # Na macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Instalacja Zależności:** Zainstaluj wszystkie wymagane biblioteki z pliku `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Instalacja Modeli spaCy:** Dla funkcji ekstrakcji kluczowych bytów (NER) potrzebny jest model języka polskiego spaCy. Zainstaluj większy model (`pl_core_news_lg`):
    ```bash
    python -m spacy download pl_core_news_lg
    ```
    *(Jeśli instalacja `lg` się nie powiedzie lub brakuje pamięci, spróbuj zainstalować mniejszy model `pl_core_news_md`: `python -m spacy download pl_core_news_md`)*.
5.  **Uruchomienie Qdrant:** Upewnij się, że instancja Qdrant jest uruchomiona i dostępna pod adresem `localhost:6333`. Zapoznaj się z dokumentacją Qdrant dotyczącą instalacji.
6.  **Uruchomienie LM Studio (dla lokalnych modeli):** Uruchom LM Studio, pobierz i załaduj wybrany model LLM i model embeddingowy. Upewnij się, że ich API są dostępne.

### Uruchomienie Projektu

Po zainstalowaniu zależności i uruchomieniu Qdrant oraz LM Studio (jeśli używasz lokalnych modeli), uruchom interfejs użytkownika Streamlit z głównego katalogu projektu:

```bash
streamlit run ui_streamlit.py
```

Spowoduje to otwarcie aplikacji w przeglądarce internetowej.

## Globalny Opis Funkcjonalności

### Główne Komponenty

*   **Moduły Przetwarzania Danych (`data_processing/`):** Odpowiedzialne za ładowanie dokumentów, dzielenie tekstu na chunki, generowanie embeddingów i reranking.
*   **Moduły Bazy Danych (`database/`):** Obsługują połączenie i interakcję z bazą danych wektorową Qdrant.
*   **Moduł Klastrowania (`clustering_module.py`):** Implementuje algorytmy klastrowania i generowanie metadanych klastrów.
*   **Główna Logika Aplikacji (`app.py`):** Inicjalizuje wszystkie komponenty, zarządza przepływem RAG i GrafRAG, w tym ekstrakcją grafu wiedzy.
*   **Interfejs Użytkownika (`ui_streamlit.py`):** Zapewnia graficzny interfejs użytkownika oparty na Streamlit, umożliwiający interakcję z chatbotem, zarządzanie dokumentami, przeglądanie wyników klastrowania i wizualizację grafu.

### Rozwiązywane Problemy

System rozwiązuje problem efektywnego wyszukiwania informacji i analizy dużych, nieustrukturyzowanych zbiorów dokumentów. Pozwala na:

*   Szybkie znajdowanie odpowiedzi na konkretne pytania w dużej liczbie dokumentów.
*   Identyfikację głównych tematów i grupowanie dokumentów (klasteryzacja).
*   Zrozumienie powiązań między kluczowymi bytami (osobami, organizacjami, miejscami itp.) poprzez wizualizację grafu wiedzy.
*   Analizę konkretnych obszarów tematycznych (klastrów) w izolacji.

### Sposób Użycia

Użytkownik może:

1.  **Przetworzyć dokumenty:** Dodać lokalne pliki lub podać URL dokumentu do zaindeksowania w bazie Qdrant.
2.  **Analizować Klastry:** Uruchomić proces klastrowania, przeglądać zidentyfikowane klastry, ich etykiety, podsumowania i kluczowe byty.
3.  **Zadawać Pytania:** Wprowadzać pytania w języku naturalnym i otrzymywać odpowiedzi oparte na treści przetworzonych dokumentów (tryby RAG/GrafRAG).
4.  **Filtrować Zapytania:** Ograniczyć wyszukiwanie RAG do dokumentów z konkretnego klastra.
5.  **Wizualizować Graf:** W trybie GrafRAG, oglądać graf wiedzy wyekstrahowany z odpowiedzi LLM.
6.  **Zarządzać Czatami:** Tworzyć, przełączać się między różnymi sesjami czatu i usuwać je.

### Dostępne Tryby Działania

*   **Tryb Zwykły Chat:** Interakcja z LLM bez wykorzystania bazy dokumentów (bez RAG).
*   **Tryb Analityk Tekstu (RAG):** Standardowy tryb RAG, generujący odpowiedzi na podstawie kontekstu z dokumentów.
*   **Tryb Analityk Grafów (GrafRAG):** Tryb RAG z dodatkową funkcjonalnością ekstrakcji i wizualizacji grafu wiedzy z odpowiedzi LLM.
*   **Tryb Google Gemini API (opcjonalnie):** Bezpośrednie zapytania do modelu Google Gemini Flash API, pomijające RAG.

## Schematy / Diagramy

*(Tutaj w przyszłości można dodać schematy przepływu danych dla RAG i GrafRAG, diagram architektury systemu itp. Na potrzeby tej dokumentacji, opis tekstowy jest głównym źródłem informacji.)*