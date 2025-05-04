
## Zadania do Wykonania

### 1. Rozszerzenie QdrantConnector do obsługi wielu kolekcji

- [ ] **Zadanie 1.1: Dodanie metody do pobierania listy kolekcji**
  - Dodać metodę `get_collections()` do klasy `QdrantConnector`, która zwraca listę wszystkich dostępnych kolekcji na serwerze Qdrant.
  - Metoda powinna zwracać słownik z nazwami kolekcji oraz dodatkowymi informacjami (np. liczba punktów, data utworzenia).

- [ ] **Zadanie 1.2: Modyfikacja metody inicjalizacji**
  - Zmodyfikować metodę `initialize()` w `QdrantConnector`, aby mogła być wywoływana ponownie dla innej kolekcji.
  - Dodać obsługę zmiany kolekcji bez konieczności ponownego tworzenia całego obiektu `QdrantConnector`.

- [ ] **Zadanie 1.3: Dodanie metody do tworzenia nowej kolekcji**
  - Dodać metodę `create_collection()` do klasy `QdrantConnector`, która tworzy nową kolekcję o podanej nazwie i parametrach.

- [ ] **Zadanie 1.4: Rozszerzenie metody usuwania kolekcji**
  - Rozszerzyć metodę `clear_collection()` lub dodać nową metodę `delete_collection()`, która pozwala na usunięcie kolekcji bez natychmiastowego tworzenia nowej.

### 2. Modyfikacja klasy RAGChatbot do obsługi zmiany kolekcji

- [ ] **Zadanie 2.1: Dodanie metody do zmiany kolekcji**
  - Dodać metodę `switch_collection(collection_name)` do klasy `RAGChatbot`, która zmienia aktualną kolekcję.
  - Metoda powinna:
    - Inicjalizować QdrantConnector z nową kolekcją
    - Resetować stan chatbota związany z poprzednią kolekcją (np. cache wyników klastrowania)

- [ ] **Zadanie 2.2: Modyfikacja konstruktora**
  - Zmodyfikować konstruktor klasy `RAGChatbot`, aby obsługiwał inicjalizację z domyślną lub podaną kolekcją.
  - Dodać obsługę błędów w przypadku nieudanej inicjalizacji kolekcji.
### 3. Implementacja interfejsu użytkownika do zarządzania kolekcjami

- [ ] **Zadanie 3.1: Dodanie sekcji wyboru kolekcji w UI**
  - Dodać nową sekcję "Wybór Kolekcji" w panelu bocznym (`ui_streamlit.py`).
  - Implementacja rozwijanej listy (dropdown) wyświetlającej dostępne kolekcje.
  - Dodać przycisk odświeżania listy kolekcji.

- [ ] **Zadanie 3.2: Implementacja funkcjonalności tworzenia nowej kolekcji**
  - Dodać pole do wprowadzenia nazwy nowej kolekcji.
  - Dodać pole numeryczne do określenia rozmiaru wektorów (z wartością domyślną 1024).
  - Dodać przycisk "Utwórz Nową Kolekcję".

- [ ] **Zadanie 3.3: Implementacja funkcjonalności usuwania kolekcji**
  - Dodać przycisk "Usuń Kolekcję" obok wybranej kolekcji.
  - Dodać okno dialogowe potwierdzenia przed usunięciem kolekcji.

- [ ] **Zadanie 3.4: Obsługa zmiany kolekcji**
  - Zaimplementować funkcję obsługującą zmianę wybranej kolekcji w UI.
  - Dodać logikę resetowania stanu aplikacji przy zmianie kolekcji.

### 4. Zarządzanie stanem aplikacji przy zmianie kolekcji

- [ ] **Zadanie 4.1: Rozszerzenie stanu sesji Streamlit**
  - Dodać nowe klucze do `st.session_state` do przechowywania:
    - Aktualnie wybranej kolekcji
    - Listy dostępnych kolekcji
    - Stanu potwierdzenia usunięcia kolekcji

- [ ] **Zadanie 4.2: Implementacja resetu stanu przy zmianie kolekcji**
  - Zaimplementować funkcję resetującą stan aplikacji związany z klastrami przy zmianie kolekcji.
  - Resetowanie: `cluster_assignments`, `cluster_labels`, `qdrant_data_cache`, `selected_cluster_id`, `cluster_summaries`, `cluster_entities`.

- [ ] **Zadanie 4.3: Obsługa błędów połączenia**
  - Dodać obsługę błędów w przypadku problemów z połączeniem do serwera Qdrant.
  - Wyświetlać odpowiednie komunikaty błędów w UI.
### 5. Zapamiętywanie ostatnio używanej kolekcji

- [ ] **Zadanie 5.1: Dodanie zapisu ostatnio używanej kolekcji**
  - Dodać zapis ostatnio używanej kolekcji do pliku JSON.
  - Implementacja funkcji `save_collection_settings()` i `load_collection_settings()`.

- [ ] **Zadanie 5.2: Automatyczne ładowanie ostatnio używanej kolekcji**
  - Zmodyfikować inicjalizację aplikacji, aby automatycznie ładowała ostatnio używaną kolekcję.
  - Dodać obsługę przypadku, gdy ostatnio używana kolekcja nie istnieje.

### 6. Testowanie i finalizacja

- [ ] **Zadanie 6.1: Testowanie tworzenia i usuwania kolekcji**
  - Przeprowadzić testy tworzenia nowych kolekcji z różnymi parametrami.
  - Testowanie usuwania kolekcji i weryfikacja, czy kolekcje są poprawnie usuwane z serwera Qdrant.

- [ ] **Zadanie 6.2: Testowanie zmiany kolekcji**
  - Przeprowadzić testy zmiany kolekcji podczas pracy z aplikacją.
  - Weryfikacja, czy stan aplikacji jest poprawnie resetowany przy zmianie kolekcji.

- [ ] **Zadanie 6.3: Testowanie obsługi błędów**
  - Testowanie zachowania aplikacji w przypadku błędów połączenia z Qdrant.
  - Weryfikacja komunikatów błędów i obsługi wyjątków.

## Implementacja Krok po Kroku

### Krok 1: Rozszerzenie QdrantConnector

Implementacja metod w pliku `database/qdrant_connector.py`:
- `get_collections()` - pobieranie listy kolekcji
- `switch_collection()` - zmiana aktualnej kolekcji
- `create_collection()` - tworzenie nowej kolekcji
- `delete_collection()` - usuwanie istniejącej kolekcji

### Krok 2: Rozszerzenie RAGChatbot

Implementacja metod w pliku `app.py`:
- `switch_collection()` - zmiana kolekcji w chatbocie
- `get_collections()` - pobieranie listy kolekcji
- `create_collection()` - tworzenie nowej kolekcji
- `delete_collection()` - usuwanie kolekcji

### Krok 3: Implementacja interfejsu użytkownika

Dodanie nowej sekcji w UI w pliku `ui_streamlit.py`:
- Rozwijana lista kolekcji
- Formularz tworzenia nowej kolekcji
- Funkcjonalność usuwania kolekcji
- Obsługa zmiany kolekcji

### Krok 4: Obsługa stanu aplikacji

Implementacja mechanizmu resetowania stanu aplikacji przy zmianie kolekcji:
- Resetowanie stanu klastrowania
- Resetowanie cache danych
- Obsługa błędów

### Krok 5: Zapamiętywanie ustawień

Implementacja mechanizmu zapamiętywania ostatnio używanej kolekcji:
- Zapis ustawień do pliku JSON
- Automatyczne ładowanie ostatnio używanej kolekcji

### Krok 6: Testowanie i finalizacja

Przeprowadzenie testów kompletnej implementacji:
- Testowanie tworzenia i usuwania kolekcji
- Testowanie zmiany kolekcji
- Testowanie obsługi błędów