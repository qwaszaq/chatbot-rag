# 🤖 Chatbot RAG z Qdrant

To projekt chatbota wykorzystującego Retrieval-Augmented Generation (RAG) z bazą wektorową Qdrant. Aplikacja pozwala na przetwarzanie dokumentów PDF, DOCX i TXT, a następnie odpowiada na pytania użytkownika wykorzystując informacje z tych dokumentów.

## 🧩 Funkcjonalności

- Ładowanie dokumentów PDF, DOCX i TXT
- Dzielenie dokumentów na fragmenty (chunks)
- Generowanie embeddingów z wykorzystaniem LM Studio
- Przechowywanie wektorów w bazie danych Qdrant
- Wyszukiwanie wektorowe w Qdrant
- Reranking wyników wyszukiwania
- Generowanie odpowiedzi na podstawie kontekstu
- Interfejs użytkownika oparty na Streamlit
- Ładowanie dokumentów z adresu URL

## 🛠️ Technologie

- Python 3.10+
- Qdrant
- LM Studio
- Streamlit
- LangChain
- Sentence Transformers

## 📦 Instalacja

1. Upewnij się, że masz zainstalowane:
   - Python 3.10+
   - Qdrant (można zainstalować przez Docker)
   - LM Studio

2. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/twoja-nazwa/chatbot-rag.git
   cd chatbot-rag
   ```

3. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```

4. Uruchom Qdrant (jeśli nie jest jeszcze uruchomiony):
   ```bash
   docker run -d -p 6333:6333 qdrant/qdrant
   ```

5. Uruchom serwer LM Studio i wybierz odpowiedni model embeddingowy.

## ▶️ Uruchamianie

1. Uruchom aplikację Streamlit:
   ```bash
   streamlit run ui_streamlit.py
   ```

2. W przeglądarce otworzy się interfejs użytkownika.

3. Załaduj dokument przez panel boczny (z pliku lub URL) i kliknij "Przetwórz dokument".

4. Zadaj pytanie w polu czatu, a chatbot wykorzysta informacje z przetworzonych dokumentów aby wygenerować odpowiedź.

## 🧪 Testy jednostkowe

Możesz uruchomić testy jednostkowe, aby sprawdzić poprawność działania poszczególnych modułów:

```bash
python -m unittest discover tests
```

## 📐 Architektura

System składa się z kilku modułów:

1. **document_loader.py** - ładowanie dokumentów z różnych formatów
2. **text_splitter.py** - dzielenie dokumentów na mniejsze fragmenty
3. **embedding_generator.py** - generowanie embeddingów z tekstu
4. **reranker.py** - reranking wyników wyszukiwania
5. **qdrant_connector.py** - integracja z bazą danych Qdrant
6. **app.py** - logika biznesowa chatbota
7. **ui_streamlit.py** - interfejs użytkownika

## 📚 Przykładowe użycie

```python
from app import RAGChatbot

# Inicjalizacja chatbota
chatbot = RAGChatbot()

# Przetwarzanie dokumentu
chatbot.process_document("ścieżka/do/dokumentu.pdf")

# Zadawanie pytań
answer, sources = chatbot.query("Jakie są główne punkty strategii firmy?")

print(f"Odpowiedź: {answer}")
print(f"Źródła: {sources}")
```

## 📝 TODO

- [ ] Dodanie obsługi wielu modeli embeddingowych
- [ ] Dodanie dokumentacji API
- [ ] Dodanie systemu logowania do pliku
- [ ] Dodanie obsługi wielu użytkowników