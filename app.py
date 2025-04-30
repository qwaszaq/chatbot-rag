"""
Główny plik aplikacji chatbota RAG z Qdrant
"""
import logging
import os
import time # Dodano import time
from data_processing.document_loader import DocumentLoader
from data_processing.text_splitter import TextSplitter
from data_processing.embedding_generator import EmbeddingGenerator
from database.qdrant_connector import QdrantConnector
from data_processing.reranker import Reranker
from langchain_core.documents import Document
# Zmieniono import LLM na ChatOpenAI z langchain_openai do połączenia z LM Studio API
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    # Zmieniono domyślną nazwę kolekcji na "nowa" i vector_size na 1024
    def __init__(self, qdrant_collection="nowa", vector_size=1024, top_k=99, top_k_reranker=33, relevance_threshold=0.0):
        """
        Inicjalizacja chatbota RAG

        Args:
            qdrant_collection (str): Nazwa kolekcji w Qdrant
            vector_size (int): Rozmiar wektorów embeddingów
            top_k (int): Liczba dokumentów do pobrania z Qdrant
            top_k_reranker (int): Liczba dokumentów po rerankingu
            relevance_threshold (float): Próg istotności po rerankingu
        """
        self.top_k = top_k
        self.top_k_reranker = top_k_reranker
        self.relevance_threshold = relevance_threshold

        # Inicjalizacja komponentów
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)

        # Inicjalizacja modelu embeddingowego
        self.embedding_generator = EmbeddingGenerator(
            api_url="http://localhost:1234/v1/embeddings",
            model_name="default" # Domyślny model LM Studio dla embeddingów (nazwa w LM Studio)
        )

        # Inicjalizacja Qdrant
        # Przekazujemy vector_size do QdrantConnector
        self.qdrant_connector = QdrantConnector(
            collection_name=qdrant_collection,
            vector_size=vector_size # Używamy vector_size z konstruktora RAGChatbot
        )
        # **Zmiana:** Jeśli inicjalizacja Qdrant zawiedzie, rzuć wyjątek
        if not self.qdrant_connector.initialize(self.embedding_generator):
             logger.error("❌ Błąd krytyczny: Nie udało się zainicjalizować połączenia z Qdrant.")
             # Rzuć wyjątek, aby przerwać tworzenie obiektu RAGChatbot
             raise ConnectionError("Nie udało się połączyć z bazą danych Qdrant. Sprawdź logi.")


        # Inicjalizacja rerankera
        self.reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")

        # Inicjalizacja modelu LLM (LM Studio) - używamy ChatOpenAI
        # LM Studio udostępnia endpoint zgodny z OpenAI API
        self.llm = ChatOpenAI(
            model="local-model",  # Dowolna nazwa modelu, LM Studio ją zignoruje lub użyje załadowanej
            base_url="http://localhost:1234/v1", # Adres endpointu LM Studio
            api_key="sk-no-key-required", # Klucz API nie jest wymagany przez LM Studio
            temperature=0.7,
            max_tokens=1000,
            # Dostosuj inne parametry według potrzeb
        )

        # Usunięto definicję self.prompt_template, ponieważ prompt będzie przekazywany dynamicznie z UI


    def process_document(self, file_path):
        """
        Przetwarza dokument: ładuje, dzieli na chunki i zapisuje do Qdrant

        Args:
            file_path (str): Ścieżka do pliku dokumentu

        Returns:
            bool: True jeśli przetworzenie zakończyło się sukcesem, False w przeciwnym wypadku
        """
        if not self.qdrant_connector.get_vectorstore():
             logger.error("❌ Qdrant nie jest zainicjalizowany. Nie można przetworzyć dokumentu.")
             return False

        try:
            logger.info(f"📄 Przetwarzanie dokumentu: {file_path}")

            # Załaduj dokument
            documents = self.document_loader.load(file_path)
            if not documents:
                logger.error(f"❌ Nie udało się załadować dokumentu: {file_path}")
                return False

            # Podziel dokument na chunki
            all_splits = []
            for doc in documents:
                splits = self.text_splitter.split_document(doc)
                all_splits.extend(splits)

            if not all_splits:
                logger.error("❌ Nie udało się podzielić dokumentu na chunki")
                return False

            logger.info(f"✂️ Dokument podzielony na {len(all_splits)} chunków")

            # Wygeneruj embeddingi i zapisz do Qdrant
            logger.info("🧠 Generowanie embeddingów i zapisywanie do Qdrant")
            # Metoda add_documents w QdrantConnector użyje wewnętrznego embedding generatora
            self.qdrant_connector.add_documents(all_splits)

            logger.info("✅ Dokument został pomyślnie przetworzony i zapisany do Qdrant")
            return True

        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania dokumentu: {str(e)}")
            return False

    def query(self, question, system_prompt_override=None):
        """
        Udziela odpowiedzi na pytanie korzystając z RAG

        Args:
            question (str): Pytanie użytkownika
            system_prompt_override (str, optional): Prompt systemowy do użycia zamiast domyślnego.

        Returns:
            str: Odpowiedź na pytanie
            list: Lista źródeł wykorzystanych w odpowiedzi
        """
        # Dodano sprawdzenie, czy LLM i reranker są zainicjalizowane
        if not hasattr(self, 'reranker') or not hasattr(self, 'llm') or not self.qdrant_connector.get_vectorstore():
             logger.error("❌ System RAG nie jest w pełni zainicjalizowany. Brakuje rerankera, LLM lub połączenia z Qdrant.")
             return "System RAG nie jest w pełni zainicjalizowany.", []

        try:
            logger.info(f"❓ Otrzymałem pytanie: {question}")

            # Wyszukaj podobne dokumenty w Qdrant
            logger.info(f"🔍 Wyszukiwanie {self.top_k} podobnych dokumentów w Qdrant")
            # Metoda similarity_search w QdrantConnector użyje swojego wewnętrznego embedding generatora do zapytania
            relevant_docs = self.qdrant_connector.similarity_search(question, k=self.top_k)

            if not relevant_docs:
                logger.info("❌ Nie znaleziono żadnych dokumentów pasujących do pytania")
                # Możemy spróbować wywołać LLM bez kontekstu dla prostych zapytań, ale na razie zwracamy informację
                return "Nie znaleziono żadnych dokumentów pasujących do pytania.", []

            logger.info(f"🎯 Znaleziono {len(relevant_docs)} dokumentów pasujących do pytania przed rerankingiem")

            # Reranking dokumentów
            logger.info(f"🔄 Reranking dokumentów (top {self.top_k_reranker}, próg istotności {self.relevance_threshold})")
            # Reranker działa na parach [zapytanie, treść dokumentu]
            reranked_scored_docs = self.reranker.rerank(
                question,
                [doc.page_content for doc in relevant_docs], # Przekazujemy tylko treść dokumentów
                self.top_k_reranker
            )

            # Przygotowanie kontekstu i listy TOP 3 źródeł na podstawie rerankingu
            context_parts = []
            top_sources = []
            added_sources = set()
            # Stwórz mapowanie content -> metadata dla łatwego dostępu
            # Uwaga: To zakłada w miarę unikalne treści chunków. Jeśli chunki mają identyczną treść,
            # to mapowanie weźmie metadane ostatniego napotkanego.
            content_to_metadata = {doc.page_content: doc.metadata for doc in relevant_docs}

            logger.info(f"📝 Budowanie kontekstu i listy źródeł (max 3) z rerankowanych dokumentów...")
            for content, score in reranked_scored_docs:
                if score > self.relevance_threshold:
                    context_parts.append(content) # Dodaj do kontekstu
                    metadata = content_to_metadata.get(content)
                    if metadata:
                        source = metadata.get("source", "nieznane źródło")
                        if source not in added_sources:
                             if len(top_sources) < 3: # Dodaj tylko top 3 unikalne źródła
                                top_sources.append(source)
                                added_sources.add(source)
                    # Nie przerywamy pętli, aby zbudować pełny kontekst z dokumentów powyżej progu
                else:
                    # Wyniki są posortowane, więc możemy przerwać, gdy napotkamy wynik poniżej progu
                    break

            logger.info(f"✅ Zbudowano kontekst z {len(context_parts)} chunków.")
            logger.info(f"✅ Wybrano {len(top_sources)} unikalnych źródeł do wyświetlenia.")

            if not context_parts:
                logger.info("❌ Brak dokumentów spełniających próg istotności po rerankingu")
                return "Nie znaleziono wystarczająco istotnych informacji w dokumentach, aby odpowiedzieć na to pytanie.", []

            # Połącz kontekst
            context = "\n\n".join(context_parts)

            # Wygeneruj odpowiedź za pomocą LLM, przekazując prompt systemowy
            logger.info("🤖 Generowanie odpowiedzi za pomocą LLM")
            if system_prompt_override is None:
                 logger.warning("⚠️ Nie podano system_prompt_override do metody query. Używanie pustego promptu systemowego.")
                 system_prompt_override = ""

            # Generowanie odpowiedzi zwraca teraz słownik z treścią i metadanymi
            response_dict = self._generate_answer(question, context, system_prompt_text=system_prompt_override)

            # Użyj listy top_sources zbudowanej wcześniej
            sources = top_sources

            logger.info("✅ Odpowiedź została pomyślnie wygenerowana wraz z metadanymi")
            # Zwracamy słownik odpowiedzi i listę źródeł (top_sources)
            return response_dict, sources # Zwracamy słownik, a nie tylko treść

        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}")
            return f"Wystąpił błąd podczas przetwarzania pytania: {e}", []

    def _generate_answer(self, question, context, system_prompt_text):
        """
        Generuje odpowiedź na podstawie kontekstu i pytania przy użyciu modelu LLM,
        dołączając metadane dotyczące generowania.

        Args:
            question (str): Pytanie użytkownika
            context (str): Kontekst z dokumentów
            system_prompt_text (str): Tekst promptu systemowego (instrukcji)

        Returns:
            dict: Słownik zawierający treść odpowiedzi i metadane
                  np. {"content": AIMessage(...), "metadata": {"model": "...", "tokens": ..., "time": ...}}
                  W przypadku błędu zwraca słownik z komunikatem o błędzie.
        """
        # Dodano sprawdzenie, czy LLM jest zainicjalizowane przed użyciem
        if not hasattr(self, 'llm'):
             error_msg = "Model LLM nie został zainicjalizowany."
             logger.error(f"❌ {error_msg}")
             # Zwracamy słownik z błędem, zachowując spójny typ zwracany
             return {"content": error_msg, "metadata": None}

        response_content = None
        token_count = None
        model_name = "N/A" # Domyślna wartość
        response_time = 0.0

        try:
            # Pobierz nazwę modelu (jeśli dostępna)
            model_name = getattr(self.llm, 'model', "N/A") # Użyj getattr dla bezpieczeństwa

            # Ręczne budowanie promptu
            prompt_to_send = f"""{system_prompt_text}

            --- DOSTARCZONY KONTEKST ---
            {context}

            --- PYTANIE UŻYTKOWNIKA ---
            {question}

            --- ODPOWIEDŹ ANALITYKA ---
            """
            logger.info("➡️ Wysłanie ręcznie zbudowanego promptu do LLM...")

            # Pomiar czasu i wywołanie LLM
            start_time = time.time()
            response = self.llm.invoke(prompt_to_send)
            end_time = time.time()
            response_time = end_time - start_time

            response_content = response # Zachowaj cały obiekt odpowiedzi (np. AIMessage)
            logger.info(f"⬅️ Otrzymano odpowiedź od LLM w {response_time:.2f}s.")

            # --- Pobieranie metadanych z obiektu odpowiedzi ---
            try:
                # 1. Spróbuj pobrać nazwę modelu - priorytet dla klucza 'model' w response_metadata
                resp_meta = getattr(response, 'response_metadata', {})
                model_name_from_direct_key = resp_meta.get('model') # Bezpośredni dostęp do klucza 'model'

                if model_name_from_direct_key:
                    model_name = model_name_from_direct_key
                    logger.info(f"📊 Nazwa modelu (z response_metadata['model']): {model_name}")
                # Fallback 1: spróbuj system_fingerprint (jak w JSON z LM Studio)
                elif 'system_fingerprint' in resp_meta and resp_meta['system_fingerprint']:
                     model_name = resp_meta['system_fingerprint']
                     logger.info(f"📊 Nazwa modelu (z response_metadata['system_fingerprint']): {model_name}")
                # Fallback 2: spróbuj z atrybutu obiektu LLM (mało prawdopodobne, że zadziała poprawnie)
                elif hasattr(self.llm, 'model') and self.llm.model != "local-model":
                    model_name = self.llm.model
                    logger.info(f"📊 Nazwa modelu (z obiektu self.llm - placeholder?): {model_name}")
                else:
                    logger.warning("⚠️ Nie udało się uzyskać nazwy modelu z response_metadata ani obiektu LLM.")
                    model_name = "N/A" # Ostateczny fallback

                # 2. Spróbuj pobrać liczbę tokenów (logika pozostaje ta sama)
                # Sprawdź usage_metadata (nowszy standard LangChain)
                usage_meta = getattr(response, 'usage_metadata', None)
                if usage_meta and 'completion_tokens' in usage_meta:
                    token_count = usage_meta.get('completion_tokens')
                    logger.info(f"📊 Użycie tokenów (odpowiedź, z usage_metadata): {token_count}")
                # Sprawdź token_usage w response_metadata (starszy/inny standard)
                elif resp_meta and 'token_usage' in resp_meta:
                    token_usage = resp_meta.get('token_usage', {})
                    token_count = token_usage.get('completion_tokens')
                    if token_count:
                         logger.info(f"📊 Użycie tokenów (odpowiedź, z response_metadata): {token_count}")
                    else:
                         logger.warning("⚠️ Nie znaleziono 'completion_tokens' w response_metadata['token_usage'].")
                else:
                    logger.warning("⚠️ Nie znaleziono danych o użyciu tokenów w 'usage_metadata' ani 'response_metadata'.")

            except AttributeError as attr_err:
                 logger.warning(f"⚠️ Obiekt odpowiedzi nie ma oczekiwanych atrybutów metadanych ({attr_err}).")
            except Exception as meta_e:
                 logger.warning(f"⚠️ Nieoczekiwany błąd podczas pobierania metadanych z odpowiedzi: {meta_e}.")
            # -----------------------------------------------

        except Exception as e:
            error_msg = f"Wystąpił błąd podczas generowania odpowiedzi przez LLM: {e}"
            logger.error(f"❌ {error_msg}")
            # Zwracamy słownik z błędem
            return {"content": error_msg, "metadata": None}

        # Zwróć słownik z treścią i metadanymi
        return {
            "content": response_content,
            "metadata": {
                "model": model_name,
                "tokens": token_count, # Może być None
                "time": response_time
            }
        }


# Usunięto metodę get_sources, ponieważ logika została przeniesiona do metody query


if __name__ == "__main__":
    # Przykład użycia
    chatbot = RAGChatbot()

    # Przetwarzanie dokumentu (można to wykonać raz dla każdego dokumentu)
    # Pamiętaj, że Streamlit UI obsługuje ładowanie plików, więc ten blok jest głównie dla testów skryptowych
    # try:
    #     # Załóżmy, że masz plik testowy 'test_document.txt' w tym samym katalogu
    #     if os.path.exists("test_document.txt"):
    #          chatbot.process_document("test_document.txt")
    #     else:
    #          print("Plik 'test_document.txt' nie znaleziono. Pomiń przetwarzanie dokumentu.")
    # except Exception as e:
    #      print(f"Wystąpił błąd podczas przetwarzania dokumentu: {e}")


    # Przykładowe pytanie
    # answer, sources = chatbot.query("Co to jest RAG?")

    # print(f"Odpowiedź: {answer}")
    # print(f"Źródła: {sources}")

    # Aby uruchomić UI Streamlit, użyj komendy w terminalu:
    # streamlit run app.py
    # lub uruchom ui_streamlit.py bezpośrednio
    print("Aby uruchomić interfejs użytkownika, uruchom: streamlit run ui_streamlit.py")