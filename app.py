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
# Usunięto import PromptTemplate, bo nie jest już używany w __init__
# from langchain_core.prompts import PromptTemplate
import spacy # Dodano import spacy
import networkx as nx # Dodano import networkx
import json # Dodano import json do parsowania odpowiedzi LLM
from langchain_core.messages import AIMessage # Dodano brakujący import

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    # Zmieniono domyślną nazwę kolekcji na "nowa1"
    def __init__(self, qdrant_collection="nowa1", vector_size=1024, top_k=99, top_k_reranker=33, relevance_threshold=0.0):
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

        # --- Ładowanie modelu spaCy dla NER ---
        try:
             self.nlp = spacy.load("pl_core_news_md")
             logger.info("✅ Załadowano model spaCy 'pl_core_news_md' dla NER.")
        except OSError:
             logger.error("❌ Nie znaleziono modelu spaCy 'pl_core_news_md'.")
             logger.error("Aby go zainstalować, uruchom w terminalu: python -m spacy download pl_core_news_md")
             # Można rzucić wyjątek lub ustawić self.nlp na None i obsłużyć to dalej
             self.nlp = None # Ustawienie na None, logika przetwarzania musi to obsłużyć
        # ------------------------------------


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

            # --- Przetwarzanie NER przed zapisem ---
            if self.nlp: # Sprawdź, czy model spaCy został załadowany
                 logger.info("🔍 Rozpoznawanie obiektów (NER) za pomocą spaCy...")
                 processed_splits = []
                 for chunk in all_splits:
                      # Przetwórz tekst chunka przez spaCy
                      doc_spacy = self.nlp(chunk.page_content)
                      # Wyodrębnij encje
                      entities = [{"text": ent.text, "label": ent.label_} for ent in doc_spacy.ents]
                      # Dodaj encje do metadanych (lub zaktualizuj istniejące)
                      if entities: # Dodaj tylko jeśli znaleziono encje
                           if chunk.metadata is None: # Upewnij się, że metadata istnieje
                                chunk.metadata = {}
                           chunk.metadata["entities"] = entities
                           # logger.debug(f"   Znaleziono encje w chunku: {entities}") # Opcjonalne logowanie
                      processed_splits.append(chunk)
                 logger.info(f"✅ Zakończono NER dla {len(processed_splits)} chunków.")
                 splits_to_save = processed_splits
            else:
                 logger.warning("⚠️ Model spaCy niezaładowany. Pomijanie kroku NER.")
                 splits_to_save = all_splits # Zapisz oryginalne chunki bez NER
            # --------------------------------------

            # Wygeneruj embeddingi i zapisz do Qdrant (przetworzone lub oryginalne chunki)
            logger.info("🧠 Generowanie embeddingów i zapisywanie do Qdrant")
            self.qdrant_connector.add_documents(splits_to_save)

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
            tuple: Zawiera:
                   - dict: Słownik odpowiedzi z kluczami 'content' i 'metadata'.
                   - list: Lista źródeł (max 3 unikalne).
                   - dict or None: Dane grafu wiedzy w formacie NetworkX node-link lub None.
        """
        # Dodano sprawdzenie, czy LLM i reranker są zainicjalizowane
        if not hasattr(self, 'reranker') or not hasattr(self, 'llm') or not self.qdrant_connector.get_vectorstore():
             logger.error("❌ System RAG nie jest w pełni zainicjalizowany. Brakuje rerankera, LLM lub połączenia z Qdrant.")
             return {"content": "System RAG nie jest w pełni zainicjalizowany.", "metadata": None}, [], None

        try:
            logger.info(f"❓ Otrzymałem pytanie: {question}")

            # Wyszukaj podobne dokumenty w Qdrant
            logger.info(f"🔍 Wyszukiwanie {self.top_k} podobnych dokumentów w Qdrant")
            relevant_docs = self.qdrant_connector.similarity_search(question, k=self.top_k)

            if not relevant_docs:
                logger.info("❌ Nie znaleziono żadnych dokumentów pasujących do pytania")
                return {"content": "Nie znaleziono żadnych dokumentów pasujących do pytania.", "metadata": None}, [], None

            logger.info(f"🎯 Znaleziono {len(relevant_docs)} dokumentów pasujących do pytania przed rerankingiem")

            # Reranking dokumentów
            logger.info(f"🔄 Reranking dokumentów (top {self.top_k_reranker}, próg istotności {self.relevance_threshold})")
            reranked_scored_docs = self.reranker.rerank(
                question,
                [doc.page_content for doc in relevant_docs],
                self.top_k_reranker
            )

            # Przygotowanie kontekstu i listy TOP 3 źródeł na podstawie rerankingu
            context_parts = []
            top_sources = []
            added_sources = set()
            content_to_metadata = {doc.page_content: doc.metadata for doc in relevant_docs}

            logger.info(f"📝 Budowanie kontekstu i listy źródeł (max 3) z rerankowanych dokumentów...")
            for content, score in reranked_scored_docs:
                if score > self.relevance_threshold:
                    context_parts.append(content)
                    metadata = content_to_metadata.get(content)
                    if metadata:
                        source = metadata.get("source", "nieznane źródło")
                        if source not in added_sources:
                             if len(top_sources) < 3:
                                top_sources.append(source)
                                added_sources.add(source)
                else:
                    break # Wyniki są posortowane

            logger.info(f"✅ Zbudowano kontekst z {len(context_parts)} chunków.")
            logger.info(f"✅ Wybrano {len(top_sources)} unikalnych źródeł do wyświetlenia.")

            if not context_parts:
                logger.info("❌ Brak dokumentów spełniających próg istotności po rerankingu")
                return {"content": "Nie znaleziono wystarczająco istotnych informacji w dokumentach, aby odpowiedzieć na to pytanie.", "metadata": None}, [], None

            context = "\n\n".join(context_parts)

            # Generowanie odpowiedzi
            logger.info("🤖 Generowanie odpowiedzi za pomocą LLM")
            if system_prompt_override is None:
                 logger.warning("⚠️ Nie podano system_prompt_override do metody query. Używanie pustego promptu systemowego.")
                 system_prompt_override = ""

            response_dict = self._generate_answer(question, context, system_prompt_text=system_prompt_override)
            sources = top_sources

            logger.info("✅ Odpowiedź została pomyślnie wygenerowana wraz z metadanymi")

            # --- Ekstrakcja grafu wiedzy z odpowiedzi ---
            graph_data = None
            answer_text_content = None
            if response_dict and response_dict.get("content"):
                 if isinstance(response_dict["content"], AIMessage):
                      answer_text_content = response_dict["content"].content
                 elif isinstance(response_dict["content"], str):
                      # Sprawdź, czy to nie jest komunikat o błędzie
                      if response_dict.get("metadata") is not None:
                           answer_text_content = response_dict["content"]
                      else:
                           logger.warning("Pomijanie ekstrakcji grafu z komunikatu o błędzie LLM.")
                 else:
                      logger.warning(f"Nieoczekiwany typ treści odpowiedzi: {type(response_dict['content'])}. Pomijanie ekstrakcji grafu.")
            else:
                 logger.warning("Brak treści w odpowiedzi LLM. Pomijanie ekstrakcji grafu.")


            if answer_text_content:
                 logger.info("🕸️ Próba ekstrakcji grafu wiedzy z odpowiedzi LLM...")
                 try:
                      # === TUTAJ JEST WYWOŁANIE METODY ===
                      graph = self._extract_graph_from_response(answer_text_content)
                      # ====================================
                      if graph and graph.nodes:
                           graph_data = nx.node_link_data(graph)
                           logger.info(f"✅ Pomyślnie wyekstrahowano graf z {len(graph.nodes)} węzłami i {len(graph.edges)} krawędziami.")
                      else:
                           logger.info("ℹ️ Nie udało się wyekstrahować znaczących relacji do grafu z odpowiedzi lub LLM zwrócił pustą listę.")
                 except Exception as graph_e:
                      # Logujemy konkretny błąd, który wystąpił TUTAJ
                      logger.error(f"❌ Błąd podczas ekstrakcji lub serializacji grafu wiedzy: {graph_e}", exc_info=True) # Dodano exc_info dla pełnego śladu
            # -----------------------------------------

            return response_dict, sources, graph_data

        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}", exc_info=True) # Dodano exc_info
            return {"content": f"Wystąpił błąd podczas przetwarzania pytania: {e}", "metadata": None}, [], None


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
        if not hasattr(self, 'llm'):
             error_msg = "Model LLM nie został zainicjalizowany."
             logger.error(f"❌ {error_msg}")
             return {"content": error_msg, "metadata": None}

        response_content = None
        token_count = None
        model_name = "N/A"
        response_time = 0.0

        try:
            model_name = getattr(self.llm, 'model', "N/A")

            prompt_to_send = f"""{system_prompt_text}

            --- DOSTARCZONY KONTEKST ---
            {context}

            --- PYTANIE UŻYTKOWNIKA ---
            {question}

            --- ODPOWIEDŹ ANALITYKA ---
            """
            logger.info("➡️ Wysłanie ręcznie zbudowanego promptu do LLM...")

            start_time = time.time()
            response = self.llm.invoke(prompt_to_send)
            end_time = time.time()
            response_time = end_time - start_time

            response_content = response
            logger.info(f"⬅️ Otrzymano odpowiedź od LLM w {response_time:.2f}s.")

            try:
                resp_meta = getattr(response, 'response_metadata', {})
                model_name_from_direct_key = resp_meta.get('model')

                # Priorytet dla 'system_fingerprint' jeśli jest dostępny (jak w twoich logach)
                sys_fingerprint = resp_meta.get('system_fingerprint')
                if sys_fingerprint:
                    model_name = sys_fingerprint
                    logger.info(f"📊 Nazwa modelu (z response_metadata['system_fingerprint']): {model_name}")
                elif model_name_from_direct_key:
                    model_name = model_name_from_direct_key
                    logger.info(f"📊 Nazwa modelu (z response_metadata['model']): {model_name}")
                elif hasattr(self.llm, 'model') and self.llm.model != "local-model":
                    model_name = self.llm.model
                    logger.info(f"📊 Nazwa modelu (z obiektu self.llm - placeholder?): {model_name}")
                else:
                    logger.warning("⚠️ Nie udało się uzyskać nazwy modelu z response_metadata ani obiektu LLM.")
                    model_name = "N/A" # Ostateczny fallback

                usage_meta = getattr(response, 'usage_metadata', None)
                if usage_meta and 'completion_tokens' in usage_meta:
                    token_count = usage_meta.get('completion_tokens')
                    logger.info(f"📊 Użycie tokenów (odpowiedź, z usage_metadata): {token_count}")
                elif resp_meta and 'token_usage' in resp_meta:
                    token_usage = resp_meta.get('token_usage', {})
                    token_count = token_usage.get('completion_tokens')
                    if token_count:
                         logger.info(f"📊 Użycie tokenów (odpowiedź, z response_metadata): {token_count}")
                    else:
                         logger.warning("⚠️ Nie znaleziono 'completion_tokens' w response_metadata['token_usage'].")
                         # Spróbujmy pobrać z ogólnego 'total_tokens', jeśli 'completion_tokens' brak
                         total_tokens = token_usage.get('total_tokens')
                         if total_tokens:
                             logger.info(f"📊 Użycie tokenów (całkowite, z response_metadata): {total_tokens}")
                             token_count = total_tokens # Użyjemy tego jako fallback, choć to nie to samo
                         else:
                             logger.warning("⚠️ Nie znaleziono 'total_tokens' w response_metadata['token_usage'].")
                else:
                    logger.warning("⚠️ Nie znaleziono danych o użyciu tokenów w 'usage_metadata' ani 'response_metadata'.")

            except AttributeError as attr_err:
                 logger.warning(f"⚠️ Obiekt odpowiedzi nie ma oczekiwanych atrybutów metadanych ({attr_err}).")
            except Exception as meta_e:
                 logger.warning(f"⚠️ Nieoczekiwany błąd podczas pobierania metadanych z odpowiedzi: {meta_e}.")

        except Exception as e:
            error_msg = f"Wystąpił błąd podczas generowania odpowiedzi przez LLM: {e}"
            logger.error(f"❌ {error_msg}", exc_info=True) # Dodano exc_info
            return {"content": error_msg, "metadata": None}

        return {
            "content": response_content,
            "metadata": {
                "model": model_name,
                "tokens": token_count,
                "time": response_time
            }
        }

    # ========================================================================
    # == TUTAJ JEST DEFINICJA METODY _extract_graph_from_response ==
    # == Upewnij się, że jest wcięta na tym samym poziomie co inne metody ==
    # ========================================================================
    def _extract_graph_from_response(self, text: str):
        """
        Używa LLM do ekstrakcji relacji z tekstu odpowiedzi i buduje graf NetworkX.
        Koncentruje się na relacjach osobowych, czasowych, adresowych i między dokumentami.

        Args:
            text (str): Tekst odpowiedzi wygenerowanej przez LLM.

        Returns:
            networkx.Graph or None: Zbudowany graf lub None w przypadku błędu/braku relacji.
        """
        # Sprawdzenie, czy LLM istnieje i tekst nie jest pusty
        if not hasattr(self, 'llm') or not text:
            logger.warning("⚠️ Pomijanie ekstrakcji grafu: LLM nie istnieje lub tekst jest pusty.")
            return None

        # Prosty prompt do ekstrakcji trójek
        extraction_prompt = f"""
        Przeanalizuj poniższy tekst i wyekstrahuj z niego relacje w formie trójek [podmiot, relacja, obiekt].
        Skup się na relacjach dotyczących:
        - Osób (np. kto co zrobił, kto gdzie pracuje, kto kogo zna).
        - Czasu (np. co kiedy się wydarzyło).
        - Adresów/Lokalizacji (np. co gdzie się znajduje).
        - Dokumentów (np. co jest wspomniane w dokumencie X, dokument Y dotyczy tematu Z).

        Zwróć wynik **WYŁĄCZNIE** jako obiekt JSON zawierający klucz "triples", którego wartością jest lista znalezionych trójek. Przykład:
        {{"triples": [["Jan Kowalski", "pracuje w", "XYZ Corp"], ["Raport.pdf", "opisuje", "Wyniki Q3"]]}}
        Jeśli nie znajdziesz żadnych istotnych relacji pasujących do kryteriów, zwróć: {{"triples": []}}

        Tekst do analizy:
        ---
        {text}
        ---

        JSON z wynikiem:
        """

        try:
            logger.info("➡️ Wywołanie LLM w celu ekstrakcji relacji dla grafu...")
            extraction_response = self.llm.invoke(extraction_prompt)
            logger.info("⬅️ Otrzymano odpowiedź ekstrakcji.")

            # Wyciągnij treść odpowiedzi
            response_text = ""
            if hasattr(extraction_response, 'content'):
                response_text = extraction_response.content
            elif isinstance(extraction_response, str):
                response_text = extraction_response
            else:
                 logger.error(f"❌ Nieoczekiwany format odpowiedzi ekstrakcji LLM: {type(extraction_response)}")
                 return None

            # Spróbuj sparsować JSON
            logger.info(f"   Odpowiedź ekstrakcji (surowa): {response_text[:300]}...") # Zwiększono długość logu

            # Czyszczenie potencjalnych ```json ... ```
            response_text_cleaned = response_text.strip()
            if response_text_cleaned.startswith("```json"):
                response_text_cleaned = response_text_cleaned[7:].strip()
                if response_text_cleaned.endswith("```"):
                     response_text_cleaned = response_text_cleaned[:-3].strip()
            elif response_text_cleaned.startswith("```"):
                 # Czasami brakuje 'json'
                 response_text_cleaned = response_text_cleaned[3:].strip()
                 if response_text_cleaned.endswith("```"):
                      response_text_cleaned = response_text_cleaned[:-3].strip()
            # Dodatkowe sprawdzenie - czasami LLM zwraca tylko JSON bez ```
            elif not (response_text_cleaned.startswith("{") and response_text_cleaned.endswith("}")):
                 # Jeśli nie wygląda jak JSON, spróbuj znaleźć JSON wewnątrz tekstu
                 json_start = response_text_cleaned.find('{')
                 json_end = response_text_cleaned.rfind('}')
                 if json_start != -1 and json_end != -1 and json_start < json_end:
                      response_text_cleaned = response_text_cleaned[json_start:json_end+1]
                      logger.info(f"   Znaleziono potencjalny JSON wewnątrz odpowiedzi: {response_text_cleaned[:100]}...")
                 else:
                      logger.error(f"❌ Odpowiedź ekstrakcji nie wygląda jak JSON i nie znaleziono w niej obiektu JSON: {response_text[:300]}...")
                      return None # Nie udało się znaleźć JSON

            # Parsowanie JSON
            extracted_data = json.loads(response_text_cleaned)

            if "triples" in extracted_data and isinstance(extracted_data["triples"], list):
                triples = extracted_data["triples"]
                if not triples:
                     logger.info("   LLM nie znalazł żadnych relacji do ekstrakcji (zwrócił pustą listę).")
                     return None # Zwracamy None, jeśli lista trójek jest pusta

                # Zbuduj graf NetworkX
                G = nx.DiGraph()
                triples_added = 0
                for triple in triples:
                    if isinstance(triple, list) and len(triple) == 3:
                        # Oczyść dane wejściowe - usuń białe znaki z początku/końca
                        subject = str(triple[0]).strip()
                        relation = str(triple[1]).strip()
                        obj = str(triple[2]).strip()

                        # Podstawowa walidacja - pomiń puste elementy
                        if subject and relation and obj:
                            G.add_node(subject, label=subject)
                            G.add_node(obj, label=obj)
                            G.add_edge(subject, obj, label=relation)
                            triples_added += 1
                        else:
                            logger.warning(f"   Pominięto trójkę z pustymi elementami: {triple}")
                    else:
                        logger.warning(f"   Pominięto nieprawidłowy format trójki: {triple}")

                if triples_added > 0:
                     logger.info(f"   Dodano {triples_added} trójek do grafu.")
                     return G
                else:
                     logger.info("   Nie dodano żadnych poprawnych trójek do grafu.")
                     return None # Zwracamy None, jeśli żadna trójka nie była poprawna

            else:
                logger.error(f"❌ Odpowiedź ekstrakcji LLM po czyszczeniu nie zawiera klucza 'triples' lub nie jest listą: {response_text_cleaned}")
                return None

        except json.JSONDecodeError as json_err:
            logger.error(f"❌ Błąd dekodowania JSON z odpowiedzi ekstrakcji LLM (po czyszczeniu): {json_err}")
            logger.error(f"   Tekst powodujący błąd: {response_text_cleaned}") # Loguj tekst, który zawiódł
            return None
        except Exception as e:
            logger.error(f"❌ Nieoczekiwany błąd podczas ekstrakcji grafu: {e}", exc_info=True) # Dodano exc_info
            return None
    # ========================================================================
    # == Koniec definicji metody _extract_graph_from_response ==
    # ========================================================================


    def clear_database(self):
        """Wywołuje metodę czyszczenia kolekcji w QdrantConnector."""
        logger.info("⚡ Żądanie wyczyszczenia bazy danych Qdrant...")
        success = self.qdrant_connector.clear_collection()
        if success:
            logger.info("✅ Baza danych Qdrant została wyczyszczona (kolekcja usunięta i utworzona na nowo).")
        else:
            logger.error("❌ Nie udało się wyczyścić bazy danych Qdrant.")
        return success

# Kod poza klasą
if __name__ == "__main__":
    logger.info("Inicjalizacja RAGChatbot w trybie skryptowym (test)...")
    try:
        # Inicjalizacja chatbota (może rzucić ConnectionError)
        chatbot = RAGChatbot()
        logger.info("✅ RAGChatbot zainicjalizowany pomyślnie.")

        # Przykładowe użycie (opcjonalne, głównie dla testów)
        # print("Próba przetworzenia dokumentu testowego (jeśli istnieje)...")
        # if os.path.exists("test_document.txt"):
        #     chatbot.process_document("test_document.txt")
        # else:
        #     print("   Plik 'test_document.txt' nie znaleziono. Pomiń przetwarzanie.")

        # Przykładowe zapytanie (jeśli chatbot się zainicjalizował)
        # print("\nPróba zadania pytania testowego...")
        # test_question = "Co wiesz o RAG?"
        # response_data, sources_list, graph_info = chatbot.query(test_question, "Odpowiedz zwięźle.")
        #
        # print(f"\nPytanie: {test_question}")
        # if response_data and response_data.get('content'):
        #     if isinstance(response_data['content'], AIMessage):
        #         print(f"Odpowiedź: {response_data['content'].content}")
        #     else:
        #         print(f"Odpowiedź: {response_data['content']}") # Np. komunikat błędu
        # else:
        #     print("Odpowiedź: Brak odpowiedzi lub błąd.")
        #
        # if response_data and response_data.get('metadata'):
        #     print(f"Metadane odpowiedzi: {response_data['metadata']}")
        #
        # print(f"Źródła: {sources_list}")
        # print(f"Dane grafu: {'Dostępne' if graph_info else 'Brak'}")
        # if graph_info:
        #      print(f"   Węzły: {len(graph_info.get('nodes', []))}, Krawędzie: {len(graph_info.get('links', []))}")

    except ConnectionError as ce:
        logger.error(f"❌ Nie udało się uruchomić chatbota: {ce}")
        print(f"Krytyczny błąd: Nie można połączyć się z Qdrant. Sprawdź, czy Qdrant działa i jest dostępny. Szczegóły w logach.")
    except ImportError as ie:
         logger.error(f"❌ Brakująca biblioteka: {ie}. Uruchom 'pip install -r requirements.txt'.")
         print(f"Brakująca biblioteka: {ie}. Zainstaluj wymagane pakiety.")
    except Exception as e:
        logger.error(f"❌ Nieoczekiwany błąd podczas inicjalizacji lub testu: {e}", exc_info=True)
        print(f"Wystąpił nieoczekiwany błąd: {e}")

    print("\n----------------------------------------------------")
    print("Aby uruchomić interfejs użytkownika, uruchom w terminalu:")
    print("streamlit run ui_streamlit.py")
    print("----------------------------------------------------")