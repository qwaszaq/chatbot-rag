# -*- coding: utf-8 -*-
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
# DODANO: Import dla Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
import spacy # Dodano import spacy
import networkx as nx # Dodano import networkx
import json # Dodano import json do parsowania odpowiedzi LLM
from langchain_core.messages import AIMessage # Dodano brakujący import
# DODANO: Import Counter do zliczania klastrów
from collections import Counter
# DODANO: Import typowania
from typing import Optional, Dict, List, Tuple, Any

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# UWAGA DOTYCZĄCA BEZPIECZEŃSTWA: Przechowywanie klucza API w kodzie jest niebezpieczne.
# Rozważ użycie zmiennych środowiskowych lub mechanizmu sekretów Streamlit.
GOOGLE_API_KEY = "AIzaSyAaOq6wmol8FR3DC39g2iuqF68I8_Edfj0" # <- Zastąp bezpiecznym mechanizmem

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
            max_tokens=10000,
            # Dostosuj inne parametry według potrzeb
        )

        # DODANO: Inicjalizacja modelu Google Gemini
        try:
            self.gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-04-17", # Poprawiona nazwa modelu flash
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                # convert_system_message_to_human=True # Może być potrzebne dla niektórych promptów systemowych
            )
            logger.info("✅ Pomyślnie zainicjalizowano model Google Gemini.")
        except Exception as e:
            logger.error(f"❌ Błąd inicjalizacji Google Gemini: {e}. Funkcjonalność Gemini będzie niedostępna.")
            self.gemini_llm = None # Ustaw na None w przypadku błędu

        # --- Ładowanie modelu spaCy dla NER ---
        try:
             # Użyjmy większego modelu dla potencjalnie lepszego NER
             self.nlp = spacy.load("pl_core_news_lg")
             logger.info("✅ Załadowano model spaCy 'pl_core_news_lg' dla NER.")
        except OSError:
             logger.warning("⚠️ Nie znaleziono modelu spaCy 'pl_core_news_lg'. Próbuję załadować 'pl_core_news_md'.")
             try:
                 self.nlp = spacy.load("pl_core_news_md")
                 logger.info("✅ Załadowano model spaCy 'pl_core_news_md' dla NER.")
             except OSError:
                 logger.error("❌ Nie znaleziono żadnego z modeli spaCy ('lg' ani 'md'). NER w grafie nie będzie dostępny.")
                 logger.error("Aby zainstalować model, uruchom w terminalu: python -m spacy download pl_core_news_lg")
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

            # --- Usunięto przetwarzanie NER PRZED zapisem - robimy to teraz podczas ekstrakcji grafu ---
            # Pozostawiamy tylko logikę zapisu oryginalnych chunków
            splits_to_save = all_splits
            # ---------------------------------------------------------------------------------------

            # Wygeneruj embeddingi i zapisz do Qdrant (oryginalne chunki)
            logger.info("🧠 Generowanie embeddingów i zapisywanie do Qdrant")
            self.qdrant_connector.add_documents(splits_to_save)

            logger.info("✅ Dokument został pomyślnie przetworzony i zapisany do Qdrant")
            return True

        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania dokumentu: {str(e)}")
            return False

    # DODANO: argument cluster_assignments i parametry rozszerzania
    def query(self, question: str, system_prompt_override: Optional[str] = None,
              cluster_assignments: Optional[Dict[str, int]] = None,
              expand_context_clusters: int = 1, expand_context_docs: int = 2) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Udziela odpowiedzi na pytanie korzystając z RAG.
        Opcjonalnie rozszerza kontekst o dodatkowe dokumenty z dominujących klastrów.

        Args:
            question (str): Pytanie użytkownika.
            system_prompt_override (str, optional): Prompt systemowy do użycia zamiast domyślnego.
            cluster_assignments (dict, optional): Słownik mapujący ID punktu na ID klastra.
                                                   Jeśli podany i niepusty, włącza logikę rozszerzania kontekstu.
            expand_context_clusters (int): Liczba dominujących klastrów do rozważenia przy rozszerzaniu kontekstu.
            expand_context_docs (int): Maksymalna liczba dodatkowych dokumentów do dodania z każdego dominującego klastra.

        Returns:
            tuple: Zawiera:
                   - dict: Słownik odpowiedzi z kluczami 'content' i 'metadata'.
                   - list[dict]: Lista słowników źródeł (max 3 unikalne), każdy z kluczami 'name' i 'id'.
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

            # Przygotowanie kontekstu i listy TOP 3 źródeł
            context_parts = []
            top_sources = []
            added_source_names = set()
            content_to_document = {doc.page_content: doc for doc in relevant_docs}
            initial_context_point_ids = set()
            reranked_docs_in_context = [] # Przechowuje obiekty Document, które trafiły do kontekstu

            logger.info(f"📝 Budowanie początkowego kontekstu i listy źródeł (max 3) z rerankowanych dokumentów...")
            for content, score in reranked_scored_docs:
                if score > self.relevance_threshold:
                    document = content_to_document.get(content)
                    if document:
                        context_parts.append(content)
                        reranked_docs_in_context.append(document) # Dodaj do listy użytych
                        metadata = document.metadata
                        if metadata:
                            point_id = metadata.get("id", metadata.get("_id"))
                            if point_id is not None:
                                initial_context_point_ids.add(point_id)
                            else:
                                logger.warning(f"   Nie znaleziono ID punktu ('id' lub '_id') w metadanych dla treści: {content[:50]}...")

                            source_name = metadata.get("source", metadata.get("file_path", metadata.get("filename", "nieznane źródło")))
                            if source_name not in added_source_names:
                                if len(top_sources) < 3:
                                    top_sources.append({"name": source_name, "id": point_id})
                                    added_source_names.add(source_name)
                        else:
                             logger.warning(f"   Dokument dla treści '{content[:50]}...' nie ma metadanych.")
                    else:
                         logger.warning(f"   Nie znaleziono obiektu Document dla treści '{content[:50]}...' w mapowaniu.")
                else:
                    pass # Dokument odrzucony przez reranker

            logger.info(f"✅ Zbudowano początkowy kontekst z {len(context_parts)} chunków.")
            logger.info(f"✅ Wybrano {len(top_sources)} unikalnych źródeł do wyświetlenia: {top_sources}")

            # === Logika rozszerzania kontekstu na podstawie klastrów ===
            if cluster_assignments and initial_context_point_ids:
                logger.info(f"🧩 Rozszerzanie kontekstu na podstawie klastrów (dominujące: {expand_context_clusters}, dodatkowe: {expand_context_docs})...")
                # Znajdź klastry dla punktów w początkowym kontekście
                context_clusters = [cluster_assignments.get(pid) for pid in initial_context_point_ids if cluster_assignments.get(pid) is not None and cluster_assignments.get(pid) != -1] # Ignoruj szum (-1)

                if context_clusters:
                    # Znajdź dominujące klastry
                    cluster_counts = Counter(context_clusters)
                    dominant_clusters = [cid for cid, count in cluster_counts.most_common(expand_context_clusters)]
                    logger.info(f"   Dominujące klastry w kontekście: {dominant_clusters}")

                    added_expansion_docs_count = 0
                    # Przejrzyj wszystkie *pierwotnie* relewantne dokumenty
                    for doc in relevant_docs:
                        metadata = doc.metadata
                        point_id = metadata.get("id", metadata.get("_id")) if metadata else None

                        # Sprawdź, czy dokument ma ID, należy do dominującego klastra i NIE był już w początkowym kontekście
                        if point_id and point_id in cluster_assignments and \
                           cluster_assignments[point_id] in dominant_clusters and \
                           point_id not in initial_context_point_ids:

                            if added_expansion_docs_count < expand_context_docs * len(dominant_clusters):
                                logger.info(f"   ➕ Dodawanie rozszerzenia kontekstu z klastra {cluster_assignments[point_id]} (ID: {point_id}): {doc.page_content[:100]}...")
                                context_parts.append(doc.page_content)
                                initial_context_point_ids.add(point_id) # Dodaj do użytych, aby nie powtórzyć
                                added_expansion_docs_count += 1
                            else:
                                logger.info("   Osiągnięto limit dodatkowych dokumentów do rozszerzenia kontekstu.")
                                break # Osiągnięto limit

                    if added_expansion_docs_count > 0:
                         logger.info(f"✅ Dodano {added_expansion_docs_count} dodatkowych chunków do kontekstu na podstawie klastrów.")
                else:
                    logger.info("   Brak informacji o klastrach (innych niż szum) dla dokumentów w początkowym kontekście.")
            elif cluster_assignments:
                 logger.info("ℹ️ Przypisania klastrów dostępne, ale brak punktów w początkowym kontekście do określenia dominujących klastrów.")
            else:
                logger.info("ℹ️ Brak przypisań klastrów, pomijanie rozszerzania kontekstu.")
            # === KONIEC LOGIKI ROZSZERZANIA KONTEKSTU ===


            if not context_parts:
                logger.info("❌ Brak dokumentów spełniających próg istotności po rerankingu (i ewentualnym rozszerzeniu).")
                return {"content": "Nie znaleziono wystarczająco istotnych informacji w dokumentach, aby odpowiedzieć na to pytanie.", "metadata": None}, [], None

            context = "\n\n".join(context_parts)
            logger.info(f"📊 Finalny kontekst ma długość: {len(context)} znaków ({len(context_parts)} chunków).")


            # Generowanie odpowiedzi
            logger.info("🤖 Generowanie odpowiedzi za pomocą LLM")
            if system_prompt_override is None:
                 logger.warning("⚠️ Nie podano system_prompt_override do metody query. Używanie pustego promptu systemowego.")
                 system_prompt_override = ""

            response_dict = self._generate_answer(question, context, system_prompt_text=system_prompt_override)
            sources = top_sources # Używamy zebranych źródeł (tylko top 3, nawet jeśli kontekst rozszerzony)

            logger.info("✅ Odpowiedź została pomyślnie wygenerowana wraz z metadanymi")

            # --- Ekstrakcja grafu wiedzy z odpowiedzi ---
            graph_data = None
            answer_text_content = None
            if response_dict and response_dict.get("content"):
                 content_data = response_dict["content"]
                 if isinstance(content_data, AIMessage):
                      answer_text_content = content_data.content
                 elif isinstance(content_data, str):
                      # Sprawdź, czy to nie jest komunikat o błędzie (brak metadanych)
                      if response_dict.get("metadata") is not None:
                           answer_text_content = content_data
                      else:
                           logger.warning("Pomijanie ekstrakcji grafu z komunikatu o błędzie LLM.")
                 else:
                      logger.warning(f"Nieoczekiwany typ treści odpowiedzi: {type(content_data)}. Pomijanie ekstrakcji grafu.")
            else:
                 logger.warning("Brak treści w odpowiedzi LLM. Pomijanie ekstrakcji grafu.")


            if answer_text_content:
                 logger.info("🕸️ Próba ekstrakcji grafu wiedzy z odpowiedzi LLM...")
                 try:
                      # Wywołanie metody ekstrakcji grafu
                      graph = self._extract_graph_from_response(answer_text_content)

                      if graph and graph.nodes:
                           # Serializacja grafu do formatu node-link oczekiwanego przez streamlit-agraph
                           graph_data = nx.node_link_data(graph)
                           logger.info(f"✅ Pomyślnie wyekstrahowano i zserializowano graf z {len(graph.nodes)} węzłami i {len(graph.edges)} krawędziami.")
                      else:
                           logger.info("ℹ️ Nie udało się wyekstrahować znaczących relacji do grafu z odpowiedzi lub LLM zwrócił pustą listę.")
                 except Exception as graph_e:
                      # Logujemy konkretny błąd, który wystąpił TUTAJ
                      logger.error(f"❌ Błąd podczas ekstrakcji lub serializacji grafu wiedzy: {graph_e}", exc_info=True)
            # -----------------------------------------

            return response_dict, sources, graph_data

        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}", exc_info=True)
            return {"content": f"Wystąpił błąd podczas przetwarzania pytania: {e}", "metadata": None}, [], None


    def _generate_answer(self, question, context, system_prompt_text):
        """
        Generuje odpowiedź na podstawie kontekstu i pytania przy użyciu modelu LLM (lokalnego),
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
             error_msg = "Lokalny model LLM (self.llm) nie został zainicjalizowany."
             logger.error(f"❌ {error_msg}")
             return {"content": error_msg, "metadata": None}

        response_content = None
        token_count = None
        model_name = "N/A"
        response_time = 0.0

        try:
            # Próba uzyskania nazwy modelu z obiektu, jeśli istnieje
            model_name = getattr(self.llm, 'model_name', getattr(self.llm, 'model', "Lokalny LLM (LM Studio)"))

            prompt_to_send = f"""{system_prompt_text}

            --- DOSTARCZONY KONTEKST ---
            {context}
            --- KONIEC KONTEKSTU ---

            --- PYTANIE UŻYTKOWNIKA ---
            {question}
            --- KONIEC PYTANIA ---

            --- ODPOWIEDŹ ANALITYKA (bazująca WYŁĄCZNIE na powyższym kontekście): ---
            """
            logger.info("➡️ Wysłanie ręcznie zbudowanego promptu do lokalnego LLM...")
            # logger.debug(f"   Pełny prompt (fragment): {prompt_to_send[:500]}...") # Opcjonalny debug

            start_time = time.time()
            # Używamy invoke, oczekujemy obiektu AIMessage lub podobnego
            response = self.llm.invoke(prompt_to_send)
            end_time = time.time()
            response_time = end_time - start_time

            response_content = response # Przechowujemy cały obiekt odpowiedzi
            logger.info(f"⬅️ Otrzymano odpowiedź od lokalnego LLM w {response_time:.2f}s.")

            # Wyciąganie metadanych z obiektu odpowiedzi LangChain
            try:
                # Sprawdź response_metadata dla nazwy modelu i użycia tokenów
                resp_meta = getattr(response, 'response_metadata', {})
                if resp_meta:
                    # Model
                    model_name_from_meta = resp_meta.get('model_name', resp_meta.get('model'))
                    if model_name_from_meta:
                        model_name = model_name_from_meta
                        logger.info(f"📊 Nazwa modelu (z response_metadata): {model_name}")
                    else:
                         logger.warning("⚠️ Nie znaleziono nazwy modelu w response_metadata dla lokalnego LLM.")

                    # Tokeny (szukamy w 'token_usage')
                    token_usage = resp_meta.get('token_usage', {})
                    if token_usage:
                        token_count = token_usage.get('completion_tokens')
                        if token_count is not None:
                             logger.info(f"📊 Użycie tokenów (odpowiedź, z response_metadata): {token_count}")
                        else:
                             # Fallback na total_tokens, jeśli completion_tokens brak
                             total_tokens = token_usage.get('total_tokens')
                             if total_tokens is not None:
                                 logger.info(f"📊 Użycie tokenów (całkowite, z response_metadata): {total_tokens}")
                                 token_count = total_tokens # Użyj jako fallback
                             else:
                                 logger.warning("⚠️ Nie znaleziono 'completion_tokens' ani 'total_tokens' w response_metadata['token_usage'] dla lokalnego LLM.")
                    else:
                         logger.warning("⚠️ Nie znaleziono 'token_usage' w response_metadata dla lokalnego LLM.")
                else:
                     logger.warning("⚠️ Brak 'response_metadata' w obiekcie odpowiedzi lokalnego LLM.")

                # Sprawdź usage_metadata jako alternatywę
                usage_meta = getattr(response, 'usage_metadata', None)
                if usage_meta and token_count is None: # Sprawdzamy tylko jeśli nie znaleźliśmy w response_metadata
                    token_count_usage = usage_meta.get('completion_tokens')
                    if token_count_usage is not None:
                        token_count = token_count_usage
                        logger.info(f"📊 Użycie tokenów (odpowiedź, z usage_metadata): {token_count}")
                    else:
                        total_tokens_usage = usage_meta.get('total_tokens')
                        if total_tokens_usage is not None:
                            logger.info(f"📊 Użycie tokenów (całkowite, z usage_metadata): {total_tokens_usage}")
                            token_count = total_tokens_usage # Fallback
                        else:
                            logger.warning("⚠️ Nie znaleziono 'completion_tokens' ani 'total_tokens' w usage_metadata dla lokalnego LLM.")
                elif token_count is None:
                    logger.warning("⚠️ Nie znaleziono danych o użyciu tokenów ani w 'response_metadata', ani w 'usage_metadata' dla lokalnego LLM.")


            except AttributeError as attr_err:
                 logger.warning(f"⚠️ Obiekt odpowiedzi lokalnego LLM nie ma oczekiwanych atrybutów metadanych ({attr_err}). Typ obiektu: {type(response)}")
            except Exception as meta_e:
                 logger.warning(f"⚠️ Nieoczekiwany błąd podczas pobierania metadanych z odpowiedzi lokalnego LLM: {meta_e}.")

        except Exception as e:
            error_msg = f"Wystąpił błąd podczas generowania odpowiedzi przez lokalny LLM: {e}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return {"content": error_msg, "metadata": None}

        return {
            "content": response_content,
            "metadata": {
                "model": model_name,
                "tokens": token_count,
                "time": response_time
            }
        }

    # DODANO: Nowa metoda do generowania odpowiedzi przez Gemini
    def _generate_gemini_answer(self, question: str) -> Dict[str, Any]:
        """
        Generuje odpowiedź na pytanie bezpośrednio przez model Google Gemini,
        dołączając metadane dotyczące generowania.

        Args:
            question (str): Pytanie użytkownika.

        Returns:
            dict: Słownik zawierający treść odpowiedzi i metadane
                  np. {"content": AIMessage(...), "metadata": {"model": "...", "tokens": ..., "time": ...}}
                  W przypadku błędu lub braku modelu Gemini zwraca słownik z komunikatem o błędzie.
        """
        if not hasattr(self, 'gemini_llm') or self.gemini_llm is None:
             error_msg = "Model Google Gemini (self.gemini_llm) nie został zainicjalizowany."
             logger.error(f"❌ {error_msg}")
             return {"content": error_msg, "metadata": None}

        response_content = None
        token_count = None
        model_name = "gemini-1.5-flash-preview-04-17" # Nazwa modelu użytego w init
        response_time = 0.0

        try:
            logger.info(f"➡️ Wysłanie promptu do Google Gemini API (model: {model_name})...")
            start_time = time.time()
            response = self.gemini_llm.invoke(question)
            end_time = time.time()
            response_time = end_time - start_time

            response_content = response # Przechowujemy cały obiekt odpowiedzi
            logger.info(f"⬅️ Otrzymano odpowiedź od Gemini API w {response_time:.2f}s.")

            # Wyciąganie metadanych z obiektu odpowiedzi Gemini (jeśli dostępne)
            try:
                # Gemini API (przez langchain-google-genai) może zwracać metadane w 'response_metadata'
                resp_meta = getattr(response, 'response_metadata', {})
                if resp_meta:
                    # Tokeny - szukamy w 'usage_metadata' wewnątrz 'response_metadata'
                    usage_metadata = resp_meta.get('usage_metadata', {})
                    if usage_metadata:
                        # Gemini zwraca 'prompt_token_count', 'candidates_token_count', 'total_token_count'
                        token_count = usage_metadata.get('candidates_token_count') # Liczba tokenów w odpowiedzi
                        if token_count is not None:
                            logger.info(f"📊 Użycie tokenów (odpowiedź, z usage_metadata): {token_count}")
                        else:
                            total_tokens = usage_metadata.get('total_token_count')
                            if total_tokens is not None:
                                logger.info(f"📊 Użycie tokenów (całkowite, z usage_metadata): {total_tokens}")
                                token_count = total_tokens # Użyj jako fallback, jeśli completion brak
                            else:
                                logger.warning("⚠️ Nie znaleziono 'candidates_token_count' ani 'total_token_count' w usage_metadata dla Gemini.")
                    else:
                         logger.warning("⚠️ Nie znaleziono 'usage_metadata' w response_metadata dla Gemini.")
                else:
                     logger.warning("⚠️ Brak 'response_metadata' w obiekcie odpowiedzi Gemini.")

                # Alternatywnie, nowsze wersje mogą mieć 'usage_metadata' bezpośrednio na obiekcie response
                usage_meta_direct = getattr(response, 'usage_metadata', None)
                if usage_meta_direct and token_count is None:
                    token_count_usage = usage_meta_direct.get('candidates_token_count')
                    if token_count_usage is not None:
                        token_count = token_count_usage
                        logger.info(f"📊 Użycie tokenów (odpowiedź, z usage_metadata - direct): {token_count}")
                    else:
                        total_tokens_usage = usage_meta_direct.get('total_token_count')
                        if total_tokens_usage is not None:
                            logger.info(f"📊 Użycie tokenów (całkowite, z usage_metadata - direct): {total_tokens_usage}")
                            token_count = total_tokens_usage # Fallback
                        else:
                            logger.warning("⚠️ Nie znaleziono tokenów w usage_metadata (direct) dla Gemini.")
                elif token_count is None:
                     logger.warning("⚠️ Nie znaleziono danych o użyciu tokenów w odpowiedzi Gemini.")

            except AttributeError as attr_err:
                 logger.warning(f"⚠️ Obiekt odpowiedzi Gemini nie ma oczekiwanych atrybutów metadanych ({attr_err}). Typ obiektu: {type(response)}")
            except Exception as meta_e:
                 logger.warning(f"⚠️ Nieoczekiwany błąd podczas pobierania metadanych z odpowiedzi Gemini: {meta_e}.")

        except Exception as e:
            error_msg = f"Wystąpił błąd podczas wywołania Google Gemini API: {e}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return {"content": error_msg, "metadata": None}

        return {
            "content": response_content,
            "metadata": {
                "model": model_name,
                "tokens": token_count,
                "time": response_time
            }
        }


    def _extract_graph_from_response(self, text: str):
        """
        Używa LLM do ekstrakcji relacji z tekstu odpowiedzi i buduje graf NetworkX,
        dodając typy NER do węzłów za pomocą spaCy.

        Args:
            text (str): Tekst odpowiedzi wygenerowanej przez LLM.

        Returns:
            networkx.DiGraph or None: Zbudowany graf lub None w przypadku błędu/braku relacji.
        """
        # Sprawdzenie, czy LLM istnieje i tekst nie jest pusty
        if not hasattr(self, 'llm') or not text:
            logger.warning("⚠️ Pomijanie ekstrakcji grafu: LLM nie istnieje lub tekst jest pusty.")
            return None

        # Sprawdzenie, czy model spaCy jest dostępny (dla NER)
        if not self.nlp:
            logger.warning("⚠️ Model spaCy (self.nlp) nie jest załadowany. Graf zostanie utworzony bez typów NER.")

        # Prompt do ekstrakcji trójek (bez zmian)
        extraction_prompt = f"""
        Przeanalizuj poniższy tekst i wyekstrahuj z niego relacje w formie trójek [podmiot, relacja, obiekt].
        Skup się na relacjach dotyczących:
        - Osób (np. kto co zrobił, kto gdzie pracuje, kto kogo zna).
        - Organizacji (np. firma X zrobiła Y, relacje między firmami).
        - Czasu (np. co kiedy się wydarzyło, daty spotkań).
        - Adresów/Lokalizacji (np. co gdzie się znajduje, spotkanie w miejscu X).
        - Dokumentów (np. co jest wspomniane w dokumencie X, dokument Y dotyczy tematu Z).

        Zwróć wynik **WYŁĄCZNIE** jako obiekt JSON zawierający klucz "triples", którego wartością jest lista znalezionych trójek. Każdy element trójki powinien być stringiem. Przykład:
        {{"triples": [["Jan Kowalski", "pracuje w", "XYZ Corp"], ["Raport Finansowy Q3", "opisuje", "Wyniki sprzedaży"], ["Spotkanie", "odbędzie się", "15 Listopada"]]}}
        Jeśli nie znajdziesz żadnych istotnych relacji pasujących do kryteriów, zwróć pustą listę: {{"triples": []}}

        Tekst do analizy:
        ---
        {text}
        ---

        JSON z wynikiem:
        """

        try:
            logger.info("➡️ Wywołanie LLM w celu ekstrakcji relacji dla grafu (max_tokens=10000)...")
            extraction_response = self.llm.invoke(extraction_prompt, config={"max_tokens": 10000})
            logger.info("⬅️ Otrzymano odpowiedź ekstrakcji.")

            response_text = ""
            if hasattr(extraction_response, 'content'):
                response_text = extraction_response.content
            elif isinstance(extraction_response, str):
                response_text = extraction_response
            else:
                 logger.error(f"❌ Nieoczekiwany format odpowiedzi ekstrakcji LLM: {type(extraction_response)}")
                 return None

            logger.info(f"   Odpowiedź ekstrakcji (surowa): {response_text[:500]}...") # Zwiększono długość logu

            # Czyszczenie odpowiedzi (usunięcie znaczników ```json, itp.)
            response_text_cleaned = response_text.strip()
            if response_text_cleaned.startswith("```json"):
                 response_text_cleaned = response_text_cleaned[7:]
                 if response_text_cleaned.endswith("```"):
                      response_text_cleaned = response_text_cleaned[:-3]
            elif response_text_cleaned.startswith("```"):
                 response_text_cleaned = response_text_cleaned[3:]
                 if response_text_cleaned.endswith("```"):
                      response_text_cleaned = response_text_cleaned[:-3]

            response_text_cleaned = response_text_cleaned.strip()
            json_start = response_text_cleaned.find('{')
            json_end = response_text_cleaned.rfind('}')

            if json_start != -1 and json_end != -1 and json_start < json_end:
                 response_text_cleaned = response_text_cleaned[json_start:json_end+1]
                 logger.info("   Tekst po czyszczeniu i przycięciu do JSON: {...}")
            else:
                 logger.error(f"❌ Nie znaleziono poprawnej struktury JSON w odpowiedzi (po czyszczeniu): {response_text_cleaned[:500]}...")
                 return None

            # Parsowanie JSON
            extracted_data = json.loads(response_text_cleaned)

            if "triples" in extracted_data and isinstance(extracted_data["triples"], list):
                triples = extracted_data["triples"]
                if not triples:
                     logger.info("   LLM nie znalazł żadnych relacji do ekstrakcji (zwrócił pustą listę).")
                     return None # Zwracamy None, jeśli lista trójek jest pusta

                # --- Budowanie grafu NetworkX z dodaniem NER ---
                G = nx.DiGraph()
                nodes_added = set() # Zbiór do śledzenia dodanych węzłów, aby nie przetwarzać NER wielokrotnie
                triples_added_count = 0 # Licznik poprawnie dodanych trójek

                for triple in triples:
                    if isinstance(triple, list) and len(triple) == 3:
                        # Oczyść i zwaliduj elementy trójki
                        subject = str(triple[0]).strip() if triple[0] is not None else ""
                        relation = str(triple[1]).strip() if triple[1] is not None else ""
                        obj = str(triple[2]).strip() if triple[2] is not None else ""

                        # Pomiń trójki z pustymi elementami kluczowymi
                        if not subject or not relation or not obj:
                            logger.warning(f"   Pominięto trójkę z brakującymi elementami: {triple}")
                            continue

                        # Przetwarzanie węzłów (Subject i Object)
                        for node_text in [subject, obj]:
                            if node_text not in nodes_added:
                                ner_type = None # Domyślnie brak typu
                                if self.nlp: # Sprawdź, czy model spaCy jest dostępny
                                    try:
                                        # Przetwórz tekst węzła przez spaCy
                                        doc_node = self.nlp(node_text)
                                        if doc_node.ents:
                                            # Pobierz etykietę *pierwszej* znalezionej encji
                                            # UWAGA: Modele 'pl_core_news' zwracają etykiety jak 'persName', 'orgName', 'geogName', 'placeName', 'date'
                                            ner_type = doc_node.ents[0].label_
                                            logger.info(f"   NER dla węzła '{node_text}': Rozpoznano typ '{ner_type}'")
                                        else:
                                            logger.info(f"   NER dla węzła '{node_text}': Nie rozpoznano konkretnej encji.")
                                    except Exception as ner_exc:
                                        logger.error(f"   Błąd podczas NER dla tekstu '{node_text}': {ner_exc}")
                                        ner_type = "ERROR" # Oznacz błąd NER

                                # Dodaj węzeł do grafu z etykietą i typem NER
                                # Atrybut 'label' to tekst węzła, 'ner_type' to etykieta spaCy
                                G.add_node(node_text, label=node_text, ner_type=ner_type)
                                nodes_added.add(node_text) # Dodaj do zbioru przetworzonych

                        # Dodaj krawędź do grafu z etykietą relacji
                        G.add_edge(subject, obj, label=relation)
                        triples_added_count += 1

                    else:
                        logger.warning(f"   Pominięto nieprawidłowy format trójki: {triple}")

                if triples_added_count > 0:
                     logger.info(f"   Dodano {triples_added_count} poprawnych trójek do grafu.")
                     # Zwróć zbudowany graf (zostanie zserializowany w metodzie query)
                     return G
                else:
                     logger.info("   Nie dodano żadnych poprawnych trójek do grafu.")
                     return None # Zwracamy None, jeśli żadna trójka nie była poprawna
                # --------------------------------------------------
            else:
                logger.error(f"❌ Odpowiedź ekstrakcji LLM (po czyszczeniu) nie zawiera klucza 'triples' lub wartość nie jest listą: {response_text_cleaned}")
                return None

        except json.JSONDecodeError as json_err:
            logger.error(f"❌ Błąd dekodowania JSON z odpowiedzi ekstrakcji LLM: {json_err}")
            logger.error(f"   Tekst powodujący błąd (po czyszczeniu): '{response_text_cleaned}'")
            return None
        except Exception as e:
            logger.error(f"❌ Nieoczekiwany błąd podczas ekstrakcji grafu: {e}", exc_info=True)
            return None


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

        # Kod testowy przeniesiony do test_clustering_data.py
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
    print("streamlit run ui_streamlit.py") # Zakładając, że plik UI nazywa się ui_streamlit.py
    print("----------------------------------------------------")