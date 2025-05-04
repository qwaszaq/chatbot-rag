"""
Moduł do integracji z bazą danych Qdrant
Implementuje interfejs LangChain Embeddings.
"""
# Zmieniono import Qdrant na ten z nowej biblioteki langchain-qdrant
from langchain_qdrant import Qdrant
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
# Zmieniono import CollectionInfo i CollectionStatus na ten z qdrant_client.http.models
# Upewnij się, że te klasy są dostępne w qdrant_client.http.models w Twojej wersji qdrant-client
from qdrant_client.http.models import (
    Distance, VectorParams, CollectionInfo, CollectionStatus,
    PointStruct, UpdateResult, PointIdsList, Payload,
    Filter, FieldCondition, MatchValue # DODANO: Importy dla filtrowania
)
import os
import logging
from typing import Dict, List, Optional # DODANO: Import Dict, List i Optional
import time # DODANO: Import time

logger = logging.getLogger(__name__)

class QdrantConnector:
    # Zmieniono domyślną nazwę kolekcji na "nowa1"
    def __init__(self, host="localhost", port=6333, collection_name="nowa1", vector_size=1024):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        self.vectorstore = None

    def initialize(self, embeddings: Embeddings):
        """Inicjalizuje połączenie z Qdrant i tworzy kolekcję jeśli nie istnieje"""
        try:
            logger.info(f"Attempting to connect to Qdrant at {self.host}:{self.port}")
            # Nawiązanie połączenia z Qdrant
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info("✅ Connected to Qdrant.")

            # Sprawdzenie czy kolekcja istnieje
            # Use try-except to check for collection existence gracefully
            collection_exists = False
            try:
                collection_info = self.client.get_collection(collection_name=self.collection_name)
                # Check if collection is active
                if collection_info.status != CollectionStatus.GREEN:
                     logger.warning(f"⚠️ Kolekcja '{self.collection_name}' istnieje, ale nie jest w stanie GREEN. Status: {collection_info.status}")
                     # Depending on status, might need to wait or recreate. For now, proceed.

                collection_exists = True
                logger.info(f"✅ Kolekcja w Qdrant '{self.collection_name}' już istnieje.")

            except Exception as e: # Catch specific exceptions like UnexpectedResponse if possible, but broad Exception is safer for now
                # If get_collection raises an error (e.g., 404 not found), the collection doesn't exist
                logger.info(f"ℹ️ Kolekcja '{self.collection_name}' nie istnieje lub wystąpił błąd przy jej pobieraniu: {e}. Spróbuję utworzyć.")
                collection_exists = False


            if not collection_exists:
                logger.info(f"✨ Tworzenie nowej kolekcji w Qdrant: {self.collection_name} z rozmiarem wektora {self.vector_size}")
                # Utworzenie nowej kolekcji
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"✅ Kolekcja '{self.collection_name}' utworzona pomyślnie z rozmiarem wektora {self.vector_size}.")

            # Utworzenie instancji Qdrant dla tej kolekcji
            # Przekazujemy obiekt embeddings tutaj. LangChain/Qdrant client powinien obsłużyć
            # połączenie z istniejącą lub nowo utworzoną kolekcją.
            logger.info(f"Initializing LangChain Qdrant vectorstore for collection '{self.collection_name}'")
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=embeddings # Przekazujemy instancję EmbeddingGenerator (która implementuje Embeddings)
            )
            logger.info("✅ LangChain Qdrant vectorstore initialized.")
            return True
        except Exception as e:
            # Catch any exception during initialization
            logger.error(f"❌ Błąd podczas inicjalizacji Qdrant: {str(e)}")
            return False

    def add_documents(self, documents):
        """Dodaje dokumenty do bazy. Vectorstore użyje swojego wewnętrznego embedding generatora."""
        if not self.vectorstore:
            logger.error("❌ Qdrant vectorstore nie jest zainicjalizowany. Nie można dodać dokumentów.")
            raise Exception("Qdrant vectorstore nie jest zainicjalizowany")

        logger.info(f"📥 Dodawanie {len(documents)} dokumentów do Qdrant.")
        try:
            # Wywołujemy add_documents na vectorstore, który ma już dostęp do embedding generatora
            # Zwraca listę identyfikatorów punktów
            result = self.vectorstore.add_documents(documents)
            logger.info(f"✅ Dodano {len(result)} dokumentów do Qdrant.")
            return result # Return the result from add_documents
        except Exception as e:
            logger.error(f"❌ Błąd podczas dodawania dokumentów do Qdrant: {str(e)}")
            raise # Propaguj błąd dalej


    def similarity_search(self, query, k=99, filter_cluster_id: Optional[int] = None):
        """
        Wyszukiwanie podobnych dokumentów z opcjonalnym filtrowaniem po cluster_id.

        Args:
            query (str): Tekst zapytania.
            k (int): Liczba wyników do zwrócenia.
            filter_cluster_id (Optional[int]): ID klastra do filtrowania. Jeśli None, brak filtrowania.

        Returns:
            List[Document]: Lista znalezionych dokumentów LangChain.
        """
        if not self.vectorstore:
            logger.error("❌ Qdrant vectorstore nie jest zainicjalizowany. Nie można wyszukać.")
            raise Exception("Qdrant vectorstore nie jest zainicjalizowany")

        query_filter = None
        log_filter_msg = ""
        if filter_cluster_id is not None:
            logger.info(f"   Applying filter for cluster_id: {filter_cluster_id}")
            # Upewnij się, że importujesz Filter, FieldCondition, MatchValue
            # DODANO: Logowanie i konwersja typu PRZED utworzeniem FieldCondition
            logger.debug(f"   Type of filter_cluster_id before int(): {type(filter_cluster_id)}, value: {filter_cluster_id}")
            try:
                # Logowanie typu przed konwersją
                logger.debug(f"   Type of filter_cluster_id before conversion: {type(filter_cluster_id)}, value: {filter_cluster_id}")

                # Użyj .item() do konwersji typów NumPy na natywne typy Python
                # lub standardowego int() dla innych typów
                if hasattr(filter_cluster_id, 'item'): # Sprawdź, czy to typ NumPy
                    converted_cluster_id = filter_cluster_id.item()
                else:
                    converted_cluster_id = int(filter_cluster_id) # Standardowa konwersja dla innych typów

                logger.debug(f"   Type of converted_cluster_id after conversion: {type(converted_cluster_id)}, value: {converted_cluster_id}")

                # Dodatkowe sprawdzenie, czy wynik jest faktycznie int
                if not isinstance(converted_cluster_id, int):
                     raise TypeError(f"Converted cluster ID is not a standard Python int after conversion: type={type(converted_cluster_id)}")

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="cluster_id", # Klucz w payloadzie
                            match=MatchValue(value=converted_cluster_id) # Użyj skonwertowanej wartości
                        )
                    ]
                )
            except (ValueError, TypeError) as conv_err:
                 logger.error(f"   Błąd konwersji filter_cluster_id na int: {conv_err}. Wyszukiwanie bez filtra klastra.")
                 query_filter = None # W razie błędu konwersji, nie filtruj
                 log_filter_msg = "" # Usuń informację o filtrze z logu

            log_filter_msg = f" z filtrem cluster_id={filter_cluster_id}"

        logger.info(f"🔍 Wyszukiwanie {k} podobnych dokumentów w Qdrant dla zapytania: '{query[:50]}'..." + log_filter_msg)
        try:
            # Używamy similarity_search_with_score, aby potencjalnie mieć dostęp do score'ów
            # Ważne: przekazujemy filtr do metody wyszukiwania vectorstore
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=query_filter # Przekazanie filtra
            )
            # Wynik to lista krotek (Document, score), bierzemy tylko dokumenty
            results = [doc for doc, score in results_with_scores]
            logger.info(f"✅ Znaleziono {len(results)} dokumentów" + log_filter_msg + ".")
            return results
        except Exception as e:
            logger.error(f"❌ Błąd podczas wyszukiwania w Qdrant{log_filter_msg}: {str(e)}", exc_info=True)
            raise # Propaguj błąd dalej

    def get_client(self):
        """Zwraca klienta Qdrant"""
        return self.client

    def get_vectorstore(self):
        """Zwraca vectorstore Qdrant"""
        return self.vectorstore

    def clear_collection(self):
        """Usuwa i tworzy na nowo kolekcję Qdrant, efektywnie czyszcząc jej zawartość."""
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można wyczyścić kolekcji.")
            return False

        collection_name = self.collection_name

        logger.warning(f"🗑️ Próba usunięcia WSZYSTKICH punktów z kolekcji: {collection_name}")
        try:
            # Sprawdź, czy kolekcja istnieje i usuń ją
            try:
                self.client.get_collection(collection_name=collection_name)
                logger.info(f"   Kolekcja '{collection_name}' istnieje. Usuwanie kolekcji...")
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"   Usunięto kolekcję '{collection_name}'.")
            except Exception as e:
                 # Jeśli kolekcja nie istnieje, get_collection rzuci wyjątek.
                 logger.info(f"   Kolekcja '{collection_name}' nie istniała lub wystąpił błąd przy sprawdzaniu/usuwaniu: {e}.")
                 # Możemy kontynuować, bo celem jest i tak stworzenie nowej.

            # Utwórz kolekcję na nowo
            vector_size = self.vector_size # Pobierz rozmiar z instancji
            logger.info(f"   Tworzenie nowej, pustej kolekcji: {collection_name} z rozmiarem wektora {vector_size}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"✅ Pomyślnie wyczyszczono (usunięto i utworzono na nowo) kolekcję '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"❌ Błąd podczas czyszczenia (usuwania/tworzenia) kolekcji '{collection_name}': {str(e)}")
            return False

    # === POPRAWNIE DODANA METODA ===
    def get_all_data_for_clustering(self, limit=None, with_vectors=True, with_payload=True):
        """
        Pobiera dane punktów (ID, wektory, payload/metadane) z kolekcji Qdrant,
        używając metody scroll do efektywnego pobierania dużych zbiorów.

        Args:
            limit (int, optional): Maksymalna liczba punktów do pobrania. Domyślnie None (pobierz wszystkie).
            with_vectors (bool): Czy pobierać wektory embeddingów. Domyślnie True.
            with_payload (bool): Czy pobierać payload (metadane). Domyślnie True.

        Returns:
            list[dict]: Lista słowników, gdzie każdy słownik reprezentuje punkt
                        i zawiera klucze 'id', 'vector', 'payload'.
                        Zwraca pustą listę w przypadku błędu lub braku punktów.
        """
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można pobrać danych.")
            return []

        logger.info(f"⬇️ Pobieranie danych z kolekcji '{self.collection_name}' do klastrowania (limit: {limit})...")
        all_points_data = []
        next_page_offset = None
        processed_count = 0

        try:
            while True:
                # Używamy scroll API do pobierania partiami
                # Poprawka: import rest musi być w zasięgu
                from qdrant_client.http import models as rest
                response, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(1000, limit - processed_count) if limit is not None else 1000, # Poprawka obsługi limitu
                    offset=next_page_offset,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                )

                if not response: # Jeśli nie ma więcej wyników
                    break

                # Przetwarzanie pobranych rekordów
                for record in response:
                     point_data = {
                         "id": record.id,
                         # Używamy getattr z domyślną wartością None, na wypadek gdyby wektor/payload nie został pobrany
                         "vector": getattr(record, 'vector', None) if with_vectors else None,
                         "payload": getattr(record, 'payload', None) if with_payload else None
                     }
                     all_points_data.append(point_data)
                     processed_count += 1

                # Sprawdź, czy osiągnięto limit
                if limit is not None and processed_count >= limit:
                    logger.info(f"   Osiągnięto limit {limit} pobranych punktów.")
                    break

                # Jeśli next_page_offset jest None, to koniec danych
                if next_page_offset is None:
                    break

            logger.info(f"✅ Pomyślnie pobrano {len(all_points_data)} punktów z kolekcji '{self.collection_name}'.")
            return all_points_data

        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania danych z Qdrant za pomocą scroll: {str(e)}")
            return [] # Zwróć pustą listę w przypadku błędu
    # === KONIEC POPRAWNIE DODANEJ METODY ===

    # === POPRAWIONA METODA AKTUALIZACJI PAYLOADU ===
    def update_payload_with_cluster_ids(self, assignments: Dict[str, int]):
        """
        Aktualizuje pole 'cluster_id' w payloadzie punktów w Qdrant na podstawie
        słownika przypisań przy użyciu client.set_payload (wsadowo).

        Args:
            assignments (Dict[str, int]): Słownik mapujący ID punktu (str) na ID klastra (int).

        Returns:
            bool: True jeśli operacja się powiodła (lub nie było nic do zrobienia),
                  False w przypadku błędu.
        """
        if not assignments:
            logger.info("ℹ️ Brak przypisań klastrów do zaktualizowania w Qdrant.")
            return True
        if not self.client:
            logger.error("❌ Błąd: Klient Qdrant nie jest zainicjalizowany w update_payload_with_cluster_ids.")
            return False

        logger.info(f"⚙️ Rozpoczynanie aktualizacji payloadów dla {len(assignments)} punktów w Qdrant (pole: cluster_id)...")
        start_time = time.time()

        # Przygotuj listy ID punktów i odpowiadających im payloadów (tylko pole cluster_id)
        point_ids = list(assignments.keys())
        # Upewnij się, że cluster_id jest typu int, a nie np. np.int32
        payloads_to_set = [{"cluster_id": int(cluster_id)} for cluster_id in assignments.values()]

        try:
            # WAŻNE: Upewnij się, że importujesz Payload i UpdateResult (zrobione na górze pliku)

            # Aktualizuj payload wsadowo używając set_payload
            update_result: UpdateResult = self.client.set_payload(
                collection_name=self.collection_name,
                payload=payloads_to_set, # Lista payloadów
                points=point_ids,       # Lista odpowiadających ID punktów
                wait=True               # Poczekaj na zakończenie
            )

            end_time = time.time()
            logger.info(f"⏱️ Aktualizacja payloadów Qdrant (set_payload wsadowo) zajęła: {end_time - start_time:.2f}s")

            update_status = getattr(update_result, 'status', 'unknown')
            if update_status in ["completed", "acknowledged"]:
                logger.info(f"✅ Pomyślnie ustawiono payload (cluster_id) dla {len(point_ids)} punktów w Qdrant.")
                return True
            else:
                logger.error(f"❌ Ustawienie payloadów Qdrant zakończone statusem: {update_status}")
                return False
        except Exception as e:
             logger.error(f"❌ Błąd podczas ustawiania payloadów w Qdrant (set_payload wsadowo): {e}", exc_info=True)
             return False
    # === KONIEC POPRAWIONEJ METODY AKTUALIZACJI PAYLOADU ===
    
    # === METODY DO ZARZĄDZANIA KOLEKCJAMI ===
    def get_collections(self):
        """
        Pobiera listę wszystkich kolekcji z serwera Qdrant wraz z dodatkowymi informacjami.
        
        Returns:
            dict: Słownik z nazwami kolekcji jako kluczami i informacjami o kolekcji jako wartościami.
                 Format: {
                     "nazwa_kolekcji": {
                         "name": "nazwa_kolekcji",
                         "vector_size": 1024,
                         "points_count": 100,
                         "status": "green"
                     },
                     ...
                 }
        """
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można pobrać listy kolekcji.")
            return {}
            
        try:
            collections_list = self.client.get_collections().collections
            collections_info = {}
            
            for collection in collections_list:
                collection_name = collection.name
                try:
                    # Pobierz szczegółowe informacje o kolekcji
                    details = self.client.get_collection(collection_name=collection_name)
                    vector_size = details.config.params.vectors.size
                    points_count = details.vectors_count
                    
                    collections_info[collection_name] = {
                        "name": collection_name,
                        "vector_size": vector_size,
                        "points_count": points_count,
                        "status": str(details.status)
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Błąd podczas pobierania szczegółów kolekcji {collection_name}: {e}")
                    collections_info[collection_name] = {
                        "name": collection_name,
                        "status": "error"
                    }
                    
            logger.info(f"✅ Pobrano informacje o {len(collections_info)} kolekcjach.")
            return collections_info
        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania listy kolekcji: {e}")
            return {}
            
    def switch_collection(self, new_collection_name, vector_size=None):
        """
        Przełącza na inną kolekcję bez konieczności tworzenia nowego obiektu QdrantConnector.
        
        Args:
            new_collection_name (str): Nazwa kolekcji, na którą należy się przełączyć.
            vector_size (int, optional): Rozmiar wektora do użycia, jeśli kolekcja nie istnieje.
                                         Jeśli None, używany jest bieżący rozmiar.
                                         
        Returns:
            bool: True jeśli operacja się powiodła, False w przypadku błędu.
        """
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można przełączyć kolekcji.")
            return False
            
        if new_collection_name == self.collection_name and self.vectorstore is not None:
            logger.info(f"ℹ️ Już używasz kolekcji '{new_collection_name}'. Nic nie zmieniono.")
            return True
            
        # Zapamiętaj stare wartości na wypadek błędu
        old_collection_name = self.collection_name
        old_vectorstore = self.vectorstore
        
        try:
            # Ustaw nową nazwę kolekcji
            self.collection_name = new_collection_name
            
            # Ustaw rozmiar wektora
            if vector_size is not None:
                self.vector_size = vector_size
                
            # Sprawdź, czy kolekcja istnieje
            collection_exists = False
            try:
                collection_info = self.client.get_collection(collection_name=new_collection_name)
                collection_exists = True
                # Aktualizuj rozmiar wektora na podstawie istniejącej kolekcji
                self.vector_size = collection_info.config.params.vectors.size
                logger.info(f"✅ Kolekcja '{new_collection_name}' istnieje. Rozmiar wektora: {self.vector_size}")
            except Exception as e:
                logger.info(f"ℹ️ Kolekcja '{new_collection_name}' nie istnieje lub wystąpił błąd: {e}")
                collection_exists = False
                
            # Jeśli kolekcja nie istnieje, utwórz ją
            if not collection_exists:
                logger.info(f"✨ Tworzenie nowej kolekcji '{new_collection_name}' z rozmiarem wektora {self.vector_size}")
                self.client.create_collection(
                    collection_name=new_collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"✅ Kolekcja '{new_collection_name}' utworzona pomyślnie.")
                
            # Zaktualizuj vectorstore dla nowej kolekcji
            embeddings = None
            if self.vectorstore:
                embeddings = self.vectorstore.embeddings
                
            if embeddings is not None:
                self.vectorstore = Qdrant(
                    client=self.client,
                    collection_name=new_collection_name,
                    embeddings=embeddings
                )
                
                logger.info(f"✅ Pomyślnie przełączono na kolekcję '{new_collection_name}'.")
                return True
            else:
                logger.error(f"❌ Nie można zaktualizować vectorstore - brak obiektu embeddings.")
                # Przywróć poprzednie wartości
                self.collection_name = old_collection_name
                self.vectorstore = old_vectorstore
                return False
        except Exception as e:
            # W przypadku błędu przywróć poprzednie wartości
            self.collection_name = old_collection_name
            self.vectorstore = old_vectorstore
            logger.error(f"❌ Błąd podczas przełączania na kolekcję '{new_collection_name}': {e}")
            return False
            
    def create_collection(self, collection_name, vector_size=1024):
        """
        Tworzy nową kolekcję o podanej nazwie i rozmiarze wektora, bez przełączania się na nią.
        
        Args:
            collection_name (str): Nazwa nowej kolekcji.
            vector_size (int): Rozmiar wektora dla nowej kolekcji.
            
        Returns:
            bool: True jeśli operacja się powiodła, False w przypadku błędu.
        """
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można utworzyć kolekcji.")
            return False
            
        try:
            # Sprawdź, czy kolekcja już istnieje
            try:
                self.client.get_collection(collection_name=collection_name)
                logger.warning(f"⚠️ Kolekcja '{collection_name}' już istnieje. Nie utworzono nowej.")
                return False
            except Exception:
                # Kolekcja nie istnieje, możemy ją utworzyć
                pass
                
            # Utwórz nową kolekcję
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            logger.info(f"✅ Utworzono nową kolekcję '{collection_name}' z rozmiarem wektora {vector_size}.")
            return True
        except Exception as e:
            logger.error(f"❌ Błąd podczas tworzenia kolekcji '{collection_name}': {e}")
            return False
            
    def delete_collection(self, collection_name):
        """
        Usuwa kolekcję o podanej nazwie.
        
        Args:
            collection_name (str): Nazwa kolekcji do usunięcia.
            
        Returns:
            bool: True jeśli operacja się powiodła, False w przypadku błędu.
        """
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można usunąć kolekcji.")
            return False
            
        try:
            # Sprawdź, czy kolekcja istnieje
            try:
                self.client.get_collection(collection_name=collection_name)
            except Exception as e:
                logger.warning(f"⚠️ Kolekcja '{collection_name}' nie istnieje lub wystąpił błąd: {e}")
                return False
                
            # Usuń kolekcję
            self.client.delete_collection(collection_name=collection_name)
            
            logger.info(f"✅ Usunięto kolekcję '{collection_name}'.")
            
            # Jeśli to była aktualnie używana kolekcja, zresetuj vectorstore
            if collection_name == self.collection_name:
                self.vectorstore = None
                logger.info(f"ℹ️ Zresetowano vectorstore, ponieważ usunięto aktualnie używaną kolekcję.")
                
            return True
        except Exception as e:
            logger.error(f"❌ Błąd podczas usuwania kolekcji '{collection_name}': {e}")
            return False
    # === KONIEC METOD DO ZARZĄDZANIA KOLEKCJAMI ===