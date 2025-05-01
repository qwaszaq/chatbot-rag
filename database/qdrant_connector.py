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
from qdrant_client.http.models import Distance, VectorParams, CollectionInfo, CollectionStatus
import os
import logging

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


    def similarity_search(self, query, k=99):
        """Wyszukiwanie podobnych dokumentów"""
        if not self.vectorstore:
            logger.error("❌ Qdrant vectorstore nie jest zainicjalizowany. Nie można wyszukać.")
            raise Exception("Qdrant vectorstore nie jest zainicjalizowany")

        logger.info(f"🔍 Wyszukiwanie {k} podobnych dokumentów w Qdrant dla zapytania: '{query[:50]}...'")
        try:
            # Metoda similarity_search w QdrantConnector użyje swojego wewnętrznego embedding generatora do zapytania
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"✅ Znaleziono {len(results)} dokumentów.")
            return results
        except Exception as e:
            logger.error(f"❌ Błąd podczas wyszukiwania w Qdrant: {str(e)}")
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
            # Sprawdź, czy kolekcja istnieje
            try:
                self.client.get_collection(collection_name=collection_name)
                logger.info(f"   Kolekcja '{collection_name}' istnieje. Usuwanie punktów...")

                # Zdefiniuj filtr, który pasuje do wszystkich punktów (pusty filtr Must)
                # Alternatywnie, jeśli znasz jakiś wspólny atrybut metadanych, można go użyć.
                # Lepszym podejściem jest użycie scroll API do pobrania ID i usunięcie po ID,
                # ale dla prostoty użyjemy delete z filtrem, który powinien działać dla mniejszych kolekcji.
                # Qdrant < 1.7: Użyj pustego filtru `models.Filter()`
                # Qdrant >= 1.7: Użyj `models.Filter(must=[])` lub po prostu `None` dla filtru
                # Sprawdź wersję qdrant-client, jeśli to konieczne. Załóżmy nowszą wersję lub None.
                from qdrant_client.http import models as rest

                # Usuń wszystkie punkty używając pustego filtru `must` lub braku filtru
                # Użycie 'wait=True' zapewnia, że operacja zostanie potwierdzona przed kontynuacją.
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=rest.PointIdsList(ids=[]), # To może nie działać zgodnie z oczekiwaniami dla "wszystkich"
                    # Spróbujmy usunąć używając filtru, który zawsze jest prawdziwy lub braku filtru.
                    # Według dokumentacji, aby usunąć wszystko, można użyć scroll API lub
                    # jeśli ID są znane, PointIdsList.
                    # Prostsze, choć potencjalnie mniej wydajne dla ogromnych kolekcji, jest delete z filtrem.
                    # Filtr, który zawsze jest prawdziwy (np. sprawdzający istnienie jakiegoś pola,
                    # które zawsze istnieje lub pusty `must` w nowszych wersjach).
                    # Sprawdźmy pusty `must` dla Qdrant >= 1.7 lub None
                    wait=True # Poczekaj na zakończenie operacji
                )
                # UWAGA: Skuteczność usuwania wszystkich punktów przez pusty filtr `delete` może zależeć
                # od wersji Qdrant. Bardziej niezawodne jest usuwanie i tworzenie kolekcji na nowo,
                # lub użycie scroll API do pobrania ID i ich usunięcie.
                # Wracamy do metody usuwania i tworzenia na nowo, ale z lepszym logowaniem.

                logger.info(f"   Próba usunięcia kolekcji: {collection_name}")
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"   Usunięto kolekcję '{collection_name}'.")

            except Exception as e:
                 # Jeśli kolekcja nie istnieje, get_collection rzuci wyjątek.
                 logger.info(f"   Kolekcja '{collection_name}' nie istniała lub wystąpił błąd przy sprawdzaniu/usuwaniu: {e}.")
                 # Upewnij się, że błąd nie jest krytyczny (np. problem z połączeniem)

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