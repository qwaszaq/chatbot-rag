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
    # Zmieniono domyślną nazwę kolekcji na "nowa" i vector_size na 1024
    def __init__(self, host="localhost", port=6333, collection_name="nowa", vector_size=1024):
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