"""
Testy jednostkowe dla modułu qdrant_connector.py
"""
import unittest
from unittest.mock import MagicMock, patch, call
# Zaktualizowano importy CollectionInfo i CollectionStatus na te z qdrant_client.http.models
from qdrant_client.http.models import Distance, VectorParams, CollectionInfo, CollectionStatus
from chatbot_rag.database.qdrant_connector import QdrantConnector

class TestQdrantConnector(unittest.TestCase):

    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_initialize_new_collection(self, MockQdrantClient):
        """Test inicjalizacji z nową kolekcją"""
        mock_client_instance = MockQdrantClient.return_value
        # Symulacja błędu przy próbie pobrania kolekcji - QdrantClient.get_collection rzuci wyjątek jeśli kolekcja nie istnieje
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")

        mock_embeddings = MagicMock()

        connector = QdrantConnector(collection_name="new_collection", vector_size=100)
        result = connector.initialize(mock_embeddings)

        self.assertTrue(result)
        # Sprawdź, czy get_collection zostało wywołane raz
        mock_client_instance.get_collection.assert_called_once_with(collection_name="new_collection")
        # Sprawdź, czy create_collection zostało wywołane raz z poprawnymi parametrami
        mock_client_instance.create_collection.assert_called_once_with(
            collection_name="new_collection",
            vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
        self.assertIsNotNone(connector.get_vectorstore())

    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_initialize_existing_collection(self, MockQdrantClient):
        """Test inicjalizacji z istniejącą kolekcją"""
        mock_client_instance = MockQdrantClient.return_value
        # Symulacja istniejącej i aktywnej kolekcji
        mock_collection_info = MagicMock(spec=CollectionInfo)
        mock_collection_info.status = CollectionStatus.GREEN
        # Nie musimy mockować vectors_config, bo już go nie używamy w logice sprawdzania istnienia/statusu
        # Jeśli byśmy potrzebowali testować logikę związaną z vectors_config (np. porównanie rozmiarów),
        # to musielibyśmy go zamockować, ale uproszczona logika tego nie wymaga.
        mock_client_instance.get_collection.return_value = mock_collection_info


        mock_embeddings = MagicMock()

        connector = QdrantConnector(collection_name="existing_collection", vector_size=100)
        result = connector.initialize(mock_embeddings)

        self.assertTrue(result)
        # Sprawdź, czy get_collection zostało wywołane raz
        mock_client_instance.get_collection.assert_called_once_with(collection_name="existing_collection")
        # Sprawdź, czy create_collection NIE zostało wywołane
        mock_client_instance.create_collection.assert_not_called()
        self.assertIsNotNone(connector.get_vectorstore())

    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_initialize_connection_error(self, MockQdrantClient):
        """Test błędu połączenia podczas inicjalizacji"""
        # Symulacja błędu połączenia już przy tworzeniu klienta
        MockQdrantClient.side_effect = Exception("Connection failed")

        mock_embeddings = MagicMock()

        connector = QdrantConnector()
        result = connector.initialize(mock_embeddings)

        self.assertFalse(result)
        self.assertIsNone(connector.get_client())
        self.assertIsNone(connector.get_vectorstore())

    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_add_documents(self, MockQdrantClient):
        """Test dodawania dokumentów"""
        mock_vectorstore = MagicMock()
        # Nie potrzebujemy mock_embeddings tutaj, bo add_documents w connectorze ich nie przyjmuje
        # mock_embeddings = MagicMock()

        connector = QdrantConnector()
        connector.vectorstore = mock_vectorstore # Ustaw mock vectorstore bezpośrednio

        test_docs = [MagicMock(), MagicMock()] # Użyjemy mocków dla dokumentów
        connector.add_documents(test_docs) # Usunięto argument embeddings

        # Sprawdź, czy add_documents na mock vectorstore zostało wywołane raz z poprawnymi argumentami
        mock_vectorstore.add_documents.assert_called_once_with(test_docs)


    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_add_documents_not_initialized(self, MockQdrantClient):
        """Test dodawania dokumentów bez inicjalizacji"""
        # mock_embeddings = MagicMock() # Niepotrzebne
        connector = QdrantConnector() # Nie wywołujemy initialize

        test_docs = [MagicMock()] # Użyjemy mocka dla dokumentu

        with self.assertRaisesRegex(Exception, "Qdrant vectorstore nie jest zainicjalizowany"): # Zmieniono komunikat błędu
            connector.add_documents(test_docs) # Usunięto argument embeddings

    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_similarity_search(self, MockQdrantClient):
        """Test wyszukiwania podobieństwa"""
        mock_vectorstore = MagicMock()
        # Nie potrzebujemy mock_embeddings tutaj
        # mock_embeddings = MagicMock()

        connector = QdrantConnector()
        connector.vectorstore = mock_vectorstore # Ustaw mock vectorstore bezpośrednio

        query = "test query"
        connector.similarity_search(query, k=5)

        # Sprawdź, czy similarity_search na mock vectorstore zostało wywołane raz z poprawnymi argumentami
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=5)

    @patch('chatbot_rag.database.qdrant_connector.QdrantClient')
    def test_similarity_search_not_initialized(self, MockQdrantClient):
        """Test wyszukiwania bez inicjalizacji"""
        connector = QdrantConnector() # Nie wywołujemy initialize

        query = "test query"

        with self.assertRaisesRegex(Exception, "Qdrant vectorstore nie jest zainicjalizowany"): # Zmieniono komunikat błędu
            connector.similarity_search(query, k=5)

if __name__ == '__main__':
    unittest.main()