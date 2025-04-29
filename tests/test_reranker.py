"""
Testy jednostkowe dla modułu reranker.py
"""
import unittest
from unittest.mock import patch, MagicMock
from chatbot_rag.data_processing.reranker import Reranker
from langchain_core.documents import Document

class TestReranker(unittest.TestCase):

    @patch('chatbot_rag.data_processing.reranker.CrossEncoder')
    def test_rerank(self, MockCrossEncoder):
        """Test rerankingu dokumentów"""
        mock_model_instance = MockCrossEncoder.return_value
        # Symulacja wyników modelu rerankującego
        mock_model_instance.predict.return_value = [0.8, 0.2, 0.6, 0.9] # Wyniki dla doc4, doc2, doc3, doc1

        reranker = Reranker()
        query = "test query"
        documents = [
            Document(page_content="Dokument 1"),
            Document(page_content="Dokument 2"),
            Document(page_content="Dokument 3"),
            Document(page_content="Dokument 4")
        ]
        # top_k = 2
        reranked_docs = reranker.rerank(query, [doc.page_content for doc in documents], top_k=2)

        self.assertEqual(len(reranked_docs), 2)
        # Sprawdź, czy dokumenty są posortowane według wyniku rerankingu i zwrócono tylko top_k
        # Oczekiwana kolejność: Dokument 4 (0.9), Dokument 1 (0.8)
        self.assertEqual(reranked_docs[0][0], "Dokument 4")
        self.assertEqual(reranked_docs[0][1], 0.9)
        self.assertEqual(reranked_docs[1][0], "Dokument 1")
        self.assertEqual(reranked_docs[1][1], 0.8)

        MockCrossEncoder.assert_called_once_with("BAAI/bge-reranker-v2-m3")
        mock_model_instance.predict.assert_called_once()
        # Sprawdź argumenty przekazane do predict - powinny być pary [query, document_content]
        predicted_pairs = mock_model_instance.predict.call_args[0][0]
        self.assertEqual(predicted_pairs, [
            ["test query", "Dokument 1"],
            ["test query", "Dokument 2"],
            ["test query", "Dokument 3"],
            ["test query", "Dokument 4"]
        ])

    @patch('chatbot_rag.data_processing.reranker.CrossEncoder')
    def test_rerank_empty_documents(self, MockCrossEncoder):
        """Test rerankingu pustej listy dokumentów"""
        mock_model_instance = MockCrossEncoder.return_value
        mock_model_instance.predict.return_value = []

        reranker = Reranker()
        query = "test query"
        documents = []
        reranked_docs = reranker.rerank(query, [doc.page_content for doc in documents], top_k=10)

        self.assertEqual(len(reranked_docs), 0)
        mock_model_instance.predict.assert_called_once_with([])

if __name__ == '__main__':
    unittest.main()