"""
Testy jednostkowe dla modułu embedding_generator.py
"""
import unittest
from unittest.mock import patch, MagicMock
from chatbot_rag.data_processing.embedding_generator import EmbeddingGenerator
import requests

class TestEmbeddingGenerator(unittest.TestCase):

    @patch('chatbot_rag.data_processing.embedding_generator.requests.post')
    def test_generate_embedding_success(self, mock_post):
        """Test generowania embeddingu dla pojedynczego tekstu - sukces"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding("test text")

        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        mock_post.assert_called_once_with(
            "http://localhost:1234/v1/embeddings",
            json={"input": "test text", "model": "default"},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30
        )

    @patch('chatbot_rag.data_processing.embedding_generator.requests.post')
    def test_generate_embedding_empty_text(self, mock_post):
        """Test generowania embeddingu dla pustego tekstu"""
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding("")

        self.assertEqual(embedding, [])
        mock_post.assert_not_called()

    @patch('chatbot_rag.data_processing.embedding_generator.requests.post')
    def test_generate_embedding_connection_error(self, mock_post):
        """Test błędu połączenia podczas generowania embeddingu"""
        mock_post.side_effect = requests.exceptions.ConnectionError

        generator = EmbeddingGenerator()

        with self.assertRaises(requests.exceptions.ConnectionError):
            generator.generate_embedding("test text")

    @patch('chatbot_rag.data_processing.embedding_generator.requests.post')
    def test_generate_embedding_http_error(self, mock_post):
        """Test błędu HTTP podczas generowania embeddingu"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        generator = EmbeddingGenerator()

        with self.assertRaises(requests.exceptions.HTTPError):
            generator.generate_embedding("test text")

    @patch('chatbot_rag.data_processing.embedding_generator.EmbeddingGenerator.generate_embedding')
    def test_batch_generate_embeddings(self, mock_generate_embedding):
        """Test generowania embeddingów dla wielu tekstów"""
        mock_generate_embedding.side_effect = [[0.1, 0.2], [0.4, 0.5]]

        generator = EmbeddingGenerator()
        texts = ["text1", "text2"]
        embeddings = generator.batch_generate_embeddings(texts)

        self.assertEqual(embeddings, [[0.1, 0.2], [0.4, 0.5]])
        mock_generate_embedding.assert_any_call("text1")
        mock_generate_embedding.assert_any_call("text2")
        self.assertEqual(mock_generate_embedding.call_count, 2)

    @patch('chatbot_rag.data_processing.embedding_generator.EmbeddingGenerator.generate_embedding')
    def test_batch_generate_embeddings_empty_list(self, mock_generate_embedding):
        """Test generowania embeddingów dla pustej listy tekstów"""
        generator = EmbeddingGenerator()
        texts = []
        embeddings = generator.batch_generate_embeddings(texts)

        self.assertEqual(embeddings, [])
        mock_generate_embedding.assert_not_called()


if __name__ == '__main__':
    unittest.main()