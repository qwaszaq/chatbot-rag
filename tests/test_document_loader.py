"""
Testy jednostkowe dla modułu document_loader.py
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
from chatbot_rag.data_processing.document_loader import DocumentLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_core.documents import Document
import os
import tempfile

class TestDocumentLoader(unittest.TestCase):

    def test_load_pdf(self):
        """Test ładowania pliku PDF"""
        with patch('chatbot_rag.data_processing.document_loader.PyPDFLoader') as MockPyPDFLoader:
            mock_loader_instance = MockPyPDFLoader.return_value
            mock_loader_instance.load.return_value = [Document(page_content="test content from pdf")]

            loader = DocumentLoader("dummy.pdf")
            documents = loader.load()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "test content from pdf")
            MockPyPDFLoader.assert_called_once_with("dummy.pdf")

    def test_load_docx(self):
        """Test ładowania pliku DOCX"""
        with patch('chatbot_rag.data_processing.document_loader.UnstructuredWordDocumentLoader') as MockDocxLoader:
            mock_loader_instance = MockDocxLoader.return_value
            mock_loader_instance.load.return_value = [Document(page_content="test content from docx")]

            loader = DocumentLoader("dummy.docx")
            documents = loader.load()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "test content from docx")
            MockDocxLoader.assert_called_once_with("dummy.docx")

    def test_load_txt(self):
        """Test ładowania pliku TXT"""
        with patch('chatbot_rag.data_processing.document_loader.TextLoader') as MockTextLoader:
            mock_loader_instance = MockTextLoader.return_value
            mock_loader_instance.load.return_value = [Document(page_content="test content from txt")]

            loader = DocumentLoader("dummy.txt")
            documents = loader.load()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "test content from txt")
            MockTextLoader.assert_called_once_with("dummy.txt", encoding='utf-8')

    def test_load_unsupported_format(self):
        """Test ładowania nieobsługiwanego formatu"""
        loader = DocumentLoader("dummy.unsupported")
        documents = loader.load()

        self.assertEqual(len(documents), 0)

    def test_load_file_not_found(self):
        """Test ładowania nieistniejącego pliku"""
        with patch('chatbot_rag.data_processing.document_loader.PyPDFLoader') as MockPyPDFLoader:
             MockPyPDFLoader.side_effect = FileNotFoundError

             loader = DocumentLoader("non_existent.pdf")
             documents = loader.load()

             self.assertEqual(len(documents), 0)

    @patch('chatbot_rag.data_processing.document_loader.requests.get')
    @patch('chatbot_rag.data_processing.document_loader.tempfile.NamedTemporaryFile', autospec=True)
    @patch('chatbot_rag.data_processing.document_loader.os.unlink')
    @patch('chatbot_rag.data_processing.document_loader.DocumentLoader.load')
    def test_load_from_url(self, mock_load, mock_unlink, MockNamedTemporaryFile, mock_get):
        """Test ładowania z URL"""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"file content"]
        mock_get.return_value = mock_response

        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/tempfile.pdf"
        MockNamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file

        mock_load.return_value = [Document(page_content="content from url")]

        loader = DocumentLoader()
        documents = loader.load_from_url("http://example.com/document.pdf")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "content from url")
        mock_get.assert_called_once_with("http://example.com/document.pdf", stream=True)
        mock_load.assert_called_once_with("/tmp/tempfile.pdf")
        mock_unlink.assert_called_once_with("/tmp/tempfile.pdf")

    @patch('chatbot_rag.data_processing.document_loader.requests.get')
    def test_load_from_url_request_error(self, mock_get):
        """Test błędu podczas ładowania z URL (Request Error)"""
        mock_get.side_effect = Exception("Request failed")

        loader = DocumentLoader()
        documents = loader.load_from_url("http://example.com/document.pdf")

        self.assertEqual(len(documents), 0)

if __name__ == '__main__':
    unittest.main()