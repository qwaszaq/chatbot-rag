"""
Testy jednostkowe dla modułu text_splitter.py
"""
import unittest
from unittest.mock import patch, MagicMock
from chatbot_rag.data_processing.text_splitter import TextSplitter
from langchain_core.documents import Document

class TestTextSplitter(unittest.TestCase):

    def test_split_text(self):
        """Test dzielenia pojedynczego tekstu"""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        text = "To jest przykładowy tekst do podzielenia na chunki."
        chunks = splitter.split_text(text)

        # Sprawdź, czy zwrócono listę
        self.assertIsInstance(chunks, list)
        # Sprawdź, czy chunki nie są puste
        self.assertTrue(all(chunks))
        # Sprawdź, czy rozmiar chunków jest w przybliżeniu zgodny z oczekiwanym
        self.assertTrue(len(chunks) > 0) # Powinny być jakieś chunki
        # Można dodać bardziej szczegółowe testy rozmiaru i overlapa, ale są one zależne od implementacji LangChain

    def test_split_empty_text(self):
        """Test dzielenia pustego tekstu"""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        text = ""
        chunks = splitter.split_text(text)

        self.assertEqual(len(chunks), 0)

    def test_split_document(self):
        """Test dzielenia dokumentu LangChain"""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        document = Document(page_content="To jest przykładowy dokument do podzielenia.", metadata={"source": "test_doc"})
        chunks = splitter.split_document(document)

        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, Document) for chunk in chunks))
        self.assertTrue(len(chunks) > 0)
        # Sprawdź, czy metadane są zachowane
        self.assertTrue(all("source" in chunk.metadata and chunk.metadata["source"] == "test_doc" for chunk in chunks))

    def test_split_invalid_document_type(self):
        """Test dzielenia obiektu innego niż Document"""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        invalid_object = "To nie jest dokument"
        chunks = splitter.split_document(invalid_object)

        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()