"""
Moduł do dzielenia tekstu na fragmenty (chunks) z wykorzystaniem LangChain
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(self, chunk_size=750, chunk_overlap=120):
        """
        Inicjalizacja text splitter z wykorzystaniem RecursiveCharacterTextSplitter z LangChain
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
    def split_text(self, text):
        """
        Dzieli tekst na fragmenty z możliwością nakładania się
        """
        if not text:
            logger.warning("⚠️ Próba podziału pustego tekstu")
            return []
            
        try:
            logger.info(f"✂️ Dzielenie tekstu na fragmenty (rozmiar: {self.chunk_size}, overlap: {self.chunk_overlap})")
            return self.splitter.split_text(text)
        except Exception as e:
            logger.error(f"❌ Błąd podczas dzielenia tekstu: {str(e)}")
            return []
        
    def split_document(self, document):
        """
        Dzieli dokument na fragmenty
        """
        if not isinstance(document, Document):
            logger.error("❌ Nieprawidłowy typ dokumentu - oczekiwano instancji Document")
            return []
            
        try:
            logger.info(f"📄 Dzielenie dokumentu '{document.metadata.get('source', 'nieznany')}' na fragmenty")
            return self.splitter.split_documents([document])
        except Exception as e:
            logger.error(f"❌ Błąd podczas dzielenia dokumentu: {str(e)}")
            return []