"""
Moduł do ładowania dokumentów z różnych formatów
"""
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_core.documents import Document
import requests
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path
        
    def load(self, file_path=None):
        """
        Ładuje dokument z podanej ścieżki
        Obsługuje formaty: PDF, DOCX, TXT
        """
        if file_path:
            self.file_path = file_path
            
        if not self.file_path:
            raise ValueError("Brak ścieżki do pliku")
            
        # Wykryj typ pliku po rozszerzeniu
        _, ext = os.path.splitext(self.file_path.lower())
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(self.file_path)
                return loader.load()
                
            elif ext in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(self.file_path)
                return loader.load()
                
            elif ext in ['.txt', '.text']:
                loader = TextLoader(self.file_path, encoding='utf-8')
                return loader.load()
                
            else:
                raise ValueError(f"Nieobsługiwany format pliku: {ext}")
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas ładowania dokumentu: {str(e)}")
            return []
            
    def load_from_url(self, url):
        """
        Pobiera dokument z URL i ładuje jego zawartość
        """
        try:
            logger.info(f"📥 Pobieranie dokumentu z URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Utwórz tymczasowy plik
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        tmpfile.write(chunk)
                tmpfile_path = tmpfile.name
                
            # Załaduj dokument z tymczasowego pliku
            documents = self.load(tmpfile_path)
            
            # Usuń tymczasowy plik
            os.unlink(tmpfile_path)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas ładowania z URL: {str(e)}")
            return []