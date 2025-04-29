"""
Test skrypt do weryfikacji integracji z Qdrant
"""
from chatbot_rag.database.qdrant_connector import QdrantConnector
from langchain_community.embeddings import LlamaCppEmbeddings
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qdrant_integration():
    try:
        logger.info("🔄 Inicjalizacja testu integracji z Qdrant")
        
        # Inicjalizacja modelu embeddingowego (LM Studio)
        embeddings = LlamaCppEmbeddings(model_path="/models/ggml-model-q4_0.gguf")
        
        # Inicjalizacja Qdrant
        qdrant = QdrantConnector(collection_name="test_collection")
        if not qdrant.initialize(embeddings):
            logger.error("❌ Nie udało się zainicjalizować Qdrant")
            return False
            
        logger.info("✅ Pomyślnie połączono z Qdrant")
        
        # Testowe dane
        test_document = "To jest testowy dokument do integracji z Qdrant."
        metadata = {"source": "test"}
        
        # Dodanie dokumentu
        logger.info("📥 Dodawanie testowego dokumentu do Qdrant")
        qdrant.add_documents([test_document], embeddings)
        logger.info("✅ Dokument dodany pomyślnie")
        
        # Wyszukiwanie podobne
        logger.info("🔍 Wykonywanie testowego wyszukiwania")
        results = qdrant.similarity_search("test", k=1)
        
        if results:
            logger.info("✅ Znaleziono wyniki wyszukiwania")
            logger.info(f"🔍 Znaleziony dokument: {results[0].page_content}")
            return True
        else:
            logger.warning("❌ Nie znaleziono wyników wyszukiwania")
            return False
            
    except Exception as e:
        logger.error(f"❌ Błąd podczas testu Qdrant: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_qdrant_integration()
    print("✅ Test Qdrant zakończony powodzeniem" if result else "❌ Test Qdrant zakończony niepowodzeniem")