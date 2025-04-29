"""
Moduł do generowania embeddingów tekstu z wykorzystaniem LM Studio
Implementuje interfejs LangChain Embeddings.
"""
import requests
import logging
import time
from typing import List # Dodano import List
from langchain_core.embeddings import Embeddings # Dodano import Embeddings

logger = logging.getLogger(__name__)

# Klasa EmbeddingGenerator dziedzicząca po Embeddings
class EmbeddingGenerator(Embeddings):
    def __init__(self, api_url="http://localhost:1234/v1/embeddings", model_name="default", timeout=30):
        """
        Inicjalizacja generatora embeddingów dla LM Studio
        
        Args:
            api_url (str): URL endpointu API LM Studio
            model_name (str): Nazwa modelu do generowania embeddingów
            timeout (int): Czas oczekiwania na odpowiedź od API w sekundach
        """
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    # Implementacja metody embed_query wymaganej przez interfejs Embeddings
    def embed_query(self, text: str) -> List[float]:
        """
        Generuje embedding dla pojedynczego tekstu (zapytania)
        
        Args:
            text (str): Tekst do przekonwertowania na embedding
            
        Returns:
            list: Wektor embeddingu (lista float)
        """
        if not text or not text.strip():
            logger.warning("⚠️ Próba wygenerowania embeddingu dla pustego tekstu")
            return []

        payload = {
            "input": text,
            "model": self.model_name
        }

        logger.info(f"🧠 Generowanie embeddingu dla zapytania (długość: {len(text)} znaków)")

        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            duration = time.time() - start_time
            logger.info(f"✅ Pomyślnie wygenerowano embedding w {duration:.2f} sekund")

            # Oczekujemy listy float zgodnie z interfejsem
            return response.json()["data"][0]["embedding"]

        except requests.exceptions.Timeout:
            logger.error(f"❌ Przekroczono limit czasu przy generowaniu embeddingu ({self.timeout} sekund)")
            raise # Propaguj błąd dalej

        except requests.exceptions.ConnectionError:
            logger.error("❌ Błąd połączenia z serwerem LM Studio. Upewnij się, że jest uruchomiony.")
            raise # Propaguj błąd dalej

        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Błąd HTTP {response.status_code}: {str(e)}")
            logger.error(f"Odpowiedź serwera: {response.text}")
            raise # Propaguj błąd dalej

        except Exception as e:
            logger.error(f"❌ Nieoczekiwany błąd podczas generowania embeddingu: {str(e)}")
            raise # Propaguj błąd dalej

    # Implementacja metody embed_documents wymaganej przez interfejs Embeddings
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generuje embeddingi dla listy tekstów (dokumentów)
        
        Args:
            texts (List[str]): Lista tekstów do przekonwertowania na embeddingi
            
        Returns:
            List[List[float]]: Lista wektorów embeddingów
        """
        if not texts or not any(text.strip() for text in texts):
            logger.warning("⚠️ Próba wygenerowania embeddingów dla pustej listy tekstów")
            return []

        logger.info(f"📚 Generowanie embeddingów dla {len(texts)} dokumentów")

        # LM Studio API v1/embeddings obsługuje batching, ale nasza obecna implementacja generate_embedding
        # wysyła pojedyncze żądanie na tekst. Możemy to zoptymalizować później, ale na razie
        # użyjemy pętli, która wywołuje embed_query (dawniej generate_embedding) dla każdego tekstu.
        # W przyszłości można by zaimplementować logikę batchingu tutaj, jeśli LM Studio to wspiera wprost.
        
        embeddings = []
        for text in texts:
            try:
                # Wywołujemy embed_query dla każdego tekstu
                embedding = self.embed_query(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"❌ Błąd podczas generowania embeddingu dla dokumentu: {str(e)}")
                # Możesz zdecydować, czy przerwać, czy kontynuować i zwrócić puste embeddingi dla tego dokumentu
                embeddings.append([]) # Zwracamy pustą listę dla błędu, aby zachować strukturę listy embeddingów
        
        return embeddings

    # Usunięto starą metodę batch_generate_embeddings, ponieważ została zastąpiona przez embed_documents

# Przykład użycia (pozostawiony dla celów testowych, ale nie będzie uruchamiany bezpośrednio)
if __name__ == "__main__":
    # Wymaga uruchomionego serwera LM Studio na localhost:1234 z załadowanym modelem embeddingowym
    try:
        logger.info("Testowanie EmbeddingGenerator...")
        generator = EmbeddingGenerator()
        
        # Test pojedynczego embeddingu
        test_text_single = "To jest przykładowy tekst."
        embedding_single = generator.embed_query(test_text_single)
        logger.info(f"Embedding dla '{test_text_single[:20]}...': {embedding_single[:5]}...") # Wyświetl tylko fragment

        # Test batch embeddingów
        test_texts_batch = ["Tekst numer jeden.", "Tekst numer dwa.", "Tekst numer trzy."]
        embeddings_batch = generator.embed_documents(test_texts_batch)
        logger.info(f"Wygenerowano {len(embeddings_batch)} embeddingów dla batcha.")
        if embeddings_batch:
             logger.info(f"Pierwszy embedding w batchu: {embeddings_batch[0][:5]}...") # Wyświetl fragment

    except Exception as e:
        logger.error(f"❌ Test EmbeddingGenerator zakończony błędem: {str(e)}")