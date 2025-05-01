# -*- coding: utf-8 -*-
"""
Skrypt do testowania pobierania danych do klastrowania z Qdrant.
"""
import logging
from app import RAGChatbot # Importujemy klasę chatbota

# Konfiguracja logowania (może być taka sama jak w app.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Inicjalizacja RAGChatbot w trybie skryptowym (test)...")
    try:
        # Inicjalizacja chatbota (może rzucić ConnectionError)
        chatbot = RAGChatbot()
        logger.info("✅ RAGChatbot zainicjalizowany pomyślnie.")

        # === POCZĄTEK TESTU get_all_data_for_clustering ===
        logger.info("\n--- Testowanie get_all_data_for_clustering ---")
        # Pobierzmy np. 5 punktów dla testu
        # Sprawdzamy, czy qdrant_connector został zainicjalizowany
        if chatbot.qdrant_connector and hasattr(chatbot.qdrant_connector, 'get_all_data_for_clustering'):
             clustering_data = chatbot.qdrant_connector.get_all_data_for_clustering(limit=5)
             if clustering_data:
                 logger.info(f"Pobrano {len(clustering_data)} punktów.")
                 logger.info("Przykładowy pierwszy punkt:")
                 # Wypisz ID i payload (metadane), pomiń wektor dla czytelności
                 first_point = clustering_data[0]
                 logger.info(f"  ID: {first_point.get('id')}")
                 logger.info(f"  Payload: {first_point.get('payload')}")
                 # logger.info(f"  Vector (fragment): {str(first_point.get('vector'))[:100]}...") # Opcjonalnie
             else:
                 logger.warning("Nie pobrano żadnych danych do klastrowania (kolekcja pusta lub błąd).")
        else:
             logger.error("❌ Obiekt chatbot.qdrant_connector nie istnieje lub nie ma metody 'get_all_data_for_clustering'.")
        logger.info("--- Koniec testu get_all_data_for_clustering ---")
        # === KONIEC TESTU ===

    except ConnectionError as ce:
        logger.error(f"❌ Nie udało się uruchomić chatbota: {ce}")
        print(f"Krytyczny błąd: Nie można połączyć się z Qdrant. Sprawdź, czy Qdrant działa i jest dostępny. Szczegóły w logach.")
    except ImportError as ie:
         logger.error(f"❌ Brakująca biblioteka: {ie}. Uruchom 'pip install -r requirements.txt'.")
         print(f"Brakująca biblioteka: {ie}. Zainstaluj wymagane pakiety.")
    except Exception as e:
        logger.error(f"❌ Nieoczekiwany błąd podczas inicjalizacji lub testu: {e}", exc_info=True)
        print(f"Wystąpił nieoczekiwany błąd: {e}")

    print("\n----------------------------------------------------")
    print("Test zakończony.")
    print("----------------------------------------------------")