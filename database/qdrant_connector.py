"""
Moduł do integracji z bazą danych Qdrant
Implementuje interfejs LangChain Embeddings.
"""
# Zmieniono import Qdrant na ten z nowej biblioteki langchain-qdrant
from langchain_qdrant import Qdrant
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
# Zmieniono import CollectionInfo i CollectionStatus na ten z qdrant_client.http.models
# Upewnij się, że te klasy są dostępne w qdrant_client.http.models w Twojej wersji qdrant-client
from qdrant_client.http.models import (
    Distance, VectorParams, CollectionInfo, CollectionStatus,
    PointStruct, UpdateResult, PointIdsList, Payload,
    Filter, FieldCondition, MatchValue, # Dodano przecinek
    # Brak importu Distance, VectorParams, PointStruct, bo już są
)
from qdrant_client.http import models as rest # Dodano import rest dla upsert
import os
import logging
from typing import Dict, List, Optional, Any # Zmieniono na Any
import time # DODANO: Import time

logger = logging.getLogger(__name__)

class QdrantConnector:
    # Zmieniono domyślną nazwę kolekcji na "nowa1"
    def __init__(self, host="localhost", port=6333, collection_name="nowa1", vector_size=1024):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.cluster_metadata_collection_name = f"{collection_name}_cluster_metadata" # Nowa nazwa kolekcji na metadane klastrów
        self.vector_size = vector_size
        self.client = None
        self.vectorstore = None

    def initialize(self, embeddings: Embeddings):
        """Inicjalizuje połączenie z Qdrant i tworzy kolekcję jeśli nie istnieje"""
        try:
            logger.info(f"Attempting to connect to Qdrant at {self.host}:{self.port}")
            # Nawiązanie połączenia z Qdrant
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info("✅ Connected to Qdrant.")

            # Sprawdzenie czy kolekcja istnieje
            # Use try-except to check for collection existence gracefully
            collection_exists = False
            try:
                collection_info = self.client.get_collection(collection_name=self.collection_name)
                # Check if collection is active
                if collection_info.status != CollectionStatus.GREEN:
                     logger.warning(f"⚠️ Kolekcja '{self.collection_name}' istnieje, ale nie jest w stanie GREEN. Status: {collection_info.status}")
                     # Depending on status, might need to wait or recreate. For now, proceed.

                collection_exists = True
                logger.info(f"✅ Kolekcja w Qdrant '{self.collection_name}' już istnieje.")

            except Exception as e: # Catch specific exceptions like UnexpectedResponse if possible, but broad Exception is safer for now
                # If get_collection raises an error (e.g., 404 not found), the collection doesn't exist
                logger.info(f"ℹ️ Kolekcja '{self.collection_name}' nie istnieje lub wystąpił błąd przy jej pobieraniu: {e}. Spróbuję utworzyć.")
                collection_exists = False


            if not collection_exists:
                logger.info(f"✨ Tworzenie nowej kolekcji w Qdrant: {self.collection_name} z rozmiarem wektora {self.vector_size}")
                # Utworzenie nowej kolekcji
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"✅ Kolekcja '{self.collection_name}' utworzona pomyślnie z rozmiarem wektora {self.vector_size}.")

            # Utworzenie instancji Qdrant dla tej kolekcji
            # Przekazujemy obiekt embeddings tutaj. LangChain/Qdrant client powinien obsłużyć
            # połączenie z istniejącą lub nowo utworzoną kolekcją.
            logger.info(f"Initializing LangChain Qdrant vectorstore for collection '{self.collection_name}'")
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=embeddings # Przekazujemy instancję EmbeddingGenerator (która implementuje Embeddings)
            )
            logger.info("✅ LangChain Qdrant vectorstore initialized.")
            # Inicjalizacja kolekcji metadanych klastrów po udanej inicjalizacji klienta
            self._initialize_metadata_collection()
            return True # Zwróć True jeśli główna inicjalizacja się powiodła

        except Exception as e:
            # Catch any exception during initialization
            logger.error(f"❌ Błąd podczas inicjalizacji Qdrant (główna kolekcja): {str(e)}")
            return False
    def add_documents(self, documents):
        """Dodaje dokumenty do bazy. Vectorstore użyje swojego wewnętrznego embedding generatora."""
        if not self.vectorstore:
            logger.error("❌ Qdrant vectorstore nie jest zainicjalizowany. Nie można dodać dokumentów.")
            raise Exception("Qdrant vectorstore nie jest zainicjalizowany")

        logger.info(f"📥 Dodawanie {len(documents)} dokumentów do Qdrant.")
        try:
            # Wywołujemy add_documents na vectorstore, który ma już dostęp do embedding generatora
            # Zwraca listę identyfikatorów punktów
            result = self.vectorstore.add_documents(documents)
            logger.info(f"✅ Dodano {len(result)} dokumentów do Qdrant.")
            return result # Return the result from add_documents
        except Exception as e:
            logger.error(f"❌ Błąd podczas dodawania dokumentów do Qdrant: {str(e)}")
            raise # Propaguj błąd dalej


    def similarity_search(self, query, k=99, filter_cluster_id: Optional[int] = None):
        """
        Wyszukiwanie podobnych dokumentów z opcjonalnym filtrowaniem po cluster_id.

        Args:
            query (str): Tekst zapytania.
            k (int): Liczba wyników do zwrócenia.
            filter_cluster_id (Optional[int]): ID klastra do filtrowania. Jeśli None, brak filtrowania.

        Returns:
            List[Document]: Lista znalezionych dokumentów LangChain.
        """
        if not self.vectorstore:
            logger.error("❌ Qdrant vectorstore nie jest zainicjalizowany. Nie można wyszukać.")
            raise Exception("Qdrant vectorstore nie jest zainicjalizowany")

        query_filter = None
        log_filter_msg = ""
        if filter_cluster_id is not None:
            logger.info(f"   Applying filter for cluster_id: {filter_cluster_id}")
            # Upewnij się, że importujesz Filter, FieldCondition, MatchValue
            # DODANO: Logowanie i konwersja typu PRZED utworzeniem FieldCondition
            logger.debug(f"   Type of filter_cluster_id before int(): {type(filter_cluster_id)}, value: {filter_cluster_id}")
            try:
                # Logowanie typu przed konwersją
                logger.debug(f"   Type of filter_cluster_id before conversion: {type(filter_cluster_id)}, value: {filter_cluster_id}")

                # Użyj .item() do konwersji typów NumPy na natywne typy Python
                # lub standardowego int() dla innych typów
                if hasattr(filter_cluster_id, 'item'): # Sprawdź, czy to typ NumPy
                    converted_cluster_id = filter_cluster_id.item()
                else:
                    converted_cluster_id = int(filter_cluster_id) # Standardowa konwersja dla innych typów

                logger.debug(f"   Type of converted_cluster_id after conversion: {type(converted_cluster_id)}, value: {converted_cluster_id}")

                # Dodatkowe sprawdzenie, czy wynik jest faktycznie int
                if not isinstance(converted_cluster_id, int):
                     raise TypeError(f"Converted cluster ID is not a standard Python int after conversion: type={type(converted_cluster_id)}")

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="cluster_id", # Klucz w payloadzie
                            match=MatchValue(value=converted_cluster_id) # Użyj skonwertowanej wartości
                        )
                    ]
                )
            except (ValueError, TypeError) as conv_err:
                 logger.error(f"   Błąd konwersji filter_cluster_id na int: {conv_err}. Wyszukiwanie bez filtra klastra.")
                 query_filter = None # W razie błędu konwersji, nie filtruj
                 log_filter_msg = "" # Usuń informację o filtrze z logu

            log_filter_msg = f" z filtrem cluster_id={filter_cluster_id}"

        logger.info(f"🔍 Wyszukiwanie {k} podobnych dokumentów w Qdrant dla zapytania: '{query[:50]}'..." + log_filter_msg)
        try:
            # Używamy similarity_search_with_score, aby potencjalnie mieć dostęp do score'ów
            # Ważne: przekazujemy filtr do metody wyszukiwania vectorstore
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=query_filter # Przekazanie filtra
            )
            # Wynik to lista krotek (Document, score), bierzemy tylko dokumenty
            results = [doc for doc, score in results_with_scores]
            logger.info(f"✅ Znaleziono {len(results)} dokumentów" + log_filter_msg + ".")
            return results
        except Exception as e:
            logger.error(f"❌ Błąd podczas wyszukiwania w Qdrant{log_filter_msg}: {str(e)}", exc_info=True)
            raise # Propaguj błąd dalej

    def get_client(self):
        """Zwraca klienta Qdrant"""
        return self.client

    def get_vectorstore(self):
        """Zwraca vectorstore Qdrant"""
        return self.vectorstore

    def _initialize_metadata_collection(self):
        """Inicjalizuje kolekcję na metadane klastrów, jeśli nie istnieje."""
        if not self.client:
            logger.error("❌ Klient Qdrant nie zainicjalizowany, nie można inicjalizować kolekcji metadanych.")
            return
        try:
            logger.info(f"Attempting to check/create cluster metadata collection: {self.cluster_metadata_collection_name}")
            try:
                self.client.get_collection(collection_name=self.cluster_metadata_collection_name)
                logger.info(f"✅ Kolekcja metadanych klastrów '{self.cluster_metadata_collection_name}' już istnieje.")
            except Exception:
                logger.info(f"ℹ️ Kolekcja metadanych klastrów '{self.cluster_metadata_collection_name}' nie istnieje. Spróbuję utworzyć.")
                # Kolekcja metadanych nie potrzebuje wektorów
                self.client.recreate_collection(
                    collection_name=self.cluster_metadata_collection_name,
                    vectors_config={} # Pusta konfiguracja wektorów
                    # Można dodać shard_number=1, replication_factor=1 dla optymalizacji, jeśli to mała kolekcja
                )
                logger.info(f"✅ Kolekcja metadanych klastrów '{self.cluster_metadata_collection_name}' utworzona pomyślnie.")
        except Exception as e:
            logger.error(f"❌ Błąd podczas inicjalizacji Qdrant (kolekcja metadanych klastrów): {str(e)}")
            # Nie przerywamy głównej inicjalizacji, ale logujemy błąd

    def clear_collection(self):
        """Usuwa i tworzy na nowo główną kolekcję Qdrant."""
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można wyczyścić kolekcji.")
            return False

        collection_name = self.collection_name

        logger.warning(f"🗑️ Próba usunięcia WSZYSTKICH punktów z kolekcji: {collection_name}")
        try:
            # Sprawdź, czy kolekcja istnieje i usuń ją
            try:
                self.client.get_collection(collection_name=collection_name)
                logger.info(f"   Kolekcja '{collection_name}' istnieje. Usuwanie kolekcji...")
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"   Usunięto kolekcję '{collection_name}'.")
            except Exception as e:
                 # Jeśli kolekcja nie istnieje, get_collection rzuci wyjątek.
                 logger.info(f"   Kolekcja '{collection_name}' nie istniała lub wystąpił błąd przy sprawdzaniu/usuwaniu: {e}.")
                 # Możemy kontynuować, bo celem jest i tak stworzenie nowej.

            # Utwórz kolekcję na nowo
            vector_size = self.vector_size # Pobierz rozmiar z instancji
            logger.info(f"   Tworzenie nowej, pustej kolekcji: {collection_name} z rozmiarem wektora {vector_size}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"✅ Pomyślnie wyczyszczono (usunięto i utworzono na nowo) kolekcję '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"❌ Błąd podczas czyszczenia (usuwania/tworzenia) kolekcji '{collection_name}': {str(e)}")
            return False

    # === POPRAWNIE DODANA METODA ===
    def get_all_data_for_clustering(self, limit=None, with_vectors=True, with_payload=True):
        """
        Pobiera dane punktów (ID, wektory, payload/metadane) z kolekcji Qdrant,
        używając metody scroll do efektywnego pobierania dużych zbiorów.

        Args:
            limit (int, optional): Maksymalna liczba punktów do pobrania. Domyślnie None (pobierz wszystkie).
            with_vectors (bool): Czy pobierać wektory embeddingów. Domyślnie True.
            with_payload (bool): Czy pobierać payload (metadane). Domyślnie True.

        Returns:
            list[dict]: Lista słowników, gdzie każdy słownik reprezentuje punkt
                        i zawiera klucze 'id', 'vector', 'payload'.
                        Zwraca pustą listę w przypadku błędu lub braku punktów.
        """
        if not self.client:
            logger.error("❌ Qdrant client nie jest zainicjalizowany. Nie można pobrać danych.")
            return []

        logger.info(f"⬇️ Pobieranie danych z kolekcji '{self.collection_name}' do klastrowania (limit: {limit})...")
        all_points_data = []
        next_page_offset = None
        processed_count = 0

        try:
            while True:
                # Używamy scroll API do pobierania partiami
                # Poprawka: import rest musi być w zasięgu
                from qdrant_client.http import models as rest
                response, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(1000, limit - processed_count) if limit is not None else 1000, # Poprawka obsługi limitu
                    offset=next_page_offset,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                )

                if not response: # Jeśli nie ma więcej wyników
                    break

                # Przetwarzanie pobranych rekordów
                for record in response:
                     point_data = {
                         "id": record.id,
                         # Używamy getattr z domyślną wartością None, na wypadek gdyby wektor/payload nie został pobrany
                         "vector": getattr(record, 'vector', None) if with_vectors else None,
                         "payload": getattr(record, 'payload', None) if with_payload else None
                     }
                     all_points_data.append(point_data)
                     processed_count += 1

                # Sprawdź, czy osiągnięto limit
                if limit is not None and processed_count >= limit:
                    logger.info(f"   Osiągnięto limit {limit} pobranych punktów.")
                    break

                # Jeśli next_page_offset jest None, to koniec danych
                if next_page_offset is None:
                    break

            logger.info(f"✅ Pomyślnie pobrano {len(all_points_data)} punktów z kolekcji '{self.collection_name}'.")
            return all_points_data

        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania danych z Qdrant za pomocą scroll: {str(e)}")
            return [] # Zwróć pustą listę w przypadku błędu
    # === KONIEC POPRAWNIE DODANEJ METODY get_all_data_for_clustering ===

    # === POPRAWIONA METODA AKTUALIZACJI PAYLOADU ===
    def update_payload_with_cluster_ids(self, assignments: Dict[str, int]):
        """
        Aktualizuje pole 'cluster_id' w payloadzie punktów w Qdrant na podstawie
        słownika przypisań przy użyciu client.upsert.

        Args:
            assignments (Dict[str, int]): Słownik mapujący ID punktu (str) na ID klastra (int).

        Returns:
            bool: True jeśli operacja się powiodła (lub nie było nic do zrobienia),
                  False w przypadku błędu.
        """
        if not assignments:
            logger.info("ℹ️ Brak przypisań klastrów do zaktualizowania w Qdrant.")
            return True
        if not self.client:
            logger.error("❌ Błąd: Klient Qdrant nie jest zainicjalizowany w update_payload_with_cluster_ids.")
            return False

        logger.info(f"⚙️ Rozpoczynanie aktualizacji payloadów dla {len(assignments)} punktów w Qdrant (pole: cluster_id)...")
        start_time = time.time()

        # Przygotuj punkty do aktualizacji
        points_to_update = []
        point_ids_processed = set() # Do śledzenia unikalnych ID

        for pid, cluster_id_val_raw in assignments.items():
            if not isinstance(pid, str):
                 logger.warning(f"Pominięto aktualizację dla punktu - nieprawidłowy typ ID: {type(pid)} (wartość: {repr(pid)})")
                 continue

            if pid in point_ids_processed:
                 logger.warning(f"Pominięto duplikat ID punktu w assignments: {pid}")
                 continue
            point_ids_processed.add(pid)

            try:
                # Upewnij się, że cluster_id jest poprawnym intem
                if isinstance(cluster_id_val_raw, (int, float)):
                     cluster_id_val = int(cluster_id_val_raw)
                elif hasattr(cluster_id_val_raw, 'item'): # Obsługa typów NumPy
                     cluster_id_val = int(cluster_id_val_raw.item())
                else:
                     raise TypeError(f"Nieprawidłowy typ dla cluster_id: {type(cluster_id_val_raw)}")

                points_to_update.append(
                    rest.PointStruct(
                        id=pid,
                        payload={"cluster_id": cluster_id_val}, # Tworzymy payload do aktualizacji
                        vector={} # Pusty wektor, bo tylko aktualizujemy payload
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Pominięto aktualizację dla punktu {pid} z powodu nieprawidłowego cluster_id '{cluster_id_val_raw}' (typ: {type(cluster_id_val_raw)}): {e}")

        if not points_to_update:
             logger.error("❌ Brak prawidłowych punktów do aktualizacji payloadu cluster_id po walidacji.")
             return False # Zwracamy False, bo były przypisania, ale żadne nie było poprawne

        logger.info(f"   Przygotowano {len(points_to_update)} punktów do aktualizacji payloadu (cluster_id) za pomocą upsert.")

        try:
            # Użyj upsert do aktualizacji payloadu punktów wsadowo
            update_result: UpdateResult = self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_update,
                wait=True
            )

            end_time = time.time()
            logger.info(f"⏱️ Aktualizacja payloadów Qdrant (upsert wsadowo) zajęła: {end_time - start_time:.2f}s")

            # Sprawdzanie statusu operacji upsert (bardziej odporne)
            update_status = "unknown"
            op_info = getattr(update_result, 'result', getattr(update_result, 'operation_info', None)) # Sprawdź oba możliwe atrybuty
            if op_info and hasattr(op_info, 'status'):
                 update_status = op_info.status
            elif hasattr(update_result, 'status'): # Fallback dla starszych wersji?
                 update_status = update_result.status

            # Konwersja statusu z obiektu enum na string, jeśli to konieczne
            if hasattr(update_status, 'value'): update_status = update_status.value

            if str(update_status).lower() in ["completed", "acknowledged"]:
                logger.info(f"✅ Pomyślnie ustawiono payload (cluster_id) dla {len(points_to_update)} punktów w Qdrant.")
                return True
            else:
                logger.error(f"❌ Ustawienie payloadów Qdrant zakończone statusem: {update_status}")
                return False
        except Exception as e:
             logger.error(f"❌ Błąd podczas ustawiania payloadów w Qdrant (upsert wsadowo): {e}", exc_info=True)
             return False
    # === KONIEC POPRAWIONEJ METODY AKTUALIZACJI PAYLOADU ===

    # === POPRAWIONA METODA ZAPISU METADANYCH KLASTRÓW ===
    def save_cluster_metadata_to_qdrant(self, cluster_metadata: Dict[int, Dict[str, Any]]):
        """
        Zapisuje metadane klastrów (etykiety, podsumowania, encje) do dedykowanej kolekcji w Qdrant.
        Każdy klaster jest zapisywany jako osobny punkt, gdzie ID punktu to ID klastra (int),
        a payload to słownik z metadanymi (label, summary, entities).

        Args:
            cluster_metadata (Dict[Any, Dict[str, Any]]): Słownik mapujący ID klastra (może być np.int64)
                                                          na słownik metadanych.

        Returns:
            bool: True jeśli operacja się powiodła, False w przypadku błędu lub braku danych do zapisu.
        """
        if not cluster_metadata:
            logger.info("ℹ️ Brak metadanych klastrów do zapisania w Qdrant.")
            return True # Nie ma nic do zrobienia
        if not self.client:
            logger.error("❌ Błąd: Klient Qdrant nie jest zainicjalizowany w save_cluster_metadata_to_qdrant.")
            return False

        # Upewnij się, że kolekcja metadanych istnieje
        self._initialize_metadata_collection()

        logger.info(f"⚙️ Rozpoczynanie zapisu metadanych dla {len(cluster_metadata)} klastrów do kolekcji '{self.cluster_metadata_collection_name}'...")
        start_time = time.time()

        points_to_upsert = []
        skipped_clusters = []
        processed_clusters = 0

        for cluster_id_raw, metadata in cluster_metadata.items():
            processed_clusters += 1
            # Dodano bardziej szczegółowe logowanie PRZED próbą konwersji
            logger.info(f"--- ({processed_clusters}/{len(cluster_metadata)}) Rozpoczynam przetwarzanie metadanych dla cluster_id: {repr(cluster_id_raw)} (typ: {type(cluster_id_raw)}) ---")

            try:
                # Bardziej odporna konwersja ID klastra na int
                int_cluster_id = -999 # Wartość wskazująca na błąd
                if isinstance(cluster_id_raw, (int, float)):
                    int_cluster_id = int(cluster_id_raw)
                elif isinstance(cluster_id_raw, str) and cluster_id_raw.isdigit():
                    int_cluster_id = int(cluster_id_raw)
                elif hasattr(cluster_id_raw, 'item') and callable(getattr(cluster_id_raw, 'item')):
                    try:
                        int_cluster_id = int(cluster_id_raw.item())
                        logger.debug(f"   Konwersja z typu NumPy na int: {int_cluster_id}")
                    except (ValueError, TypeError) as numpy_conv_err:
                         raise TypeError(f"Błąd konwersji .item() dla typu NumPy: {numpy_conv_err}") from numpy_conv_err
                else:
                    raise TypeError(f"Nieobsługiwany typ lub format cluster_id: {type(cluster_id_raw)}")

                logger.info(f"   ID klastra po konwersji: {int_cluster_id} (typ: {type(int_cluster_id)})")

                if int_cluster_id == -1:
                    logger.info(f"   Pominięto klaster szumu (-1): {cluster_id_raw}")
                    skipped_clusters.append(f"{cluster_id_raw} (szum)")
                    continue # Idź do następnej iteracji pętli

                # Walidacja pozostałych metadanych
                label = metadata.get("label")
                summary = metadata.get("summary")
                entities_list_raw = metadata.get("entities", []) # Pobierz surową listę

                # --- Walidacja etykiety ---
                if not label or not isinstance(label, str):
                     logger.warning(f"   Pominięto klaster {int_cluster_id}: Brak lub nieprawidłowy typ etykiety: {repr(label)} (typ: {type(label)})")
                     skipped_clusters.append(f"{int_cluster_id} (błąd etykiety)")
                     continue

                # --- Walidacja i serializacja encji ---
                if not isinstance(entities_list_raw, list):
                    logger.warning(f"   Oczekiwano listy encji dla klastra {int_cluster_id}, otrzymano {type(entities_list_raw)}. Ustawiam na pustą listę.")
                    serializable_entities = []
                else:
                    serializable_entities = []
                    for i, entity in enumerate(entities_list_raw):
                        if isinstance(entity, (list, tuple)) and len(entity) == 3:
                            # Sprawdź, czy elementy wewnętrzne są serializowalne (np. stringi, liczby)
                            try:
                                # Próba konwersji na listę, która jest bezpieczna dla JSON
                                entity_as_list = list(entity)
                                # Dodatkowe sprawdzenie, czy elementy nie są jakimiś obiektami
                                if all(isinstance(item, (str, int, float, bool)) or item is None for item in entity_as_list):
                                     serializable_entities.append(entity_as_list)
                                else:
                                     logger.warning(f"   Pominięto encję {i} w klastrze {int_cluster_id} z powodu nieserializowalnych typów: {entity_as_list}")
                            except Exception as inner_conv_err:
                                logger.warning(f"   Pominięto encję {i} w klastrze {int_cluster_id} z powodu błędu konwersji na listę: {inner_conv_err}")
                        else:
                            logger.warning(f"   Pominięto nieprawidłowy format encji {i} w klastrze {int_cluster_id}: {repr(entity)} (typ: {type(entity)}, długość: {len(entity) if hasattr(entity, '__len__') else 'N/A'})")

                # --- Przygotowanie Payload ---
                final_summary = summary if isinstance(summary, str) else "Brak podsumowania."
                payload_dict: Dict[str, Any] = {
                    "label": label,
                    "summary": final_summary,
                    "entities": serializable_entities
                }

                # --- Logowanie tuż przed dodaniem ---
                logger.info(f"   >>> Przygotowano do dodania: ID={int_cluster_id}, Payload={payload_dict}")

                # --- Dodanie PointStruct ---
                points_to_upsert.append(
                    rest.PointStruct(
                        id=int_cluster_id, # Używamy int jako ID
                        vector={}, # Brak wektorów
                        payload=payload_dict
                    )
                )
                logger.info(f"   +++ Pomyślnie dodano PointStruct dla klastra {int_cluster_id} do listy upsert.")

            except (ValueError, TypeError, Exception) as e: # Rozszerzono obsługę błędów
                 logger.error(f"   Pominięto klaster z ID '{cluster_id_raw}' z powodu błędu przetwarzania: {e}", exc_info=True)
                 skipped_clusters.append(f"{cluster_id_raw} (błąd przetwarzania)")
                 continue # Idź do następnej iteracji pętli

        if skipped_clusters:
             logger.warning(f"Pominięto zapis metadanych dla {len(skipped_clusters)}/{processed_clusters} klastrów z powodu szumu lub błędów: {skipped_clusters}")

        if not points_to_upsert:
            logger.error("❌ Brak prawidłowych metadanych klastrów do zapisania w Qdrant (po odfiltrowaniu/błędach). Operacja zapisu przerwana.")
            # Zwracamy False, bo nie udało się przygotować żadnych danych do zapisu, mimo że były na wejściu.
            return False

        # Jeśli mamy punkty do zapisania, kontynuujemy
        try:
            logger.info(f"Rozpoczynanie operacji upsert dla {len(points_to_upsert)} punktów metadanych...")
            update_result: UpdateResult = self.client.upsert(
                collection_name=self.cluster_metadata_collection_name,
                points=points_to_upsert,
                wait=True # Poczekaj na zakończenie
            )

            end_time = time.time()
            logger.info(f"⏱️ Zapis metadanych klastrów do Qdrant (upsert wsadowo) zajął: {end_time - start_time:.2f}s")

            # Sprawdzanie statusu operacji upsert (bardziej odporne)
            update_status = "unknown"
            op_info = getattr(update_result, 'result', getattr(update_result, 'operation_info', None)) # Sprawdź oba możliwe atrybuty
            if op_info and hasattr(op_info, 'status'):
                 update_status = op_info.status
            elif hasattr(update_result, 'status'): # Fallback dla starszych wersji?
                 update_status = update_result.status

            # Konwersja statusu z obiektu enum na string, jeśli to konieczne
            if hasattr(update_status, 'value'): update_status = update_status.value

            if str(update_status).lower() in ["completed", "acknowledged"]:
                logger.info(f"✅ Pomyślnie zapisano/zaktualizowano metadane dla {len(points_to_upsert)} klastrów w Qdrant (status: {update_status}).")
                return True
            else:
                logger.error(f"❌ Zapis metadanych klastrów do Qdrant zakończony statusem: {update_status}")
                return False
        except Exception as e:
             logger.error(f"❌ Błąd podczas zapisu metadanych klastrów do Qdrant (upsert wsadowo): {e}", exc_info=True)
             return False
    # === KONIEC POPRAWIONEJ METODY ZAPISU METADANYCH KLASTRÓW ===

    # === NOWA METODA DO ŁADOWANIA METADANYCH KLASTRÓW ===
    def load_cluster_metadata_from_qdrant(self) -> Dict[int, Dict[str, Any]]:
        """
        Ładuje metadane klastrów (etykiety, podsumowania, encje) z dedykowanej kolekcji w Qdrant.

        Returns:
            Dict[int, Dict[str, Any]]: Słownik mapujący ID klastra (int) na słownik metadanych.
                                        Zwraca pusty słownik w przypadku błędu lub braku danych.
        """
        if not self.client:
            logger.error("❌ Błąd: Klient Qdrant nie jest zainicjalizowany w load_cluster_metadata_from_qdrant.")
            return {}

        # Upewnij się, że kolekcja metadanych istnieje (choć powinna być już zainicjalizowana)
        self._initialize_metadata_collection()

        logger.info(f"⬇️ Ładowanie metadanych klastrów z kolekcji '{self.cluster_metadata_collection_name}'...")
        loaded_metadata = {}
        next_page_offset = None

        try:
            while True:
                # Używamy scroll API do pobierania partiami
                from qdrant_client.http import models as rest # Upewnijmy się, że jest w zasięgu
                response, next_page_offset = self.client.scroll(
                    collection_name=self.cluster_metadata_collection_name,
                    limit=256, # Pobieramy w rozsądnych partiach
                    offset=next_page_offset,
                    with_payload=True, # Potrzebujemy payloadu
                    with_vectors=False, # Wektory nie są potrzebne
                )

                if not response: # Jeśli nie ma więcej wyników
                    break

                # Przetwarzanie pobranych rekordów
                for record in response:
                    cluster_id = record.id # ID punktu to ID klastra (int)
                    payload = getattr(record, 'payload', {})
                    if isinstance(cluster_id, int) and payload:
                        # Odtwarzamy listę krotek dla encji
                        entities_list_of_lists = payload.get("entities", [])
                        reconstructed_entities = [tuple(entity) for entity in entities_list_of_lists if isinstance(entity, list) and len(entity) == 3]

                        loaded_metadata[cluster_id] = {
                            "label": payload.get("label"),
                            "summary": payload.get("summary"),
                            "entities": reconstructed_entities # Zapisz odtworzoną listę krotek
                        }
                    else:
                         logger.warning(f"Pominięto rekord metadanych z nieprawidłowym ID ({cluster_id}, typ: {type(cluster_id)}) lub pustym payloadem.")

                # Jeśli next_page_offset jest None, to koniec danych
                if next_page_offset is None:
                    break

            logger.info(f"✅ Pomyślnie załadowano metadane dla {len(loaded_metadata)} klastrów z kolekcji '{self.cluster_metadata_collection_name}'.")
            return loaded_metadata

        except Exception as e:
            # Sprawdź, czy błąd wynika z braku kolekcji
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                 logger.warning(f"⚠️ Kolekcja metadanych '{self.cluster_metadata_collection_name}' nie istnieje. Zwracam puste metadane.")
            else:
                 logger.error(f"❌ Błąd podczas ładowania metadanych klastrów z Qdrant za pomocą scroll: {str(e)}", exc_info=True)
            return {} # Zwróć pusty słownik w przypadku błędu
    # === KONIEC NOWEJ METODY DO ŁADOWANIA METADANYCH KLASTRÓW ===