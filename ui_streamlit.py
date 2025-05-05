"""
Interfejs użytkownika dla chatbota RAG z wykorzystaniem Streamlit
"""
import streamlit as st
from app import RAGChatbot
import logging
import os
import tempfile
import uuid # Dodano import uuid
import json # Dodano import json
import networkx as nx # Dodano import networkx
from streamlit_agraph import agraph, Node, Edge, Config # Dodano importy dla wizualizacji grafu
import textwrap # Dodano import textwrap do zawijania etykiet
import io # DODANO: Do obsługi plików w pamięci
from pypdf import PdfReader # DODANO: Do odczytu PDF
# Dodano import dla obiektu odpowiedzi LLM z LangChain, aby móc sprawdzić jego typ
from langchain_core.messages import AIMessage, HumanMessage
import warnings # Dodano do obsługi FutureWarning
import numpy as np # Dodano import numpy
# Zaktualizowano importy z clustering_module
# DODANO: import generate_cluster_summary i extract_key_entities
from clustering_module import (
    perform_clustering,
    generate_cluster_labels_llm,
    generate_cluster_summary,
    extract_key_entities
)
from collections import Counter # Do zliczania punktów w klastrach
# DODANO: Import dla typu LLM Google
from langchain_google_genai import ChatGoogleGenerativeAI
# Dodano import cdist
from scipy.spatial.distance import cdist
# Dodano import time (potrzebny w QdrantConnector, ale też może się przydać)
import time
# Dodano import typowania (potrzebny w QdrantConnector, ale też może się przydać)
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd # DODANO: Import pandas

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = "chat_history.json" # Ścieżka do pliku historii
# CLUSTER_METADATA_FILE = "cluster_metadata.json" # USUNIĘTO: Metadane będą w Qdrant

# === POCZĄTEK POPRAWKI KOLORÓW ===
# --- Mapowanie typów NER na kolory (DOSTOSOWANE DO pl_core_news_md/lg) ---
NER_COLORS = {
    "persName": "#FF6347",  # Tomato (Osoba)
    "orgName": "#4682B4",   # SteelBlue (Organizacja)
    "geogName": "#32CD32",  # LimeGreen (Nazwa geograficzna - np. kraj, rzeka)
    "placeName": "#9ACD32", # YellowGreen (Miejsce - np. miasto, adres)
    "date": "#FFD700",      # Gold (Data)
    "time": "#FFA500",      # Orange (Czas - jeśli model go rozpoznaje)
    "nam": "#A9A9A9",       # DarkGray (Inne nazwy własne - np. marki, wydarzenia - fallback)
    "ERROR": "#FF0000",     # Czerwony dla błędów NER (jeśli wystąpią w app.py)
    "DOC": "#BA55D3",       # MediumOrchid (Dokument - jeśli LLM dodałby taki typ)
}
DEFAULT_NODE_COLOR = "#D3D3D3" # LightGray (Domyślny/Nieznany/Brak encji/Nieznany typ NER)
# ------------------------------------
# === KONIEC POPRAWKI KOLORÓW ===

# --- Predefiniowane Prompty Systemowe ---
PROMPT_ANALITYK_TEKSTU = """Rola:\n\nJesteś doświadczonym analitykiem tekstu. Otrzymujesz jeden lub kilka dokumentów jednocześnie (raporty, artykuły, prezentacje, sprawozdania, e-maile, pliki PDF, Word, itp.) oraz pytania od członków zespołu. Twoim zadaniem jest znalezienie konkretnych informacji w tych materiałach i udzielenie jasnych, precyzyjnych odpowiedzi.\n\n\n\n\nTwoje zadania:\n\n\n\n\nPrzeczytaj uważnie wszystkie dostarczone źródła.\n\nDla każdego pytania:\n\nZidentyfikuj i porównaj informacje we wszystkich dostępnych dokumentach.\n\nPodaj konkretną odpowiedź opartą wyłącznie na treści źródłowej.\n\nJeśli ta sama informacja występuje w kilku miejscach – wybierz najbardziej wiarygodną i aktualną wersję.\n\nJeśli są sprzeczne dane – zaznacz to i podaj możliwe wyjaśnienie.\n\nZawsze wskaż dokładne źródło w tekście (cytat, numer akapitu, nazwa pliku lub lokalizacja).\n\nJeżeli odpowiedź nie występuje bezpośrednio w dokumentach, zaznacz to i dodaj krótką interpretację (jeśli to możliwe).\n\nFormat odpowiedzi dla każdego pytania:\n\n\n\n\nPytanie: [tu wpisz pytanie]\n\nOdpowiedź: [jasna, konkretna odpowiedź]\n\nŹródło w dokumentach: [cytat, numer akapitu, nazwa pliku, strona lub opis fragmentu]\n\nKomentarz (jeśli potrzebny): [jeśli są sprzeczności lub brak informacji – wyjaśnij to]\n\n\n\n\nRodzaje pytań, które możesz otrzymać (i jak na nie reagować):\n\n\n\n\nPytanie o osobę (np. „Kto jest szefem tej organizacji?”):\n\n→ Wskaż imię, nazwisko, stanowisko i dokument, w którym to się znajduje.\n\nPytanie o liczby (np. „Ile pieniędzy zostało zabezpieczonych?”):\n\n→ Podaj konkretną kwotę, powołując się na dane z odpowiedniego pliku. Jeśli kwoty różnią się – opisz to.\n\nPytanie o przyczyny, działania, efekty (np. „Dlaczego projekt się opóźnił?”):\n\n→ Podaj powody, działania lub skutki, nawet jeśli są rozproszone w różnych źródłach.\n\nPytania o fakty i szczegóły:\n\n→ Wydobądź najważniejsze detale z różnych dokumentów i połącz je w spójną odpowiedź.\n\nDostarczone materiały:\n\n[lista lub zestaw dokumentów i źródeł – np. Raport_finansowy_Q4.pdf, Spotkanie_zarządu_dnotatki.docx, E-mail_od_dostawcy.msg]\n\n\n\n\n📌 Przykład odpowiedzi z wieloma źródłami:\n\nPytanie: Ile pieniędzy zostało zabezpieczonych?\n\nOdpowiedź: Zabezpieczono 4,5 mln zł (Raport_finansowy_Q4.pdf), jednak w notatce ze spotkania (Spotkanie_zarządu_dnotatki.docx) pojawia się kwota 4,2 mln zł – możliwa aktualizacja danych w późniejszym okresie.\n\nŹródło w dokumentach:\n\n\n\nRaport_finansowy_Q4.pdf, strona 5: „Zabezpieczone środki wynoszą 4,5 mln zł.”\n\nSpotkanie_zarządu_dnotatki.docx, akapit 3: „Dysponujemy obecnie 4,2 mln zł zarezerwowanych środków.”\n\nKomentarz: Możliwa różnica wynika z częściowego wydatkowania środków po publikacji raportu."""

PROMPT_ANALITYK_GRAFOW = """Rola:

Jesteś doświadczonym analitykiem tekstu, specjalizującym się w wydobywaniu precyzyjnych informacji, identyfikowaniu kluczowych bytów (osób, organizacji, miejsc, dokumentów, pseudonimów, metod działania, powiązań finansowych itp.) oraz relacji między nimi w analizowanych materiałach. Otrzymujesz jeden lub kilka dokumentów jednocześnie (raporty, artykuły, prezentacje, sprawozdania, e-maile, pliki PDF, Word, itp.) oraz pytania od członków zespołu. Twoim zadaniem jest znalezienie konkretnych informacji w tych materiałach, udzielenie jasnych, precyzyjnych odpowiedzi tekstowych, które jednocześnie ułatwią późniejszą ekstrakcję powiązań do grafu wiedzy.

Twoje zadania:

*   Przeczytaj uważnie wszystkie dostarczone źródła.
*   Dla każdego pytania:
    *   Zidentyfikuj i porównaj informacje we wszystkich dostępnych dokumentach.
    *   W swojej odpowiedzi, oprócz udzielenia informacji tekstowej, jasno zidentyfikuj kluczowe byty (np. osoby używając pełnego imienia i nazwiska oraz pseudonimu, organizacje, konkretne pliki, specyficzne metody działania) oraz opisz relacje między nimi (np. "Agnieszka Malinowska ('Czas') zarządzała finansami 'Projektu Ślimak' prowadzonego przez Agnieszkę Wójcik ('Ślimak')", "Agnieszka Zając ('Guma') uzyskała papier ze znakiem wodnym od Agnieszki Jankowskiej ('Dźwięk')"). Używaj precyzyjnych czasowników opisujących relację.
    *   Podaj konkretną odpowiedź opartą wyłącznie na treści źródłowej.
    *   Jeśli ta sama informacja występuje w kilku miejscach – wybierz najbardziej wiarygodną i aktualną wersję.
    *   Jeśli są sprzeczne dane – zaznacz to i podaj możliwe wyjaśnienie.
    *   Zawsze wskaż dokładne źródło w tekście (cytat, numer akapitu, nazwa pliku lub lokalizacja). Jeśli informacja o relacji pochodzi z wielu źródeł, wskaż je.
    *   Jeżeli odpowiedź nie występuje bezpośrednio w dokumentach, zaznacz to i dodaj krótką interpretację (jeśli to możliwe).

Format odpowiedzi dla każdego pytania:

*   Pytanie: [tu wpisz pytanie]
*   Odpowiedź: [jasna, konkretna odpowiedź, zawierająca jasno zidentyfikowane byty i relacje między nimi, istotne dla odpowiedzi na pytanie]
*   Źródło w dokumentach: [cytat, numer akapitu, nazwa pliku, strona lub opis fragmentu potwierdzający odpowiedź i zidentyfikowane relacje]
*   Komentarz (jeśli potrzebny): [jeśli są sprzeczności lub brak informacji – wyjaśnij to]

Rodzaje pytań, które możesz otrzymać (i jak na nie reagować):

*   Pytanie o osobę (np. „Kto jest szefem tej organizacji?”):
    → Wskaż imię, nazwisko, stanowisko i dokument, w którym to się znajduje. **Jawnie opisz relację (np. "Jan Kowalski JEST SZEFEM organizacji ABC Corp").**
*   Pytanie o liczby (np. „Ile pieniędzy zostało zabezpieczonych?”):
    → Podaj konkretną kwotę, powołując się na dane z odpowiedniego pliku. Jeśli kwoty różnią się – opisz to. **Jeśli liczba dotyczy relacji (np. kwota przelewu), opisz ją (np. "Kwota 850 000 PLN ZOSTAŁA PRZELANA przez firmę X na konto Y").**
*   Pytanie o przyczyny, działania, efekty (np. „Dlaczego projekt się opóźnił?”):
    → Podaj powody, działania lub skutki, nawet jeśli są rozproszone w różnych źródłach. **Opisz relacje przyczynowo-skutkowe (np. "Opóźnienie projektu BYŁO SPOWODOWANE przez brak dostawy komponentów od firmy Z").**
*   Pytania o fakty i szczegóły:
    → Wydobądź najważniejsze detale z różnych dokumentów i połącz je w spójną odpowiedź. **Jeśli detale opisują cechy bytu lub relacje, zaznacz to (np. "Agnieszka Pawlak UŻYWAŁA futerału Yamaha YFL-221 DO PRZECHOWYWANIA sztabek platyny").**

Dostarczone materiały:

[lista lub zestaw dokumentów i źródeł – np. Raport_finansowy_Q4.pdf, Spotkanie_zarządu_dnotatki.docx, E-mail_od_dostawcy.msg]

📌 Przykład odpowiedzi z wieloma źródłami i uwzględnieniem relacji:

Pytanie: Ile pieniędzy zostało zabezpieczonych i kto nimi dysponował?
Odpowiedź: Zabezpieczono 4,5 mln zł środków finansowych (według Raport_finansowy_Q4.pdf). Jednak w późniejszej notatce (Spotkanie_zarządu_dnotatki.docx) wskazano, że Zarząd organizacji ABC Corp DYSPONUJE kwotą 4,2 mln zł zarezerwowanych środków.
Źródło w dokumentach:
*   Raport_finansowy_Q4.pdf, strona 5: „Zabezpieczone środki wynoszą 4,5 mln zł.”
*   Spotkanie_zarządu_dnotatki.docx, akapit 3: „Zarząd ABC Corp dysponuje obecnie 4,2 mln zł zarezerwowanych środków.”
Komentarz: Możliwa różnica wynika z częściowego wydatkowania środków przez Zarząd ABC Corp po publikacji raportu."""

# Zmieniono klucze, aby zawierały "(RAG)"
AVAILABLE_PROMPTS = {
    "Analityk Tekstu (RAG)": PROMPT_ANALITYK_TEKSTU,
    "Analityk Grafów (RAG)": PROMPT_ANALITYK_TEKSTU # ZMIANA: Użyj tego samego promptu co Analityk Tekstu do generowania odpowiedzi
}
# ---------------------------------------

# --- Funkcje do obsługi historii czatów ---
def load_chat_history():
    """Ładuje historię czatów z pliku JSON."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "chats" in data and "active_chat_id" in data:
                     logger.info(f"💾 Załadowano historię czatów z {CHAT_HISTORY_FILE}")
                     return data
                else:
                     logger.warning(f"⚠️ Plik {CHAT_HISTORY_FILE} ma nieprawidłową strukturę.")
                     return None
        except json.JSONDecodeError:
            logger.error(f"❌ Błąd dekodowania JSON w pliku {CHAT_HISTORY_FILE}. Plik może być uszkodzony.")
            return None
        except Exception as e:
            logger.error(f"❌ Nieoczekiwany błąd podczas ładowania historii z {CHAT_HISTORY_FILE}: {e}")
            return None
    else:
        logger.info(f"ℹ️ Plik historii czatów {CHAT_HISTORY_FILE} nie istnieje. Inicjalizacja nowego stanu.")
        return None

def save_chat_history(chats_data, active_chat_id):
    """Zapisuje historię czatów do pliku JSON, konwertując wiadomości i zachowując dane grafu."""
    try:
        serializable_chats = {}
        for chat_id, chat_content in chats_data.items():
            serializable_messages = []
            for msg in chat_content.get("messages", []):
                role = msg.get("role")
                msg_to_save = {"role": role}

                # --- Logika zapisu treści ---
                content_str = msg.get("content")
                if content_str is not None:
                    msg_to_save["content"] = content_str
                # ----------------------------

                # --- Logika zapisu źródeł, grafu, promptu i metadanych ---
                if "sources" in msg: msg_to_save["sources"] = msg["sources"]
                if "graph_data" in msg: msg_to_save["graph_data"] = msg["graph_data"]
                if "prompt_answered" in msg: msg_to_save["prompt_answered"] = msg["prompt_answered"]
                if "metadata" in msg: msg_to_save["metadata"] = msg["metadata"] # DODANO: Zapis metadanych
                # ------------------------------------

                if role and "content" in msg_to_save: # Zapisz tylko jeśli jest rola i treść
                    serializable_messages.append(msg_to_save)

            serializable_chats[chat_id] = {
                "name": chat_content.get("name", f"Czat {chat_id[:8]}"),
                "messages": serializable_messages
            }

        data_to_save = {
            "chats": serializable_chats,
            "active_chat_id": active_chat_id
        }

        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        # logger.info(f"💾 Zapisano historię czatów do {CHAT_HISTORY_FILE}")
    except Exception as e:
        logger.error(f"❌ Błąd podczas zapisywania historii do {CHAT_HISTORY_FILE}: {e}")
# -----------------------------------------

# --- Usunięto funkcje load_cluster_metadata i save_cluster_metadata ---

# Funkcja pomocnicza do mapowania źródła na ID punktu (wymaga dostosowania!)
# USUNIĘTO: Już niepotrzebna, bo ID jest przekazywane w sources
# def find_point_id_for_source(source_name, qdrant_data):
#     ...

def main():

    st.title("🤖 Chatbot RAG z Qdrant")

    # --- Inicjalizacja stanu sesji ---
    if "chats" not in st.session_state:
         loaded_data = load_chat_history()
         if loaded_data:
              # Przywróć stan z pliku
              st.session_state.chats = loaded_data.get("chats", {})
              active_chat_id_from_file = loaded_data.get("active_chat_id")
              if active_chat_id_from_file and active_chat_id_from_file in st.session_state.chats:
                   st.session_state.active_chat_id = active_chat_id_from_file
              elif st.session_state.chats:
                   st.session_state.active_chat_id = next(iter(st.session_state.chats))
                   logger.warning("Załadowany active_chat_id był nieprawidłowy. Ustawiono pierwszy dostępny.")
              else: # Plik istnieje, ale jest pusty lub uszkodzony
                   first_chat_id = str(uuid.uuid4())
                   st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
                   st.session_state.active_chat_id = first_chat_id
                   logger.warning("Plik historii istniał, ale był pusty lub nieprawidłowy. Utworzono nowy domyślny czat.")
                   save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
         else: # Plik nie istnieje lub błąd ładowania
              first_chat_id = str(uuid.uuid4())
              st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
              st.session_state.active_chat_id = first_chat_id
              logger.info("Nie załadowano historii. Utworzono nowy domyślny czat.")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
    # Sprawdź, czy active_chat_id jest nadal prawidłowy po potencjalnym usunięciu czatu
    elif "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
         if st.session_state.chats:
              st.session_state.active_chat_id = next(iter(st.session_state.chats))
              logger.warning(f"Active_chat_id brakujący lub nieprawidłowy po inicjalizacji. Ustawiono na: {st.session_state.active_chat_id}")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
         else: # Stan awaryjny - brak czatów
              first_chat_id = str(uuid.uuid4())
              st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
              st.session_state.active_chat_id = first_chat_id
              logger.error("Stan awaryjny: Brak czatów i active_chat_id po inicjalizacji. Utworzono nowy domyślny czat.")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)

    # Inicjalizacja pozostałych kluczy stanu sesji
    if 'selected_chat_mode' not in st.session_state:
        st.session_state.selected_chat_mode = "Analityk Tekstu (RAG)"
    if 'use_gemini_api' not in st.session_state:
        st.session_state.use_gemini_api = False
    if 'filter_small_graphs' not in st.session_state:
        st.session_state.filter_small_graphs = True
    if 'cluster_assignments' not in st.session_state:
        st.session_state.cluster_assignments = None # Inicjalizuj jako None
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = {} # Inicjalizuj jako pusty słownik
    if 'qdrant_data_cache' not in st.session_state:
        st.session_state.qdrant_data_cache = None
    if 'selected_cluster_id' not in st.session_state:
        st.session_state.selected_cluster_id = None
    if 'cluster_summaries' not in st.session_state:
        st.session_state.cluster_summaries = {} # Pusty słownik
    if 'cluster_entities' not in st.session_state:
        st.session_state.cluster_entities = {} # Pusty słownik
    if 'cluster_centroids' not in st.session_state:
        st.session_state.cluster_centroids = None
    if 'cluster_indices_map' not in st.session_state:
        st.session_state.cluster_indices_map = None
    if 'point_id_to_text_map' not in st.session_state:
        st.session_state.point_id_to_text_map = None
    if 'point_id_list_ordered' not in st.session_state:
        st.session_state.point_id_list_ordered = None
    if 'embeddings_matrix_cache' not in st.session_state:
        st.session_state.embeddings_matrix_cache = None

    if 'new_user_input_submitted' not in st.session_state:
        st.session_state.new_user_input_submitted = False

    if 'filter_by_selected_cluster' not in st.session_state:
        st.session_state.filter_by_selected_cluster = False
    # DODANO: Inicjalizacja dla plików kontekstowych
    if 'context_files' not in st.session_state:
        st.session_state.context_files = {} # Słownik: {nazwa_pliku: tresc}

    # --- Inicjalizacja chatbota ---
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = RAGChatbot()
            logger.info("✅ Pomyślnie zainicjalizowano chatbota RAG")
        except Exception as e:
            logger.error(f"❌ Błąd inicjalizacji chatbota: {str(e)}")
            st.error(f"Nie udało się zainicjalizować chatbota. Sprawdź logi lub upewnij się, że Qdrant i LM Studio działają. Błąd: {e}")
            return
    # -----------------------------

    # --- Ładowanie danych klastrowania (assignments z Qdrant, reszta z pliku JSON) ---
    if 'cluster_load_status' not in st.session_state:
        st.session_state.cluster_load_status = "not_loaded" # Możliwe statusy: not_loaded, loaded_ok, loaded_empty, error

    if 'cluster_data_loaded' not in st.session_state: # Flaga, aby ładować tylko raz na sesję
        # Najpierw załaduj przypisania z Qdrant
        if "chatbot" in st.session_state and hasattr(st.session_state.chatbot, 'qdrant_connector'):
            logger.info("🌀 Próba załadowania przypisań klastrów z metadanych Qdrant na starcie aplikacji...")
            try:
                qdrant_data_on_start = st.session_state.chatbot.qdrant_connector.get_all_data_for_clustering(
                    with_payload=True, with_vectors=False, limit=None
                )
                if qdrant_data_on_start:
                    loaded_assignments = {}
                    found_any_cluster_id = False
                    for point in qdrant_data_on_start:
                        point_id = point.get('id')
                        payload = point.get('payload')
                        if point_id and payload and 'cluster_id' in payload:
                             cluster_id_value = payload['cluster_id']
                             if isinstance(cluster_id_value, int):
                                 loaded_assignments[point_id] = cluster_id_value
                                 found_any_cluster_id = True
                             else:
                                 logger.warning(f"   Znaleziono nieprawidłowy typ dla cluster_id w payloadzie punktu {point_id}: {type(cluster_id_value)}")
                        elif point_id and payload: pass
                        elif point_id: logger.warning(f"   Punkt {point_id} nie ma payloadu w danych z Qdrant.")

                    if loaded_assignments:
                        st.session_state.cluster_assignments = loaded_assignments
                        st.session_state.cluster_load_status = "loaded_ok" # Ustawiamy OK, bo mamy przypisania
                        logger.info(f"✅ Załadowano {len(loaded_assignments)} przypisań klastrów z metadanych Qdrant.")
                    elif found_any_cluster_id:
                         st.session_state.cluster_assignments = {} # Puste, ale poprawne
                         st.session_state.cluster_load_status = "loaded_empty"
                         logger.warning("⚠️ Znaleziono pola cluster_id, ale miały nieprawidłowy typ.")
                    else:
                         logger.info("ℹ️ Nie znaleziono informacji o klastrach ('cluster_id') w metadanych Qdrant.")
                         st.session_state.cluster_assignments = None # Ustaw na None, jeśli brak jakichkolwiek danych
                         st.session_state.cluster_load_status = "loaded_empty"
                else:
                    logger.warning("⚠️ Nie udało się pobrać danych z Qdrant na starcie lub kolekcja pusta.")
                    st.session_state.cluster_assignments = None
                    st.session_state.cluster_load_status = "error"
            except Exception as load_exc:
                logger.error(f"❌ Wyjątek podczas ładowania danych klastrowania z Qdrant: {load_exc}", exc_info=True)
                st.session_state.cluster_assignments = None
                st.session_state.cluster_load_status = "error"
        else:
            logger.error("❌ Nie można załadować danych klastrowania - obiekt chatbot lub qdrant_connector niedostępny.")
            st.session_state.cluster_assignments = None
            st.session_state.cluster_load_status = "error"

        # --- ŁADOWANIE METADANYCH KLASTRÓW (ETYKIETY, PODSUMOWANIA, ENCJE) Z QDRANT ---
        logger.info("🌀 Próba załadowania metadanych klastrów (etykiety, podsumowania, encje) z Qdrant...")
        loaded_metadata = None
        if "chatbot" in st.session_state and hasattr(st.session_state.chatbot, 'qdrant_connector'):
            try:
                # Wywołaj nową (lub istniejącą) metodę do ładowania metadanych
                loaded_metadata = st.session_state.chatbot.qdrant_connector.load_cluster_metadata_from_qdrant()
                if loaded_metadata:
                    logger.info(f"✅ Załadowano {len(loaded_metadata)} rekordów metadanych klastrów z Qdrant.")
                    # Rozpakuj metadane do odpowiednich stanów sesji
                    st.session_state.cluster_labels = {k: v.get('label') for k, v in loaded_metadata.items() if v.get('label')}
                    st.session_state.cluster_summaries = {k: v.get('summary') for k, v in loaded_metadata.items() if v.get('summary')}
                    st.session_state.cluster_entities = {k: v.get('entities', []) for k, v in loaded_metadata.items()} # Zawsze przypisz listę, nawet pustą
                else:
                    logger.info("ℹ️ Nie znaleziono zapisanych metadanych klastrów w Qdrant.")
                    # Upewnij się, że stany są puste, jeśli nic nie załadowano
                    st.session_state.cluster_labels = {}
                    st.session_state.cluster_summaries = {}
                    st.session_state.cluster_entities = {}
            except AttributeError as ae:
                 # Jeśli metoda load_cluster_metadata_from_qdrant jeszcze nie istnieje
                 if 'load_cluster_metadata_from_qdrant' in str(ae):
                      logger.warning("⚠️ Metoda 'load_cluster_metadata_from_qdrant' nie istnieje w QdrantConnector. Metadane nie zostaną załadowane.")
                 else:
                      logger.error(f"❌ Błąd atrybutu podczas ładowania metadanych klastrów: {ae}")
                 st.session_state.cluster_labels = {}
                 st.session_state.cluster_summaries = {}
                 st.session_state.cluster_entities = {}
            except Exception as meta_load_exc:
                logger.error(f"❌ Wyjątek podczas ładowania metadanych klastrów z Qdrant: {meta_load_exc}", exc_info=True)
                st.session_state.cluster_labels = {}
                st.session_state.cluster_summaries = {}
                st.session_state.cluster_entities = {}
        else:
            logger.error("❌ Nie można załadować metadanych klastrów - obiekt chatbot lub qdrant_connector niedostępny.")
            st.session_state.cluster_labels = {}
            st.session_state.cluster_summaries = {}
            st.session_state.cluster_entities = {}
        # ---------------------------------------------------------------------------------

        # Inicjalizacja pozostałych stanów metadanych (jeśli nie istnieją)
        # Te linie są teraz mniej krytyczne, bo powyższy blok powinien je ustawić, ale zostawmy dla bezpieczeństwa
        if 'cluster_labels' not in st.session_state: st.session_state.cluster_labels = {}
        if 'cluster_summaries' not in st.session_state: st.session_state.cluster_summaries = {}
        if 'cluster_entities' not in st.session_state: st.session_state.cluster_entities = {}
        if 'show_cluster_inconsistency_warning' not in st.session_state: st.session_state.show_cluster_inconsistency_warning = False # Reset flagi

        # --- Sprawdzanie spójności danych klastrowania (opcjonalne, na razie zakomentowane) ---
        # Można dodać logikę sprawdzania spójności między liczbą przypisań a liczbą metadanych,
        # ale wymagałoby to dodania np. liczby punktów do zapisywanych metadanych.
        # Na razie zakładamy, że dane są spójne lub ostrzeżenie nie jest konieczne.
        # if 'show_cluster_inconsistency_warning' not in st.session_state:
        #      st.session_state.show_cluster_inconsistency_warning = False
        # assignments_from_qdrant = st.session_state.get('cluster_assignments')
        # qdrant_assignments_count = len(assignments_from_qdrant) if assignments_from_qdrant is not None else 0
        # metadata_count = len(st.session_state.get('cluster_labels', {})) # Sprawdzamy np. po etykietach
        # if qdrant_assignments_count > 0 and metadata_count > 0 and qdrant_assignments_count != metadata_count:
        #      logger.warning(f"⚠️ Potencjalna niespójność danych klastrowania! Liczba przypisań w Qdrant ({qdrant_assignments_count}) różni się od liczby załadowanych metadanych ({metadata_count}).")
        #      st.session_state.show_cluster_inconsistency_warning = True
        # else:
        #      st.session_state.show_cluster_inconsistency_warning = False
        # --------------------------------------------------------------------------------

        # USUNIĘTO stary blok warunkowy 'if loaded_metadata:' i powiązaną logikę ładowania z pliku JSON
        # oraz logikę sprawdzania spójności opartą na pliku JSON.

        # Inicjalizacja flagi ostrzeżenia (jeśli jej nie ma) - teraz zawsze False na starcie,
        # ponieważ nie sprawdzamy już spójności z nieistniejącym plikiem JSON.
        if 'show_cluster_inconsistency_warning' not in st.session_state:
             st.session_state.show_cluster_inconsistency_warning = False

        st.session_state.cluster_data_loaded = True # Oznacz próbę ładowania jako zakończoną
    # ---------------------------------------------------

    # Panel boczny z opcjami
    with st.sidebar:
        st.header("💬 Czaty")

        if st.button("+ Nowy Czat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            chat_count = len(st.session_state.chats) + 1
            st.session_state.chats[new_chat_id] = {"name": f"Czat {chat_count}", "messages": []}
            st.session_state.active_chat_id = new_chat_id
            if "editing_chat_id" in st.session_state: del st.session_state.editing_chat_id
            # Resetuj metadane klastrów przy tworzeniu nowego czatu (przypisania zostają w Qdrant)
            st.session_state.cluster_labels = {}
            st.session_state.qdrant_data_cache = None
            st.session_state.selected_cluster_id = None
            st.session_state.cluster_summaries = {}
            st.session_state.cluster_entities = {}
            # Usunięto logikę usuwania pliku metadanych
            # try:
            #      if os.path.exists(CLUSTER_METADATA_FILE): os.remove(CLUSTER_METADATA_FILE)
            # except OSError as e: logger.error(f"Nie udało się usunąć pliku {CLUSTER_METADATA_FILE} przy tworzeniu nowego czatu: {e}")
            save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
            st.rerun()

        st.markdown("---")

        editing_chat_id_local = st.session_state.get("editing_chat_id")
        chat_to_delete_id_local = st.session_state.get("chat_to_delete_id")

        # Sortowanie czatów alfabetycznie dla lepszej organizacji
        sorted_chat_items = sorted(st.session_state.chats.items(), key=lambda item: item[1]['name'])

        for chat_id, chat_data in sorted_chat_items:
            if chat_id == editing_chat_id_local or chat_id == chat_to_delete_id_local:
                 continue

            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col1:
                button_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
                if st.button(f"{chat_data['name']}", key=f"select_{chat_id}", use_container_width=True, type=button_type):
                    st.session_state.active_chat_id = chat_id
                    if "editing_chat_id" in st.session_state: del st.session_state.editing_chat_id
                    if "chat_to_delete_id" in st.session_state: del st.session_state.chat_to_delete_id
                    # Resetuj wybór klastra i flagę filtrowania przy zmianie czatu
                    st.session_state.selected_cluster_id = None
                    st.session_state.filter_by_selected_cluster = False
                    st.rerun()
            with col2:
                 if st.button("✏️", key=f"edit_{chat_id}", help="Edytuj nazwę czatu"):
                     st.session_state.editing_chat_id = chat_id
                     if "chat_to_delete_id" in st.session_state: del st.session_state.chat_to_delete_id
                     st.rerun()
            with col3:
                 can_delete = len(st.session_state.chats) > 1
                 if st.button("🗑️", key=f"delete_{chat_id}", help="Usuń czat", disabled=not can_delete):
                     st.session_state.chat_to_delete_id = chat_id
                     if "editing_chat_id" in st.session_state: del st.session_state.editing_chat_id
                     st.rerun()

        st.markdown("---")

        if editing_chat_id_local:
            st.subheader("Edytuj nazwę czatu")
            current_name = st.session_state.chats[editing_chat_id_local]["name"]
            new_name = st.text_input("Nowa nazwa:", value=current_name, key=f"input_edit_{editing_chat_id_local}")
            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                if st.button("Zapisz", key=f"save_edit_{editing_chat_id_local}", use_container_width=True):
                    if new_name.strip():
                         st.session_state.chats[editing_chat_id_local]["name"] = new_name.strip()
                         del st.session_state.editing_chat_id
                         save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                         st.rerun()
                    else:
                         st.warning("Nazwa czatu nie może być pusta.")
            with edit_col2:
                if st.button("Anuluj", key=f"cancel_edit_{editing_chat_id_local}", use_container_width=True):
                    del st.session_state.editing_chat_id
                    st.rerun()

        if chat_to_delete_id_local:
             st.warning(f"Czy na pewno chcesz usunąć czat '{st.session_state.chats[chat_to_delete_id_local]['name']}'?")
             del_col1, del_col2 = st.columns(2)
             with del_col1:
                  if st.button("Tak, usuń", key=f"confirm_delete_{chat_to_delete_id_local}", use_container_width=True, type="primary"):
                       deleted_chat_id = chat_to_delete_id_local
                       del st.session_state.chats[deleted_chat_id]
                       del st.session_state.chat_to_delete_id
                       if st.session_state.active_chat_id == deleted_chat_id:
                            if st.session_state.chats:
                                 st.session_state.active_chat_id = sorted(st.session_state.chats.keys(), key=lambda k: st.session_state.chats[k]['name'])[0]
                            else:
                                 if "active_chat_id" in st.session_state: del st.session_state.active_chat_id
                       save_chat_history(st.session_state.chats, st.session_state.get("active_chat_id"))
                       st.rerun()
             with del_col2:
                  if st.button("Nie, anuluj", key=f"cancel_delete_{chat_to_delete_id_local}", use_container_width=True):
                       del st.session_state.chat_to_delete_id
                       st.rerun()

        st.markdown("---")
        st.header("🤖 Wybór Modelu")

        # Checkbox do przełączania na Gemini API
        st.session_state.use_gemini_api = st.checkbox(
            "✨ Użyj Google Gemini API (pomija RAG)",
            key="gemini_api_checkbox",
            value=st.session_state.use_gemini_api,
            help="Jeśli zaznaczone, zapytania będą kierowane bezpośrednio do Google Gemini Flash API (wymaga klucza API). Pomija wyszukiwanie w dokumentach (RAG)."
        )

        # Ukryj wybór trybu RAG i ustawienia, jeśli Gemini API jest aktywne
        if not st.session_state.use_gemini_api:
            st.header("🎭 Wybierz Tryb Czatu (Lokalny LLM)")
            ALL_CHAT_MODES = ["Zwykły Chat"] + list(AVAILABLE_PROMPTS.keys())
            # Upewnij się, że wybrany tryb jest w opcjach, fallback na pierwszy
            current_mode_index = 0
            if st.session_state.selected_chat_mode in ALL_CHAT_MODES:
                current_mode_index = ALL_CHAT_MODES.index(st.session_state.selected_chat_mode)
            else:
                st.session_state.selected_chat_mode = ALL_CHAT_MODES[0] # Fallback

            st.session_state.selected_chat_mode = st.radio(
                "Wybierz tryb:",
                options=ALL_CHAT_MODES,
                key="mode_selector",
                index=current_mode_index
            )
            if st.session_state.selected_chat_mode != "Zwykły Chat":
                with st.expander("Podgląd wybranego promptu systemowego"):
                     st.markdown(f"```\n{AVAILABLE_PROMPTS[st.session_state.selected_chat_mode]}\n```")
            st.markdown("---")

            st.header("⚙️ Ustawienia RAG")
            st.session_state.top_k = st.slider("Liczba dok. z Qdrant", 1, 200, st.session_state.get('top_k', 99))
            st.session_state.top_k_reranker = st.slider("Liczba dok. po rerankingu", 1, 100, st.session_state.get('top_k_reranker', 33))
            st.session_state.relevance_threshold = st.slider("Próg istotności rerankera", 0.0, 1.0, st.session_state.get('relevance_threshold', 0.0), 0.01)
            st.markdown("---")
        else:
            st.info("Tryby RAG i ich ustawienia są niedostępne, gdy aktywny jest tryb Google Gemini API.")
            st.markdown("---")


        # === SEKCJA KLASTROWANIA ===
        st.header("🔬 Analiza Klastrowania")
        num_clusters_kmeans = st.number_input("Liczba klastrów (dla K-Means):", min_value=2, max_value=50, value=8, step=1, key="kmeans_clusters")

        if st.button("🚀 Analizuj / Odśwież Klastry", key="analyze_clusters_button"):
            with st.spinner("Pobieranie danych, wykonywanie klastrowania i generowanie metadanych..."):
                try:
                    # Krok 1: Pobierz dane
                    qdrant_data = st.session_state.chatbot.qdrant_connector.get_all_data_for_clustering(with_payload=True, with_vectors=True)
                    st.session_state.qdrant_data_cache = qdrant_data # Cache dla UI
                    st.session_state.selected_cluster_id = None # Resetuj wybór

                    if not qdrant_data:
                         st.error("Nie udało się pobrać danych z Qdrant lub kolekcja jest pusta.")
                         st.session_state.cluster_assignments = None
                         st.session_state.cluster_labels = {}
                         st.session_state.cluster_summaries = {}
                         st.session_state.cluster_entities = {}
                         st.session_state.cluster_load_status = "loaded_empty"
                         # Usunięto usuwanie pliku metadanych
                    else:
                         # Przygotuj dane wejściowe dla klastrowania
                         point_ids = [d['id'] for d in qdrant_data if d.get('vector') is not None]
                         embeddings_matrix = np.array([d['vector'] for d in qdrant_data if d.get('vector') is not None])
                         point_id_to_text_map = {d['id']: d.get('payload', {}).get('page_content', '') for d in qdrant_data if d.get('payload')}
                         point_id_list_ordered = point_ids
                         # Zapisz potrzebne dane w stanie sesji
                         st.session_state.point_id_to_text_map = point_id_to_text_map
                         st.session_state.point_id_list_ordered = point_id_list_ordered
                         st.session_state.embeddings_matrix_cache = embeddings_matrix

                         if embeddings_matrix.shape[0] > 0:
                             # Krok 2: Wykonaj klastrowanie
                             cluster_labels_array, centroids_array, cluster_indices_map = perform_clustering(
                                 embeddings_matrix, algorithm='kmeans', n_clusters=num_clusters_kmeans
                             )

                             if cluster_labels_array is not None and cluster_indices_map is not None:
                                 # Zapisz wyniki klastrowania w stanie sesji
                                 st.session_state.cluster_assignments = dict(zip(point_ids, cluster_labels_array))
                                 st.session_state.cluster_centroids = centroids_array
                                 st.session_state.cluster_indices_map = cluster_indices_map
                                 st.success(f"✅ Klastrowanie zakończone. Znaleziono {len(set(cluster_labels_array) - {-1})} klastrów dla {len(point_ids)} punktów.")

                                 # Krok 3: Zaktualizuj Qdrant
                                 if st.session_state.cluster_assignments:
                                     with st.spinner("Aktualizacja ID klastrów w bazie wektorowej..."):
                                         update_success = st.session_state.chatbot.qdrant_connector.update_payload_with_cluster_ids(
                                             st.session_state.cluster_assignments
                                         )
                                         if update_success: logger.info("Aktualizacja payloadów Qdrant zakończona sukcesem.")
                                         else: st.error("⚠️ Wystąpił błąd podczas aktualizacji ID klastrów w bazie Qdrant.")
                                 else: st.warning("Nie utworzono przypisań klastrów, pomijanie aktualizacji Qdrant.")

                                 # Krok 4: Generuj i zapisz metadane (etykiety, podsumowania, encje) DLA WSZYSTKICH KLASTRÓW
                                 generated_labels = {}
                                 generated_summaries = {}
                                 generated_entities = {}
                                 all_metadata_generated_successfully = True

                                 llm_available = 'chatbot' in st.session_state and hasattr(st.session_state.chatbot, 'llm')
                                 spacy_available = 'chatbot' in st.session_state and hasattr(st.session_state.chatbot, 'nlp') and st.session_state.chatbot.nlp is not None

                                 cluster_ids_to_process = list(cluster_indices_map.keys())
                                 progress_bar_meta = st.progress(0.0)
                                 status_text_meta = st.empty()
                                 status_text_meta.text("Rozpoczynanie generowania metadanych dla klastrów...")

                                 for i, cluster_id in enumerate(cluster_ids_to_process):
                                     current_progress = (i + 1) / len(cluster_ids_to_process)
                                     status_text_meta.text(f"Generowanie metadanych dla Klastra {cluster_id} ({i+1}/{len(cluster_ids_to_process)})...")
                                     logger.info(f"Generowanie metadanych dla Klastra {cluster_id}...")

                                     # Generuj etykietę
                                     if llm_available:
                                          try:
                                               label = generate_cluster_labels_llm(
                                                   st.session_state.cluster_assignments,
                                                   st.session_state.qdrant_data_cache,
                                                   st.session_state.chatbot.llm
                                               ).get(cluster_id, f"Klaster {cluster_id}")
                                               generated_labels[cluster_id] = label
                                          except Exception as label_err:
                                               logger.error(f"Błąd generowania etykiety dla klastra {cluster_id}: {label_err}")
                                               generated_labels[cluster_id] = f"Klaster {cluster_id} (błąd)"
                                               all_metadata_generated_successfully = False
                                     else:
                                          generated_labels[cluster_id] = f"Klaster {cluster_id}"
                                          if i == 0: logger.warning("LLM niedostępny, pomijanie generowania etykiet LLM.")
                                          all_metadata_generated_successfully = False

                                     # Generuj podsumowanie
                                     if llm_available:
                                         if (st.session_state.embeddings_matrix_cache is not None and
                                             st.session_state.cluster_centroids is not None and
                                             st.session_state.cluster_indices_map is not None and
                                             st.session_state.point_id_to_text_map is not None and
                                             st.session_state.point_id_list_ordered is not None):
                                             try:
                                                 summary = generate_cluster_summary(
                                                     cluster_id=cluster_id,
                                                     cluster_indices=cluster_indices_map[cluster_id],
                                                     embeddings=st.session_state.embeddings_matrix_cache,
                                                     centroids=st.session_state.cluster_centroids,
                                                     point_id_to_text=st.session_state.point_id_to_text_map,
                                                     point_id_list=st.session_state.point_id_list_ordered,
                                                     llm=st.session_state.chatbot.llm
                                                 )
                                                 generated_summaries[cluster_id] = summary
                                             except Exception as summary_err:
                                                  logger.error(f"Błąd generowania podsumowania dla klastra {cluster_id}: {summary_err}")
                                                  generated_summaries[cluster_id] = f"Błąd generowania podsumowania: {summary_err}"
                                                  all_metadata_generated_successfully = False
                                         else:
                                              generated_summaries[cluster_id] = "Brak danych do generowania podsumowania (cache)."
                                              if i==0: logger.warning("Brak danych cache do generowania podsumowań.")
                                              all_metadata_generated_successfully = False
                                     else:
                                         generated_summaries[cluster_id] = "LLM niedostępny do generowania podsumowania."
                                         if i == 0: logger.warning("LLM niedostępny, pomijanie generowania podsumowań.")
                                         all_metadata_generated_successfully = False

                                     # Generuj encje
                                     if spacy_available:
                                         cluster_texts_for_ner = [
                                             st.session_state.point_id_to_text_map.get(st.session_state.point_id_list_ordered[idx])
                                             for idx in cluster_indices_map.get(cluster_id, [])
                                             if st.session_state.point_id_to_text_map.get(st.session_state.point_id_list_ordered[idx])
                                         ]
                                         if cluster_texts_for_ner:
                                             try:
                                                 entities = extract_key_entities(cluster_texts_for_ner, st.session_state.chatbot.nlp)
                                                 generated_entities[cluster_id] = entities
                                             except Exception as ner_err:
                                                  logger.error(f"Błąd ekstrakcji encji dla klastra {cluster_id}: {ner_err}")
                                                  generated_entities[cluster_id] = []
                                                  all_metadata_generated_successfully = False
                                         else:
                                              generated_entities[cluster_id] = []
                                              if i==0: logger.warning("Brak tekstów do ekstrakcji encji.")
                                              all_metadata_generated_successfully = False
                                     else:
                                         generated_entities[cluster_id] = []
                                         if i == 0: logger.warning("Model spaCy niedostępny, pomijanie ekstrakcji encji.")
                                         all_metadata_generated_successfully = False

                                     progress_bar_meta.progress(current_progress)
                                     time.sleep(0.1) # Małe opóźnienie

                                 status_text_meta.empty() # Usuń tekst postępu
                                 progress_bar_meta.empty() # Usuń pasek postępu

                                 # Zapisz wszystkie wygenerowane metadane do stanu sesji i pliku JSON
                                 st.session_state.cluster_labels = generated_labels
                                 st.session_state.cluster_summaries = generated_summaries
                                 st.session_state.cluster_entities = generated_entities
                                 # ZAPIS DO QDRANT zamiast do pliku
                                 cluster_metadata_to_save = {}
                                 for cid in generated_labels.keys():
                                     # Pomijamy klaster -1 (szum) przy zapisie metadanych
                                     if cid != -1:
                                         cluster_metadata_to_save[cid] = {
                                             "label": generated_labels.get(cid),
                                             "summary": generated_summaries.get(cid),
                                             "entities": generated_entities.get(cid, []) # Upewnij się, że entities jest listą
                                         }
                                 if cluster_metadata_to_save:
                                     with st.spinner("Zapisywanie metadanych klastrów do Qdrant..."):
                                         save_meta_success = st.session_state.chatbot.qdrant_connector.save_cluster_metadata_to_qdrant(
                                             cluster_metadata_to_save
                                         )
                                         # POPRAWKA: Sprawdź wynik przed wyświetleniem komunikatu
                                         if save_meta_success is True: # Jawne sprawdzenie True
                                             logger.info("✅ Pomyślnie zapisano metadane klastrów do Qdrant.")
                                             st.success("✅ Metadane klastrów zapisane w Qdrant.")
                                         elif save_meta_success is False: # Jawne sprawdzenie False
                                              # Logowanie błędu już powinno być w metodzie QdrantConnector
                                              st.error("⚠️ Wystąpił błąd podczas zapisywania metadanych klastrów do Qdrant (funkcja zwróciła False).")
                                         # else: # Obsługa przypadków, gdyby funkcja zwróciła coś innego niż bool (nie powinno się zdarzyć)
                                         #    logger.warning(f"Nieoczekiwana wartość zwrócona przez save_cluster_metadata_to_qdrant: {save_meta_success}")
                                         #    st.warning("Otrzymano nieoczekiwany status zapisu metadanych.")
                                 else:
                                     logger.warning("Brak metadanych klastrów (poza szumem) do zapisania.")

                                 # Usunięto wywołanie save_cluster_metadata(...)

                                 if all_metadata_generated_successfully:
                                     st.success("✅ Etykiety, podsumowania i encje dla wszystkich klastrów wygenerowane.") # Zmieniono komunikat
                                 else:
                                     st.warning("⚠️ Niektóre metadane klastrów mogły nie zostać wygenerowane z powodu braku danych lub modeli.")

                                 # Ustaw status na OK po udanej analizie i zapisie
                                 st.session_state.cluster_load_status = "loaded_ok"

                             else: # Błąd podczas klastrowania
                                 st.error("❌ Wystąpił błąd podczas klastrowania.")
                                 st.session_state.cluster_assignments = None
                                 st.session_state.cluster_labels = {}
                                 st.session_state.cluster_centroids = None
                                 st.session_state.cluster_indices_map = None
                                 st.session_state.point_id_to_text_map = None
                                 st.session_state.point_id_list_ordered = None
                                 st.session_state.embeddings_matrix_cache = None
                                 st.session_state.cluster_load_status = "error"
                         else:
                              st.warning("⚠️ Brak wektorów w pobranych danych do klastrowania.")
                              st.session_state.cluster_assignments = None
                              st.session_state.cluster_labels = {}
                              st.session_state.cluster_load_status = "loaded_empty"
                except Exception as cluster_e:
                    logger.error(f"❌ Błąd podczas procesu klastrowania: {cluster_e}", exc_info=True)
                    st.error(f"Wystąpił błąd: {cluster_e}")
                    st.session_state.cluster_assignments = None
                    st.session_state.cluster_labels = {}
                    st.session_state.qdrant_data_cache = None
                    st.session_state.cluster_load_status = "error"
            st.rerun()

        # Wyświetlanie informacji o klastrach (jeśli istnieją lub wystąpił błąd ładowania)
        cluster_summary_container = st.container()
        with cluster_summary_container:
             load_status = st.session_state.get('cluster_load_status', 'not_loaded')
             assignments_map = st.session_state.get('cluster_assignments')
             assignments_exist = assignments_map is not None and len(assignments_map) > 0

             # Wyświetl subheader tylko jeśli są klastry lub był błąd ładowania
             if assignments_exist or load_status == 'error':
                 st.subheader("Wyniki Klastrowania")
                 # DODANO: Wyświetl ostrzeżenie o niespójności, jeśli flaga jest ustawiona
                 if st.session_state.get('show_cluster_inconsistency_warning', False):
                     st.warning("⚠️ **Uwaga:** Wyświetlone metadane klastrów (etykiety, itp.) mogą być nieaktualne, ponieważ liczba dokumentów w bazie zmieniła się od ostatniej analizy. Zalecane jest ponowne uruchomienie '🚀 Analizuj / Odśwież Klastry'.")

             # Komunikaty o stanie ładowania
             if load_status == 'error':
                 st.warning("⚠️ Nie udało się załadować przypisań klastrów z Qdrant. Uruchom analizę ponownie.")
             elif load_status == 'loaded_empty':
                 st.info("ℹ️ Nie znaleziono zapisanych wyników klastrowania w Qdrant lub poprzednia analiza nie znalazła grup. Uruchom analizę, aby spróbować ponownie.")
             elif not assignments_exist and load_status != 'error' and load_status != 'loaded_empty':
                 st.info("ℹ️ Brak zapisanych wyników klastrowania. Uruchom analizę, aby wygenerować klastry.")

             # Wyświetl listę klastrów tylko jeśli istnieją
             if assignments_exist:
                 try:
                     cluster_counts = Counter(assignments_map.values())
                     noise_points = cluster_counts.pop(-1, 0)
                     num_clusters_found = len(cluster_counts)
                     if num_clusters_found > 0:
                        st.write(f"Liczba znalezionych klastrów: {num_clusters_found}")
                     elif noise_points > 0 :
                         st.write(f"Nie znaleziono klastrów (tylko {noise_points} punktów szumu).")

                     if noise_points > 0: st.write(f"Liczba punktów szumu/outlierów: {noise_points}")

                     sorted_clusters = sorted(cluster_counts.items())
                     for cluster_id, count in sorted_clusters:
                          label_text = f"Klaster {cluster_id}"
                          cluster_label = st.session_state.get('cluster_labels', {}).get(cluster_id)
                          if cluster_label:
                              label_text += f": **{cluster_label}**"
                          if st.button(f"{label_text} ({count} punktów)", key=f"view_cluster_{cluster_id}"):
                              if st.session_state.selected_cluster_id == cluster_id:
                                   st.session_state.selected_cluster_id = None
                                   st.session_state.filter_by_selected_cluster = False # Resetuj flagę przy deselekcji
                              else:
                                   st.session_state.selected_cluster_id = cluster_id
                                   st.session_state.filter_by_selected_cluster = False # Resetuj flagę przy wyborze nowego
                              st.rerun()
                 except Exception as e:
                      logger.error(f"Błąd wyświetlania wyników klastrowania: {e}", exc_info=True)
                      st.error("Błąd przy wyświetlaniu podsumowania klastrów.")

        # Wyświetlanie szczegółów wybranego klastra
        if st.session_state.get('selected_cluster_id') is not None:
             selected_id = st.session_state.selected_cluster_id
             assignments_map = st.session_state.get('cluster_assignments')
             if assignments_map and any(cid == selected_id for cid in assignments_map.values()):
                  selected_label_text = f"Klaster {selected_id}"
                  cluster_label = st.session_state.get('cluster_labels', {}).get(selected_id)
                  if cluster_label:
                       selected_label_text += f": **{cluster_label}**"
                  st.subheader(f"Szczegóły - {selected_label_text}")
                  try:
                      # DODANO: Checkbox do filtrowania RAG
                      st.session_state.filter_by_selected_cluster = st.checkbox(
                          f"Filtruj następne zapytanie do Klastra {selected_id}",
                          key=f"filter_checkbox_{selected_id}",
                          value=st.session_state.get('filter_by_selected_cluster', False),
                          help="Zaznacz, aby następne pytanie w polu poniżej było zadane tylko w kontekście dokumentów z tego klastra."
                      )
                      st.markdown("---")

                      # Wyświetl Podsumowanie (jeśli istnieje w stanie sesji)
                      cluster_summary = st.session_state.get('cluster_summaries', {}).get(selected_id)
                      st.markdown("**Podsumowanie klastra:**") # Wyświetl nagłówek zawsze
                      if cluster_summary:
                          st.markdown(cluster_summary)
                      else:
                          st.caption("_Brak zapisanego podsumowania dla tego klastra._")
                      st.markdown("---") # Separator zawsze

                      # Wyświetl Kluczowe Obiekty (jeśli istnieją w stanie sesji)
                      entities_list = st.session_state.get('cluster_entities', {}).get(selected_id)
                      st.markdown("**Kluczowe Obiekty:**") # Wyświetl nagłówek zawsze
                      if entities_list is not None: # Sprawdź, czy klucz istnieje (nawet jeśli lista jest pusta)
                          if entities_list:
                              for entity_text, entity_label, count in entities_list:
                                  st.markdown(f"- `{entity_text}` ({entity_label}): {count}")
                          else:
                              st.caption("_Brak zapisanych kluczowych obiektów dla tego klastra (lub ekstrakcja nie powiodła się)._")
                      else:
                           st.caption("_Brak zapisanych kluczowych obiektów dla tego klastra (metadane nie załadowane)._")
                      st.markdown("---") # Separator zawsze

                      # Pobierz dane punktów (jeśli potrzebne i nie ma w cache)
                      point_id_to_data = {}
                      cluster_point_ids = [pid for pid, cid in assignments_map.items() if cid == selected_id]
                      with st.expander(f"Pokaż/Ukryj listę punktów ({len(cluster_point_ids)})"):
                          if 'qdrant_data_cache' not in st.session_state or st.session_state.qdrant_data_cache is None:
                              with st.spinner("Pobieranie danych punktów..."):
                                  st.session_state.qdrant_data_cache = st.session_state.chatbot.qdrant_connector.get_all_data_for_clustering(with_payload=True, with_vectors=False)
                              if st.session_state.qdrant_data_cache is not None:
                                  point_id_to_data = {d['id']: d for d in st.session_state.qdrant_data_cache}
                              else:
                                  st.warning("Nie udało się pobrać danych punktów z Qdrant.")
                          else:
                              point_id_to_data = {d['id']: d for d in st.session_state.qdrant_data_cache}

                          # Wyświetlanie listy punktów za pomocą st.dataframe (jeśli mamy dane)
                          if not cluster_point_ids:
                              st.write("Brak punktów w klastrze.")
                          elif not point_id_to_data:
                              st.warning("Nie można wyświetlić listy punktów (brak danych).")
                          else:
                              # Przygotuj dane do tabeli
                              table_data = []
                              full_content_map = {} # Słownik do przechowywania pełnej treści
                              PREVIEW_LENGTH = 150 # Długość podglądu treści

                              for point_id in cluster_point_ids:
                                  point_data = point_id_to_data.get(point_id)
                                  if point_data and point_data.get('payload'):
                                      payload = point_data['payload']
                                      metadata = payload.get('metadata', {})
                                      content = payload.get('page_content', '_brak treści_')
                                      source_name = metadata.get('source', metadata.get('file_path', metadata.get('filename', 'nieznane źródło')))
                                      display_name = os.path.basename(source_name)
                                      point_id_short = point_id[:8] # Skrócone ID

                                      table_data.append({
                                          "ID Fragmentu": point_id_short,
                                          "Nazwa Pliku": display_name,
                                          "Początek Treści": content[:PREVIEW_LENGTH] + ('...' if len(content) > PREVIEW_LENGTH else '')
                                      })
                                      full_content_map[point_id_short] = content # Zapisz pełną treść
                                  else:
                                      logger.warning(f"Brak danych payload dla punktu {point_id} przy tworzeniu tabeli.")

                              if table_data:
                                  df = pd.DataFrame(table_data)
                                  st.dataframe(
                                       df,
                                       hide_index=True, # Ukryj domyślny indeks pandas
                                       use_container_width=True # Rozciągnij tabelę na całą szerokość
                                  )
                                  # DODANO: Możliwość wyświetlenia pełnej treści po wybraniu ID z selectboxa
                                  selected_point_id_short = st.selectbox(
                                       "Wybierz ID Fragmentu, aby zobaczyć pełną treść:",
                                       options=[""] + list(full_content_map.keys()), # Dodaj pustą opcję na początek
                                       key=f"select_full_content_{selected_id}"
                                  )
                                  if selected_point_id_short and selected_point_id_short in full_content_map:
                                       st.text_area(
                                           f"Pełna treść fragmentu {selected_point_id_short}:",
                                           value=full_content_map[selected_point_id_short],
                                           height=200,
                                           disabled=True,
                                           key=f"full_content_view_{selected_id}_{selected_point_id_short}"
                                       )
                              else:
                                   st.write("Brak danych punktów do wyświetlenia w tabeli.")
                  except Exception as detail_e:
                       logger.error(f"Błąd wyświetlania szczegółów klastra {selected_id}: {detail_e}", exc_info=True)
                       st.error(f"Błąd przy wyświetlaniu szczegółów klastra {selected_id}.")
             else:
                  st.session_state.selected_cluster_id = None
                  st.warning("Wybrany klaster nie jest już dostępny.")
        st.markdown("---")
        # === KONIEC SEKCJI KLASTROWANIA ===

        st.header("📊 Ustawienia Grafu")
        st.session_state.filter_small_graphs = st.checkbox(
            "Filtruj małe grafy (min. 3 węzły)",
            value=st.session_state.filter_small_graphs,
            key="filter_small_graphs_checkbox"
        )
        st.markdown("---")

        # === SEKCJA PLIKÓW KONTEKSTOWYCH (TYLKO DLA LLM) ===
        st.header("📄 Pliki Kontekstowe (dla LLM)")
        context_file_uploader = st.file_uploader(
            "Dodaj pliki TXT/PDF jako dodatkowy kontekst (nie będą indeksowane w RAG)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="context_uploader"
        )

        if context_file_uploader:
            new_context_files = {}
            context_upload_errors = []
            with st.spinner("Przetwarzanie plików kontekstowych..."):
                for uploaded_file in context_file_uploader:
                    file_name = uploaded_file.name
                    try:
                        if file_name.lower().endswith(".txt"):
                            # Odczyt pliku TXT
                            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                            content = stringio.read()
                            new_context_files[file_name] = content
                            logger.info(f"📄 Odczytano plik TXT jako kontekst: {file_name}")
                        elif file_name.lower().endswith(".pdf"):
                            # Odczyt pliku PDF za pomocą pypdf
                            pdf_bytes = io.BytesIO(uploaded_file.getvalue())
                            reader = PdfReader(pdf_bytes)
                            content = ""
                            for page in reader.pages:
                                content += page.extract_text() + "\n"
                            new_context_files[file_name] = content
                            logger.info(f"📄 Odczytano plik PDF jako kontekst: {file_name}")
                    except Exception as e:
                        error_msg = f"Błąd odczytu pliku kontekstowego '{file_name}': {e}"
                        logger.error(error_msg, exc_info=True)
                        context_upload_errors.append(error_msg)

            # Aktualizuj stan sesji tylko nowymi/zaktualizowanymi plikami
            # To pozwala na dodawanie kolejnych plików bez nadpisywania poprzednich
            st.session_state.context_files.update(new_context_files)

            if context_upload_errors:
                for error in context_upload_errors:
                    st.error(error)

        # Wyświetlanie listy załadowanych plików kontekstowych i przycisku czyszczenia
        if st.session_state.context_files:
            st.subheader("Załadowane pliki kontekstowe:")
            for filename in st.session_state.context_files.keys():
                st.markdown(f"- `{filename}`")
            if st.button("Wyczyść pliki kontekstowe", key="clear_context_files"):
                st.session_state.context_files = {}
                # Wyczyść też sam uploader, ustawiając jego wartość na pustą listę
                # To wymaga specyficznego klucza, który nadaliśmy uploaderowi
                # Używamy st.session_state['key'] do ustawienia wartości uploadera
                st.session_state.context_uploader = [] # Resetuj stan uploadera
                logger.info("🧹 Wyczyściłem listę plików kontekstowych.")
                st.rerun()
        st.markdown("---")
        # === KONIEC SEKCJI PLIKÓW KONTEKSTOWYCH ===


        st.header("📁 Zarządzanie dokumentami (RAG)")

        st.markdown("---")
        st.subheader("⚠️ Strefa niebezpieczna")
        if 'confirm_clear_db' not in st.session_state:
             st.session_state.confirm_clear_db = False

        collection_name_to_clear = "nieznana"
        if "chatbot" in st.session_state and hasattr(st.session_state.chatbot, 'qdrant_connector'):
             collection_name_to_clear = st.session_state.chatbot.qdrant_connector.collection_name

        if st.button(f"🗑️ Wyczyść kolekcję '{collection_name_to_clear}'", type="secondary"):
             st.session_state.confirm_clear_db = True

        if st.session_state.confirm_clear_db:
             st.warning(f"**Czy na pewno chcesz usunąć WSZYSTKIE dane z kolekcji '{collection_name_to_clear}'?** Tej operacji nie można cofnąć.")
             col_confirm, col_cancel = st.columns(2)
             with col_confirm:
                  if st.button(f"Tak, wyczyść '{collection_name_to_clear}'", type="primary", use_container_width=True):
                       with st.spinner(f"Czyszczenie kolekcji '{collection_name_to_clear}'..."):
                            if "chatbot" in st.session_state:
                                 try:
                                      success = st.session_state.chatbot.clear_database()
                                      if success:
                                           st.success(f"✅ Kolekcja '{collection_name_to_clear}' została wyczyszczona.")
                                           # Po wyczyszczeniu resetujemy stan klastrowania
                                           st.session_state.cluster_assignments = None
                                           st.session_state.cluster_labels = {}
                                           st.session_state.qdrant_data_cache = None
                                           st.session_state.selected_cluster_id = None
                                           st.session_state.cluster_summaries = {}
                                           st.session_state.cluster_entities = {}
                                           st.session_state.cluster_load_status = "loaded_empty" # Ustawiamy na pusty
                                           # Usunięto logikę usuwania pliku metadanych
                                           # try:
                                           #      if os.path.exists(CLUSTER_METADATA_FILE): os.remove(CLUSTER_METADATA_FILE)
                                           # except OSError as e: logger.error(f"Nie udało się usunąć pliku {CLUSTER_METADATA_FILE} po czyszczeniu bazy: {e}")
                                      else:
                                           st.error(f"❌ Wystąpił błąd podczas czyszczenia kolekcji '{collection_name_to_clear}'.")
                                 except AttributeError:
                                      st.error("❌ Błąd: Obiekt chatbota niepoprawny. Spróbuj zrestartować aplikację.")
                                 except Exception as e:
                                      st.error(f"❌ Wystąpił nieoczekiwany błąd: {e}")
                            else:
                                 st.error("❌ Błąd: Obiekt chatbota niedostępny.")
                       st.session_state.confirm_clear_db = False
                       st.rerun()
             with col_cancel:
                  if st.button("Anuluj", use_container_width=True):
                       st.session_state.confirm_clear_db = False
                       st.rerun()
        st.markdown("---")


        # Ładowanie z pliku lokalnego
        uploaded_files = st.file_uploader(
            "Wybierz dokumenty do przetworzenia",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        # Ładowanie z URL
        st.subheader("lub z adresu URL")
        url_input = st.text_input("Wprowadź adres URL dokumentu")

        st.markdown("---")
        # Przycisk przetwarzania - uproszczony, bez oddzielnego przycisku URL
        if st.button("🧠 Przetwórz dodane dokumenty"):
            files_to_process = []
            if uploaded_files: files_to_process.extend(uploaded_files)

            url_to_process = url_input if url_input else None

            if not files_to_process and not url_to_process:
                 st.warning("⚠️ Najpierw wybierz pliki lub wprowadź URL.")
            else:
                 # --- Przetwarzanie plików ---
                 if files_to_process:
                      num_files = len(files_to_process)
                      st.info(f"Rozpoczynam przetwarzanie {num_files} plików...")
                      processed_count = 0
                      error_count = 0
                      progress_bar = st.progress(0)
                      status_text = st.empty()

                      for i, uploaded_file in enumerate(files_to_process):
                          temp_dir = None
                          try:
                              temp_dir = tempfile.mkdtemp()
                              file_path = os.path.join(temp_dir, uploaded_file.name)
                              status_text.text(f"Plik {i+1}/{num_files}: {uploaded_file.name}")
                              with open(file_path, "wb") as f: f.write(uploaded_file.getvalue())
                              logger.info(f"📥 Zapisano tymczasowy plik: {file_path}")

                              with st.spinner(f"Przetwarzanie {uploaded_file.name}..."):
                                   success = st.session_state.chatbot.process_document(file_path)

                              if success:
                                  logger.info(f"✅ Pomyślnie przetworzono: {uploaded_file.name}")
                                  processed_count += 1
                              else:
                                  logger.error(f"❌ Błąd przetwarzania: {uploaded_file.name}")
                                  error_count += 1
                                  st.warning(f"Błąd podczas przetwarzania: {uploaded_file.name}")

                          except Exception as e:
                              logger.error(f"❌ Krytyczny błąd obsługi pliku {uploaded_file.name}: {str(e)}", exc_info=True)
                              st.error(f"Krytyczny błąd obsługi pliku {uploaded_file.name}: {e}")
                              error_count += 1
                          finally:
                              if temp_dir and os.path.exists(temp_dir):
                                  try:
                                      if 'file_path' in locals() and os.path.exists(file_path): os.unlink(file_path)
                                      os.rmdir(temp_dir)
                                      logger.info(f"🧹 Posprzątano: {uploaded_file.name}")
                                  except Exception as cleanup_e:
                                      logger.error(f"🧹❌ Błąd sprzątania {uploaded_file.name}: {cleanup_e}")
                          progress_bar.progress((i + 1) / num_files)

                      status_text.text("Zakończono przetwarzanie plików.")
                      if processed_count > 0:
                          st.success(f"✅ Przetworzono pomyślnie {processed_count} z {num_files} plików.")
                          # DODANO: Komunikat o konieczności ponownej analizy klastrów
                          st.info("ℹ️ Dodano nowe dokumenty. Aby uwzględnić je w analizie, uruchom ponownie '🚀 Analizuj / Odśwież Klastry'.")
                      if error_count > 0: st.error(f"❌ Wystąpiły błędy dla {error_count} z {num_files} plików.")
                      # Po przetworzeniu dokumentów, resetuj stan klastrowania
                      st.session_state.cluster_assignments = None
                      st.session_state.cluster_labels = {}
                      st.session_state.qdrant_data_cache = None
                      st.session_state.selected_cluster_id = None
                      st.session_state.cluster_summaries = {}
                      st.session_state.cluster_entities = {}
                      st.session_state.cluster_load_status = "not_loaded"
                      # Usunięto logikę usuwania pliku metadanych
                      # try:
                      #      if os.path.exists(CLUSTER_METADATA_FILE): os.remove(CLUSTER_METADATA_FILE)
                      # except OSError as e: logger.error(f"Nie udało się usunąć pliku {CLUSTER_METADATA_FILE} po przetworzeniu dokumentów: {e}")


                 # --- Przetwarzanie URL ---
                 if url_to_process:
                      st.info(f"Rozpoczynam przetwarzanie dokumentu z URL: {url_to_process}")
                      with st.spinner("Przetwarzanie URL..."):
                          try:
                              st.warning("Przetwarzanie z URL nie jest jeszcze w pełni zaimplementowane w tym przycisku.")
                              # Po przetworzeniu dokumentów z URL, resetuj stan klastrowania
                              st.session_state.cluster_assignments = None
                              st.session_state.cluster_labels = {}
                              st.session_state.qdrant_data_cache = None
                              st.session_state.selected_cluster_id = None
                              st.session_state.cluster_summaries = {}
                              st.session_state.cluster_entities = {}
                              st.session_state.cluster_load_status = "not_loaded"
                              # Usunięto logikę usuwania pliku metadanych
                              # try:
                              #      if os.path.exists(CLUSTER_METADATA_FILE): os.remove(CLUSTER_METADATA_FILE)
                              # except OSError as e: logger.error(f"Nie udało się usunąć pliku {CLUSTER_METADATA_FILE} po przetworzeniu URL: {e}")
                          # Naprawienie bloku try-except (dodanie except i poprawne wcięcie)
                          except Exception as e:
                               logger.error(f"❌ Błąd przetwarzania URL: {str(e)}", exc_info=True)
                               st.error(f"Błąd podczas przetwarzania URL: {e}")


        st.markdown("---")
        if st.button("🔄 Resetuj czat"):
            if "active_chat_id" in st.session_state and st.session_state.active_chat_id in st.session_state.chats:
                 st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
                 save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                 logger.info(f"Wyczyszczono historię aktywnego czatu: {st.session_state.active_chat_id}")
                 # Resetuj metadane klastrów przy resetowaniu czatu
                 st.session_state.cluster_labels = {}
                 st.session_state.qdrant_data_cache = None
                 st.session_state.selected_cluster_id = None
                 st.session_state.cluster_summaries = {}
                 st.session_state.cluster_entities = {}
                 st.session_state.filter_by_selected_cluster = False # Resetuj flagę filtrowania
                 # Usunięto logikę usuwania pliku metadanych
                 # try:
                 #      if os.path.exists(CLUSTER_METADATA_FILE): os.remove(CLUSTER_METADATA_FILE)
                 # except OSError as e: logger.error(f"Nie udało się usunąć pliku {CLUSTER_METADATA_FILE} przy resecie czatu: {e}")
                 # Resetuj status ładowania, aby wymusić ponowne ładowanie z Qdrant (w Iteracji 2)
                 if 'cluster_data_loaded' in st.session_state: del st.session_state.cluster_data_loaded
                 if 'cluster_load_status' in st.session_state: del st.session_state.cluster_load_status
            else:
                 logger.warning("Nie można zresetować czatu - brak aktywnego czatu.")
            st.rerun()


    # --- Wyświetlanie aktywnego czatu ---
    if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            st.session_state.active_chat_id = next(iter(st.session_state.chats))
            logger.warning("Aktywny czat był nieprawidłowy. Przełączono na pierwszy dostępny.")
            save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
            st.rerun()
        else:
            first_chat_id = str(uuid.uuid4())
            st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
            st.session_state.active_chat_id = first_chat_id
            logger.info("Brak czatów. Utworzono nowy domyślny czat.")
            save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
            st.rerun()

    active_chat_id = st.session_state.active_chat_id
    active_chat_data = st.session_state.chats[active_chat_id]
    st.header(f"Czat: {active_chat_data['name']}")

    message_container = st.container()
    with message_container:
         # --- Pętla wyświetlania historii ---
         for message_index, message in enumerate(active_chat_data.get("messages", [])):
              with st.chat_message(message["role"]):
                   content_to_display = message.get("content", "_Brak treści_")
                   sources_to_display = message.get("sources", []) # Oczekujemy listy słowników {'name': ..., 'id': ...} lub listy stringów (stara historia)
                   graph_data_to_display = message.get("graph_data")
                   metadata_to_display = message.get("metadata") # DODANO: Pobierz metadane

                   st.markdown(content_to_display)

                   # --- Dodawanie przycisków "Skasuj" i "Wyślij ponownie" ---
                   if message["role"] == "assistant" and message_index > 0:
                       # Sprawdź, czy poprzednia wiadomość to zapytanie użytkownika
                       previous_message = active_chat_data["messages"][message_index - 1]
                       if previous_message["role"] == "user":
                           # Utwórz kolumny dla przycisków, znacznie węższe dla ikon
                           col1_btn, col2_btn, _ = st.columns([0.1, 0.1, 0.8]) # Zmniejszone proporcje dla ikon

                           with col1_btn:
                               delete_key = f"delete_msg_{active_chat_id}_{message_index}"
                               # Użyj tylko ikony jako etykiety przycisku
                               if st.button("🗑️", key=delete_key, help="Usuń tę odpowiedź i poprzedzające ją zapytanie użytkownika.", use_container_width=True):
                                   try:
                                       messages_list = st.session_state.chats[active_chat_id]["messages"]
                                       # Usuwamy najpierw wiadomość o wyższym indeksie (asystenta), aby nie psuć indeksu niższej (użytkownika)
                                       del messages_list[message_index]
                                       del messages_list[message_index - 1]
                                       save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                                       logger.info(f"Usunięto wiadomości o indeksach {message_index} (asystent) i {message_index - 1} (użytkownik) z czatu {active_chat_id}")
                                       st.rerun()
                                   except IndexError:
                                       logger.error(f"Błąd indeksu podczas próby usunięcia wiadomości {message_index} i {message_index - 1} z czatu {active_chat_id}")
                                       st.error("Wystąpił błąd podczas usuwania wiadomości.")
                                   except Exception as e:
                                       logger.error(f"Nieoczekiwany błąd podczas usuwania wiadomości: {e}", exc_info=True)
                                       st.error(f"Wystąpił nieoczekiwany błąd: {e}")


                           with col2_btn:
                               resend_key = f"resend_msg_{active_chat_id}_{message_index}"
                               # Użyj tylko ikony jako etykiety przycisku
                               if st.button("🔄", key=resend_key, help="Usuń tę odpowiedź i wyślij poprzednie zapytanie ponownie do LLM.", use_container_width=True):
                                   try:
                                       messages_list = st.session_state.chats[active_chat_id]["messages"]
                                       # Usuwamy tylko odpowiedź asystenta
                                       del messages_list[message_index]
                                       st.session_state.new_user_input_submitted = True # Ustawiamy flagę, aby wywołać LLM przy następnym rerun
                                       save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                                       logger.info(f"Usunięto wiadomość o indeksie {message_index} (asystent) z czatu {active_chat_id} i ustawiono flagę ponownego wysłania.")
                                       st.rerun()
                                   except IndexError:
                                        logger.error(f"Błąd indeksu podczas próby usunięcia wiadomości {message_index} z czatu {active_chat_id} do ponownego wysłania.")
                                        st.error("Wystąpił błąd podczas przygotowania do ponownego wysłania.")
                                   except Exception as e:
                                       logger.error(f"Nieoczekiwany błąd podczas ponownego wysyłania: {e}", exc_info=True)
                                       st.error(f"Wystąpił nieoczekiwany błąd: {e}")
                   # --- Koniec dodawania przycisków ---

                   # Istniejący kod wyświetlania metadanych, źródeł i grafu poniżej...

                   # DODANO: Wyświetlanie metadanych, jeśli istnieją
                   if metadata_to_display:
                       model_name = metadata_to_display.get('model', 'N/A')
                       tokens = metadata_to_display.get('tokens')
                       resp_time = metadata_to_display.get('time')
                       meta_parts = [f"Model: {model_name}"]
                       if tokens is not None: meta_parts.append(f"Tokeny: {tokens}")
                       if resp_time is not None: meta_parts.append(f"Czas: {resp_time:.2f}s")
                       st.caption(" | ".join(meta_parts))

                   if sources_to_display:
                           with st.expander("Źródła"):
                               # Iterujemy po liście źródeł, która może zawierać stringi (stara historia) lub słowniki (nowe odpowiedzi)
                               for source_item in sources_to_display:
                                   source_name = 'nieznane źródło'
                                   point_id = None

                                   if isinstance(source_item, dict):
                                       # Nowy format: {'name': ..., 'id': ...}
                                       source_name = source_item.get('name', 'nieznane źródło')
                                       point_id = source_item.get('id')
                                   elif isinstance(source_item, str):
                                       # Stary format: tylko nazwa pliku (string)
                                       source_name = source_item
                                       # point_id pozostaje None, więc informacja o klastrze nie zostanie dodana dla starej historii
                                   else:
                                       # Nieoczekiwany format
                                       logger.warning(f"Nieoczekiwany typ elementu w sources_to_display: {type(source_item)}")
                                       source_name = str(source_item) # Spróbuj wyświetlić jako string

                                   cluster_info_str = ""
                                   # Sprawdzamy, czy mamy przypisania klastrów i czy ID punktu istnieje i jest w przypisaniach
                                   assignments_map_disp = st.session_state.get('cluster_assignments')
                                   if assignments_map_disp and point_id and point_id in assignments_map_disp:
                                       cluster_id = assignments_map_disp[point_id]
                                       if cluster_id != -1: # Ignoruj szum
                                           cluster_name = f"Klaster {cluster_id}"
                                           # Sprawdź, czy mamy etykiety i czy dla tego klastra istnieje etykieta
                                           cluster_label_disp = st.session_state.get('cluster_labels', {}).get(cluster_id)
                                           if cluster_label_disp:
                                               cluster_name = cluster_label_disp
                                           cluster_info_str = f" (**{cluster_name}**)" # Dodano pogrubienie dla lepszej widoczności
                                   elif point_id is None and isinstance(source_item, dict): # Tylko jeśli spodziewaliśmy się ID, ale go nie było
                                        cluster_info_str = " (ID źródła niedostępne)"

                                   # Używamy os.path.basename, aby wyświetlić tylko nazwę pliku
                                   display_name = os.path.basename(source_name)
                                   st.markdown(f"- `{display_name}`{cluster_info_str}")


                   # --- Sekcja wizualizacji grafu ---
                   if graph_data_to_display:
                        # Logika filtrowania grafu wg liczby węzłów z checkboxem
                        MIN_NODES_TO_DISPLAY = 3
                        num_nodes_in_graph = 0
                        should_render_graph_section = False

                        if isinstance(graph_data_to_display, dict) and 'nodes' in graph_data_to_display and 'links' in graph_data_to_display:
                             num_nodes_in_graph = len(graph_data_to_display.get('nodes', []))
                             # Pokaż graf jeśli checkbox odznaczony LUB (checkbox zaznaczony ORAZ liczba węzłów >= 3)
                             should_render_graph_section = not st.session_state.filter_small_graphs or num_nodes_in_graph >= MIN_NODES_TO_DISPLAY
                        else:
                             logger.warning(f"Nieprawidłowy format graph_data_to_display dla indeksu {message_index}")

                        if should_render_graph_section:
                            toggle_key = f"graph_toggle_{active_chat_id}_{message_index}"
                            if st.toggle("Wyświetl Graf Wiedzy", key=toggle_key, value=False):
                                try:
                                    nodes_agraph = []
                                    edges_agraph = []

                                    # Rekonstrukcja grafu NetworkX (tylko do potencjalnego obliczenia rozmiaru)
                                    degrees = {}
                                    max_degree = 0
                                    temp_G = nx.DiGraph()
                                    try:
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore", FutureWarning)
                                            temp_G = nx.node_link_graph(graph_data_to_display, directed=True, multigraph=False)
                                        degrees = dict(temp_G.degree())
                                        if degrees and any(d > 0 for d in degrees.values()): max_degree = max(degrees.values())
                                    except Exception as graph_reconstruction_e:
                                        logger.error(f"Błąd rekonstrukcji grafu dla indeksu {message_index}: {graph_reconstruction_e}")
                                        degrees, max_degree = {}, 0 # Reset stopni przy błędzie

                                    # Blok obliczania rozmiaru węzłów
                                    BASE_NODE_SIZE = 15
                                    LARGE_NODE_SIZE = 30

                                    # Tworzenie węzłów (wszystkich z danych)
                                    for node_data in graph_data_to_display.get('nodes', []):
                                         node_id = node_data.get('id', str(uuid.uuid4()))
                                         label = node_data.get('label', node_id)
                                         wrapped_label = textwrap.fill(label, width=20, replace_whitespace=False, drop_whitespace=False)
                                         ner_type = node_data.get('ner_type')
                                         node_color = NER_COLORS.get(ner_type, DEFAULT_NODE_COLOR)
                                         node_size = BASE_NODE_SIZE
                                         current_degree = degrees.get(node_id, 0)
                                         if max_degree > 0 and current_degree == max_degree: node_size = LARGE_NODE_SIZE
                                         tooltip = f"{label}" + (f" ({ner_type})" if ner_type else "")
                                         nodes_agraph.append(Node(id=node_id, label=wrapped_label, size=node_size, color=node_color, title=tooltip))

                                    # Tworzenie krawędzi (wszystkich z danych)
                                    for link_data in graph_data_to_display.get('links', []):
                                         source_id, target_id = link_data.get('source'), link_data.get('target')
                                         if source_id and target_id:
                                              edge_label = link_data.get('label', '')
                                              display_label = edge_label.lower()
                                              edges_agraph.append(Edge(source=source_id, target=target_id, label=display_label, color="#cccccc"))

                                    # Konfiguracja grafu
                                    config = Config(width='100%', height=600, directed=True, physics=False, hierarchical=True, layout={'hierarchical': {'direction': 'UD', 'sortMethod': 'hubsize'}}, nodeHighlightBehavior=True, highlightColor='#F7A7A6', collapsible=False)

                                    # Wyświetlanie grafu
                                    if nodes_agraph:
                                         with st.spinner("Ładowanie grafu..."):
                                              agraph(nodes=nodes_agraph, edges=edges_agraph, config=config)
                                    else:
                                         st.caption("_Brak węzłów do wyświetlenia w grafie._")
                                except Exception as e:
                                    logger.error(f"Błąd podczas renderowania grafu dla indeksu {message_index}: {e}", exc_info=True)
                                    st.error("Wystąpił błąd podczas wyświetlania grafu.")
                        # Informacja, jeśli graf został ukryty z powodu filtrowania
                        elif num_nodes_in_graph > 0: # Tylko jeśli graf istniał, ale był za mały
                             st.caption(f"_Graf został ukryty (zawiera {num_nodes_in_graph} węzłów, wymagane >= {MIN_NODES_TO_DISPLAY}). Odznacz opcję 'Filtruj małe grafy' w panelu bocznym, aby go zobaczyć._")
                        # Nie pokazujemy nic, jeśli graph_data_to_display było None lub nieprawidłowe
                   # ------------------------------------

    # --- Obsługa nowej wiadomości ---
    user_prompt = st.chat_input("Zadaj pytanie...")

    if user_prompt:
        st.session_state.chats[st.session_state.active_chat_id]["messages"].append({"role": "user", "content": user_prompt})
        save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
        # Ustaw flagę wskazującą na nowe dane wejściowe od użytkownika
        st.session_state.new_user_input_submitted = True
        st.rerun()

    # --- Logika generowania odpowiedzi ---
    current_chat_messages = st.session_state.chats[st.session_state.active_chat_id].get("messages", [])
    # Sprawdź flagę, czy użytkownik właśnie wysłał nową wiadomość
    if st.session_state.get("new_user_input_submitted", False):
        # Resetuj flagę natychmiast, aby uniknąć wielokrotnego wywołania
        st.session_state.new_user_input_submitted = False

        # Sprawdź, czy ostatnia wiadomość faktycznie pochodzi od użytkownika (dodatkowe zabezpieczenie)
        if current_chat_messages and current_chat_messages[-1]["role"] == "user":
            last_user_prompt = current_chat_messages[-1]["content"]
            response_placeholder = message_container.empty()
            with response_placeholder.chat_message("assistant"):
                with st.spinner("Myślę..."):
                    try:
                        response_content_str = ""
                        response_metadata = {} # Słownik na metadane
                        sources = []
                        graph_data = None
                        prompt_answered = last_user_prompt

                        # Sprawdź, czy używać Gemini API
                        if st.session_state.use_gemini_api:
                            logger.info("✨ Tryb: Google Gemini API")
                            if st.session_state.chatbot.gemini_llm:
                                # Wywołaj dedykowaną metodę dla Gemini, przekazując pliki kontekstowe
                                context_files_content_dict = st.session_state.get('context_files', {}) # Zmieniono nazwę zmiennej dla jasności
                                response_dict = st.session_state.chatbot._generate_gemini_answer(
                                    question=last_user_prompt,
                                    context_files_content=context_files_content_dict # Przekazujemy słownik
                                )
                                if response_dict and response_dict.get("content"):
                                    gemini_response = response_dict["content"] # Treść odpowiedzi
                                    if hasattr(gemini_response, 'content'):
                                        response_content_str = gemini_response.content
                                    elif isinstance(gemini_response, str):
                                        response_content_str = gemini_response
                                    else:
                                        response_content_str = str(gemini_response)
                                    response_metadata = response_dict.get("metadata", {}) # Pobierz metadane
                                    logger.info("✅ Odpowiedź z Gemini API otrzymana.")
                                else:
                                     response_content_str = "Błąd: Otrzymano nieprawidłową odpowiedź z Gemini API."
                                     logger.error(response_content_str)
                                sources = []
                                graph_data = None
                            else:
                                response_content_str = "Błąd: Model Google Gemini nie został poprawnie zainicjalizowany."
                                logger.error(response_content_str)
                                sources = []
                                graph_data = None
                        # Logika dla lokalnego LLM (Zwykły Chat lub RAG)
                        elif st.session_state.selected_chat_mode == "Zwykły Chat":
                            logger.info("💬 Tryb: Zwykły Chat (Lokalny LLM) - wywołanie LLM bez RAG")
                            # Użyj _generate_answer dla spójności metadanych
                            response_dict = st.session_state.chatbot._generate_answer(last_user_prompt, "", "") # Pusty kontekst i prompt systemowy
                            if response_dict and response_dict.get("content"):
                                plain_response = response_dict["content"]
                                if hasattr(plain_response, 'content'):
                                    response_content_str = plain_response.content
                                elif isinstance(plain_response, str):
                                    response_content_str = plain_response
                                else:
                                    response_content_str = str(plain_response)
                                response_metadata = response_dict.get("metadata", {})
                            else:
                                 response_content_str = "Błąd: Otrzymano nieprawidłową odpowiedź z lokalnego LLM."
                                 logger.error(response_content_str)
                            sources = []
                            graph_data = None
                        else: # Tryb RAG z lokalnym LLM
                            logger.info(f"⚙️ Tryb: RAG ({st.session_state.selected_chat_mode}) z lokalnym LLM")
                            st.session_state.chatbot.top_k = st.session_state.top_k
                            st.session_state.chatbot.top_k_reranker = st.session_state.top_k_reranker
                            st.session_state.chatbot.relevance_threshold = st.session_state.relevance_threshold
                            selected_prompt_text = AVAILABLE_PROMPTS[st.session_state.selected_chat_mode]
                            # Sprawdź, czy ekstrahować graf
                            should_extract_graph = (st.session_state.selected_chat_mode == "Analityk Grafów (RAG)")
                            logger.info(f"   Ekstrakcja grafu: {'Włączona' if should_extract_graph else 'Wyłączona'}")
                            # Sprawdź, czy filtrować po klastrze
                            query_cluster_filter_id = None
                            if st.session_state.get('filter_by_selected_cluster', False) and st.session_state.selected_cluster_id is not None:
                                query_cluster_filter_id = st.session_state.selected_cluster_id
                                logger.info(f"   Zapytanie będzie filtrowane do klastra ID: {query_cluster_filter_id}")
                            # Pobierz pliki kontekstowe
                            context_files_content_dict = st.session_state.get('context_files', {}) # Zmieniono nazwę zmiennej dla jasności
                            # Przekazanie cluster_assignments, extract_graph ORAZ filter_cluster_id do query
                            # DODANO: context_files_content
                            response_dict, sources, graph_data = st.session_state.chatbot.query(
                                question=last_user_prompt,
                                context_files_content=context_files_content_dict, # Przekazujemy słownik
                                system_prompt_override=selected_prompt_text,
                                cluster_assignments=st.session_state.get('cluster_assignments'), # Przekaż, jeśli istnieje
                                extract_graph=should_extract_graph, # Przekaż flagę ekstrakcji
                                filter_cluster_id=query_cluster_filter_id # Przekaż ID klastra do filtrowania lub None
                            )
                            # Resetuj flagę filtrowania PO wysłaniu zapytania
                            st.session_state.filter_by_selected_cluster = False
                            # _generate_answer jest wywoływane wewnątrz query, więc response_dict już zawiera 'content' i 'metadata'
                            if response_dict and response_dict.get("content"):
                                rag_response_content = response_dict["content"]
                                if isinstance(rag_response_content, AIMessage):
                                     response_content_str = rag_response_content.content
                                elif isinstance(rag_response_content, str):
                                     response_content_str = rag_response_content
                                else:
                                     response_content_str = str(rag_response_content)
                                response_metadata = response_dict.get("metadata", {})
                            else:
                                 response_content_str = "Błąd: Otrzymano nieprawidłową odpowiedź z RAG."
                                 logger.error(response_content_str)


                        # Zapis odpowiedzi do historii (z metadanymi)
                        st.session_state.chats[st.session_state.active_chat_id]["messages"].append({
                            "role": "assistant",
                            "content": response_content_str,
                            "sources": sources,
                            "graph_data": graph_data,
                            "prompt_answered": prompt_answered,
                            "metadata": response_metadata # DODANO: Zapis metadanych
                        })
                        save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                        # Resetuj flagę filtrowania również po udanej odpowiedzi (na wszelki wypadek)
                        st.session_state.filter_by_selected_cluster = False
                        # Flaga new_user_input_submitted już zresetowana na początku bloku
                        st.rerun() # Odśwież, aby pokazać odpowiedź

                    except StopIteration as si:
                        logger.debug(f"Przerwano generowanie odpowiedzi: {si}")
                        # Flaga już zresetowana
                        pass # st.rerun() może być pomocne do wyczyszczenia spinnera
                        st.rerun()

                    except Exception as e:
                        logger.error(f"❌ Błąd podczas przetwarzania pytania w trybie '{st.session_state.selected_chat_mode}' lub Gemini: {str(e)}", exc_info=True)
                        error_message = f"Wystąpił błąd: {e}"
                        st.session_state.chats[st.session_state.active_chat_id]["messages"].append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                            "graph_data": None,
                            "prompt_answered": last_user_prompt,
                            "metadata": {"model": "Błąd", "tokens": None, "time": 0.0} # Dodaj metadane błędu
                        })
                        save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                        # Flaga już zresetowana
                        st.rerun() # Odśwież, aby pokazać błąd
        else:
             # Ten przypadek nie powinien się zdarzyć przy poprawnej logice flagi,
             # ale warto zalogować, jeśli tak się stanie.
             logger.warning("Flaga 'new_user_input_submitted' była True, ale ostatnia wiadomość nie pochodziła od użytkownika. Resetowanie flagi.")
             # Flaga i tak jest resetowana na początku bloku 'if st.session_state.get("new_user_input_submitted", False):'


if __name__ == "__main__":
    main()
