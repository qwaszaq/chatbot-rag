# -*- coding: utf-8 -*-
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
# Dodano import dla obiektu odpowiedzi LLM z LangChain, aby móc sprawdzić jego typ
from langchain_core.messages import AIMessage, HumanMessage

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = "chat_history.json" # Ścieżka do pliku historii

# === POCZĄTEK POPRAWKI ===
# --- Mapowanie typów NER na kolory (DOSTOSOWANE DO pl_core_news_md/lg) ---
NER_COLORS = {
    "persName": "#FF6347",  # Tomato (Osoba)
    "orgName": "#4682B4",   # SteelBlue (Organizacja)
    "geogName": "#32CD32",  # LimeGreen (Nazwa geograficzna - np. kraj, rzeka)
    "placeName": "#9ACD32", # YellowGreen (Miejsce - np. miasto, adres)
    "date": "#FFD700",      # Gold (Data)
    "time": "#FFA500",      # Orange (Czas - jeśli model go rozpoznaje)
    # Możesz dodać inne etykiety z modelu spaCy:
    # https://spacy.io/models/pl#pl_core_news_md-labels
    "nam": "#A9A9A9",       # DarkGray (Inne nazwy własne - np. marki, wydarzenia - fallback)
    "ERROR": "#FF0000",     # Czerwony dla błędów NER (jeśli wystąpią w app.py)
    # Typy, które może wygenerować LLM, a nie spaCy:
    "DOC": "#BA55D3",       # MediumOrchid (Dokument - jeśli LLM dodałby taki typ)
    # Można dodać MISC jako fallback, jeśli spaCy zwraca taką etykietę
    # "MISC": "#A9A9A9",
}
DEFAULT_NODE_COLOR = "#D3D3D3" # LightGray (Domyślny/Nieznany/Brak encji/Nieznany typ NER)
# ------------------------------------
# === KONIEC POPRAWKI ===


# Domyślny prompt systemowy (pobrany z app.py)
DEFAULT_SYSTEM_PROMPT = """Rola:\n\nJesteś doświadczonym analitykiem tekstu. Otrzymujesz jeden lub kilka dokumentów jednocześnie (raporty, artykuły, prezentacje, sprawozdania, e-maile, pliki PDF, Word, itp.) oraz pytania od członków zespołu. Twoim zadaniem jest znalezienie konkretnych informacji w tych materiałach i udzielenie jasnych, precyzyjnych odpowiedzi.\n\n\n\n\nTwoje zadania:\n\n\n\n\nPrzeczytaj uważnie wszystkie dostarczone źródła.\n\nDla każdego pytania:\n\nZidentyfikuj i porównaj informacje we wszystkich dostępnych dokumentach.\n\nPodaj konkretną odpowiedź opartą wyłącznie na treści źródłowej.\n\nJeśli ta sama informacja występuje w kilku miejscach – wybierz najbardziej wiarygodną i aktualną wersję.\n\nJeśli są sprzeczne dane – zaznacz to i podaj możliwe wyjaśnienie.\n\nZawsze wskaż dokładne źródło w tekście (cytat, numer akapitu, nazwa pliku lub lokalizacja).\n\nJeżeli odpowiedź nie występuje bezpośrednio w dokumentach, zaznacz to i dodaj krótką interpretację (jeśli to możliwe).\n\nFormat odpowiedzi dla każdego pytania:\n\n\n\n\nPytanie: [tu wpisz pytanie]\n\nOdpowiedź: [jasna, konkretna odpowiedź]\n\nŹródło w dokumentach: [cytat, numer akapitu, nazwa pliku, strona lub opis fragmentu]\n\nKomentarz (jeśli potrzebny): [jeśli są sprzeczności lub brak informacji – wyjaśnij to]\n\n\n\n\nRodzaje pytań, które możesz otrzymać (i jak na nie reagować):\n\n\n\n\nPytanie o osobę (np. „Kto jest szefem tej organizacji?”):\n\n→ Wskaż imię, nazwisko, stanowisko i dokument, w którym to się znajduje.\n\nPytanie o liczby (np. „Ile pieniędzy zostało zabezpieczonych?”):\n\n→ Podaj konkretną kwotę, powołując się na dane z odpowiedniego pliku. Jeśli kwoty różnią się – opisz to.\n\nPytanie o przyczyny, działania, efekty (np. „Dlaczego projekt się opóźnił?”):\n\n→ Podaj powody, działania lub skutki, nawet jeśli są rozproszone w różnych źródłach.\n\nPytania o fakty i szczegóły:\n\n→ Wydobądź najważniejsze detale z różnych dokumentów i połącz je w spójną odpowiedź.\n\nDostarczone materiały:\n\n[lista lub zestaw dokumentów i źródeł – np. Raport_finansowy_Q4.pdf, Spotkanie_zarządu_dnotatki.docx, E-mail_od_dostawcy.msg]\n\n\n\n\n📌 Przykład odpowiedzi z wieloma źródłami:\n\nPytanie: Ile pieniędzy zostało zabezpieczonych?\n\nOdpowiedź: Zabezpieczono 4,5 mln zł (Raport_finansowy_Q4.pdf), jednak w notatce ze spotkania (Spotkanie_zarządu_dnotatki.docx) pojawia się kwota 4,2 mln zł – możliwa aktualizacja danych w późniejszym okresie.\n\nŹródło w dokumentach:\n\n\n\nRaport_finansowy_Q4.pdf, strona 5: „Zabezpieczone środki wynoszą 4,5 mln zł.”\n\nSpotkanie_zarządu_dnotatki.docx, akapit 3: „Dysponujemy obecnie 4,2 mln zł zarezerwowanych środków.”\n\nKomentarz: Możliwa różnica wynika z częściowego wydatkowania środków po publikacji raportu."""

# --- Funkcje do obsługi historii czatów ---
def load_chat_history():
    """Ładuje historię czatów z pliku JSON."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Podstawowa walidacja struktury
                if isinstance(data, dict) and "chats" in data and "active_chat_id" in data:
                     logger.info(f"💾 Załadowano historię czatów z {CHAT_HISTORY_FILE}")
                     return data
                else:
                     logger.warning(f"⚠️ Plik {CHAT_HISTORY_FILE} ma nieprawidłową strukturę.")
                     return None
        except json.JSONDecodeError:
            logger.error(f"❌ Błąd dekodowania JSON w pliku {CHAT_HISTORY_FILE}. Plik może być uszkodzony.")
            # Opcjonalnie: można dodać logikę backupu uszkodzonego pliku
            return None
        except Exception as e:
            logger.error(f"❌ Nieoczekiwany błąd podczas ładowania historii z {CHAT_HISTORY_FILE}: {e}")
            return None
    else:
        logger.info(f"ℹ️ Plik historii czatów {CHAT_HISTORY_FILE} nie istnieje. Inicjalizacja nowego stanu.")
        return None

def save_chat_history(chats_data, active_chat_id):
    """Zapisuje historię czatów do pliku JSON, konwertując wiadomości."""
    try:
        # Przygotuj dane do zapisu (konwersja wiadomości)
        serializable_chats = {}
        for chat_id, chat_content in chats_data.items():
            serializable_messages = []
            # Zapisujemy teraz CAŁĄ wiadomość, łącznie z graph_data
            for msg in chat_content.get("messages", []):
                 serializable_msg = {}
                 serializable_msg["role"] = msg.get("role")

                 content_data = msg.get("response_dict", msg.get("content"))
                 content_str = None

                 if isinstance(content_data, dict) and "content" in content_data:
                     raw_content = content_data.get("content")
                     if isinstance(raw_content, AIMessage):
                         content_str = raw_content.content
                     elif isinstance(raw_content, str):
                         content_str = raw_content
                     else:
                         content_str = str(raw_content) # Fallback
                 elif isinstance(content_data, AIMessage):
                     content_str = content_data.content
                 elif isinstance(content_data, str):
                     content_str = content_data # Wiadomość użytkownika lub starszy format
                 else:
                     content_str = str(content_data) # Fallback

                 if serializable_msg["role"] and content_str is not None:
                     if serializable_msg["role"] == "user":
                          serializable_msg["content"] = content_str
                     elif serializable_msg["role"] == "assistant":
                          # Dla asystenta zapisujemy słownik, jeśli jest, inaczej samą treść
                          if isinstance(msg.get("response_dict"), dict):
                               # Zapisujemy tylko treść i metadane (bez obiektu AIMessage)
                               metadata = msg["response_dict"].get("metadata")
                               serializable_msg["response_dict"] = {
                                   "content": content_str,
                                   "metadata": metadata
                               }
                          else:
                               serializable_msg["content"] = content_str # Fallback na starszy format lub błąd

                          # Zapisz źródła i graf
                          serializable_msg["sources"] = msg.get("sources", [])
                          serializable_msg["graph_data"] = msg.get("graph_data") # Zapisz dane grafu

                     serializable_messages.append(serializable_msg)

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

def main():
    # Wymuś reinicjalizację chatbota przy każdym uruchomieniu skryptu/odświeżeniu
    if "chatbot" not in st.session_state: # Uruchom reinicjalizację tylko jeśli chatbot nie istnieje
        if "chatbot" in st.session_state:
             del st.session_state.chatbot
             logger.info("Usunięto stary obiekt chatbota ze stanu sesji, aby wymusić reinicjalizację.")
        # Inicjalizacja chatbota przeniesiona niżej, po inicjalizacji czatów

    st.set_page_config(page_title="🤖 Chatbot RAG z Qdrant", layout="wide")

    st.title("🤖 Chatbot RAG z Qdrant")

    # --- Inicjalizacja stanu sesji z pliku lub domyślnie ---
    if "chats" not in st.session_state:
         loaded_data = load_chat_history()
         if loaded_data:
             st.session_state.chats = loaded_data["chats"]
             # Sprawdź poprawność załadowanego active_chat_id
             if loaded_data["active_chat_id"] in st.session_state.chats:
                 st.session_state.active_chat_id = loaded_data["active_chat_id"]
             elif st.session_state.chats:
                 st.session_state.active_chat_id = next(iter(st.session_state.chats))
                 logger.warning("Załadowany active_chat_id był nieprawidłowy. Ustawiono pierwszy dostępny.")
             else: # Pusty plik historii
                 first_chat_id = str(uuid.uuid4())
                 st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
                 st.session_state.active_chat_id = first_chat_id
                 logger.info("Załadowano pustą historię, utworzono nowy domyślny czat.")
                 save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
         else: # Nie udało się załadować pliku
             first_chat_id = str(uuid.uuid4())
             st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
             st.session_state.active_chat_id = first_chat_id
             logger.info("Nie załadowano historii. Utworzono nowy domyślny czat.")
             save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
    # Zabezpieczenie na wypadek utraty active_chat_id w trakcie sesji
    elif "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
         if st.session_state.chats:
             st.session_state.active_chat_id = next(iter(st.session_state.chats))
             logger.warning(f"Active_chat_id brakujący lub nieprawidłowy. Ustawiono na: {st.session_state.active_chat_id}")
             save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
         else: # Sytuacja awaryjna
             first_chat_id = str(uuid.uuid4())
             st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
             st.session_state.active_chat_id = first_chat_id
             logger.error("Stan awaryjny: Brak czatów i active_chat_id. Utworzono nowy domyślny czat.")
             save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
    # ---------------------------------------------------

    # Inicjalizacja chatbota (przeniesiona tutaj, po upewnieniu się, że stan czatów istnieje)
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = RAGChatbot()
            logger.info("✅ Pomyślnie zainicjalizowano chatbota RAG")
        except Exception as e:
            logger.error(f"❌ Błąd inicjalizacji chatbota: {str(e)}", exc_info=True) # Dodano exc_info
            st.error(f"Nie udało się zainicjalizować chatbota. Sprawdź logi aplikacji lub upewnij się, że Qdrant i LM Studio działają. Błąd: {e}")
            st.stop() # Zatrzymaj aplikację, jeśli chatbot się nie zainicjalizował
            # return # Alternatywnie można użyć return

    # Panel boczny z opcjami
    with st.sidebar:
        st.header("💬 Czaty")

        # Przycisk do tworzenia nowego czatu
        if st.button("+ Nowy Czat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            chat_count = len(st.session_state.chats) + 1
            st.session_state.chats[new_chat_id] = {"name": f"Czat {chat_count}", "messages": []}
            st.session_state.active_chat_id = new_chat_id
            if "editing_chat_id" in st.session_state: del st.session_state.editing_chat_id
            if "chat_to_delete_id" in st.session_state: del st.session_state.chat_to_delete_id
            save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
            st.rerun()

        st.markdown("---")

        # Wyświetlanie listy czatów i przycisków
        editing_chat_id_local = st.session_state.get("editing_chat_id")
        chat_to_delete_id_local = st.session_state.get("chat_to_delete_id")

        # Sortowanie czatów alfabetycznie wg nazwy dla lepszej organizacji
        sorted_chat_items = sorted(st.session_state.chats.items(), key=lambda item: item[1]['name'])

        for chat_id, chat_data in sorted_chat_items:
            # Pomijanie czatów w trybie edycji/usuwania
            if chat_id == editing_chat_id_local or chat_id == chat_to_delete_id_local:
                 continue

            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col1:
                button_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
                if st.button(f"{chat_data['name']}", key=f"select_{chat_id}", use_container_width=True, type=button_type):
                    st.session_state.active_chat_id = chat_id
                    if "editing_chat_id" in st.session_state: del st.session_state.editing_chat_id
                    if "chat_to_delete_id" in st.session_state: del st.session_state.chat_to_delete_id
                    # Zapis historii nie jest tu konieczny, chyba że chcemy zapisać tylko zmianę active_chat_id
                    # save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
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

        # Sekcja edycji nazwy
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

        # Sekcja potwierdzenia usunięcia
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
                                 # Ustaw na pierwszy czat z posortowanej listy
                                 st.session_state.active_chat_id = sorted(st.session_state.chats.keys(), key=lambda k: st.session_state.chats[k]['name'])[0]
                            else: # Usunięto ostatni czat
                                 if "active_chat_id" in st.session_state: del st.session_state.active_chat_id
                                 # Logika inicjalizacji na górze utworzy nowy czat po rerun
                       save_chat_history(st.session_state.chats, st.session_state.get("active_chat_id"))
                       st.rerun()
             with del_col2:
                  if st.button("Nie, anuluj", key=f"cancel_delete_{chat_to_delete_id_local}", use_container_width=True):
                       del st.session_state.chat_to_delete_id
                       st.rerun()

        st.markdown("---")
        st.header("📄 Zarządzanie dokumentami") # Zmieniono emoji

        # Przycisk do czyszczenia bazy danych
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
             st.warning(f"**Czy na pewno chcesz usunąć WSZYSTKIE dane z kolekcji '{collection_name_to_clear}' w Qdrant?** Tej operacji nie można cofnąć.")
             col_confirm, col_cancel = st.columns(2)
             with col_confirm:
                  if st.button(f"Tak, wyczyść '{collection_name_to_clear}'", type="primary", use_container_width=True): # Skrócono tekst przycisku
                       with st.spinner(f"Czyszczenie kolekcji '{collection_name_to_clear}'..."):
                            if "chatbot" in st.session_state:
                                 try:
                                      success = st.session_state.chatbot.clear_database()
                                      if success:
                                           st.success(f"✅ Kolekcja '{collection_name_to_clear}' została wyczyszczona.")
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

        # Przycisk przetwarzania przeniesiony pod oba inputy
        st.markdown("---")
        if st.button("🧠 Przetwórz dodane dokumenty"):
            files_to_process = []
            url_to_process = None

            if uploaded_files:
                files_to_process.extend(uploaded_files)
            if url_input:
                url_to_process = url_input

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
                              with open(file_path, "wb") as f:
                                  f.write(uploaded_file.getvalue())
                              logger.info(f"📥 Zapisano tymczasowy plik: {file_path}")

                              with st.spinner(f"Przetwarzanie {uploaded_file.name}..."):
                                   success = st.session_state.chatbot.process_document(file_path)

                              if success:
                                  logger.info(f"✅ Pomyślnie przetworzono: {uploaded_file.name}")
                                  processed_count += 1
                              else:
                                  logger.error(f"❌ Błąd przetwarzania: {uploaded_file.name}")
                                  error_count += 1
                                  st.warning(f"Błąd podczas przetwarzania pliku: {uploaded_file.name}")

                          except Exception as e:
                              logger.error(f"❌ Krytyczny błąd (plik {uploaded_file.name}): {str(e)}", exc_info=True)
                              st.error(f"Krytyczny błąd (plik {uploaded_file.name}): {e}")
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
                      if processed_count > 0: st.success(f"✅ Przetworzono pomyślnie {processed_count} z {num_files} plików.")
                      if error_count > 0: st.error(f"❌ Wystąpiły błędy dla {error_count} z {num_files} plików.")

                 # --- Przetwarzanie URL ---
                 if url_to_process:
                      st.info(f"Rozpoczynam przetwarzanie dokumentu z URL: {url_to_process}")
                      with st.spinner("Przetwarzanie URL..."):
                          try:
                              # Zakładamy, że process_document może obsłużyć URL lub potrzebna osobna metoda
                              # Jeśli RAGChatbot nie ma metody dla URL, trzeba ją dodać
                              # Na razie zakładamy, że process_document może przyjąć URL jako ścieżkę
                              # lub lepiej: dodać metodę process_url w RAGChatbot
                              # Dla uproszczenia - teraz zakładamy, że process_document nie działa z URL
                              # Trzeba by dodać logikę podobną jak w poprzedniej wersji
                              st.warning("Przetwarzanie z URL nie jest jeszcze w pełni zaimplementowane w tym przycisku.")
                              # Poniżej przykład jak mogłoby to wyglądać (wymaga dostosowania app.py)
                              """
                              documents = st.session_state.chatbot.document_loader.load_from_url(url_to_process)
                              if documents:
                                  all_splits = []
                                  for doc in documents:
                                      splits = st.session_state.chatbot.text_splitter.split_document(doc)
                                      all_splits.extend(splits)
                                  if all_splits:
                                      st.session_state.chatbot.qdrant_connector.add_documents(all_splits)
                                      st.success(f"✅ Dokument z URL przetworzony pomyślnie.")
                                  else:
                                      st.error("❌ Nie udało się podzielić dokumentu z URL.")
                              else:
                                  st.error("❌ Nie udało się załadować dokumentu z URL.")
                              """
                          except Exception as e:
                              logger.error(f"❌ Błąd przetwarzania URL: {str(e)}", exc_info=True)
                              st.error(f"Błąd podczas przetwarzania URL: {e}")


        st.markdown("---")
        st.header("⚙️ Ustawienia RAG")
        # Użycie kluczy sesji do przechowywania wartości suwaków
        st.session_state.top_k = st.slider("Liczba dokumentów z Qdrant", 1, 200, st.session_state.get('top_k', 99))
        st.session_state.top_k_reranker = st.slider("Liczba dokumentów po rerankingu", 1, 100, st.session_state.get('top_k_reranker', 33))
        st.session_state.relevance_threshold = st.slider("Próg istotności rerankera", 0.0, 1.0, st.session_state.get('relevance_threshold', 0.0), 0.01)

        st.markdown("---")
        st.header("📝 Prompt systemowy")
        if 'system_prompt' not in st.session_state:
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.session_state.system_prompt = st.text_area(
            "Edytuj instrukcję dla LLM:", # Zmieniono etykietę
            value=st.session_state.system_prompt,
            height=300
        )

        # Usunięto przycisk "Resetuj czat", bo mamy zarządzanie czatami


    # --- Główny obszar czatu ---
    active_chat_id = st.session_state.active_chat_id
    # Upewnijmy się, że aktywny chat istnieje (na wypadek usunięcia)
    if active_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            st.session_state.active_chat_id = next(iter(st.session_state.chats))
            active_chat_id = st.session_state.active_chat_id
            logger.warning("Aktywny czat nie istniał. Ustawiono pierwszy dostępny.")
            st.rerun() # Odśwież, aby załadować właściwy czat
        else:
            # Jeśli nie ma żadnych czatów, logika na górze powinna stworzyć nowy
            logger.error("Brak dostępnych czatów.")
            st.error("Wystąpił błąd - brak dostępnych czatów.")
            st.stop()

    active_chat_data = st.session_state.chats[active_chat_id]
    st.header(f"Czat: {active_chat_data['name']}")

    # Wyświetlanie historii aktywnego czatu
    if not active_chat_data.get("messages"): # Bezpieczniejsze sprawdzenie
         st.info("Rozpocznij rozmowę, zadając pytanie poniżej.")

    for message_index, message in enumerate(active_chat_data.get("messages", [])): # Dodano enumerate dla unikalnego klucza grafu
        with st.chat_message(message["role"]):
            content_to_display = ""
            metadata_to_display = None
            sources_to_display = message.get("sources", [])
            graph_data = message.get("graph_data") # Pobierz dane grafu
            is_error = False

            if message["role"] == "user":
                content_to_display = message.get("content", "_Brak treści_")
            elif message["role"] == "assistant":
                # Sprawdzamy, czy wiadomość została załadowana z JSON (ma content/response_dict jako string/dict)
                # czy jest z bieżącej sesji (response_dict zawiera obiekt AIMessage)
                response_data = message.get("response_dict", message.get("content"))

                if isinstance(response_data, dict): # Format z response_dict (nowy lub załadowany)
                    content_or_obj = response_data.get("content")
                    metadata_to_display = response_data.get("metadata")

                    if isinstance(content_or_obj, AIMessage): # Z bieżącej sesji
                         content_to_display = content_or_obj.content
                    elif isinstance(content_or_obj, str): # Załadowany z JSON lub błąd
                         content_to_display = content_or_obj
                         if metadata_to_display is None: is_error = True # Zakładamy, że błąd nie ma metadanych
                    else:
                         content_to_display = "_Nieznany format treści w response_dict_"
                         logger.warning(f"Nieznany typ treści w response_dict: {type(content_or_obj)}")
                elif isinstance(response_data, str): # Starszy format (tylko content) lub wiadomość użytkownika
                     content_to_display = response_data
                else:
                     content_to_display = "_Brak treści lub nieznany format_"
                     logger.warning(f"Nieznany format wiadomości: {message}")

            # Wyświetl treść
            if content_to_display:
                st.markdown(content_to_display)
            else:
                st.markdown("_Pusta wiadomość_")

            # Wyświetl metadane (jeśli są i nie jest to błąd)
            if metadata_to_display and not is_error:
                model = metadata_to_display.get("model", "N/A")
                tokens = metadata_to_display.get("tokens", "N/A")
                response_time = metadata_to_display.get("time", 0.0)
                tokens_str = str(tokens) if tokens is not None else "N/A"
                st.caption(f"Model: {model} | Tokeny: {tokens_str} | Czas: {response_time:.2f}s")

            # Wyświetl źródła (jeśli są)
            if sources_to_display:
                    with st.expander("Źródła"):
                        st.markdown("Wykorzystane dokumenty:")
                        # Wyświetlamy tylko nazwy plików, bez pełnych ścieżek
                        unique_sources = set()
                        for source in sources_to_display:
                             # Sprawdź, czy source jest stringiem przed użyciem os.path.basename
                             if isinstance(source, str):
                                  unique_sources.add(os.path.basename(source))
                             else:
                                  unique_sources.add(str(source)) # Fallback na string
                        for base_name in sorted(list(unique_sources)):
                             st.markdown(f"- `{base_name}`")


            # --- Sekcja wizualizacji grafu ---
            if graph_data:
                 # Używamy indeksu wiadomości dla unikalnego klucza
                 toggle_key = f"graph_toggle_{active_chat_id}_{message_index}"
                 if st.toggle("Wyświetl Graf Wiedzy", key=toggle_key, value=False): # Domyślnie zwinięty
                      try:
                          nodes = []
                          edges = []
                          # Sprawdź, czy graph_data ma oczekiwaną strukturę node-link
                          if isinstance(graph_data, dict) and 'nodes' in graph_data and 'links' in graph_data:

                              # === POCZĄTEK ZMIAN DLA ROZMIARU WĘZŁÓW ===
                              try:
                                  # 1. Odtwórz graf NetworkX w UI, aby obliczyć stopnie
                                  # Użyj DiGraph, bo krawędzie mogą być skierowane
                                  temp_G = nx.node_link_graph(graph_data, directed=True, multigraph=False)

                                  # 2. Oblicz stopnie (suma połączeń wchodzących i wychodzących dla każdego węzła)
                                  degrees = dict(temp_G.degree())

                                  # 3. Znajdź maksymalny stopień (jeśli są jakieś połączenia)
                                  max_degree = 0 # Domyślnie
                                  if degrees: # Sprawdź, czy słownik stopni nie jest pusty (są jakieś węzły)
                                       # Sprawdź, czy są jakiekolwiek krawędzie (max > 0)
                                       if any(d > 0 for d in degrees.values()):
                                            max_degree = max(degrees.values())
                                       else:
                                            max_degree = 0 # Brak krawędzi

                                  # 4. Definicja rozmiarów
                                  BASE_NODE_SIZE = 15
                                  LARGE_NODE_SIZE = 30 # Rozmiar dla "najważniejszego" węzła/węzłów
                                  logger.debug(f"Obliczanie rozmiaru węzłów. Max degree: {max_degree}")

                              except Exception as graph_reconstruction_e:
                                  logger.error(f"Błąd podczas rekonstrukcji grafu lub obliczania stopni: {graph_reconstruction_e}")
                                  # W przypadku błędu użyjemy domyślnych rozmiarów
                                  degrees = {}
                                  max_degree = 0
                                  BASE_NODE_SIZE = 15
                                  LARGE_NODE_SIZE = 15 # Bez powiększania przy błędzie
                              # === KONIEC ZMIAN DLA ROZMIARU WĘZŁÓW ===


                              for node_data in graph_data.get('nodes', []):
                                   node_id = node_data.get('id', str(uuid.uuid4())) # Fallback na UUID jeśli brak id
                                   label = node_data.get('label', node_id)
                                   wrapped_label = textwrap.fill(label, width=20, replace_whitespace=False, drop_whitespace=False) # Zawijanie etykiet
                                   ner_type = node_data.get('ner_type')
                                   node_color = NER_COLORS.get(ner_type, DEFAULT_NODE_COLOR) # Pobierz kolor, fallback na domyślny

                                   # === ZMIANA: Ustawienie rozmiaru węzła ===
                                   node_size = BASE_NODE_SIZE # Ustaw domyślny rozmiar
                                   current_degree = degrees.get(node_id, 0) # Pobierz stopień dla tego węzła

                                   # Powiększ węzeł, jeśli ma maksymalny stopień (i jest on większy niż 0)
                                   # Sprawdzamy max_degree > 0, aby nie powiększać węzłów w grafie bez krawędzi
                                   if max_degree > 0 and current_degree == max_degree:
                                        node_size = LARGE_NODE_SIZE
                                        logger.debug(f"Węzeł '{node_id}' ma max stopień ({current_degree}), rozmiar={LARGE_NODE_SIZE}")
                                   # === KONIEC ZMIANY ===

                                   # Dodaj podpowiedź (tooltip) z pełną etykietą i typem NER
                                   tooltip = f"{label}"
                                   if ner_type:
                                       tooltip += f" ({ner_type})"

                                   nodes.append(Node(id=node_id,
                                                      label=wrapped_label,
                                                      size=node_size,  # <-- Tutaj używany jest obliczony rozmiar
                                                      color=node_color,
                                                      title=tooltip)) # Dodano title dla tooltipu

                              for link_data in graph_data.get('links', []):
                                   # Upewnij się, że source i target istnieją jako węzły
                                   source_id = link_data.get('source')
                                   target_id = link_data.get('target')
                                   if source_id and target_id: # Pomijaj krawędzie bez source/target
                                       edges.append(Edge(source=source_id,
                                                         target=target_id,
                                                         label=link_data.get('label', ''),
                                                         color="#cccccc" # Jasnoszary dla krawędzi
                                                         ))

                              # Konfiguracja grafu - Statyczny układ hierarchiczny (bez zmian)
                              config = Config(width='100%',
                                              height=600,
                                              directed=True,
                                              physics=False,
                                              hierarchical=True,
                                              layout={
                                                  'hierarchical': {
                                                      'direction': 'UD',
                                                      'sortMethod': 'hubsize'
                                                  }
                                              },
                                              nodeHighlightBehavior=True,
                                              highlightColor='#F7A7A6',
                                              collapsible=False,
                                             )

                              if nodes:
                                   with st.spinner("Ładowanie grafu..."):
                                        agraph(nodes=nodes, edges=edges, config=config) # Użycie poprawionej konfiguracji
                              else:
                                   st.caption("_Brak węzłów do wyświetlenia w grafie._")
                          else:
                               logger.warning(f"Nieprawidłowa struktura graph_data: {type(graph_data)}")
                               st.caption("_Nieprawidłowy format danych grafu._")

                      except Exception as e:
                           logger.error(f"Błąd podczas renderowania grafu: {e}", exc_info=True)
                           st.error("Wystąpił błąd podczas wyświetlania grafu.")
            # -----------------------------------------

    # Pole do wprowadzania nowego pytania
    if prompt := st.chat_input("Zadaj pytanie..."):
        # Dodaj pytanie użytkownika do historii aktywnego czatu
        st.session_state.chats[active_chat_id]["messages"].append({"role": "user", "content": prompt})
        # Zapisz historię od razu po dodaniu pytania użytkownika
        save_chat_history(st.session_state.chats, active_chat_id)
        # Natychmiastowe wyświetlenie pytania użytkownika przez rerun
        st.rerun()

    # Jeśli ostatnia wiadomość była od użytkownika, wygeneruj odpowiedź
    last_message = active_chat_data.get("messages", [])[-1] if active_chat_data.get("messages") else None
    if last_message and last_message["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Myślę..."):
                try:
                    # Ustaw parametry RAG z sesji tuż przed zapytaniem
                    if "chatbot" in st.session_state:
                         st.session_state.chatbot.top_k = st.session_state.get('top_k', 99)
                         st.session_state.chatbot.top_k_reranker = st.session_state.get('top_k_reranker', 33)
                         st.session_state.chatbot.relevance_threshold = st.session_state.get('relevance_threshold', 0.0)
                    else:
                         st.error("Błąd: Chatbot nie jest zainicjalizowany.")
                         st.stop() # Zatrzymaj przetwarzanie

                    # Pobierz odpowiedź, źródła i dane grafu
                    response_dict, sources, graph_data = st.session_state.chatbot.query(
                        question=last_message["content"], # Użyj ostatniego pytania użytkownika
                        system_prompt_override=st.session_state.system_prompt
                    )

                    # Dodaj odpowiedź asystenta do historii aktywnego czatu
                    st.session_state.chats[active_chat_id]["messages"].append({
                        "role": "assistant",
                        "response_dict": response_dict,
                        "sources": sources,
                        "graph_data": graph_data
                    })
                    # Zapisz historię po otrzymaniu odpowiedzi
                    save_chat_history(st.session_state.chats, active_chat_id)
                    # Odśwież, aby wyświetlić odpowiedź
                    st.rerun()

                except Exception as e:
                    logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}", exc_info=True)
                    error_message = f"Wystąpił błąd: {e}"
                    st.error(error_message)
                    # Zapisz informację o błędzie w historii (jako wiadomość asystenta)
                    st.session_state.chats[active_chat_id]["messages"].append({
                        "role": "assistant",
                        "response_dict": {"content": error_message, "metadata": None}, # Format błędu
                        "sources": [],
                        "graph_data": None
                    })
                    save_chat_history(st.session_state.chats, active_chat_id)
                    # Nie ma potrzeby rerun, błąd jest już wyświetlony

if __name__ == "__main__":
    # nest_asyncio jest zwykle potrzebne w środowiskach takich jak Jupyter,
    # ale w Streamlit może nie być konieczne lub nawet powodować problemy.
    # Spróbujmy bez tego. Jeśli pojawią się błędy asyncio, można odkomentować.
    # import nest_asyncio
    # nest_asyncio.apply()
    main()