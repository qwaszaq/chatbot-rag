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
            for msg in chat_content.get("messages", []):
                role = msg.get("role")
                content_data = msg.get("response_dict", msg.get("content")) # Sprawdź nowy i stary format

                content_str = ""
                if isinstance(content_data, dict) and "content" in content_data:
                    # Obsługa nowego formatu (słownik odpowiedzi)
                    raw_content = content_data["content"]
                    if isinstance(raw_content, AIMessage):
                        content_str = raw_content.content
                    elif isinstance(raw_content, str):
                        content_str = raw_content # Błąd lub zwykły string
                    else:
                         content_str = str(raw_content) # Fallback
                elif isinstance(content_data, AIMessage):
                    # Obsługa AIMessage bezpośrednio w kluczu 'content' (starszy format)
                    content_str = content_data.content
                elif isinstance(content_data, str):
                     content_str = content_data # Wiadomość użytkownika lub starszy format
                else:
                     content_str = str(content_data) # Fallback

                if role and content_str is not None: # Zapisz tylko jeśli rola i treść istnieją
                    serializable_messages.append({"role": role, "content": content_str})
                # Uwaga: Metadane (model, tokeny, czas) i źródła są tracone przy zapisie

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
        # logger.info(f"💾 Zapisano historię czatów do {CHAT_HISTORY_FILE}") # Można odkomentować dla debugowania
    except Exception as e:
        logger.error(f"❌ Błąd podczas zapisywania historii do {CHAT_HISTORY_FILE}: {e}")
# -----------------------------------------

def main():
# Wymuś reinicjalizację chatbota przy każdym uruchomieniu skryptu/odświeżeniu
    if "chatbot" in st.session_state:
        del st.session_state.chatbot
        logger.info("Usunięto stary obiekt chatbota ze stanu sesji, aby wymusić reinicjalizację.")
    st.set_page_config(page_title="🤖 Chatbot RAG z Qdrant", layout="wide")

    st.title("🤖 Chatbot RAG z Qdrant")
    # Usunięto opis RAG
    # --- Inicjalizacja stanu sesji z pliku lub domyślnie ---
    if "chats" not in st.session_state:
         loaded_data = load_chat_history()
         if loaded_data:
              st.session_state.chats = loaded_data["chats"]
              # Sprawdź, czy załadowany active_chat_id jest prawidłowy
              if loaded_data["active_chat_id"] in st.session_state.chats:
                   st.session_state.active_chat_id = loaded_data["active_chat_id"]
              elif st.session_state.chats: # Jeśli nieprawidłowy, ale są inne czaty
                   st.session_state.active_chat_id = next(iter(st.session_state.chats))
                   logger.warning("Załadowany active_chat_id był nieprawidłowy. Ustawiono pierwszy dostępny.")
              else: # Jeśli nie ma czatów po załadowaniu (plik był pusty?)
                   # Utwórz domyślny czat
                   first_chat_id = str(uuid.uuid4())
                   st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
                   st.session_state.active_chat_id = first_chat_id
                   logger.info("Załadowano pustą historię, utworzono nowy domyślny czat.")
                   save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz nowy stan
         else:
              # Jeśli nie udało się załadować, utwórz domyślny czat
              first_chat_id = str(uuid.uuid4())
              st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
              st.session_state.active_chat_id = first_chat_id
              logger.info("Nie załadowano historii. Utworzono nowy domyślny czat.")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz nowy stan
    # Upewnij się, że active_chat_id istnieje (dodatkowe zabezpieczenie)
    elif "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
         if st.session_state.chats:
              st.session_state.active_chat_id = next(iter(st.session_state.chats))
              logger.warning(f"Active_chat_id brakujący lub nieprawidłowy po inicjalizacji. Ustawiono na: {st.session_state.active_chat_id}")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz poprawny stan
         else: # Sytuacja awaryjna - brak czatów i active_id
              first_chat_id = str(uuid.uuid4())
              st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
              st.session_state.active_chat_id = first_chat_id
              logger.error("Stan awaryjny: Brak czatów i active_chat_id po inicjalizacji. Utworzono nowy domyślny czat.")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
    # ---------------------------------------------------

    # Inicjalizacja chatbota (pozostaje bez zmian)
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = RAGChatbot()
            logger.info("✅ Pomyślnie zainicjalizowano chatbota RAG")
        except Exception as e:
            logger.error(f"❌ Błąd inicjalizacji chatbota: {str(e)}")
            st.error(f"Nie udało się zainicjalizować chatbota. Sprawdź logi aplikacji lub upewnij się, że Qdrant i LM Studio działają. Błąd: {e}")
            return

    # Panel boczny z opcjami
    with st.sidebar:
        st.header("💬 Czaty")

        # Przycisk do tworzenia nowego czatu
        if st.button("+ Nowy Czat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            chat_count = len(st.session_state.chats) + 1
            st.session_state.chats[new_chat_id] = {"name": f"Czat {chat_count}", "messages": []}
            st.session_state.active_chat_id = new_chat_id
            # Wyczyść stan edycji, jeśli był aktywny
            if "editing_chat_id" in st.session_state:
                del st.session_state.editing_chat_id
            save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz stan po dodaniu czatu
            st.rerun()

        st.markdown("---")

        # Wyświetlanie listy czatów i przycisków
        # chat_ids_to_delete = [] # Niepotrzebne, użyjemy stanu sesji
        editing_chat_id_local = st.session_state.get("editing_chat_id")
        chat_to_delete_id_local = st.session_state.get("chat_to_delete_id") # Pobierz ID czatu do usunięcia

        for chat_id, chat_data in st.session_state.chats.items():
             # Nie wyświetlaj normalnego wiersza dla czatu w trakcie edycji lub usuwania
            if chat_id == editing_chat_id_local or chat_id == chat_to_delete_id_local:
                 continue

            col1, col2, col3 = st.columns([0.7, 0.15, 0.15]) # Kolumny dla nazwy, edycji, usuwania
            with col1:
                # Przycisk do przełączania aktywnego czatu
                button_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
                if st.button(f"{chat_data['name']}", key=f"select_{chat_id}", use_container_width=True, type=button_type):
                    st.session_state.active_chat_id = chat_id
                    # Wyczyść stan edycji przy zmianie czatu
                    if "editing_chat_id" in st.session_state:
                         del st.session_state.editing_chat_id
                    # Wyczyść też stan usuwania, jeśli był aktywny
                    if "chat_to_delete_id" in st.session_state:
                         del st.session_state.chat_to_delete_id
                    # Nie ma potrzeby zapisywać tutaj, bo stan wiadomości się nie zmienił
                    st.rerun()
            with col2:
                 # Przycisk do rozpoczęcia edycji nazwy
                 if st.button("✏️", key=f"edit_{chat_id}", help="Edytuj nazwę czatu"):
                     st.session_state.editing_chat_id = chat_id
                     # Wyczyść stan usuwania, jeśli był aktywny
                     if "chat_to_delete_id" in st.session_state:
                          del st.session_state.chat_to_delete_id
                     # Nie ma potrzeby zapisywać tutaj
                     st.rerun() # Odśwież, aby pokazać pole edycji
            with col3:
                 # Przycisk do rozpoczęcia usuwania czatu
                 # Nie pozwól usunąć ostatniego czatu
                 can_delete = len(st.session_state.chats) > 1
                 if st.button("🗑️", key=f"delete_{chat_id}", help="Usuń czat", disabled=not can_delete):
                     st.session_state.chat_to_delete_id = chat_id
                     # Wyczyść stan edycji, jeśli był aktywny
                     if "editing_chat_id" in st.session_state:
                          del st.session_state.editing_chat_id
                     # Nie ma potrzeby zapisywać tutaj
                     st.rerun() # Odśwież, aby pokazać potwierdzenie

        st.markdown("---")

        # Sekcja edycji nazwy (wyświetlana warunkowo)
        if editing_chat_id_local:
            st.subheader("Edytuj nazwę czatu")
            current_name = st.session_state.chats[editing_chat_id_local]["name"]
            new_name = st.text_input("Nowa nazwa:", value=current_name, key=f"input_edit_{editing_chat_id_local}")

            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                if st.button("Zapisz", key=f"save_edit_{editing_chat_id_local}", use_container_width=True):
                    if new_name.strip(): # Upewnij się, że nazwa nie jest pusta
                         st.session_state.chats[editing_chat_id_local]["name"] = new_name.strip()
                         edited_chat_id = editing_chat_id_local # Zapisz ID przed usunięciem ze stanu
                         del st.session_state.editing_chat_id # Zakończ edycję
                         save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz stan po edycji nazwy
                         st.rerun()
                    else:
                         st.warning("Nazwa czatu nie może być pusta.")
            with edit_col2:
                if st.button("Anuluj", key=f"cancel_edit_{editing_chat_id_local}", use_container_width=True):
                    del st.session_state.editing_chat_id # Zakończ edycję
                    # Nie ma potrzeby zapisywać przy anulowaniu
                    st.rerun()

        # Sekcja potwierdzenia usunięcia (wyświetlana warunkowo)
        if chat_to_delete_id_local:
             st.warning(f"Czy na pewno chcesz usunąć czat '{st.session_state.chats[chat_to_delete_id_local]['name']}'?")
             del_col1, del_col2 = st.columns(2)
             with del_col1:
                  if st.button("Tak, usuń", key=f"confirm_delete_{chat_to_delete_id_local}", use_container_width=True, type="primary"):
                       deleted_chat_id = chat_to_delete_id_local
                       del st.session_state.chats[deleted_chat_id] # Usuń czat ze słownika
                       del st.session_state.chat_to_delete_id # Wyczyść stan usuwania

                       # Jeśli usunięto aktywny czat, ustaw nowy aktywny
                       if st.session_state.active_chat_id == deleted_chat_id:
                            if st.session_state.chats: # Jeśli są jeszcze jakieś czaty
                                 st.session_state.active_chat_id = next(iter(st.session_state.chats))
                            else: # Jeśli usunięto ostatni czat
                                 # Utwórz nowy domyślny czat (logika inicjalizacji to obsłuży po rerun)
                                 # Wystarczy wyczyścić active_chat_id, inicjalizacja na górze to złapie
                                 if "active_chat_id" in st.session_state: # Dodatkowe sprawdzenie
                                     del st.session_state.active_chat_id
                       save_chat_history(st.session_state.chats, st.session_state.get("active_chat_id")) # Zapisz stan po usunięciu
                       st.rerun()
             with del_col2:
                  if st.button("Nie, anuluj", key=f"cancel_delete_{chat_to_delete_id_local}", use_container_width=True):
                       del st.session_state.chat_to_delete_id # Wyczyść stan usuwania
                       # Nie ma potrzeby zapisywać przy anulowaniu
                       st.rerun()

        st.markdown("---")
        st.header("� Zarządzanie dokumentami") # Przeniesiono niżej

        # Przycisk do czyszczenia bazy danych
        st.markdown("---")
        st.subheader("⚠️ Strefa niebezpieczna")
        if 'confirm_clear_db' not in st.session_state:
             st.session_state.confirm_clear_db = False

        # Pobierz nazwę kolekcji z obiektu chatbota (jeśli istnieje)
        collection_name_to_clear = "nieznana"
        if "chatbot" in st.session_state and hasattr(st.session_state.chatbot, 'qdrant_connector'):
             collection_name_to_clear = st.session_state.chatbot.qdrant_connector.collection_name

        if st.button(f"🗑️ Wyczyść kolekcję '{collection_name_to_clear}'", type="secondary"):
             st.session_state.confirm_clear_db = True

        if st.session_state.confirm_clear_db:
             st.warning(f"**Czy na pewno chcesz usunąć WSZYSTKIE dane z kolekcji '{collection_name_to_clear}' w Qdrant?** Tej operacji nie można cofnąć.")
             col_confirm, col_cancel = st.columns(2)
             with col_confirm:
                  if st.button(f"Tak, wyczyść kolekcję '{collection_name_to_clear}'", type="primary"):
                       with st.spinner(f"Czyszczenie kolekcji '{collection_name_to_clear}'..."):
                            # Upewnij się, że chatbot istnieje przed wywołaniem metody
                            if "chatbot" in st.session_state:
                                 try:
                                      success = st.session_state.chatbot.clear_database()
                                      if success:
                                           st.success(f"✅ Kolekcja '{collection_name_to_clear}' została wyczyszczona.")
                                      else:
                                           st.error(f"❌ Wystąpił błąd podczas czyszczenia kolekcji '{collection_name_to_clear}'.")
                                 except AttributeError:
                                      st.error("❌ Błąd: Wygląda na to, że obiekt chatbota nie został poprawnie załadowany lub zaktualizowany. Spróbuj zrestartować aplikację.")
                                 except Exception as e:
                                      st.error(f"❌ Wystąpił nieoczekiwany błąd: {e}")
                            else:
                                 st.error("❌ Błąd: Obiekt chatbota nie jest dostępny w sesji.")

                       st.session_state.confirm_clear_db = False # Zresetuj stan potwierdzenia
                       st.rerun() # Odśwież, aby ukryć potwierdzenie
             with col_cancel:
                  if st.button("Anuluj"):
                       st.session_state.confirm_clear_db = False # Zresetuj stan potwierdzenia
                       st.rerun() # Odśwież, aby ukryć potwierdzenie
        st.markdown("---")


        # Ładowanie z pliku lokalnego (umożliwia wybór wielu plików)
        uploaded_files = st.file_uploader(
            "Wybierz dokumenty do przetworzenia",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        # Logika przetwarzania plików zostanie przeniesiona do obsługi przycisku poniżej.
        # Usunięto kod zapisujący pojedynczy plik tymczasowy tutaj.

        # Ładowanie z URL
        st.markdown("---")
        st.subheader("Z adresu URL")
        url_input = st.text_input("Wprowadź adres URL dokumentu")
        if st.button("🔗 Przetwórz z URL"):
            if url_input:
                 with st.spinner("Przetwarzanie dokumentu z URL..."):
                    try:
                        documents = st.session_state.chatbot.document_loader.load_from_url(url_input)
                        if documents:
                            st.info(f"Pobrano {len(documents)} dokument(ów) z URL. Dzielenie na chunki i zapis do Qdrant...")

                            all_splits = []
                            for doc in documents:
                                splits = st.session_state.chatbot.text_splitter.split_document(doc)
                                all_splits.extend(splits)

                            if all_splits:
                                st.session_state.chatbot.qdrant_connector.add_documents(all_splits) # Usunięto embeddings argument
                                st.success("✅ Dokument z URL został pomyślnie przetworzony i zapisany do Qdrant")
                            else:
                                st.error("❌ Nie udało się podzielić dokumentu z URL na chunki lub zapisać do Qdrant")

                        else:
                            st.error("❌ Nie udało się pobrać lub załadować dokumentu z podanego URL")

                    except Exception as e:
                        logger.error(f"❌ Błąd podczas przetwarzania dokumentu z URL: {str(e)}")
                        st.error(f"Wystąpił błąd podczas przetwarzania dokumentu z URL: {e}")
            else:
                st.warning("⚠️ Wprowadź adres URL")


        st.markdown("---")
        if st.button("🧠 Przetwórz wybrane dokumenty"):
            if uploaded_files: # Sprawdź, czy lista plików nie jest pusta
                num_files = len(uploaded_files)
                st.info(f"Rozpoczynam przetwarzanie {num_files} plików...")
                processed_count = 0
                error_count = 0

                # Użyj paska postępu
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    temp_dir = None # Inicjalizuj przed blokiem try
                    try:
                        # Utwórz tymczasowy katalog dla każdego pliku
                        temp_dir = tempfile.mkdtemp()
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        status_text.text(f"Przetwarzanie pliku {i+1}/{num_files}: {uploaded_file.name}")

                        # Zapisz plik tymczasowo
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        logger.info(f"📥 Zapisano tymczasowy plik: {file_path}")

                        # Przetwórz dokument za pomocą RAGChatbot
                        with st.spinner(f"Przetwarzanie {uploaded_file.name}..."):
                             success = st.session_state.chatbot.process_document(file_path)

                        if success:
                            logger.info(f"✅ Pomyślnie przetworzono: {uploaded_file.name}")
                            processed_count += 1
                        else:
                            logger.error(f"❌ Błąd przetwarzania: {uploaded_file.name}")
                            error_count += 1
                            st.warning(f"Wystąpił błąd podczas przetwarzania pliku: {uploaded_file.name}")

                    except Exception as e:
                        logger.error(f"❌ Krytyczny błąd podczas obsługi pliku {uploaded_file.name}: {str(e)}")
                        st.error(f"Wystąpił krytyczny błąd podczas obsługi pliku {uploaded_file.name}: {e}")
                        error_count += 1
                    finally:
                        # Posprzątaj plik i katalog tymczasowy
                        if temp_dir and os.path.exists(temp_dir):
                            try:
                                if 'file_path' in locals() and os.path.exists(file_path):
                                    os.unlink(file_path)
                                os.rmdir(temp_dir)
                                logger.info(f"🧹 Posprzątano tymczasowe zasoby dla: {uploaded_file.name}")
                            except Exception as cleanup_e:
                                logger.error(f"🧹❌ Błąd podczas sprzątania zasobów tymczasowych dla {uploaded_file.name}: {cleanup_e}")
                    # Aktualizuj pasek postępu
                    progress_bar.progress((i + 1) / num_files)

                # Podsumowanie po zakończeniu pętli
                status_text.text("Zakończono przetwarzanie wszystkich plików.")
                if processed_count > 0:
                    st.success(f"✅ Pomyślnie przetworzono {processed_count} z {num_files} plików.")
                if error_count > 0:
                    st.error(f"❌ Wystąpiły błędy podczas przetwarzania {error_count} z {num_files} plików. Sprawdź logi po więcej szczegółów.")

            else:
                st.warning("⚠️ Najpierw wybierz dokumenty do przetworzenia.")


        st.markdown("---")
        st.header("⚙️ Ustawienia RAG")
        st.session_state.top_k = st.slider("Liczba dokumentów do pobrania z Qdrant", 1, 200, 99)
        st.session_state.top_k_reranker = st.slider("Liczba dokumentów po rerankingu", 1, 100, 33)
        st.session_state.relevance_threshold = st.slider("Próg istotności rerankera", 0.0, 1.0, 0.0, 0.01)
        st.markdown("---")
        st.header("📝 Prompt systemowy")
        # Inicjalizacja wartości w sesji, jeśli jeszcze nie istnieje
        if 'system_prompt' not in st.session_state:
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        # Pole do edycji promptu
        st.session_state.system_prompt = st.text_area(
            "Edytuj prompt systemowy (instrukcję dla LLM):",
            value=st.session_state.system_prompt,
            height=300
        )
        st.markdown("---")
        if st.button("🔄 Resetuj czat"):
            st.session_state.messages = []
            st.rerun() # Używamy st.rerun() do odświeżenia po resecie czatu


    # --- Wyświetlanie aktywnego czatu ---
    active_chat_id = st.session_state.active_chat_id
    active_chat_data = st.session_state.chats[active_chat_id]
    st.header(f"Czat: {active_chat_data['name']}") # Wyświetl nazwę aktywnego czatu

    # Wyświetl historię aktywnego czatu
    if not active_chat_data["messages"]:
         st.info("Rozpocznij rozmowę, zadając pytanie poniżej.")

    for message in active_chat_data["messages"]: # Iteruj po wiadomościach aktywnego czatu
        with st.chat_message(message["role"]):
            content_to_display = ""
            metadata_to_display = None
            sources_to_display = message.get("sources", []) # Pobierz źródła, jeśli istnieją
            is_error = False

            if message["role"] == "user":
                content_to_display = message.get("content", "_Brak treści_")
            elif message["role"] == "assistant":
                # Priorytetowo sprawdzamy, czy istnieje pełny słownik odpowiedzi (wiadomość z bieżącej sesji)
                if isinstance(message.get("response_dict"), dict):
                    response_content = message["response_dict"].get("content")
                    metadata_to_display = message["response_dict"].get("metadata")
                    logger.debug(f"Wyświetlanie wiadomości asystenta (z response_dict). Typ response_content: {type(response_content)}")

                    # Wyodrębnij treść z AIMessage lub stringa błędu
                    if hasattr(response_content, 'content'): # Sprawdź, czy obiekt ma atrybut 'content'
                        content_to_display = getattr(response_content, 'content', None)
                        if content_to_display is None:
                            logger.warning(f"Atrybut 'content' w obiekcie odpowiedzi (response_dict) był None. Typ obiektu: {type(response_content)}")
                            content_to_display = "_Problem z odczytem treści odpowiedzi_"
                    elif isinstance(response_content, str): # Obsługa komunikatu o błędzie
                        content_to_display = response_content
                        if metadata_to_display is None: # Zakładamy, że błędy nie mają metadanych
                            is_error = True
                    else:
                        content_to_display = "_Nieznany format odpowiedzi asystenta (w response_dict)_"
                        logger.warning(f"Nie udało się wyodrębnić treści z obiektu typu (w response_dict): {type(response_content)}")

                # Fallback: jeśli nie ma response_dict, użyj klucza 'content' (wiadomość załadowana z JSON)
                elif "content" in message:
                    content_to_display = message.get("content", "_Brak treści_")
                    logger.debug("Wyświetlanie wiadomości asystenta (z klucza 'content' - załadowana z historii).")
                    # Metadane i źródła nie są dostępne w tym przypadku (chyba że zapis zostałby rozszerzony)
                else:
                    content_to_display = "_Brak treści lub nieznany format wiadomości asystenta_"
                    logger.warning(f"Nie znaleziono ani 'response_dict' ani 'content' w wiadomości asystenta: {message}")

            # Wyświetl główną treść wiadomości
            if content_to_display:
                st.markdown(content_to_display)
            else:
                st.markdown("_Pusta wiadomość_") # Na wszelki wypadek

            # Wyświetl metadane odpowiedzi LLM (tylko jeśli dostępne z bieżącej sesji i nie jest to błąd)
            if metadata_to_display and not is_error:
                model = metadata_to_display.get("model", "N/A")
                tokens = metadata_to_display.get("tokens", "N/A")
                response_time = metadata_to_display.get("time", 0.0)
                tokens_str = str(tokens) if tokens is not None else "N/A"
                st.caption(f"Model: {model} | Tokeny: {tokens_str} | Czas: {response_time:.2f}s")

            # Wyświetl źródła (jeśli istnieją w message)
            # Pamiętaj: obecna funkcja save_chat_history ich nie zapisuje.
            if sources_to_display:
                    with st.expander("Źródła"):
                        st.markdown("Wykorzystane dokumenty:")
                        for source in message["sources"]:
                            st.markdown(f"- `{source}`")

    # --- Obsługa nowej wiadomości ---
# --- Dodano sekcję wizualizacji grafu ---
            graph_data = message.get("graph_data")
            if graph_data: # Sprawdź, czy dane grafu istnieją dla tej wiadomości
                # Użyj unikalnego klucza dla toggle - np. indeks wiadomości + chat_id
                # Potrzebujemy indeksu wiadomości w ramach czatu
                try:
                     # Bezpieczniejsze uzyskanie indeksu
                     message_list = active_chat_data.get("messages", [])
                     # Znajdź indeks na podstawie obiektu wiadomości (referencja)
                     message_index = -1
                     for idx, msg_in_list in enumerate(message_list):
                          if msg_in_list is message: # Porównanie referencji
                               message_index = idx
                               break

                     if message_index != -1:
                          toggle_key = f"graph_toggle_{active_chat_id}_{message_index}"

                          if st.toggle("Wyświetl Graf Wiedzy", key=toggle_key):
                               try:
                                    # Przygotuj dane dla agraph z formatu node_link_data, dodając zawijanie etykiet
                                    nodes = [
                                        Node(
                                            id=node['id'],
                                            # Użyj textwrap do zawijania długich etykiet co ~25 znaków
                                            label=textwrap.fill(node.get('label', node['id']), width=25),
                                            size=15
                                        ) for node in graph_data.get('nodes', [])
                                    ]
                                    # Krawędzie pozostają bez zmian formatowania etykiet (zwykle są krótsze)
                                    edges = [Edge(source=link['source'], target=link['target'], label=link.get('label', '')) for link in graph_data.get('links', [])]

                                    # Konfiguracja wyglądu grafu - zmiana na układ hierarchiczny
                                    config = Config(width='100%',
                                                    height=600,
                                                    directed=True,
                                                    physics=False, # Wyłącz fizykę dla układu hierarchicznego
                                                    hierarchical=True, # Włącz układ hierarchiczny
                                                    layout={'hierarchical': {'direction': 'UD', 'sortMethod': 'hubsize'}}, # Podstawowe opcje hierarchii (Up-Down, sortuj wg liczby połączeń)
                                                    nodeHighlightBehavior=True,
                                                    highlightColor='#F7A7A6',
                                                    collapsible=True # Opcja zwijania węzłów (może wymagać dodatkowej konfiguracji)
                                                    )

                                    if nodes and edges:
                                         with st.spinner("Ładowanie grafu..."):
                                              agraph(nodes=nodes, edges=edges, config=config)
                                    elif nodes: # Wyświetl same węzły, jeśli nie ma krawędzi
                                         st.caption("_Graf zawiera tylko pojedyncze obiekty (brak relacji)._")
                                         agraph(nodes=nodes, edges=[], config=config)
                                    else:
                                         st.caption("_Brak danych do wyświetlenia grafu (brak węzłów)._")

                               except Exception as e:
                                    logger.error(f"Błąd podczas renderowania grafu: {e}")
                                    st.error("Wystąpił błąd podczas próby wyświetlenia grafu.")
                     else:
                          logger.error("Nie udało się znaleźć indeksu wiadomości dla klucza toggle grafu.")

                except Exception as e:
                    logger.error(f"Nieoczekiwany błąd przy tworzeniu klucza toggle grafu: {e}")
            # -----------------------------------------
    if prompt := st.chat_input("Zadaj pytanie..."):
        # Dodaj pytanie użytkownika do historii AKTYWNEGO czatu
        st.session_state.chats[st.session_state.active_chat_id]["messages"].append({"role": "user", "content": prompt})

        # Natychmiastowe wyświetlenie pytania użytkownika (bez zmian)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Myślę..."):
                try:
                    # Ustaw parametry z sesji
                    st.session_state.chatbot.top_k = st.session_state.top_k
                    st.session_state.chatbot.top_k_reranker = st.session_state.top_k_reranker
                    st.session_state.chatbot.relevance_threshold = st.session_state.relevance_threshold

                    # Pobierz odpowiedź, źródła i dane grafu (query zwraca teraz 3 wartości)
                    response_dict, sources, graph_data = st.session_state.chatbot.query(
                        question=prompt,
                        system_prompt_override=st.session_state.system_prompt # Przekazujemy prompt z GUI
                    )

                    # Zapisz odpowiedź (słownik), źródła i dane grafu do historii AKTYWNEGO czatu
                    st.session_state.chats[st.session_state.active_chat_id]["messages"].append({
                        "role": "assistant",
                        "response_dict": response_dict, # Zapisujemy cały słownik odpowiedzi
                        "sources": sources,
                        "graph_data": graph_data # Zapisujemy dane grafu (format node_link_data)
                    })
                    save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz stan po dodaniu wiadomości (sukces)
                    st.rerun()


                except Exception as e:
                    logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}")
                    error_message = f"Wystąpił błąd podczas przetwarzania pytania: {e}"
                    st.markdown(error_message)
                    # Zapisz wiadomość o błędzie w nowym formacie (jako słownik)
                    st.session_state.chats[st.session_state.active_chat_id]["messages"].append({
                        "role": "assistant",
                        "response_dict": {"content": error_message, "metadata": None}, # Zapisz błąd w strukturze słownika
                        "sources": [] # Brak źródeł przy błędzie
                    })
                    save_chat_history(st.session_state.chats, st.session_state.active_chat_id) # Zapisz stan po dodaniu wiadomości (nawet błędu)
                # Po przetworzeniu pytania, Streamlit automatycznie odświeży interfejs
                # Usunięto st.experimental_rerun()
                # st.rerun() # Dodano st.rerun() tutaj, aby odświeżyć po odpowiedzi


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    main()