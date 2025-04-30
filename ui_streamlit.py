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

# --- Mapowanie typów NER na kolory ---
# Usunięto zdublowany fragment kodu (importy itp.)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = "chat_history.json" # Ścieżka do pliku historii

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

AVAILABLE_PROMPTS = {
    "Analityk Tekstu": PROMPT_ANALITYK_TEKSTU,
    "Analityk Grafów": PROMPT_ANALITYK_GRAFOW
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
                # Używamy teraz klucza 'content' zapisanego w poprzedniej iteracji
                content_str = msg.get("content")

                if content_str is not None:
                    msg_to_save["content"] = content_str
                # ----------------------------

                # --- Logika zapisu źródeł i grafu ---
                if "sources" in msg: msg_to_save["sources"] = msg["sources"]
                if "graph_data" in msg: msg_to_save["graph_data"] = msg["graph_data"]
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

def main():
    
    st.title("🤖 Chatbot RAG z Qdrant")

    # --- Inicjalizacja stanu sesji z pliku lub domyślnie ---
    if "chats" not in st.session_state:
         loaded_data = load_chat_history()
         if loaded_data:
              st.session_state.chats = loaded_data["chats"]
              if loaded_data["active_chat_id"] in st.session_state.chats:
                   st.session_state.active_chat_id = loaded_data["active_chat_id"]
              elif st.session_state.chats:
                   st.session_state.active_chat_id = next(iter(st.session_state.chats))
                   logger.warning("Załadowany active_chat_id był nieprawidłowy. Ustawiono pierwszy dostępny.")
              else:
                   first_chat_id = str(uuid.uuid4())
                   st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
                   st.session_state.active_chat_id = first_chat_id
                   logger.info("Załadowano pustą historię, utworzono nowy domyślny czat.")
                   save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
         else:
              first_chat_id = str(uuid.uuid4())
              st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
              st.session_state.active_chat_id = first_chat_id
              logger.info("Nie załadowano historii. Utworzono nowy domyślny czat.")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
    elif "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
         if st.session_state.chats:
              st.session_state.active_chat_id = next(iter(st.session_state.chats))
              logger.warning(f"Active_chat_id brakujący lub nieprawidłowy po inicjalizacji. Ustawiono na: {st.session_state.active_chat_id}")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
         else:
              first_chat_id = str(uuid.uuid4())
              st.session_state.chats = {first_chat_id: {"name": "Czat 1", "messages": []}}
              st.session_state.active_chat_id = first_chat_id
              logger.error("Stan awaryjny: Brak czatów i active_chat_id po inicjalizacji. Utworzono nowy domyślny czat.")
              save_chat_history(st.session_state.chats, st.session_state.active_chat_id)

    # Inicjalizacja wybranego promptu w stanie sesji
    if 'selected_prompt_name' not in st.session_state:
        st.session_state.selected_prompt_name = "Analityk Tekstu" # Domyślny wybór
    # ---------------------------------------------------

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


    # Panel boczny z opcjami
    with st.sidebar:
        st.header("💬 Czaty")

        if st.button("+ Nowy Czat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            chat_count = len(st.session_state.chats) + 1
            st.session_state.chats[new_chat_id] = {"name": f"Czat {chat_count}", "messages": []}
            st.session_state.active_chat_id = new_chat_id
            if "editing_chat_id" in st.session_state: del st.session_state.editing_chat_id
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
                                 # Ustaw na pierwszy czat z posortowanej listy
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
        st.header("🎭 Wybierz Rolę (Prompt Systemowy)")
        st.session_state.selected_prompt_name = st.radio(
            "Wybierz prompt:",
            options=list(AVAILABLE_PROMPTS.keys()),
            key="prompt_selector",
            index=list(AVAILABLE_PROMPTS.keys()).index(st.session_state.selected_prompt_name) # Zapewnia zapamiętanie wyboru
        )
        with st.expander("Podgląd wybranego promptu"):
             st.markdown(f"```\n{AVAILABLE_PROMPTS[st.session_state.selected_prompt_name]}\n```")
        st.markdown("---")

        st.header("⚙️ Ustawienia RAG")
        # Użycie kluczy sesji dla suwaków
        st.session_state.top_k = st.slider("Liczba dok. z Qdrant", 1, 200, st.session_state.get('top_k', 99))
        st.session_state.top_k_reranker = st.slider("Liczba dok. po rerankingu", 1, 100, st.session_state.get('top_k_reranker', 33))
        st.session_state.relevance_threshold = st.slider("Próg istotności rerankera", 0.0, 1.0, st.session_state.get('relevance_threshold', 0.0), 0.01)
        st.markdown("---")

        st.header("📁 Zarządzanie dokumentami")

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
                      # ... (reszta logiki przetwarzania plików bez zmian) ...
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
                      if processed_count > 0: st.success(f"✅ Przetworzono pomyślnie {processed_count} z {num_files} plików.")
                      if error_count > 0: st.error(f"❌ Wystąpiły błędy dla {error_count} z {num_files} plików.")

                 # --- Przetwarzanie URL ---
                 if url_to_process:
                      st.info(f"Rozpoczynam przetwarzanie dokumentu z URL: {url_to_process}")
                      with st.spinner("Przetwarzanie URL..."):
                          try:
                              # Zakładamy, że RAGChatbot ma metodę process_document_from_url
                              # lub że process_document potrafi wykryć URL
                              # Wymaga to implementacji w app.py
                              # Poniżej przykładowa logika, jeśli process_document nie obsługuje URL
                              # documents = st.session_state.chatbot.document_loader.load_from_url(url_to_process)
                              # if documents:
                              #     all_splits = []
                              #     for doc in documents:
                              #         splits = st.session_state.chatbot.text_splitter.split_document(doc)
                              #         all_splits.extend(splits)
                              #     if all_splits:
                              #         st.session_state.chatbot.qdrant_connector.add_documents(all_splits)
                              #         st.success(f"✅ Dokument z URL przetworzony pomyślnie.")
                              #     else: st.error("❌ Nie udało się podzielić dokumentu z URL.")
                              # else: st.error("❌ Nie udało się załadować dokumentu z URL.")
                              # --- Tymczasowe ostrzeżenie ---
                              st.warning("Przetwarzanie z URL nie jest jeszcze w pełni zaimplementowane w tym przycisku.")
                          except Exception as e:
                              logger.error(f"❌ Błąd przetwarzania URL: {str(e)}", exc_info=True)
                              st.error(f"Błąd podczas przetwarzania URL: {e}")


        st.markdown("---")
        if st.button("🔄 Resetuj czat"):
            if "active_chat_id" in st.session_state and st.session_state.active_chat_id in st.session_state.chats:
                 st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
                 save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                 logger.info(f"Wyczyszczono historię aktywnego czatu: {st.session_state.active_chat_id}")
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
                   sources_to_display = message.get("sources", [])
                   graph_data_to_display = message.get("graph_data")

                   st.markdown(content_to_display)

                   if sources_to_display:
                           with st.expander("Źródła"):
                               unique_sources = set()
                               for source in sources_to_display:
                                    if isinstance(source, str): unique_sources.add(os.path.basename(source))
                                    else: unique_sources.add(str(source))
                               for base_name in sorted(list(unique_sources)):
                                    st.markdown(f"- `{base_name}`")

                   # --- Sekcja wizualizacji grafu ---
                   if graph_data_to_display:
                        toggle_key = f"graph_toggle_{active_chat_id}_{message_index}"
                        # Domyślnie zwinięty (value=False)
                        if st.toggle("Wyświetl Graf Wiedzy", key=toggle_key, value=False):
                             try:
                                 nodes = []
                                 edges = []
                                 if isinstance(graph_data_to_display, dict) and 'nodes' in graph_data_to_display and 'links' in graph_data_to_display:
                                     # Blok obliczania rozmiaru węzłów
                                     try:
                                         temp_G = nx.node_link_graph(graph_data_to_display, directed=True, multigraph=False)
                                         degrees = dict(temp_G.degree())
                                         max_degree = 0
                                         if degrees and any(d > 0 for d in degrees.values()): max_degree = max(degrees.values())
                                         BASE_NODE_SIZE = 15
                                         LARGE_NODE_SIZE = 30
                                         logger.debug(f"Rozmiar grafu dla indeksu {message_index}. Max degree: {max_degree}")
                                     except Exception as graph_reconstruction_e:
                                         logger.error(f"Błąd rekonstrukcji grafu dla indeksu {message_index}: {graph_reconstruction_e}")
                                         degrees, max_degree, BASE_NODE_SIZE, LARGE_NODE_SIZE = {}, 0, 15, 15

                                     # Tworzenie węzłów
                                     for node_data in graph_data_to_display.get('nodes', []):
                                          node_id = node_data.get('id', str(uuid.uuid4()))
                                          label = node_data.get('label', node_id)
                                          wrapped_label = textwrap.fill(label, width=20, replace_whitespace=False, drop_whitespace=False)
                                          ner_type = node_data.get('ner_type')
                                          # === POPRAWNE POBIERANIE KOLORU ===
                                          node_color = NER_COLORS.get(ner_type, DEFAULT_NODE_COLOR)
                                          # ==================================
                                          node_size = BASE_NODE_SIZE
                                          current_degree = degrees.get(node_id, 0)
                                          if max_degree > 0 and current_degree == max_degree: node_size = LARGE_NODE_SIZE
                                          tooltip = f"{label}" + (f" ({ner_type})" if ner_type else "")
                                          nodes.append(Node(id=node_id, label=wrapped_label, size=node_size, color=node_color, title=tooltip))

                                     # Tworzenie krawędzi
                                     for link_data in graph_data_to_display.get('links', []):
                                          source_id, target_id = link_data.get('source'), link_data.get('target')
                                          if source_id and target_id:
                                               edges.append(Edge(source=source_id, target=target_id, label=link_data.get('label', ''), color="#cccccc"))

                                     # Konfiguracja grafu (hierarchiczna, statyczna)
                                     config = Config(width='100%', height=600, directed=True, physics=False, hierarchical=True, layout={'hierarchical': {'direction': 'UD', 'sortMethod': 'hubsize'}}, nodeHighlightBehavior=True, highlightColor='#F7A7A6', collapsible=False)

                                     # Wyświetlanie grafu
                                     if nodes:
                                          with st.spinner("Ładowanie grafu..."):
                                               agraph(nodes=nodes, edges=edges, config=config)
                                     else: st.caption("_Brak węzłów do wyświetlenia w grafie._")
                                 else: st.caption("_Nieprawidłowy format danych grafu._")
                             except Exception as e:
                                 logger.error(f"Błąd podczas renderowania grafu dla indeksu {message_index}: {e}", exc_info=True)
                                 st.error("Wystąpił błąd podczas wyświetlania grafu.")
                   # ------------------------------------

    # --- Obsługa nowej wiadomości ---
    user_prompt = st.chat_input("Zadaj pytanie...")

    if user_prompt:
        st.session_state.chats[st.session_state.active_chat_id]["messages"].append({"role": "user", "content": user_prompt})
        save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
        st.rerun()

    # --- Logika generowania odpowiedzi ---
    current_chat_messages = st.session_state.chats[st.session_state.active_chat_id].get("messages", [])
    # Sprawdź, czy ostatnia wiadomość jest od użytkownika i czy *nie jest* już przetworzona (nie ma klucza 'sources' lub 'graph_data' - uproszczenie)
    if current_chat_messages and current_chat_messages[-1]["role"] == "user" and "sources" not in current_chat_messages[-1]:
        last_user_prompt = current_chat_messages[-1]["content"]
        response_placeholder = message_container.empty() # Użyj kontenera zdefiniowanego wyżej
        with response_placeholder.chat_message("assistant"):
            with st.spinner("Myślę..."):
                try:
                    # Ustaw parametry RAG
                    st.session_state.chatbot.top_k = st.session_state.top_k
                    st.session_state.chatbot.top_k_reranker = st.session_state.top_k_reranker
                    st.session_state.chatbot.relevance_threshold = st.session_state.relevance_threshold

                    # Pobierz wybrany prompt
                    selected_prompt_text = AVAILABLE_PROMPTS[st.session_state.selected_prompt_name]

                    # Wywołanie query
                    response_dict, sources, graph_data = st.session_state.chatbot.query(
                        question=last_user_prompt,
                        system_prompt_override=selected_prompt_text
                    )

                    # Wyciągnij treść z odpowiedzi
                    response_content_str = ""
                    if isinstance(response_dict.get("content"), AIMessage):
                         response_content_str = response_dict.get("content").content
                    elif isinstance(response_dict.get("content"), str):
                         response_content_str = response_dict.get("content")
                    else:
                         response_content_str = str(response_dict.get("content","")) # Fallback

                    # Zaktualizuj OSTATNIĄ wiadomość (która była promptem użytkownika)
                    # To jest podejście alternatywne - modyfikujemy istniejącą listę
                    # Lepszym może być DODANIE nowej wiadomości asystenta. Wróćmy do dodawania.

                    # Dodaj odpowiedź asystenta jako NOWĄ wiadomość
                    st.session_state.chats[st.session_state.active_chat_id]["messages"].append({
                        "role": "assistant",
                        "content": response_content_str, # Zapisujemy tylko string treści
                        "sources": sources,
                        "graph_data": graph_data
                    })
                    save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                    st.rerun()

                except Exception as e:
                    logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}", exc_info=True)
                    error_message = f"Wystąpił błąd: {e}"
                    # Dodaj wiadomość o błędzie jako NOWĄ wiadomość
                    st.session_state.chats[st.session_state.active_chat_id]["messages"].append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "graph_data": None
                    })
                    save_chat_history(st.session_state.chats, st.session_state.active_chat_id)
                    st.rerun() # Odśwież, aby pokazać błąd

if __name__ == "__main__":
    # Usunięto nest_asyncio - zwykle nie jest potrzebne w Streamlit
    main()