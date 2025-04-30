"""
Interfejs użytkownika dla chatbota RAG z wykorzystaniem Streamlit
"""
import streamlit as st
from app import RAGChatbot
import logging
import os
import tempfile
# Dodano import dla obiektu odpowiedzi LLM z LangChain, aby móc sprawdzić jego typ
from langchain_core.messages import AIMessage, HumanMessage

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Domyślny prompt systemowy (pobrany z app.py)
DEFAULT_SYSTEM_PROMPT = """Rola:\n\nJesteś doświadczonym analitykiem tekstu. Otrzymujesz jeden lub kilka dokumentów jednocześnie (raporty, artykuły, prezentacje, sprawozdania, e-maile, pliki PDF, Word, itp.) oraz pytania od członków zespołu. Twoim zadaniem jest znalezienie konkretnych informacji w tych materiałach i udzielenie jasnych, precyzyjnych odpowiedzi.\n\n\n\n\nTwoje zadania:\n\n\n\n\nPrzeczytaj uważnie wszystkie dostarczone źródła.\n\nDla każdego pytania:\n\nZidentyfikuj i porównaj informacje we wszystkich dostępnych dokumentach.\n\nPodaj konkretną odpowiedź opartą wyłącznie na treści źródłowej.\n\nJeśli ta sama informacja występuje w kilku miejscach – wybierz najbardziej wiarygodną i aktualną wersję.\n\nJeśli są sprzeczne dane – zaznacz to i podaj możliwe wyjaśnienie.\n\nZawsze wskaż dokładne źródło w tekście (cytat, numer akapitu, nazwa pliku lub lokalizacja).\n\nJeżeli odpowiedź nie występuje bezpośrednio w dokumentach, zaznacz to i dodaj krótką interpretację (jeśli to możliwe).\n\nFormat odpowiedzi dla każdego pytania:\n\n\n\n\nPytanie: [tu wpisz pytanie]\n\nOdpowiedź: [jasna, konkretna odpowiedź]\n\nŹródło w dokumentach: [cytat, numer akapitu, nazwa pliku, strona lub opis fragmentu]\n\nKomentarz (jeśli potrzebny): [jeśli są sprzeczności lub brak informacji – wyjaśnij to]\n\n\n\n\nRodzaje pytań, które możesz otrzymać (i jak na nie reagować):\n\n\n\n\nPytanie o osobę (np. „Kto jest szefem tej organizacji?”):\n\n→ Wskaż imię, nazwisko, stanowisko i dokument, w którym to się znajduje.\n\nPytanie o liczby (np. „Ile pieniędzy zostało zabezpieczonych?”):\n\n→ Podaj konkretną kwotę, powołując się na dane z odpowiedniego pliku. Jeśli kwoty różnią się – opisz to.\n\nPytanie o przyczyny, działania, efekty (np. „Dlaczego projekt się opóźnił?”):\n\n→ Podaj powody, działania lub skutki, nawet jeśli są rozproszone w różnych źródłach.\n\nPytania o fakty i szczegóły:\n\n→ Wydobądź najważniejsze detale z różnych dokumentów i połącz je w spójną odpowiedź.\n\nDostarczone materiały:\n\n[lista lub zestaw dokumentów i źródeł – np. Raport_finansowy_Q4.pdf, Spotkanie_zarządu_dnotatki.docx, E-mail_od_dostawcy.msg]\n\n\n\n\n📌 Przykład odpowiedzi z wieloma źródłami:\n\nPytanie: Ile pieniędzy zostało zabezpieczonych?\n\nOdpowiedź: Zabezpieczono 4,5 mln zł (Raport_finansowy_Q4.pdf), jednak w notatce ze spotkania (Spotkanie_zarządu_dnotatki.docx) pojawia się kwota 4,2 mln zł – możliwa aktualizacja danych w późniejszym okresie.\n\nŹródło w dokumentach:\n\n\n\nRaport_finansowy_Q4.pdf, strona 5: „Zabezpieczone środki wynoszą 4,5 mln zł.”\n\nSpotkanie_zarządu_dnotatki.docx, akapit 3: „Dysponujemy obecnie 4,2 mln zł zarezerwowanych środków.”\n\nKomentarz: Możliwa różnica wynika z częściowego wydatkowania środków po publikacji raportu."""


def main():
    st.set_page_config(page_title="🤖 Chatbot RAG z Qdrant", layout="wide")

    st.title("🤖 Chatbot RAG z Qdrant")
    st.markdown("Aplikacja wykorzystuje Retrieval-Augmented Generation (RAG) z bazą wektorową Qdrant")

    # Inicjalizacja chatbota
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
        st.header("📁 Zarządzanie dokumentami")

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


    # Historia czatu
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Wyświetl historię czatu
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Sprawdź typ wiadomości przed wyświetleniem
            if isinstance(message["content"], (AIMessage, HumanMessage)):
                 st.markdown(message["content"].content) # Wyświetl tylko treść wiadomości
            else:
                 st.markdown(message["content"]) # Wyświetl jako zwykły tekst (dla starszych wiadomości lub błędów)

            if "sources" in message and message["sources"]:
                # Sprawdź, czy źródła nie są None lub puste przed wyświetleniem expandera
                if message["sources"]:
                    with st.expander("Źródła"):
                        st.markdown("Wykorzystane dokumenty:")
                        for source in message["sources"]:
                            st.markdown(f"- `{source}`")

    # Obsługa nowej wiadomości
    if prompt := st.chat_input("Zadaj pytanie..."):
        # Dodaj pytanie użytkownika do historii
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Natychmiastowe wyświetlenie pytania użytkownika
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Myślę..."):
                try:
                    # Ustaw parametry z sesji
                    st.session_state.chatbot.top_k = st.session_state.top_k
                    st.session_state.chatbot.top_k_reranker = st.session_state.top_k_reranker
                    st.session_state.chatbot.relevance_threshold = st.session_state.relevance_threshold

                    # Pobierz odpowiedź, przekazując aktualny prompt systemowy z sesji
                    response, sources = st.session_state.chatbot.query(
                        question=prompt,
                        system_prompt_override=st.session_state.system_prompt # Przekazujemy prompt z GUI
                    )

                    # Zapisz odpowiedź do historii (zapisujemy cały obiekt odpowiedzi)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response, # Zapisujemy cały obiekt odpowiedzi z LLM
                        "sources": sources
                    })

                    # Po dodaniu odpowiedzi do historii, wymuś ponowne uruchomienie skryptu
                    st.rerun()


                except Exception as e:
                    logger.error(f"❌ Błąd podczas przetwarzania pytania: {str(e)}")
                    error_message = f"Wystąpił błąd podczas przetwarzania pytania: {e}"
                    st.markdown(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message # Tutaj zapisujemy string błędu
                    })
                # Po przetworzeniu pytania, Streamlit automatycznie odświeży interfejs
                # Usunięto st.experimental_rerun()
                # st.rerun() # Dodano st.rerun() tutaj, aby odświeżyć po odpowiedzi


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    main()