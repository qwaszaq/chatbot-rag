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

        # Ładowanie z pliku lokalnego
        uploaded_file = st.file_uploader("Wybierz dokument do przetworzenia", type=["pdf", "docx", "txt"])

        if uploaded_file:
            # Zapisz przesłany plik tymczasowo
            with st.spinner("Zapisywanie dokumentu..."):
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = file_path
                logger.info(f"📥 Zapisano dokument: {file_path}")

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
        if st.button("🧠 Przetwórz wybrany dokument"):
            if hasattr(st.session_state, "uploaded_file_path") and st.session_state.uploaded_file_path:
                with st.spinner("Przetwarzanie dokumentu..."):
                    success = st.session_state.chatbot.process_document(st.session_state.uploaded_file_path)
                    if success:
                        st.success("✅ Dokument został pomyślnie przetworzony i zapisany do Qdrant")
                        # Opcjonalnie posprzątaj tymczasowy plik po przetworzeniu
                        # os.unlink(st.session_state.uploaded_file_path)
                        # del st.session_state.uploaded_file_path
                    else:
                        st.error("❌ Wystąpił błąd podczas przetwarzania dokumentu")
            else:
                st.warning("⚠️ Najpierw wybierz dokument do przetworzenia (z pliku lub URL)")


        st.markdown("---")
        st.header("⚙️ Ustawienia RAG")
        st.session_state.top_k = st.slider("Liczba dokumentów do pobrania z Qdrant", 1, 200, 99)
        st.session_state.top_k_reranker = st.slider("Liczba dokumentów po rerankingu", 1, 100, 33)
        st.session_state.relevance_threshold = st.slider("Próg istotności rerankera", 0.0, 1.0, 0.0, 0.01)
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

                    # Pobierz odpowiedź
                    response, sources = st.session_state.chatbot.query(prompt)

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