"""
Główny moduł chatbota RAG
"""
import os
import uuid
from data_processing.document_loader import DocumentLoader
from data_processing.text_splitter import TextSplitter
from data_processing.embedding_generator import EmbeddingGenerator
from database.arcadedb_connector import ArcadeDBConnector
from data_processing.reranker import Reranker

class RAGChatbot:
    def __init__(self):
        self.document_loader = DocumentLoader("")
        self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = EmbeddingGenerator()
        self.db_connector = ArcadeDBConnector()
        self.reranker = Reranker()
        
        # Inicjalizacja schematu bazy danych
        self.db_connector.initialize_schema()
        
    def process_document(self, file_path):
        """Przetwarza dokument i zapisuje embeddingi do bazy danych"""
        # Ustawiamy ścieżkę pliku w loaderze
        self.document_loader.file_path = file_path
        
        # Ładujemy dokument
        document = self.document_loader.load()
        
        # Dzielimy dokument na fragmenty
        chunks = self.text_splitter.split_document(document)
        
        # Generujemy embeddingi dla każdego fragmentu i zapisujemy do bazy
        for chunk in chunks:
            embedding = self.embedding_generator.generate_embedding(chunk)
            doc_id = str(uuid.uuid4())
            self.db_connector.store_embedding(doc_id, chunk, embedding)
            
        return len(chunks)
        
    def answer_query(self, query, best_k=99, top_k_rerank=33):
        """Odpowiada na zapytanie użytkownika"""
        # Generujemy embedding dla zapytania
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Znajdujemy najbardziej podobne dokumenty
        similar_docs = self.db_connector.find_similar(query_embedding, limit=best_k)
        
        # Rerankujemy wyniki
        ranked_docs = self.reranker.rerank(query, [doc[1] for doc in similar_docs], top_k=top_k_rerank)
        
        # Przygotowujemy kontekst dla modelu
        context = "\n\n".join([doc[0] for doc, score in ranked_docs])
        
        # Wywołujemy LLM do generowania odpowiedzi
        response = self._call_llm(query, context)
        
        return response, ranked_docs
        
    def _call_llm(self, query, context):
        """Wywołuje LLM do generowania odpowiedzi"""
        # Ta funkcja będzie integrowana z API LM Studio
        # Oto przykładowa implementacja:
        payload = {
            "prompt": f"Odpowiedz na podstawie poniższego kontekstu:\n\n{context}\n\nZapytanie: {query}",
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # W rzeczywistej implementacji należy zaktualizować URL do LM Studio
        response = requests.post("http://localhost:1234/v1/completions", json=payload)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            raise Exception(f"Failed to get response from LLM: {response.text}")