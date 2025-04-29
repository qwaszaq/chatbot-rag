"""
Moduł do rerankingu wyników wyszukiwania
"""
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query, documents, top_k=33):
        """Rerankuje dokumenty na podstawie zapytania"""
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Łączymy dokumenty z ich wynikami i sortujemy
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Zwracamy top_k najlepszych dokumentów z ich wynikami
        return scored_docs[:top_k]