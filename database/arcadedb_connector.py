"""
Moduł do integracji z bazą danych ArcadeDB
"""
import requests
import json
import time

class ArcadeDBConnector:
    def __init__(self, host="host.docker.internal", port=6333, database="rag_db", username="root", password="rootroot"):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.base_url = f"http://{host}:{port}/{database}"
        self.auth = (self.username, self.password)
        
    def _send_query(self, query):
        """Wysyła zapytanie do ArcadeDB z ponownymi próbami"""
        url = f"{self.base_url}/sql"
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=query, auth=self.auth, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503 and attempt < max_retries - 1:
                    # Baza może się nie uruchomić - czekamy i próbujemy ponownie
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Query failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                raise Exception("Nie można połączyć się z ArcadeDB. Upewnij się, że baza danych jest uruchomiona.")
                
        return None  # Zwracamy None jeśli wszystkie próby zakończyły się niepowodzeniem
        
    def check_connection(self):
        """Sprawdza czy można się połączyć z bazą danych"""
        try:
            url = f"{self.base_url}/status"
            response = requests.get(url, auth=self.auth, timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def initialize_schema(self):
        """Inicjalizuje schemat bazy danych"""
        if not self.check_connection():
            raise Exception("Nie można połączyć się z ArcadeDB. Sprawdź czy baza danych jest uruchomiona.")
            
        queries = [
            "CREATE CLASS Document IF NOT EXISTS",
            "CREATE PROPERTY Document.id STRING IF NOT EXISTS",
            "CREATE PROPERTY Document.content STRING IF NOT EXISTS",
            "CREATE PROPERTY Document.embedding EMBEDDEDLIST<DOUBLE> IF NOT EXISTS",
            "CREATE INDEX Document.id ON Document(id) UNIQUE"
        ]
        
        for query in queries:
            result = self._send_query(query)
            if not result:
                raise Exception(f"Nie udało się wykonać zapytania: {query}")
    
    def store_embedding(self, doc_id, content, embedding):
        """Zapisuje embedding dokumentu do bazy"""
        query = f"""
        UPSERT INTO Document 
        SET id = '{doc_id}', 
            content = '{content.replace("'", "''")}', 
            embedding = [{','.join(str(x) for x in embedding)}]
        """
        result = self._send_query(query)
        return result is not None
        
    def find_similar(self, query_embedding, limit=5):
        """Znajduje najbardziej podobne dokumenty do zapytania"""
        query = f"""
        SELECT id, content, VECTOR_DISTANCE(embedding, [{','.join(str(x) for x in query_embedding)}]) AS distance
        FROM Document
        ORDER BY distance
        LIMIT {limit}
        """
        result = self._send_query(query)
        if result:
            return [(item['id'], item['content'], item['distance']) for item in result['result']]
        return []