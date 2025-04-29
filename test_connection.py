"""
Test połączenia z ArcadeDB
"""
import requests

def test_arcadedb_connection():
    url = "http://localhost:2424/rag_db/status"
    auth = ("root", "rootroot")
    
    try:
        response = requests.get(url, auth=auth, timeout=5)
        if response.status_code == 200:
            print("✅ Pomyślne połączenie z ArcadeDB!")
            print("Odpowiedź:", response.json())
        else:
            print(f"❌ Nieoczekiwany kod odpowiedzi: {response.status_code}")
            print("Treść odpowiedzi:", response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Nie można połączyć się z ArcadeDB. Sprawdź:")
        print("1. Czy kontener Docker jest uruchomiony")
        print("2. Czy port 6333 jest poprawnie mapowany")
        print("3. Czy baza danych 'rag_db' istnieje")

if __name__ == "__main__":
    test_arcadedb_connection()