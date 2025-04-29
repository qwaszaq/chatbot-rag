"""
Tworzy bazę danych w ArcadeDB jeśli nie istnieje
"""
import requests

def create_arcadedb_database():
    # URL do endpointu zarządzania bazami danych
    url = "http://localhost:6333/databases"
    auth = ("root", "rootroot")  # Zaktualizowane dane logowania
    
    # Dane do utworzenia nowej bazy
    payload = {
        "name": "rag_db",
        "type": "document"
    }
    
    try:
        # Sprawdź najpierw czy baza już istnieje
        response = requests.get(f"{url}/rag_db", auth=auth, timeout=5)
        if response.status_code == 200:
            print("✅ Baza danych 'rag_db' już istnieje")
            
            # Usuń istniejącą bazę jeśli istnieje
            print("🗑️ Usuwam istniejącą bazę danych 'rag_db'...")
            delete_response = requests.delete(f"{url}/rag_db", auth=auth, timeout=10)
            if delete_response.status_code == 204:
                print("✅ Baza danych 'rag_db' została usunięta")
            else:
                print(f"❌ Nie udało się usunąć bazy danych: {delete_response.status_code}")
                print("Treść odpowiedzi:", delete_response.text)
                return False
        
        # Czekaj chwilę przed próbą utworzenia nowej bazy
        import time
        time.sleep(2)
            
        # Utwórz nową bazę danych
        print("🆕 Tworzę nową bazę danych 'rag_db'...")
        response = requests.post(url, auth=auth, json=payload, timeout=10)
        
        if response.status_code == 201:
            print("✅ Pomyślnie utworzono bazę danych 'rag_db'")
            return True
        else:
            print(f"❌ Nieoczekiwany kod odpowiedzi: {response.status_code}")
            print("Treść odpowiedzi:", response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Nie można połączyć się z ArcadeDB. Sprawdź:")
        print("1. Czy kontener Docker jest uruchomiony")
        print("2. Czy port 6333 jest poprawnie mapowany")

if __name__ == "__main__":
    create_arcadedb_database()