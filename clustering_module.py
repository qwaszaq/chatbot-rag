# -*- coding: utf-8 -*-
"""
Moduł odpowiedzialny za logikę klastrowania danych.
"""
import logging
import numpy as np
from sklearn.cluster import KMeans, DBSCAN # Importujemy K-Means i DBSCAN
# Można dodać import AgglomerativeClustering, jeśli będzie potrzebne
# from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score # Do oceny jakości klastrowania (opcjonalne)
from sklearn.preprocessing import StandardScaler # Do skalowania danych (ważne dla DBSCAN)

logger = logging.getLogger(__name__)

def perform_clustering(embeddings: np.ndarray, algorithm: str = 'kmeans', n_clusters: int = 8, **kwargs):
    """
    Wykonuje klastrowanie na podanych wektorach embeddings.

    Args:
        embeddings (np.ndarray): Tablica NumPy zawierająca wektory embeddings (wiersze=punkty, kolumny=wymiary).
        algorithm (str): Nazwa algorytmu ('kmeans' lub 'dbscan'). Domyślnie 'kmeans'.
        n_clusters (int): Liczba klastrów do znalezienia (używane przez K-Means). Domyślnie 8.
        **kwargs: Dodatkowe argumenty przekazywane do konstruktora algorytmu klastrowania
                  (np. eps, min_samples dla DBSCAN).

    Returns:
        np.ndarray or None: Tablica NumPy z etykietami klastrów dla każdego punktu wejściowego.
                           Etykieta -1 oznacza szum/outlier (w DBSCAN).
                           Zwraca None w przypadku błędu lub braku danych.
    """
    if embeddings is None or embeddings.shape[0] == 0:
        logger.warning("⚠️ Brak danych embeddings do przeprowadzenia klastrowania.")
        return None

    logger.info(f"📈 Rozpoczynanie klastrowania dla {embeddings.shape[0]} punktów przy użyciu algorytmu: {algorithm}...")
    
    labels = None
    
    try:
        if algorithm == 'kmeans':
            # Sprawdzenie, czy n_clusters nie jest większe niż liczba próbek
            actual_n_clusters = min(n_clusters, embeddings.shape[0])
            if actual_n_clusters < n_clusters:
                 logger.warning(f"⚠️ Żądana liczba klastrów ({n_clusters}) jest większa niż liczba próbek ({embeddings.shape[0]}). Używam {actual_n_clusters} klastrów.")
            if actual_n_clusters < 2: # K-Means potrzebuje co najmniej 2 klastrów (lub 1 jeśli jest tylko 1 próbka)
                 logger.warning(f"⚠️ K-Means wymaga co najmniej 2 próbek do utworzenia więcej niż 1 klastra. Zwracam etykiety [0] dla {embeddings.shape[0]} próbek.")
                 return np.zeros(embeddings.shape[0], dtype=int) # Przypisz wszystkie punkty do jednego klastra

            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto', **kwargs)
            kmeans.fit(embeddings)
            labels = kmeans.labels_
            logger.info(f"✅ K-Means zakończone. Znaleziono {actual_n_clusters} klastrów.")

        elif algorithm == 'dbscan':
            # DBSCAN jest wrażliwy na skalę danych, dobrze jest je przeskalować
            logger.info("   Skalowanie danych przed DBSCAN...")
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Domyślne parametry DBSCAN, jeśli nie podano w kwargs
            eps = kwargs.get('eps', 0.5) 
            min_samples = kwargs.get('min_samples', 5)
            logger.info(f"   Używane parametry DBSCAN: eps={eps}, min_samples={min_samples}")

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, **{k:v for k,v in kwargs.items() if k not in ['eps', 'min_samples']})
            dbscan.fit(embeddings_scaled)
            labels = dbscan.labels_
            
            # Podsumowanie wyników DBSCAN
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_points = list(labels).count(-1)
            logger.info(f"✅ DBSCAN zakończone. Znaleziono {n_clusters_found} klastrów i {n_noise_points} punktów szumu/outlierów.")

        # Można dodać obsługę innych algorytmów (np. hierarchical) tutaj
        # elif algorithm == 'hierarchical':
        #     agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        #     agg_clustering.fit(embeddings)
        #     labels = agg_clustering.labels_
        #     logger.info(f"✅ Hierarchical Clustering zakończone.")

        else:
            logger.error(f"❌ Nieznany algorytm klastrowania: {algorithm}")
            return None

        # Opcjonalna ewaluacja jakości klastrowania (jeśli są co najmniej 2 klastry)
        if labels is not None and len(set(labels)) > 1:
            try:
                score = silhouette_score(embeddings, labels)
                logger.info(f"📊 Współczynnik sylwetki (Silhouette Score): {score:.3f}")
            except Exception as score_e:
                logger.warning(f"⚠️ Nie można obliczyć Silhouette Score: {score_e}")

        return labels

    except Exception as e:
        logger.error(f"❌ Błąd podczas wykonywania klastrowania ({algorithm}): {e}", exc_info=True)
        return None

# Można dodać inne funkcje pomocnicze, np. do znajdowania optymalnego k dla K-Means
# def find_optimal_k(embeddings, max_k=15):
#     # ... implementacja metody łokcia lub silhouette ...
#     pass

# Można dodać funkcję do generowania etykiet klastrów
# def generate_cluster_labels(cluster_data):
#     # ... implementacja TF-IDF lub podsumowania LLM ...
#     pass