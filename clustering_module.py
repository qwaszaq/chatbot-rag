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
import random # Do próbkowania tekstów
# Dodano import dla typowania LLM i spacy
from typing import List, Dict, Any, Tuple
from collections import Counter # Dodano import Counter
import spacy # Dodano import spacy


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

        # Opcjonalna ewaluacja jakości klastrowania (jeśli są co najmniej 2 klastry i nie tylko szum)
        unique_labels = set(labels)
        if labels is not None and len(unique_labels - {-1}) > 1: # Sprawdź czy jest więcej niż 1 klaster (poza szumem)
            try:
                # Oblicz Silhouette Score tylko dla punktów nienależących do szumu
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1: # Potrzebujemy co najmniej 2 punktów nienależących do szumu
                    score = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
                    logger.info(f"📊 Współczynnik sylwetki (Silhouette Score dla punktów nieszumowych): {score:.3f}")
                else:
                     logger.info("📊 Nie można obliczyć Silhouette Score - za mało punktów nienależących do szumu.")
            except Exception as score_e:
                logger.warning(f"⚠️ Nie można obliczyć Silhouette Score: {score_e}")

        return labels

    except Exception as e:
        logger.error(f"❌ Błąd podczas wykonywania klastrowania ({algorithm}): {e}", exc_info=True)
        return None


# === FUNKCJA DO GENEROWANIA ETYKIET ===
def generate_cluster_labels_llm(cluster_assignments: dict, qdrant_data: list, llm: Any, sample_size: int = 5, max_text_length: int = 300):
    """
    Generuje opisowe etykiety dla klastrów za pomocą LLM.

    Args:
        cluster_assignments (dict): Słownik mapujący ID punktu na etykietę klastra.
        qdrant_data (list[dict]): Lista słowników z danymi punktów (musi zawierać 'id' i 'payload' z 'page_content').
        llm: Instancja modelu językowego LangChain (np. ChatOpenAI).
        sample_size (int): Liczba próbek tekstu do użycia dla każdego klastra do generowania etykiety.
        max_text_length (int): Maksymalna długość pojedynczego fragmentu tekstu używanego w prompcie.

    Returns:
        dict: Słownik mapujący ID klastra na wygenerowaną etykietę (np. {0: "Finanse i Dotacje", 1: "Raporty Techniczne"}).
              Zwraca pusty słownik w przypadku błędu lub braku danych.
    """
    if not cluster_assignments or not qdrant_data or not llm:
        logger.warning("⚠️ Brak danych wejściowych do wygenerowania etykiet klastrów.")
        return {}

    logger.info(f"🏷️ Rozpoczynanie generowania etykiet dla klastrów za pomocą LLM...")

    # Stwórz słownik mapujący ID punktu na jego tekst (page_content)
    point_id_to_text = {item['id']: item.get('payload', {}).get('page_content', '')
                        for item in qdrant_data if item.get('payload')}

    # Grupuj ID punktów według klastra
    clusters = {}
    for point_id, cluster_label in cluster_assignments.items():
        if cluster_label != -1: # Ignoruj punkty szumu/outliers
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(point_id)

    generated_labels = {}

    for cluster_id, point_ids in clusters.items():
        logger.info(f"   Generowanie etykiety dla Klastra {cluster_id} (zawiera {len(point_ids)} punktów)...")

        # Zbierz próbkę tekstów dla klastra
        sample_texts = []
        # Losuj próbkę ID punktów z klastra
        sample_point_ids = random.sample(point_ids, min(len(point_ids), sample_size))

        for point_id in sample_point_ids:
            text = point_id_to_text.get(point_id)
            if text:
                # Przytnij tekst, aby prompt nie był za długi
                sample_texts.append(text[:max_text_length] + "..." if len(text) > max_text_length else text)

        if not sample_texts:
            logger.warning(f"   Brak tekstów dla Klastra {cluster_id}. Pomijanie generowania etykiety.")
            generated_labels[cluster_id] = f"Klaster {cluster_id} (brak danych tekstowych)"
            continue

        # Przygotuj prompt dla LLM
        combined_texts = "\n---\n".join(sample_texts)
        prompt = f"""
        Poniżej znajdują się fragmenty tekstów należące do tej samej grupy tematycznej (klastra).
        Twoim zadaniem jest przeanalizowanie tych fragmentów i zaproponowanie krótkiej, zwięzłej etykiety (maksymalnie 3-5 słów) opisującej główny temat tej grupy. Etykieta powinna być po polsku.

        Fragmenty tekstów:
        ---
        {combined_texts}
        ---

        Zaproponuj etykietę (tylko sama etykieta, bez dodatkowych wyjaśnień):
        """

        try:
            logger.debug(f"      Wysyłanie promptu do LLM dla Klastra {cluster_id}...")
            response = llm.invoke(prompt)

            label_text = ""
            if hasattr(response, 'content'):
                label_text = response.content
            elif isinstance(response, str):
                label_text = response
            else:
                label_text = str(response)

            # Proste czyszczenie odpowiedzi LLM (usuwanie np. cudzysłowów)
            cleaned_label = label_text.strip().strip('"').strip("'").strip()
            # Ustawienie pierwszej litery na wielką
            if cleaned_label:
                 generated_labels[cluster_id] = cleaned_label[0].upper() + cleaned_label[1:]
            else:
                 generated_labels[cluster_id] = f"Klaster {cluster_id} (nie udało się wygenerować etykiety)"
                 logger.warning(f"   LLM zwrócił pustą odpowiedź dla Klastra {cluster_id}.")

            logger.info(f"   Wygenerowana etykieta dla Klastra {cluster_id}: '{generated_labels[cluster_id]}'")

        except Exception as llm_e:
            logger.error(f"   ❌ Błąd podczas wywoływania LLM dla Klastra {cluster_id}: {llm_e}")
            generated_labels[cluster_id] = f"Klaster {cluster_id} (błąd generowania)"

    logger.info("✅ Zakończono generowanie etykiet dla klastrów.")
    return generated_labels
# === KONIEC FUNKCJI DO GENEROWANIA ETYKIET ===


# === FUNKCJA DO GENEROWANIA PODSUMOWANIA ===
def generate_cluster_summary(texts: List[str], llm: Any, max_context_length: int = 8000, sample_size: int = 20) -> str:
    """
    Generuje podsumowanie dla grupy tekstów (chunków) należących do jednego klastra za pomocą LLM.

    Args:
        texts (List[str]): Lista tekstów chunków należących do klastra.
        llm: Instancja modelu językowego LangChain (np. ChatOpenAI).
        max_context_length (int): Maksymalna przybliżona długość kontekstu (w znakach) do wysłania do LLM.
                                   Teksty będą losowo próbkowane i łączone, aż do osiągnięcia tego limitu.
        sample_size (int): Alternatywnie, można użyć stałej liczby próbek, jeśli `max_context_length` nie jest priorytetem.
                           Jeśli `max_context_length` jest ustawione, `sample_size` jest ignorowane.

    Returns:
        str: Wygenerowane podsumowanie lub komunikat o błędzie.
    """
    if not texts or not llm:
        logger.warning("⚠️ Brak tekstów lub modelu LLM do wygenerowania podsumowania klastra.")
        return "Brak danych do wygenerowania podsumowania."

    logger.info(f"📝 Rozpoczynanie generowania podsumowania dla {len(texts)} tekstów...")

    # Przygotowanie kontekstu - próbkowanie lub łączenie do limitu długości
    combined_texts = ""
    if max_context_length:
        current_length = 0
        # Poprawka: Zawsze próbuj z całej listy, aby uniknąć błędu przy małej liczbie tekstów
        sampled_texts_indices = random.sample(range(len(texts)), min(len(texts), len(texts))) # Losowa kolejność indeksów
        selected_texts = []
        for index in sampled_texts_indices:
            text = texts[index]
            if current_length + len(text) + 5 < max_context_length: # +5 dla separatora "\n---\n"
                selected_texts.append(text)
                current_length += len(text) + 5
            else:
                # Dodaj przynajmniej jeden tekst, nawet jeśli przekracza limit, jeśli nic nie wybrano
                if not selected_texts:
                    selected_texts.append(text[:max_context_length-5]) # Przytnij, aby zmieścić się w limicie
                    logger.warning("   Pierwszy tekst był dłuższy niż max_context_length, został przycięty.")
                break # Osiągnięto limit długości
        combined_texts = "\n---\n".join(selected_texts)
        logger.info(f"   Użyto {len(selected_texts)} z {len(texts)} tekstów (limit długości: {max_context_length} znaków).")

    elif sample_size:
         sampled_texts = random.sample(texts, min(len(texts), sample_size))
         combined_texts = "\n---\n".join(sampled_texts)
         logger.info(f"   Użyto {len(sampled_texts)} z {len(texts)} tekstów (limit próbek: {sample_size}).")
    else: # Fallback - użyj wszystkich (może być bardzo długie!)
        combined_texts = "\n---\n".join(texts)
        logger.warning(f"   Użyto wszystkich {len(texts)} tekstów (brak limitu długości/próbek).")


    if not combined_texts:
        logger.warning("   Nie udało się zbudować kontekstu do podsumowania.")
        return "Nie można było zbudować kontekstu do podsumowania (puste teksty?)."

    # Przygotuj prompt dla LLM
    prompt = f"""
    Poniżej znajdują się fragmenty tekstów należące do tej samej grupy tematycznej (klastra), prawdopodobnie związane ze sprawą ZGP "Agnieszka".
    Twoim zadaniem jest przeanalizowanie tych fragmentów i stworzenie zwięzłego podsumowania (kilka zdań), które oddaje główne wątki, kluczowe informacje, postacie lub wydarzenia poruszane w tej grupie tekstów. Skup się na najważniejszych aspektach. Podsumowanie powinno być po polsku.

    Fragmenty tekstów:
    ---
    {combined_texts}
    ---

    Zwięzłe podsumowanie głównych wątków i kluczowych informacji:
    """

    try:
        logger.debug("      Wysyłanie promptu podsumowania do LLM...")
        response = llm.invoke(prompt)

        summary_text = ""
        if hasattr(response, 'content'):
            summary_text = response.content
        elif isinstance(response, str):
            summary_text = response
        else:
            summary_text = str(response)

        cleaned_summary = summary_text.strip()
        logger.info("✅ Podsumowanie klastra wygenerowane.")
        return cleaned_summary if cleaned_summary else "LLM zwrócił puste podsumowanie."

    except Exception as llm_e:
        logger.error(f"   ❌ Błąd podczas wywoływania LLM dla podsumowania klastra: {llm_e}")
        return f"Błąd podczas generowania podsumowania: {llm_e}"
# === KONIEC FUNKCJI DO GENEROWANIA PODSUMOWANIA ===


# === NOWA FUNKCJA DO EKSTRAKCJI KLUCZOWYCH BYTÓW (NER) ===
def extract_key_entities(texts: List[str], nlp: Any, top_n: int = 10,
                         allowed_labels: set = {"persName", "orgName", "geogName", "placeName"}) -> List[Tuple[str, str, int]]:
    """
    Ekstrahuje i zlicza nazwane encje (NER) z listy tekstów za pomocą modelu spaCy.

    Args:
        texts (List[str]): Lista tekstów chunków należących do klastra.
        nlp: Załadowana instancja modelu językowego spaCy (np. pl_core_news_lg).
        top_n (int): Liczba najczęstszych encji do zwrócenia.
        allowed_labels (set): Zbiór etykiet NER, które mają być uwzględnione w wynikach.
                              Domyślnie: osoby, organizacje, nazwy geograficzne, miejsca.

    Returns:
        List[Tuple[str, str, int]]: Lista krotek (tekst_encji, etykieta_ner, liczba_wystąpień),
                                     posortowana malejąco według liczby wystąpień.
                                     Zwraca pustą listę w przypadku błędu, braku modelu nlp lub braku encji.
    """
    if not nlp:
        logger.error("❌ Model spaCy (nlp) nie jest dostępny do ekstrakcji encji.")
        return []
    if not texts:
        logger.warning("⚠️ Brak tekstów do ekstrakcji encji.")
        return []

    logger.info(f"🧐 Rozpoczynanie ekstrakcji kluczowych bytów (NER) z {len(texts)} tekstów (top {top_n}, dozwolone: {allowed_labels})...")
    entity_counts = Counter()

    try:
        # Przetwarzanie tekstów partiami dla lepszej wydajności spaCy
        for doc in nlp.pipe(texts, disable=["tagger", "parser"]): # Wyłączamy niepotrzebne komponenty
            for ent in doc.ents:
                # Sprawdź, czy etykieta encji jest dozwolona
                if ent.label_ in allowed_labels:
                    # Normalizuj tekst encji (np. usuń białe znaki na początku/końcu)
                    entity_text = ent.text.strip()
                    if entity_text: # Ignoruj puste encje po strip()
                        entity_counts[(entity_text, ent.label_)] += 1

        # Pobierz top_n najczęstszych encji
        most_common_entities = entity_counts.most_common(top_n)

        # Konwertuj na format wyjściowy (tekst, etykieta, liczba)
        result = [(text, label, count) for (text, label), count in most_common_entities]
        logger.info(f"✅ Znaleziono {len(result)} kluczowych bytów.")
        return result

    except Exception as e:
        logger.error(f"❌ Błąd podczas ekstrakcji encji NER: {e}", exc_info=True)
        return []
# === KONIEC NOWEJ FUNKCJI ===