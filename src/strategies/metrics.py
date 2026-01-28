from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    CountVectorizer = None
    cosine_similarity = None

def calculate_novelty(candidates: List[str], distinct_set: List[str]) -> List[float]:
    """
    Novelty(x, K) = 1 - max_{k in K}(similarity(x, k))
    similarity: 3-gram cosine similarity
    
    Args:
        candidates: List of new candidate texts
        distinct_set: List of existing texts to compare against (population + history)
        
    Returns:
        List of novelty scores (0.0 to 1.0)
    """
    if not distinct_set or not candidates or CountVectorizer is None:
        return [1.0] * len(candidates)

    all_texts = candidates + distinct_set
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        candidate_vectors = tfidf_matrix[:len(candidates)]
        set_vectors = tfidf_matrix[len(candidates):]
        
        similarities = cosine_similarity(candidate_vectors, set_vectors)
        # similarities shape: (len(candidates), len(distinct_set))
        
        max_similarities = similarities.max(axis=1) # Max sim for each candidate
        # Ensure 1D array
        if hasattr(max_similarities, "toarray"):
             max_similarities = max_similarities.toarray().flatten()
        elif hasattr(max_similarities, "flatten"): # numpy array
             max_similarities = max_similarities.flatten()
             
        novelty_scores = 1.0 - max_similarities
        return novelty_scores.tolist()
        
    except ValueError:
        # Empty vocabulary or other issue
        return [0.0] * len(candidates)
