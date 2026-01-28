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
    候補テキストの新規性 (Novelty) を計算する。
    
    Novelty(x, K) = 1 - max_{k in K}(similarity(x, k))
    similarityには文字n-gram (n=3) のコサイン類似度を使用する。
    
    Args:
        candidates (List[str]): 評価対象の新規候補テキストのリスト
        distinct_set (List[str]): 比較対象となる既存テキスト群（過去の個体やknowledge baseなど）
        
    Returns:
        List[float]: 各候補の新規性スコア (0.0 〜 1.0)。
                     1.0に近いほど既存テキストと類似していないことを示す。
                     scikit-learnが利用できない場合やエラー時はデフォルト値(1.0または0.0)を返す。
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
