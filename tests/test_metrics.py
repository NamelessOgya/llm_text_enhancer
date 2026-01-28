import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# We need to patch before importing if we want to force usage of mocks when sklearn is missing
# But strategies.metrics tries import at top level.
# So we can patch `strategies.metrics.CountVectorizer` and `strategies.metrics.cosine_similarity`

from strategies import metrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Prepare patches
        self.mock_vectorizer = MagicMock()
        self.mock_cosine = MagicMock()
        
    def test_calculate_novelty_basic_mocked(self):
        """Mockを使用してロジックをテスト"""
        
        # Mocking behavior
        # fit_transform returns matrix
        # cosine_similarity returns dists
        
        mock_vect_instance = MagicMock()
        mock_vect_instance.fit_transform.return_value = "tfidf_matrix" 
        self.mock_vectorizer.return_value = mock_vect_instance
        
        # Mock cosine_similarity to return specific values
        # shape: (n_candidates, n_set)
        # candidates=2, set=2
        # return [[1.0, 0.0], [0.1, 0.2]]
        # max -> [1.0, 0.2]
        # novelty -> [0.0, 0.8]
        self.mock_cosine.return_value = np.array([[1.0, 0.0], [0.1, 0.2]])
        
        with patch('strategies.metrics.CountVectorizer', self.mock_vectorizer), \
             patch('strategies.metrics.cosine_similarity', self.mock_cosine):
            
            candidates = ["apple banana", "orange juice"]
            distinct_set = ["apple banana", "grape soda"]
            
            novelties = metrics.calculate_novelty(candidates, distinct_set)
            
            self.assertEqual(len(novelties), 2)
            self.assertAlmostEqual(novelties[0], 0.0, places=1)
            self.assertAlmostEqual(novelties[1], 0.8, places=1)

    def test_calculate_novelty_fallback(self):
        """sklearnがない(None)場合のフォールバック確認"""
        # Force None
        with patch('strategies.metrics.CountVectorizer', None):
            candidates = ["test"]
            distinct_set = ["set"]
            novelties = metrics.calculate_novelty(candidates, distinct_set)
            self.assertEqual(novelties, [1.0])

if __name__ == "__main__":
    unittest.main()
