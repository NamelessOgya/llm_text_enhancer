import unittest
import os
import json
import shutil
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluation.cache import EvaluationCache
from evaluation.llm_evaluator import LLMEvaluator
from evaluation.perspectrum_evaluator import PerspectrumLLMEvaluator
from llm.interface import LLMInterface

class TestEvaluationCache(unittest.TestCase):
    def setUp(self):
        self.test_cache_file = "tmp/test_cache.json"
        # Ensure clean state
        if os.path.exists(self.test_cache_file):
            os.remove(self.test_cache_file)
            
        self.cache = EvaluationCache(self.test_cache_file)
        
    def tearDown(self):
        if os.path.exists(self.test_cache_file):
            os.remove(self.test_cache_file)

    def test_set_and_get(self):
        self.cache.set("TestEval", "text", "target", 0.8, "reason")
        
        # Verify in memory
        score, reason = self.cache.get("TestEval", "text", "target")
        self.assertEqual(score, 0.8)
        self.assertEqual(reason, "reason")
        
        # Verify file persistence
        # Reload cache object
        new_cache = EvaluationCache(self.test_cache_file)
        score, reason = new_cache.get("TestEval", "text", "target")
        self.assertEqual(score, 0.8)

    def test_cache_miss(self):
        res = self.cache.get("TestEval", "not_exist", "target")
        self.assertIsNone(res)

class TestEvaluatorIntegration(unittest.TestCase):
    def setUp(self):
        self.test_cache_file = "tmp/evaluator_cache.json"
        
        # Mock LLM
        self.mock_llm = MagicMock(spec=LLMInterface)
    
    def tearDown(self):
        # Clean up default cache file if created
        if os.path.exists(self.test_cache_file):
             # We might not want to delete the real cache file if we are running against it,
             # but here we assume test environment? 
             # For safety, let's just leave it or use a mocked path in integration if possible.
             # Ideally we should inject cache path to Evaluators, but they init it internally.
             # We can patch the class or the instance.
             pass

    def test_llm_evaluator_cache_hit(self):
        # We need to control the cache file used by the evaluator.
        # Since we can't easily pass it in init without changing signature, 
        # we will monkeypatch the instance's cache.
        
        evaluator = LLMEvaluator(self.mock_llm)
        mock_cache = MagicMock()
        evaluator.cache = mock_cache
        
        # Case 1: Cache Hit
        mock_cache.get.return_value = (0.9, "Cached Reason")
        
        score, reason = evaluator.evaluate("text", "target")
        
        self.assertEqual(score, 0.9)
        self.assertEqual(reason, "Cached Reason")
        self.mock_llm.generate.assert_not_called()

    def test_llm_evaluator_cache_miss(self):
        evaluator = LLMEvaluator(self.mock_llm)
        mock_cache = MagicMock()
        evaluator.cache = mock_cache
        
        # Case 2: Cache Miss
        mock_cache.get.return_value = None
        self.mock_llm.generate.return_value = "Score: 0.7\nReason: New Reason"
        
        score, reason = evaluator.evaluate("text", "target")
        
        self.assertEqual(score, 0.7)
        self.assertEqual(reason, "New Reason")
        self.mock_llm.generate.assert_called_once()
        mock_cache.set.assert_called_with("LLMEvaluator", "text", "target", 0.7, "New Reason")

if __name__ == '__main__':
    unittest.main()
