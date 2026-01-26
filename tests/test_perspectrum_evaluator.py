import unittest
from unittest.mock import MagicMock
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluation.perspectrum_evaluator import PerspectrumLLMEvaluator
from llm.interface import LLMInterface

class TestPerspectrumLLMEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=LLMInterface)
        # Using a dummy config path or None should be fine if we mock _check_rules, 
        # but to be safe we can let it try to load default or fail gracefully.
        # The constructor calls PerspectrumRuleEvaluator which tries to load config.
        # We'll just let it run.
        self.evaluator = PerspectrumLLMEvaluator(self.mock_llm)
        
        # Mock the rule checker to always pass to focus on LLM logic
        # We need to bypass the actual _check_rules logic which checks stanc/wordcount
        self.evaluator._check_rules = MagicMock(return_value=(1.0, ""))
        
        # Mock cache to prevent file IO and ensure isolation
        self.evaluator.cache = MagicMock()
        self.evaluator.cache.get.return_value = None # Force cache miss by default

    def test_evaluate_max_of_three(self):
        """Test that the evaluator runs 3 times (default) and picks the max score"""
        # Ensure default is 3
        self.evaluator.llm_eval_repeat = 3
        
        # Setup mock to return 3 different scores in JSON format
        # Scores in JSON are 1-10. Evaluator normalizes to 0.1-1.0.
        # Returns: 5 (0.5), 8 (0.8), 6 (0.6)
        response_1 = json.dumps({"score": 5, "reason": "Reason 1"})
        response_2 = json.dumps({"score": 8, "reason": "Reason 2"})
        response_3 = json.dumps({"score": 6, "reason": "Reason 3"})
        
        self.mock_llm.generate.side_effect = [response_1, response_2, response_3]

        score, reason = self.evaluator.evaluate("candidate text", "reference text")
        
        # Expecting the max score (8 -> 0.8 normalized)
        self.assertAlmostEqual(score, 0.8)
        self.assertEqual(reason, "Reason 2")
        self.assertEqual(self.mock_llm.generate.call_count, 3)

    def test_evaluate_resilience(self):
        """Test that the evaluator ignores failures if at least one succeeds"""
        self.evaluator.llm_eval_repeat = 3
        
        # Setup mock to fail once (return garbage) and succeed others
        response_1 = "Garbage response"
        response_2 = json.dumps({"score": 7, "reason": "Reason 2"})
        response_3 = json.dumps({"score": 4, "reason": "Reason 3"})
        
        self.mock_llm.generate.side_effect = [response_1, response_2, response_3]

        score, reason = self.evaluator.evaluate("candidate text", "reference text")
        
        # Expecting max of valid scores (7 -> 0.7)
        self.assertAlmostEqual(score, 0.7)
        self.assertEqual(reason, "Reason 2")
        # Should still call 3 times
        self.assertEqual(self.mock_llm.generate.call_count, 3)

    def test_evaluate_all_fail(self):
        """Test that 0.0 is returned if all attempts fail"""
        self.evaluator.llm_eval_repeat = 3
        
        # All fail
        self.mock_llm.generate.return_value = "Garbage"
        
        score, reason = self.evaluator.evaluate("candidate text", "reference text")
        
        self.assertEqual(score, 0.0)
        self.assertEqual(self.mock_llm.generate.call_count, 3)

    def test_evaluate_custom_repetitions(self):
        """Test that the evaluator respects llm_eval_repeat configuration"""
        # Set repeat to 5
        self.evaluator.llm_eval_repeat = 5
        
        # Responses for 5 calls
        responses = [json.dumps({"score": i, "reason": f"R{i}"}) for i in range(1, 6)]
        self.mock_llm.generate.side_effect = responses
        
        score, reason = self.evaluator.evaluate("candidate text", "reference text")
        
        # Max should be last one (score 5 -> 0.5)
        self.assertAlmostEqual(score, 0.5)
        self.assertEqual(reason, "R5")
        self.assertEqual(self.mock_llm.generate.call_count, 5)

if __name__ == '__main__':
    unittest.main()
