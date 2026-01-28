import unittest
from unittest.mock import MagicMock
from evaluator.perspectrum.judge import PerspectrumLLMEvaluator

class MockLLM:
    def generate(self, *args, **kwargs):
        return '{"score": 8, "reason": "good"}'

class TestPerspectrumJudge(unittest.TestCase):
    def test_evaluate(self):
        judge = PerspectrumLLMEvaluator(MockLLM())
        # Mock logic rule checker
        judge._rule_checker.evaluate = lambda t, r: (1.0, "")
        
        score, reason = judge.evaluate("SUPPORT: R", "SUPPORT: R")
        self.assertEqual(score, 0.8)
        self.assertEqual(reason, "good")
        
    def test_rule_fail(self):
        judge = PerspectrumLLMEvaluator(MockLLM())
        judge._rule_checker.evaluate = lambda t, r: (0.0, "Fail")
        
        score, reason = judge.evaluate("Bad", "Ref")
        self.assertEqual(score, 0.0)
        self.assertEqual(reason, "Fail")
