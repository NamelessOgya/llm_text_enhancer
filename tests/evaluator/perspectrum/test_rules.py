import unittest
from evaluator.perspectrum.rules import PerspectrumRuleEvaluator

class TestPerspectrumRules(unittest.TestCase):
    def test_evaluate_valid(self):
        evaluator = PerspectrumRuleEvaluator()
        # Mock meteor behavior or assume it returns (0.5, "meteor")
        # Since MeteorRuleEvaluator depends on java/nltk, we might fail if not mocked.
        # But let's assume valid stucture pass
        
        # We need to mock meteor_evaluator inside
        evaluator.meteor_evaluator.evaluate = lambda c, r: (0.5, "mock")
        
        score, reason = evaluator.evaluate("SUPPORT: Reason", "SUPPORT: Reason")
        self.assertEqual(score, 0.5)

    def test_stance_mismatch(self):
        evaluator = PerspectrumRuleEvaluator()
        score, reason = evaluator.evaluate("OPPOSE: Reason", "SUPPORT: Reason")
        self.assertEqual(score, 0.0)
        self.assertIn("Stance mismatch", reason)
        
    def test_word_count(self):
        evaluator = PerspectrumRuleEvaluator()
        evaluator.max_words = 2
        score, reason = evaluator.evaluate("SUPPORT: too many words", "SUPPORT: ok")
        self.assertEqual(score, 0.0)
        self.assertIn("Word count", reason)
