import unittest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm.dummy_adapter import DummyAdapter
from generate import generate_initial_texts
# evolve_prompts was removed, so we import the strategy factory instead
from evolution_strategies import get_evolution_strategy
from evaluation.llm_evaluator import LLMEvaluator
from evaluation.rule_evaluator import KeywordRuleEvaluator, RegexRuleEvaluator, get_rule_evaluator

class TestCoreLogic(unittest.TestCase):
    def setUp(self):
        self.llm = DummyAdapter()
        self.task_def = "Test Task"

    def test_generate_initial_texts(self):
        k = 5
        texts = generate_initial_texts(self.llm, k, self.task_def)
        self.assertEqual(len(texts), k)
        for t_data in texts:
            self.assertIsInstance(t_data, tuple)
            self.assertEqual(len(t_data), 2)
            t, meta = t_data
            self.assertIsInstance(t, str)
            self.assertTrue(len(t) > 0)
            self.assertIsInstance(meta, str)
            self.assertTrue(len(meta) > 0)

    def test_evolve_ga(self):
        # Testing GA strategy via strategy class
        k = 5
        strategy = get_evolution_strategy("ga")
        
        # Create some dummy previous population
        population = [
            {"prompt": "p", "text": f"Text Content {i}", "score": float(i), "file": f"text_{i}.txt"} 
            for i in range(k)
        ]
        # logic sorts by score desc, so Text Content 4 (score 4.0) should be best (Elite)
        
        context = {"result_dir": "/tmp", "iteration": 1}
        new_texts = strategy.evolve(self.llm, population, k, self.task_def, context)
        
        self.assertEqual(len(new_texts), k)
        
        # Elitism check: Top 20% of 5 is 1. So best text "Text Content 4" should be preserved.
        # new_texts[0] is (text, meta)
        self.assertIn("Text Content 4", new_texts[0][0])
        self.assertIn("Elitism", new_texts[0][1])
        
        # Mutation check: Others should be mutated
        # DummyAdapter returns "Mutated ..." or similar based on logic.
        # Ensure that non-elite texts are new strings
        self.assertTrue(any(t[0] != "Text Content 4" for t in new_texts))

    def test_llm_evaluator_with_dummy(self):
        evaluator = LLMEvaluator(self.llm)
        score, reason = evaluator.evaluate("Pikachu is electric", "Electric Type")
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 10.0)
        self.assertIsInstance(reason, str)

    def test_keyword_rule_evaluator(self):
        evaluator = KeywordRuleEvaluator()
        # Perfect match
        score, reason = evaluator.evaluate("fire type pokemon", "fire type")
        self.assertEqual(score, 10.0)
        self.assertEqual(reason, "")
        # Partial match
        score, reason = evaluator.evaluate("fire pokemon", "fire type")
        self.assertEqual(score, 5.0)
        # No match
        score, reason = evaluator.evaluate("water", "fire")
        self.assertEqual(score, 0.0)

    def test_regex_rule_evaluator(self):
        evaluator = RegexRuleEvaluator()
        # Match
        score, reason = evaluator.evaluate("This is a Charizard", "Charizard")
        self.assertEqual(score, 10.0)
        self.assertEqual(reason, "")
        # Case insensitive match
        score, reason = evaluator.evaluate("charizard", "Charizard")
        self.assertEqual(score, 10.0)
        # No match
        score, reason = evaluator.evaluate("Pikachu", "Charizard")
        self.assertEqual(score, 0.0)

if __name__ == '__main__':
    unittest.main()
