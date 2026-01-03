import unittest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm.dummy_adapter import DummyAdapter
from generate import generate_initial_prompts, evolve_prompts
from evaluation.llm_evaluator import LLMEvaluator
from evaluation.rule_evaluator import KeywordRuleEvaluator, RegexRuleEvaluator, get_rule_evaluator

class TestCoreLogic(unittest.TestCase):
    def setUp(self):
        self.llm = DummyAdapter()
        self.task_def = "Test Task"

    def test_generate_initial_prompts(self):
        k = 5
        prompts = generate_initial_prompts(self.llm, k, self.task_def)
        self.assertEqual(len(prompts), k)
        for p in prompts:
            self.assertIsInstance(p, str)
            self.assertTrue(len(p) > 0)

    def test_evolve_prompts(self):
        k = 5
        # Create some dummy previous prompts and metrics
        prev_prompts = [f"Prompt {i}" for i in range(k)]
        metrics = [
            {"file": f"text_{i}.txt", "score": float(i)} for i in range(k)
        ]
        # logic sorts by score desc, so Prompt 4 (score 4.0) should be best
        
        new_prompts = evolve_prompts(self.llm, prev_prompts, metrics, k, self.task_def)
        
        self.assertEqual(len(new_prompts), k)
        # Elitism check: Top 20% of 5 is 1. So best prompt "Prompt 4" should be preserved.
        self.assertIn("Prompt 4", new_prompts[0]) 
        
        # Mutation check: Others should be mutated
        # DummyAdapter returns "Mutated ..." for mutation requests
        mutation_count = sum(1 for p in new_prompts if "Mutated" in p)
        self.assertTrue(mutation_count > 0)

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
        score, reason = evaluator.evaluate("fire pokemon", "fire type") # 1/2 matches = 5.0? No, intersection(fire, pokemon) & (fire, type) -> fire. 1/2 = 5.0.
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
