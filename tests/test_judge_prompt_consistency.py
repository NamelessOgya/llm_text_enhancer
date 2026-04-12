import os
import unittest
import logging
from typing import List, Tuple

# PYTHONPATH needs to be configured properly or run with PYTHONPATH=.:./src
from src.evaluator.perspectrum.judge import PerspectrumLLMEvaluator
from src.llm.factory import get_llm_adapter

# Ensure API Key is somewhat available, otherwise these tests might fail in pure CI
# The user intends to run this to verify consistency.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Disable extensive logging to prevent clutter
logging.getLogger("src.evaluator").setLevel(logging.CRITICAL)

class TestPerspectrumJudgeConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not GEMINI_API_KEY:
            raise unittest.SkipTest("GEMINI_API_KEY is not set. Skipping real API evaluation tests.")
            
        # Initialize the target model as requested
        cls.llm = get_llm_adapter("gemini", "gemini-2.5-flash")
        
        # Initialize evaluator (this will pick up the updated judge.yaml)
        # We ensure it finds the correct YAML configuring it explicitly if needed
        # but the default usually works (config/definitions/prompts/judge.yaml or the perspectrum specific one)
        # However, the task uses a specific perspective YAML
        prompt_path = os.path.join(os.getcwd(), "config", "definitions", "prompts", "perspectrum_llm", "judge.yaml")
        config_path = os.path.join(os.getcwd(), "config", "definitions", "tasks", "task_perspectrum.yaml")
        
        # Override llm_eval_repeat to 1 for tests to speed up execution
        cls.evaluator = PerspectrumLLMEvaluator(cls.llm, config_path=config_path, prompt_path=prompt_path)
        cls.evaluator.llm_eval_repeat = 1
        
    def run_evaluation_test(self, test_name: str, reference: str, generated: str, expected_score: int):
        score, reason = self.evaluator.evaluate(generated, reference)
        # Score returned is normalized 0.1 to 1.0. Multiply by 10 for 1-10 scale.
        actual_score = int(round(score * 10))
        
        # We allow a small error margin of ±2 points due to LLM subjectivity
        margin = 2
        
        print(f"\n[{test_name}]")
        print(f"Ref: {reference}")
        print(f"Gen: {generated}")
        print(f"Expected: {expected_score} | Got: {actual_score}")
        print(f"Reason: {reason}")
        
        self.assertTrue(
            abs(actual_score - expected_score) <= margin,
            f"Model assigned {actual_score} but expected {expected_score} (±{margin}). Reason: {reason}"
        )

    def test_samples(self):
        import yaml
        
        # Load external test cases
        fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures", "perspectrum_judge_test_cases.yaml")
        if not os.path.exists(fixtures_path):
            self.fail(f"Test cases file not found: {fixtures_path}")
            
        with open(fixtures_path, 'r', encoding='utf-8') as f:
            tests = yaml.safe_load(f)
            
        for test in tests:
            name = test["name"]
            ref = test["reference"]
            gen = test["generated"]
            expected = test["expected_score"]
            with self.subTest(name=name):
                self.run_evaluation_test(name, ref, gen, expected)

if __name__ == "__main__":
    unittest.main()
