import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import json
from typing import List

from llm.interface import LLMInterface
from evolution_strategies import (
    get_evolution_strategy,
    GeneticAlgorithmStrategy,
    TextGradStrategy,
    TrajectoryStrategy,
    DemonstrationStrategy,
    EnsembleStrategy
)

class MockLLM(LLMInterface):
    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls = [] # Store call arguments for verification

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = f"Generated Response {self.call_count}"
        self.call_count += 1
        return response

    def get_token_usage(self):
        return {}

class TestEvolutionStrategies(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_results"
        os.makedirs(self.test_dir, exist_ok=True)
        self.task_def = "Test Task"
        self.k = 2
        
        # Base population
        # Sorted order by score: p1(80), p2(60), p3(40)
        # Note: 'prompt' key is irrelevant but kept for compatibility simulation
        self.population = [
            {"prompt": "p1", "text": "text_content_1", "score": 80, "file": "text_0.txt"},
            {"prompt": "p2", "text": "text_content_2", "score": 60, "file": "text_1.txt"},
            {"prompt": "p3", "text": "text_content_3", "score": 40, "file": "text_2.txt"},
        ]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ga_strategy(self):
        """
        Test Genetic Algorithm Strategy (Text Optimization):
        1. Identification of Elite (highest score text).
        2. Construction of mutation prompt with correct parent text.
        """
        strategy = GeneticAlgorithmStrategy()
        # Mock responses for mutation
        llm = MockLLM(["Mutated Text 1"]) 
        
        # k=2. Elite ratio 0.2 -> 0.4 -> max(1) = 1 elite.
        # Elite should be "text_content_1" (score 80).
        
        new_texts = strategy.evolve(llm, self.population, self.k, self.task_def, {})
        
        # Check Output Count
        self.assertEqual(len(new_texts), self.k)
        
        # Check Elitism Logic
        self.assertEqual(new_texts[0][0], "text_content_1", "First output should be the elite text")
        self.assertIn("Elitism", new_texts[0][1])
        
        # Check Mutation Logic
        self.assertEqual(new_texts[1][0], "Mutated Text 1")
        
        # Check what was sent to LLM
        self.assertEqual(len(llm.calls), 1)
        mutation_call_prompt = llm.calls[0]
        
        # Verify call content
        self.assertIn(self.task_def, mutation_call_prompt, "Task def should be in mutation prompt")
        self.assertIn("text_content_1", mutation_call_prompt, "Elite parent text should be used for mutation")

    def test_textgrad_strategy(self):
        """
        Test TextGrad Strategy (Text Optimization):
        1. Elitism.
        2. Generation of Gradient (Backward) on TEXT.
        3. Application of Gradient (Update) on TEXT.
        """
        strategy = TextGradStrategy()
        # Mock responses: 
        # Loop 1: Gradient, Update
        llm = MockLLM(["Gradient Feedback", "Updated Text Content"])
        
        new_texts = strategy.evolve(llm, self.population, self.k, self.task_def, {})
        
        # Check Output Count
        self.assertEqual(len(new_texts), self.k)
        
        # Check Elitism
        self.assertEqual(new_texts[0][0], "text_content_1")
        
        # Check TextGrad Update
        self.assertEqual(new_texts[1][0], "Updated Text Content")
        
        # Check LLM Calls
        self.assertEqual(len(llm.calls), 2)
        
        gradient_prompt = llm.calls[0]
        update_prompt = llm.calls[1]
        
        # Verify Gradient Prompt Logic
        # It should critique the text being updated (text_content_1 loop logic)
        self.assertIn("text_content_1", gradient_prompt)
        self.assertIn("80", gradient_prompt)
        
        # Verify Update Prompt Logic
        self.assertIn("text_content_1", update_prompt, "Original text should be in update prompt")
        self.assertIn("Gradient Feedback", update_prompt, "Gradient should be in update prompt")

    def test_trajectory_strategy(self):
        """
        Test Trajectory Strategy (Text Optimization):
        1. History loading.
        2. Sorting of text history (Low -> High Score).
        """
        strategy = TrajectoryStrategy()
        
        # Setup mock history in file system
        iter0_dir = os.path.join(self.test_dir, "iter0")
        os.makedirs(os.path.join(iter0_dir, "texts"), exist_ok=True) # Text dir
        
        # Create history file
        with open(os.path.join(iter0_dir, "texts", "text_0.txt"), "w") as f:
            f.write("history_text_low_score_50")
        with open(os.path.join(iter0_dir, "metrics.json"), "w") as f:
            json.dump([{"file": "text_0.txt", "score": 50}], f)
            
        context = {"result_dir": self.test_dir, "iteration": 1}
        
        # Responses for k=2. Logic calls generate once per item.
        llm = MockLLM(["Predicted Next Text 1", "Predicted Next Text 2"])
        
        # Combined history: p3(40), history(50), p2(60), p1(80)
        
        new_texts = strategy.evolve(llm, self.population, self.k, self.task_def, context)
        
        # Check LLM Calls (k=2, generates one by one -> 2 calls)
        self.assertEqual(len(llm.calls), 2)
        meta_prompt = llm.calls[0]
        
        # Verify History Sorting and content
        self.assertIn("history_text_low_score_50", meta_prompt)
        self.assertIn("text_content_1", meta_prompt) # p1
        
        # Find indices to verify order
        idx_p3 = meta_prompt.find("text_content_3") # 40
        idx_hist = meta_prompt.find("history_text_low_score_50") # 50
        idx_p2 = meta_prompt.find("text_content_2") # 60
        idx_p1 = meta_prompt.find("text_content_1") # 80
        
        # Expect Low -> High
        self.assertTrue(idx_p3 < idx_hist < idx_p2 < idx_p1, "History should be sorted Low to High")

    def test_demonstration_strategy(self):
        """
        Test Demonstration Strategy (Text Optimization):
        1. Selection of High Score texts.
        """
        strategy = DemonstrationStrategy()
        
        context = {"result_dir": self.test_dir, "iteration": 1}
        
        llm = MockLLM(["New Few Shot Text", "New Few Shot Text 2"])
        
        new_texts = strategy.evolve(llm, self.population, self.k, self.task_def, context)
        
        # Check LLM calls (k=2 -> 2 calls)
        self.assertEqual(len(llm.calls), 2)
        meta_prompt = llm.calls[0]
        
        # Verify content
        # Should contain high score texts
        self.assertIn("text_content_1", meta_prompt)
        self.assertIn("text_content_2", meta_prompt)
        self.assertIn("Example 1", meta_prompt)

    def test_ensemble_strategy(self):
        """
        Test Ensemble Strategy:
        1. Ratios parsing (passed via context).
        2. Merging of TEXT outputs.
        """
        from evolution_strategies import EnsembleStrategy
        
        strategy = EnsembleStrategy()
        llm = MockLLM(["Ensemble Text Output"])
        
        context = {
            "ensemble_ratios": {"ga": 0.6, "textgrad": 0.4},
            "result_dir": "test_results"
        }
        
        k = 5
        new_texts = strategy.evolve(llm, self.population, k, self.task_def, context)
        
        self.assertEqual(len(new_texts), k)
        
        ga_count = 0
        textgrad_count = 0
        for text, meta in new_texts:
            if "[Ensemble: ga]" in meta:
                ga_count += 1
            if "[Ensemble: textgrad]" in meta:
                textgrad_count += 1
                
        self.assertEqual(ga_count, 3)
        self.assertEqual(textgrad_count, 2)

    def test_factory(self):
        self.assertIsInstance(get_evolution_strategy("ga"), GeneticAlgorithmStrategy)
        self.assertIsInstance(get_evolution_strategy("textgrad"), TextGradStrategy)
        self.assertIsInstance(get_evolution_strategy("ensemble"), EnsembleStrategy)
        self.assertRaises(ValueError, get_evolution_strategy, "unknown")

if __name__ == "__main__":
    unittest.main()
