import unittest
from unittest.mock import MagicMock
from strategies.eed.phases.exploit import run_exploit_phase
from strategies.eed.phases.diversity import run_diversity_phase
from strategies.eed.phases.explore import run_explore_phase

class MockStrategy:
    def __init__(self):
        self.prompts = {
            "exploit_gradient_prompt": "prompt {parent_text}",
            "update_prompt": "update {gradient}",
            "contrastive_gradient_prompt": "contrast",
            "contrastive_update_prompt": "contrast_up",
            "norm_prompt": "norm",
            "persona_gen_prompt": "persona",
            "explore_gen_prompt": "explore {target_persona}"
        }
    def _get_task_constraint(self, context): return ""
    def _generate_with_retry(self, llm, prompt, context): return "response"

class TestEEDPhases(unittest.TestCase):
    def test_exploit(self):
        strategy = MockStrategy()
        candidates = [{"text": "t1", "score": 1.0}]
        new_texts, grads, parents = run_exploit_phase(strategy, None, candidates, 1, "task", {})
        self.assertEqual(len(new_texts), 1)
        self.assertEqual(len(grads), 1)
        
    def test_diversity(self):
        strategy = MockStrategy()
        pop = [{"text": "t1", "score": 1.0}]
        new_texts = run_diversity_phase(strategy, None, pop, 1, "task", {}, ["grad"])
        self.assertEqual(len(new_texts), 1)
        
    def test_explore(self):
        strategy = MockStrategy()
        new_texts = run_explore_phase(strategy, None, 1, "task", {}, ["t1"])
        self.assertEqual(len(new_texts), 1)
