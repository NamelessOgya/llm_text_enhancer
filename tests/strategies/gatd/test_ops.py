import unittest
from unittest.mock import MagicMock
from strategies.gatd.components.ops import op_elitism, op_crossover

class MockStrategy:
    def __init__(self):
        self.prompts = {"crossover_prompt": "prompt"}
    def _get_task_constraint(self, c): return ""
    def _generate_with_retry(self, llm, p, c): return "child"

class TestGATDOps(unittest.TestCase):
    def test_elitism(self):
        pop = [{"text": "best", "score": 1.0}, {"text": "bad", "score": 0.0}]
        res = op_elitism(pop, 1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "best")
        
    def test_crossover(self):
        pop = [{"text": "p1", "score": 1.0}, {"text": "p2", "score": 0.5}]
        strategy = MockStrategy()
        res = op_crossover(strategy, None, pop, 1, "task", {})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "child")
