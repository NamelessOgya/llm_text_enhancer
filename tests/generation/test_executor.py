import unittest
from unittest.mock import MagicMock, patch
import argparse
from generation.executor import evolve_population

class MockLLM:
    def get_token_usage(self): return {}
    def generate(self, *args, **kwargs): return "mock"

class TestExecutor(unittest.TestCase):
    @patch("generation.executor.load_metrics")
    @patch("generation.executor.os.path.exists")
    @patch("generation.executor.open")
    def test_evolve_population(self, mock_open, mock_exists, mock_load_metrics):
        mock_exists.return_value = True
        mock_load_metrics.return_value = [{"file": "text_0.txt", "score": 1.0}]
        
        args = argparse.Namespace(
            iteration=1,
            evolution_method="ga",
            population_size=1,
            ensemble_ratios=None
        )
        
        mock_strategy = MagicMock()
        mock_strategy.evolve.return_value = [("t", "l")]
        
        with patch("generation.executor.get_evolution_strategy", return_value=mock_strategy):
            res = evolve_population(args, MockLLM(), "dir", "task", {})
            self.assertEqual(len(res), 1)
