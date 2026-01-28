import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from generation.filesystem import setup_row_directories
from generation.executor import load_or_generate_initial_population, evolve_population

class MockLLM:
    def get_token_usage(self): return {}
    def generate(self, *args, **kwargs): return "mock_text"

class TestGenerateLogic(unittest.TestCase):
    def test_setup_row_directories(self):
        """ディレクトリ構造のセットアップ確認"""
        with patch("os.makedirs") as mock_makedirs:
            dirs = setup_row_directories("/tmp/res", 0, 1)
            
            self.assertEqual(dirs["row_dir"], "/tmp/res/row_0")
            self.assertEqual(dirs["current_iter_dir"], "/tmp/res/row_0/iter1")
            self.assertTrue(mock_makedirs.called)

    def test_load_or_generate_initial_population_generate(self):
        """初期集団生成ロジック (生成パターン)"""
        args = argparse.Namespace(
            initial_population_dir=None,
            evolution_method="ga",
            population_size=2
        )
        llm = MockLLM()
        context = {}
        
        # Mock strategy factory
        mock_strategy = MagicMock()
        mock_strategy.generate_initial.return_value = [("t1", "l1"), ("t2", "l2")]
        
        with patch("generation.executor.get_evolution_strategy", return_value=mock_strategy):
            outputs = load_or_generate_initial_population(args, llm, 0, "row_0", "task", context)
            
            self.assertEqual(len(outputs), 2)
            mock_strategy.generate_initial.assert_called_once()

    @patch("generation.executor.load_metrics")
    @patch("generation.executor.os.path.exists")
    @patch("generation.executor.open", new_callable=mock_open, read_data="content")
    def test_evolve_population(self, mock_file, mock_exists, mock_load_metrics):
        """進化ロジックのフロー確認"""
        mock_exists.return_value = True
        mock_load_metrics.return_value = [
            {"file": "text_0.txt", "score": 1.0}
        ]
        
        args = argparse.Namespace(
            iteration=1,
            evolution_method="ga",
            population_size=2,
            ensemble_ratios=None
        )
        llm = MockLLM()
        context = {}
        
        mock_strategy = MagicMock()
        mock_strategy.evolve.return_value = [("new_t1", "new_l1")]
        
        with patch("generation.executor.get_evolution_strategy", return_value=mock_strategy):
            outputs = evolve_population(args, llm, "/tmp/row_0", "task", context)
            
            self.assertEqual(len(outputs), 1)
            mock_strategy.evolve.assert_called_once()

if __name__ == "__main__":
    unittest.main()
