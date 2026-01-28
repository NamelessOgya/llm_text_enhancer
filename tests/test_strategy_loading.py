import sys
import os
import unittest

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from strategies import get_evolution_strategy
from strategies.base import EvolutionStrategy
from strategies.ga import GeneticAlgorithmStrategy
from strategies.textgrad import TextGradStrategy
from strategies.textgrad_v2 import TextGradV2Strategy
from strategies.trajectory import TrajectoryStrategy
from strategies.demonstration import DemonstrationStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.he import HEStrategy
from strategies.eed import EEDStrategy
from strategies.gatd import GATDStrategy

class TestStrategyLoading(unittest.TestCase):
    def test_load_strategies(self):
        test_cases = [
            ("ga", GeneticAlgorithmStrategy),
            ("textgrad", TextGradStrategy),
            ("textgradv2", TextGradV2Strategy),
            ("trajectory", TrajectoryStrategy),
            ("demonstration", DemonstrationStrategy),
            ("ensemble", EnsembleStrategy),
            ("he", HEStrategy),
            ("eed", EEDStrategy),
            ("gatd", GATDStrategy),
        ]
        
        for name, expected_class in test_cases:
            with self.subTest(name=name):
                strategy = get_evolution_strategy(name)
                self.assertIsInstance(strategy, EvolutionStrategy)
                self.assertIsInstance(strategy, expected_class)
                print(f"Successfully loaded {name} as {expected_class.__name__}")

    def test_fallback(self):
        strategy = get_evolution_strategy("unknown_strategy")
        self.assertIsInstance(strategy, GeneticAlgorithmStrategy)
        print("Successfully validated fallback behavior")

if __name__ == "__main__":
    unittest.main()
