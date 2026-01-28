import sys
import os
import unittest

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from strategies import get_evolution_strategy
from strategies.base import EvolutionStrategy
from strategies.ga.strategy import GeneticAlgorithmStrategy
from strategies.textgrad.strategy import TextGradStrategy
from strategies.textgrad.strategy_v2 import TextGradV2Strategy
from strategies.trajectory.strategy import TrajectoryStrategy
from strategies.demonstration.strategy import DemonstrationStrategy
from strategies.ensemble.strategy import EnsembleStrategy
from strategies.he.strategy import HEStrategy
from strategies.eed.strategy import EEDStrategy
from strategies.gatd.strategy import GATDStrategy

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
        with self.assertRaises(ValueError):
            get_evolution_strategy("unknown_strategy")
        print("Successfully validated ValueError behavior")

if __name__ == "__main__":
    unittest.main()
