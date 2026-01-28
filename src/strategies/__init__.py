from .base import EvolutionStrategy
from .ga.strategy import GeneticAlgorithmStrategy
from .textgrad.strategy import TextGradStrategy
from .textgrad.strategy_v2 import TextGradV2Strategy
from .trajectory.strategy import TrajectoryStrategy
from .demonstration.strategy import DemonstrationStrategy
from .ensemble.strategy import EnsembleStrategy
from .he.strategy import HEStrategy
from .eed.strategy import EEDStrategy
from .gatd.strategy import GATDStrategy

def get_evolution_strategy(strategy_name: str) -> EvolutionStrategy:
    """
    Factory function to retrieve a specific evolution strategy instance.
    
    Args:
        strategy_name (str): Name of the strategy (e.g., 'ga', 'textgrad', 'ensemble').
        
    Returns:
        EvolutionStrategy: Instance of the requested strategy.
        
    Raises:
        ValueError: If the strategy_name is unknown.
    """
    strategy_name = strategy_name.lower()
    
    strategies = {
        "ga": GeneticAlgorithmStrategy,
        "textgrad": TextGradStrategy,
        "textgradv2": TextGradV2Strategy,
        "trajectory": TrajectoryStrategy,
        "demonstration": DemonstrationStrategy,
        "ensemble": EnsembleStrategy,
        "he": HEStrategy,
        "eed": EEDStrategy,
        "gatd": GATDStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown evolution strategy: {strategy_name}")
        
    return strategies[strategy_name]()
