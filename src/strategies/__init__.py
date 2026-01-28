from .base import EvolutionStrategy
from .ga import GeneticAlgorithmStrategy
from .textgrad import TextGradStrategy
from .textgrad_v2 import TextGradV2Strategy
from .trajectory import TrajectoryStrategy
from .demonstration import DemonstrationStrategy
from .ensemble import EnsembleStrategy
from .he import HEStrategy
from .eed import EEDStrategy
from .gatd import GATDStrategy

def get_evolution_strategy(method_name: str) -> EvolutionStrategy:
    """
    指定されたメソッド名の進化戦略インスタンスを返すファクトリ関数。
    """
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
    
    strategy_class = strategies.get(method_name.lower())
    
    if strategy_class:
        return strategy_class()
    
    # Default fallback
    return GeneticAlgorithmStrategy()
