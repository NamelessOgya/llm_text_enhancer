import logging
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

# Forward reference issue: Ensemble needs get_evolution_strategy
# But get_evolution_strategy needs to import strategies...
# Circular import likely if we put factory in __init__.py and Ensemble imports factory.
# Solution: Import factory function inside evolve method to avoid top-level circular dependency.

class EnsembleStrategy(EvolutionStrategy):
    """
    5. アンサンブル型 (Ensemble Strategy)
    
    複数の進化戦略を組み合わせて次世代テキストを生成する。
    """
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Ensemble Strategy...")
        
        # Delayed import to avoid circular dependency
        from . import get_evolution_strategy
        
        ratios = context.get("ensemble_ratios", {"ga": 1.0})
        
        counts = {}
        remaining = k
        keys = list(ratios.keys())
        
        for idx, method in enumerate(keys):
            ratio = ratios[method]
            if idx == len(keys) - 1:
                count = remaining
            else:
                count = int(k * ratio)
            counts[method] = count
            remaining -= count
            
        new_texts = []
        
        for method, count in counts.items():
            if count <= 0:
                continue
                
            logger.info(f"Ensemble sub-strategy: {method}, count: {count}")
            strategy = get_evolution_strategy(method)
            sub_results = strategy.evolve(llm, population, count, task_def, context)
            
            for text, meta in sub_results:
                new_meta = f"[Ensemble: {method}] {meta}"
                new_texts.append((text, new_meta))
                
        return new_texts[:k]
