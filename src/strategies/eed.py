import logging
import os
import random
import yaml
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from .base import EvolutionStrategy
from .eed_phases import run_exploit_phase, run_diversity_phase, run_explore_phase

logger = logging.getLogger(__name__)

class EEDStrategy(EvolutionStrategy):
    """
    8. EED (Expert-driven Evolution and Diagnosis)
    
    Exploit, Diversity, Explore の3つのフェーズを組み合わせて進化させる。
    Config: config/logic/eed.yaml
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "eed.yaml")
        default_config = {
            "exploit_sample_num": 6,
            "diversity_sample_num": 2,
            "explore_sample_num": 2,
            "grad_temperature": 1.0,
            "sample_pass_rate": 3,
            "elite_sample_num": 1,
            "persona_reuse_rate": 0.5,
            "gradient_max_words": 25 # Default
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load eed.yaml: {e}")
        return default_config

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using EED Strategy (Refactored)...")
        
        self._ensure_prompts("eed", context)
        
        # Load Config
        exploit_num = self.config["exploit_sample_num"]
        diversity_num = self.config["diversity_sample_num"]
        explore_num = self.config["explore_sample_num"]

        # Adjust k if config sums don't match (Prioritize config ratios, but fit to k)
        total_config = exploit_num + diversity_num + explore_num

        if k != total_config:
             logger.warning(f"Population size {k} != Config Sum {total_config}. Scaling config.")
             ratio_ex = exploit_num / total_config
             ratio_div = diversity_num / total_config
             
             exploit_num = int(k * ratio_ex)
             diversity_num = int(k * ratio_div)
             explore_num = k - exploit_num - diversity_num

        sorted_pop = sorted(population, key=lambda x: (x['score'], random.random()), reverse=True)
        
        new_texts = []
        
        # --- 1. Exploit Phase ---
        logger.debug(f"Phase 1: Exploit ({exploit_num} items)")
        exploit_texts, parent_gradients, parents_exploit = run_exploit_phase(
            self, llm, sorted_pop, exploit_num, task_def, context
        )
        new_texts.extend(exploit_texts)
        
        # --- 2. Diversity Phase ---
        logger.debug(f"Phase 2: Diversity ({diversity_num} items)")
        diversity_texts = run_diversity_phase(
            self, llm, sorted_pop, parents_exploit, parent_gradients, diversity_num, task_def, context
        )
        new_texts.extend(diversity_texts)

        # --- 3. Explore Phase ---
        logger.debug(f"Phase 3: Explore ({explore_num} items)")
        explore_texts = run_explore_phase(
            self, llm, sorted_pop, explore_num, task_def, context
        )
        new_texts.extend(explore_texts)
        
        # Fill/Truncate to k
        return new_texts[:k]
