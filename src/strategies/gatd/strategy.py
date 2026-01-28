import logging
import os
import random
import yaml
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from ..base import EvolutionStrategy
from .components.ops import op_elitism, op_crossover, op_textgrad, op_mutation

logger = logging.getLogger(__name__)

class GATDStrategy(EvolutionStrategy):
    """
    9. GA-TextGrad-Diversity (GATD) Hybrid
    """
    def __init__(self):
        super().__init__()
        # self.config will be loaded in evolve to support context-based config

    def _load_config(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        config_path = None
        if context and "logic_config" in context and context["logic_config"]:
             raw_config = context["logic_config"]
             # If absolute path, use as is
             if os.path.isabs(raw_config):
                 config_path = raw_config
             else:
                 # Check if it exists as relative path (e.g. config/logic/foo.yaml)
                 if os.path.exists(raw_config):
                      config_path = os.path.abspath(raw_config)
                 else:
                      # Try under config/logic/
                      possible_path = os.path.join(os.getcwd(), "config", "logic", raw_config)
                      if os.path.exists(possible_path):
                           config_path = possible_path
                      else:
                           config_path = raw_config # Let it fail later or specific log
        else:
             # Default fallback
             config_path = os.path.join(os.getcwd(), "config", "logic", "gatd.yaml")
             
        default_config = {
            "elite_sample_num": 2,
            "gd_sample_num": 4,      # Genetic Diversity (Crossover)
            "grad_sample_num": 2,    # TextGrad
            "mutation_sample_num": 2,# Persona Mutation
            "gradient_max_words": 25
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        default_config.update(user_config)
                    logger.info(f"Loaded GATD config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load GATD config from {config_path}: {e}")
        elif context and "logic_config" in context:
             logger.warning(f"Specified logic_config not found: {context['logic_config']}")
             
        return default_config

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using GATD Strategy (Refactored)...")
        
        self._ensure_prompts("gatd", context)
        
        # Load config dynamically based on context
        self.config = self._load_config(context)
        
        elite_num = self.config.get("elite_sample_num", 2)
        gd_num = self.config.get("gd_sample_num", 4)
        grad_num = self.config.get("grad_sample_num", 2)
        mutation_num = self.config.get("mutation_sample_num", 2)
        
        new_texts = []
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        
        # 1. Elitism
        new_texts.extend(op_elitism(sorted_pop, elite_num))
        
        # 2. Crossover
        if len(new_texts) < k:
             needed = min(gd_num, k - len(new_texts))
             new_texts.extend(op_crossover(self, llm, population, needed, task_def, context))
             
        # 3. TextGrad
        if len(new_texts) < k:
             needed = min(grad_num, k - len(new_texts))
             grad_max = self.config.get("gradient_max_words", 25)
             new_texts.extend(op_textgrad(self, llm, population, needed, task_def, context, grad_max))
             
        # 4. Mutation
        if len(new_texts) < k:
             needed = k - len(new_texts) # Fill remaining with mutation
             # But honor mutation_num as soft limit? Original filled with mutation then fallback.
             # Let's just fill with mutation up to k (unlimited or based on remaining)
             new_texts.extend(op_mutation(self, llm, population, needed, task_def, context))
             
        # Fallback if mutation failed or whatever
        while len(new_texts) < k:
             constraint = self._get_task_constraint(context)
             if sorted_pop:
                  best = sorted_pop[0]
                  fallback_prompt = self.prompts["fallback_prompt"].format(
                      constraint=constraint,
                      task_def=task_def,
                      best_text=best['text']
                  )
                  t = self._generate_with_retry(llm, fallback_prompt, context)
                  new_texts.append((t, "GATD Fallback Fill"))
             else:
                  t = self._generate_with_retry(llm, self.prompts["fallback_empty_prompt"].format(
                      constraint=constraint, task_def=task_def), context)
                  new_texts.append((t, "GATD Fallback (Empty)"))
                  
        return new_texts[:k]
