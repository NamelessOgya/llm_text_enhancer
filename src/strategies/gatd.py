import logging
import os
import random
import json
import yaml
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class GATDStrategy(EvolutionStrategy):
    """
    9. GA-TextGrad-Diversity (GATD) Hybrid
    
    GA (Crossover), TextGrad (Gradient), Persona (Diversity) を混合したハイブリッド戦略。
    Config: config/logic/gatd.yaml
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "gatd.yaml")
        default_config = {
            "elite_sample_num": 2,
            "gd_sample_num": 4,      # Genetic Diversity (Crossover)
            "grad_sample_num": 2,    # TextGrad
            "mutation_sample_num": 2,# Persona Mutation
            "gradient_max_words": 25
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load gatd.yaml: {e}")
        return default_config

    def _rank_selection(self, population: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Rank-based selection.
        Probability P_i = (N - i) / Sum(N - j)
        """
        n = len(population)
        if n == 0:
            return []
            
        # Sort by score descending
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        
        # Calculate weights: N, N-1, ..., 1
        weights = [n - i for i in range(n)]
        total_weight = sum(weights)
        
        if total_weight == 0:
             return random.choices(sorted_pop, k=k)
             
        # Select k individuals with replacement
        selected = random.choices(sorted_pop, weights=weights, k=k)
        return selected

    def _get_persona_history(self, context: Dict[str, Any]) -> List[str]:
        result_dir = context.get('result_dir')
        if not result_dir:
            return []
        history_path = os.path.join(result_dir, "persona_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _update_persona_history(self, context: Dict[str, Any], new_personas: List[str]):
        result_dir = context.get('result_dir')
        if not result_dir:
            return
        history_path = os.path.join(result_dir, "persona_history.json")
        
        history = self._get_persona_history(context)
        history.extend(new_personas)
        
        # Deduplicate
        history = list(set(history))
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using GATD Strategy...")
        
        # Load config params
        elite_num = self.config.get("elite_sample_num", 2)
        gd_num = self.config.get("gd_sample_num", 4)
        grad_num = self.config.get("grad_sample_num", 2)
        mutation_num = self.config.get("mutation_sample_num", 2)
        
        # Adjust numbers if k is different from sum (scale proportionally or prioritized)
        # Priority: Elite > GD > Grad > Mutation
        
        new_texts = []
        
        self._ensure_prompts("gatd", context)
        
        # 1. Elitism
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        for i in range(min(elite_num, len(sorted_pop))):
             best_individual = sorted_pop[i]
             new_texts.append((best_individual['text'], f"Elitism: Rank {i+1} preserved (Score: {best_individual['score']})"))
        
        # 2. Crossover (GD)
        gd_count = 0
        while len(new_texts) < k and gd_count < gd_num:
            parents = self._rank_selection(population, 2)
            if len(parents) < 2:
                # Fallback if population is too small, duplicate parent
                parents = parents * 2 
            
            p1 = parents[0]
            p2 = parents[1]
            
            constraint = self._get_task_constraint(context)
            
            crossover_prompt = self.prompts["crossover_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                p1_score=p1['score'],
                p1_text=p1['text'],
                p2_score=p2['score'],
                p2_text=p2['text']
            )
            child_text = self._generate_with_retry(llm, crossover_prompt, context)
            new_texts.append((child_text, f"GATD Crossover (Score {p1['score']} & {p2['score']})"))
            gd_count += 1
            
        # 3. TextGrad
        grad_count = 0
        while len(new_texts) < k and grad_count < grad_num:
            target_list = self._rank_selection(population, 1)
            if not target_list: break
            target = target_list[0]
            
            constraint = self._get_task_constraint(context)
            grad_max_words = self.config.get("gradient_max_words", 25)
            
            # Gradient Step
            gradient_prompt = self.prompts["gradient_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target['text'],
                target_score=target['score'],
                grad_max_words=grad_max_words
            )
            gradient = llm.generate(gradient_prompt).strip()
            
            # Update Step
            update_prompt = self.prompts["update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target['text'],
                gradient=gradient
            )
            updated_text = self._generate_with_retry(llm, update_prompt, context)
            new_texts.append((updated_text, f"GATD TextGrad:\nGradient: {gradient}"))
            grad_count += 1
            
        # 4. Mutation (Persona)
        mutation_count = 0
        # Load history
        persona_history = self._get_persona_history(context)
        new_personas = []
        
        while len(new_texts) < k and mutation_count < mutation_num:
             # Select parent to inherit stance from
             parent_list = self._rank_selection(population, 1)
             if not parent_list: break
             parent = parent_list[0]

             constraint = self._get_task_constraint(context)
             
             # Use recent history for prompt context to avoid immediate repeats
             history_sample = persona_history[-10:] if len(persona_history) > 10 else persona_history
             history_str = "\n".join([f"- {p}" for p in history_sample])
             
             persona_prompt = self.prompts["persona_prompt"].format(
                 task_def=task_def,
                 history_str=history_str
             )
             persona = llm.generate(persona_prompt).strip()
             # Simple validation
             if len(persona) > 200: persona = persona[:200]
             
             new_personas.append(persona)
             
             generation_prompt = self.prompts["generation_prompt"].format(
                 constraint=constraint,
                 task_def=task_def,
                 parent_text=parent['text'],
                 persona=persona
             )
             mutated_text = self._generate_with_retry(llm, generation_prompt, context)
             new_texts.append((mutated_text, f"GATD Persona Mutation (Parent Score: {parent['score']}): {persona}"))
             mutation_count += 1
             
        # Save new personas
        if new_personas:
            self._update_persona_history(context, new_personas)
        
        # Fill remaining if any (due to exact matching issues)
        while len(new_texts) < k:
             # Fallback to simple mutation of best
             if sorted_pop:
                 best = sorted_pop[0]
                 constraint = self._get_task_constraint(context)
                 fallback_prompt = self.prompts["fallback_prompt"].format(
                     constraint=constraint,
                     task_def=task_def,
                     best_text=best['text']
                 )
                 t = self._generate_with_retry(llm, fallback_prompt, context)
                 new_texts.append((t, "GATD Fallback Fill"))
             else:
                 # Should not happen if population > 0
                 constraint = self._get_task_constraint(context)
                 fallback_prompt = self.prompts["fallback_empty_prompt"].format(
                     constraint=constraint,
                     task_def=task_def
                 )
                 t = self._generate_with_retry(llm, fallback_prompt, context)
                 new_texts.append((t, "GATD Fallback (Empty Pop)"))

        return new_texts[:k]
