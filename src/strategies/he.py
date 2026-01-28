import logging
import os
import random
import json
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class HEStrategy(EvolutionStrategy):
    """
    6. Hypothesis & Exploration (HE)
    
    成功法則の「仮説(Hypothesis)」を知識ベースとして蓄積・更新しながら探索する。
    Generate (Diversity) -> Evaluate -> Update Hypothesis -> Exploit (TextGrad like)
    """

    def _get_knowledge_json_path(self, result_dir: str) -> str:
        return os.path.join(result_dir, "he_knowledge.json")

    def _create_logic_json(self, meta_prompt: str, strategy_type: str, source: str) -> str:
        data = {
            "meta_prompt": meta_prompt,
            "type": strategy_type,
            "source": source
        }
        return json.dumps(data, ensure_ascii=False)

    def _update_he_knowledge(self, llm: LLMInterface, population: List[Dict[str, Any]], task_def: str, result_dir: str, context: Dict[str, Any]):
        """
        現在の個体群のスコア分布を見て、仮説(Knowledge)を更新する。
        """
        k_path = self._get_knowledge_json_path(result_dir)
        
        current_hypothesis = "Initial stage. Exploration needed."
        if os.path.exists(k_path):
            try:
                with open(k_path, 'r') as f:
                    data = json.load(f)
                    current_hypothesis = data.get("hypothesis", current_hypothesis)
            except: pass

        # Analyze High vs Low
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        high_group = sorted_pop[:3]
        low_group = sorted_pop[-3:]
        
        summary_context = "High Score Examples:\n"
        for p in high_group:
             summary_context += f"- Score {p['score']}: {p['text'][:200]}...\n"
        
        summary_context += "\nLow Score Examples:\n"
        for p in low_group:
             summary_context += f"- Score {p['score']}: {p['text'][:200]}...\n"

        if not self.prompts:
             self._ensure_prompts("he", context)
             
        prompt = self.prompts["knowledge_update_prompt"].format(
            task_def=task_def,
            current_hypothesis=current_hypothesis,
            summary_context=summary_context
        )
        
        new_hypothesis = llm.generate(prompt).strip()
        
        # Save
        data = {
            "hypothesis": new_hypothesis,
            "updated_at_iter": context.get('iteration')
        }
        try:
            with open(k_path, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except: pass
        
        return new_hypothesis

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using HE Strategy...")
        
        self._ensure_prompts("he", context)
        
        result_dir = context.get('result_dir')
        iteration = context.get('iteration')
        
        # 1. Update Hypothesis
        hypothesis = self._update_he_knowledge(llm, population, task_def, result_dir, context)
        logger.info(f"Updated Hypothesis: {hypothesis[:100]}...")
        
        # 2. Decide Strategy (Exploration vs Exploitation)
        # Simple Logic: 50:50 or Adaptive?
        # Let's do: 30% Exploration (Diversity), 70% Exploitation (Refinement based on Hypothesis)
        num_exploration = int(k * 0.3)
        num_batch = k - num_exploration
        
        new_texts = []
        
        # --- Phase 1: Exploration (Diversity) ---
        # Generate completely new approaches challenging the hypothesis or trying new angles
        if num_exploration > 0:
            for i in range(num_exploration):
                # Generate a specific angle/variation instruction
                hyp = hypothesis
                
                constraint = self._get_task_constraint(context)
                gen_prompt = self.prompts["gen_prompt"].format(
                    constraint=constraint,
                    task_def=task_def,
                    hyp=hyp
                )
                text = llm.generate(gen_prompt).strip()
                logic = self._create_logic_json(gen_prompt, "exploration_he", hyp)
                new_texts.append((text, logic))

        # --- Phase 2: Exploitation (Batch TextGrad) ---
        if num_batch > 0:
            batch_context = ""
            for i, p in enumerate(population):
                batch_context += f"Item {i}: Score={p['score']}, Text='{p['text']}'\n"
                
            sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
            targets = []
            for i in range(num_batch):
                targets.append(sorted_pop[i % len(sorted_pop)])
                
            for target in targets:
                grad_prompt = self.prompts["grad_prompt"].format(
                    task_def=task_def,
                    batch_context=batch_context,
                    target_text=target['text']
                )
                gradient = llm.generate(grad_prompt).strip()
                
                constraint = self._get_task_constraint(context)
                update_prompt = self.prompts["update_prompt"].format(
                    constraint=constraint,
                    task_def=task_def,
                    target_text=target['text'],
                    gradient=gradient
                )
                text = llm.generate(update_prompt).strip()
                logic = self._create_logic_json(update_prompt, "batch_textgrad", gradient)
                new_texts.append((text, logic))
                
        return new_texts[:k]
