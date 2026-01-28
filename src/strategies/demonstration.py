import logging
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class DemonstrationStrategy(EvolutionStrategy):
    """
    4. Demonstration (Few-Shot Examples)
    
    高スコアのテキストを「成功事例 (Demonstration)」としてFew-shotプロンプトに含め、
    「これらと同じくらい高品質な新しいテキスト」を生成させる。
    """
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Demonstration Strategy (Text Optimization)...")
        
        self._ensure_prompts("demonstration", context)
        
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        top_examples = sorted_pop[:5] # Top 5を事例として使う
        
        examples_text = ""
        for i, ex in enumerate(top_examples):
            examples_text += f"\nExample {i+1} (High Quality):\n{ex['text']}\n"
            
        new_texts = []
        while len(new_texts) < k:
            constraint = self._get_task_constraint(context)
            meta_prompt = self.prompts["meta_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                examples_text=examples_text
            )
            generated_text = llm.generate(meta_prompt).strip()
            new_texts.append((generated_text, meta_prompt))
            
        return new_texts
