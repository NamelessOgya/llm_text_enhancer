import logging
import random
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from ..base import EvolutionStrategy

logger = logging.getLogger(__name__)

class GeneticAlgorithmStrategy(EvolutionStrategy):
    """
    1. 遺伝的アルゴリズム (Genetic Algorithm)
    
    テキストそのものを変異(Mutation)させて進化させる。
    エリート(高スコアのテキスト)を保存し、それを親として「言い換え」「拡張」「要約」「トーン変更」などの操作を行い、
    新しいテキスト変形を生成する。
    """
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Genetic Algorithm Strategy (Text Optimization)...")
        
        self._ensure_prompts("ga", context)
        
        # スコア順にソート (降順) - 同点時はランダム
        sorted_pop = sorted(population, key=lambda x: (x['score'], random.random()), reverse=True)
        
        # エリート選択 (上位20%, 最低1個)
        num_elite = max(1, int(k * 0.2))
        elites = [p for p in sorted_pop[:num_elite]]
        
        new_texts = []
        
        # エリートはそのまま次世代に残す (Elitism)
        for elite in elites:
            new_texts.append((elite['text'], "Elitism: Preserved from previous generation"))
            
        # 残りの枠を変異で埋める
        while len(new_texts) < k:
            # 親をエリートからランダム選択
            parent = random.choice(elites)
            parent_text = parent['text']
            
            # 変異タイプをランダム選択
            mutation_types = [
                "paraphrase (言い換え)", 
                "expand (詳しく・具体的にする)", 
                "shorten (簡潔にする)", 
                "change tone (より魅力的なトーンにする)"
            ]
            m_type = random.choice(mutation_types)
            
            constraint = self._get_task_constraint(context)
            mutation_prompt = self.prompts["mutation_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                parent_text=parent_text,
                m_type=m_type
            )
            mutated_text = llm.generate(mutation_prompt).strip()
            new_texts.append((mutated_text, mutation_prompt))
            
        return new_texts
