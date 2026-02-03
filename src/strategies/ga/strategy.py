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
        
        # 1. Elitism (20%)
        num_elite = max(1, int(k * 0.2))
        elites = [p for p in sorted_pop[:num_elite]]
        
        new_texts = []
        
        for elite in elites:
            new_texts.append((elite['text'], "Elitism: Preserved from previous generation"))
            
        # 2. Crossover (60%)
        # Calculate remaining slots
        remaining = k - len(new_texts)
        # We want mutation to be roughly 20% of k, so crossover gets the rest up to that point
        # Target mutation count
        target_mutation = int(k * 0.2)
        # Therefore crossover target is remaining - target_mutation
        # But we must ensure at least some crossover if possible? 
        # User said "Elite 20%, Mutation 20%". So Crossover is 60%.
        
        num_crossover = int(k * 0.6)
        
        # Adjust if rounding errors cause overflow/underflow
        # Prioritize Elite > Crossover > Mutation or Elite > Mutation > Crossover?
        # Let's try to hit the targets as close as possible without exceeding k
        
        # We process Crossover until we have enough, or leave space for mutation?
        # Actually safer to loop:
        # Fill Crossover
        crossover_target_count = num_crossover
        
        for _ in range(crossover_target_count):
            if len(new_texts) >= k: break
            
            # Rank Selection for Parents
            # Weights based on rank (higher rank = higher weight)
            n = len(sorted_pop)
            weights = [n - i for i in range(n)]
            if sum(weights) == 0:
                 parents = random.choices(sorted_pop, k=2)
            else:
                 parents = random.choices(sorted_pop, weights=weights, k=2)
            
            p1, p2 = parents[0], parents[1]
            constraint = self._get_task_constraint(context)
            
            prompt = self.prompts["crossover_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                p1_score=p1['score'], p1_text=p1['text'],
                p2_score=p2['score'], p2_text=p2['text']
            )
            child = self._generate_with_retry(llm, prompt, context)
            new_texts.append((child, f"GA Crossover (Score {p1['score']} & {p2['score']})"))

        # 3. Mutation (Remaining ~20%)
        while len(new_texts) < k:
            # 親をエリートからランダム選択 (既存ロジック踏襲) or Rank Selection?
            # 既存はエリートから選択していたが、多様性のためにはRank Selectionの方が良いかも？
            # しかし元の実装はエリートから選択していた。
            # "親をエリートからランダム選択" -> line 38
            # 元の実装を活かすならエリートから。
            
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
