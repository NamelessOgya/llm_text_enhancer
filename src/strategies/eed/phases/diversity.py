import logging
import random
import json
from typing import List, Tuple, Dict, Any
from .utils import create_logic_json
from ...metrics import calculate_novelty

logger = logging.getLogger(__name__)

def run_diversity_phase(
    strategy: Any,
    llm: Any,
    population: List[Dict[str, Any]],
    k_diversity: int,
    task_def: str,
    context: Dict[str, Any],
    parent_gradients: List[str]
) -> List[Tuple[str, str]]:
    """
    Diversityフェーズ: Noveltyに基づく選択とContrastive Update
    
    Args:
        strategy: 呼び出し元のEEDStrategy
        llm: LLM
        population: 前世代の全個体
        k_diversity: 生成数
        task_def: タスク定義
        context: コンテキスト
        parent_gradients: Exploitフェーズで生成されたGradientリスト(これと対比する)
        
    Returns:
        List[Tuple[str, str]]: 生成された(text, logic)
    """
    logger.info(f"--- Diversity Phase (k={k_diversity}) ---")
    
    constraint = strategy._get_task_constraint(context)
    
    # 1. Calculate Novelty Scores for all population
    texts = [p['text'] for p in population]
    # Novelty計算: 自分以外の全個体との類似度の最大値を1から引いたものなど
    # ここでは簡易的に、全個体セットに対するNoveltyを計算
    novelty_scores = calculate_novelty(texts, texts) 
    
    # 2. Select top-k diversity candidates
    # score * weighted + novelty * weight? 
    # EEDでは "Quality-Diversity" なので、Scoreも考慮するが、ここはNovelty優先で選んでみる
    # あるいはパレートフロント的な選択が良いが、簡易実装としてNoveltyが高い順に、ただしScoreが閾値以下のゴミは除く
    
    candidates_with_novelty = []
    for i, p in enumerate(population):
        if p['score'] > 0.1: # 最低限の品質フィルタ
            candidates_with_novelty.append({**p, "novelty": novelty_scores[i]})
            
    candidates_with_novelty.sort(key=lambda x: x['novelty'], reverse=True)
    diversity_parents = candidates_with_novelty[:k_diversity]
    
    generated_texts = []
    
    for i, parent in enumerate(diversity_parents):
        try:
            # Contrastive Gradient Generation
            # Exploitで出たGradientと被らないように指示
            existing_grad = "Generic improvement"
            if parent_gradients:
                existing_grad = random.choice(parent_gradients)
                
            contrast_prompt = strategy.prompts["contrastive_gradient_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                parent_text=parent['text'],
                score=parent['score'],
                existing_gradient=existing_grad
            )
            contrast_gradient = strategy._generate_with_retry(llm, contrast_prompt, context).strip()
            
            # Contrastive Update
            update_prompt = strategy.prompts["contrastive_update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                parent_text=parent['text'],
                contrastive_gradient=contrast_gradient
            )
            
            new_text = strategy._generate_with_retry(llm, update_prompt, context).strip()
            
            logic = create_logic_json(
                phase="Diversity",
                parent_text=parent['text'][:100] + "...",
                instruction=contrast_gradient,
                meta={
                    "novelty_score": parent['novelty'],
                    "contrast_to": existing_grad[:50] + "..."
                }
            )
            generated_texts.append((new_text, logic))
            
        except Exception as e:
            logger.error(f"Error in Diversity Phase: {e}")
            continue
            
    return generated_texts
