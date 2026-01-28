import logging
import os
import random
import json
import yaml
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from ..base import EvolutionStrategy

logger = logging.getLogger(__name__)

class TextGradStrategy(EvolutionStrategy):
    """
    2. TextGrad (Pseudo-Gradient)
    
    テキストに対する「改善の方向性(勾配)」を言語化し、それに基づいてテキストを修正する。
    1. 現状のテキストとスコアから「なぜスコアがこの値なのか」「どうすれば上がるか」を分析(Gradient Generation)。
    2. その分析に基づいてテキストをリライトする(Update)。
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "textgrad.yaml")
        default_config = {
            "elite_sample_num": 1,
            "gradient_max_words": 25 # Default
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load textgrad.yaml: {e}")
        return default_config

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using TextGrad Strategy (Text Optimization)...")
        
        self._ensure_prompts("textgrad", context)
        
        elite_num = self.config.get("elite_sample_num", 1)

        # エリートは保存
        sorted_pop = sorted(population, key=lambda x: (x['score'], random.random()), reverse=True)
        new_texts = []
        for i in range(min(elite_num, len(sorted_pop))):
             best_individual = sorted_pop[i]
             new_texts.append((best_individual['text'], f"Elitism: Rank {i+1} preserved (Score: {best_individual['score']})"))
        
        # 残りの枠を勾配更新で生成
        idx = 0
        while len(new_texts) < k:
            target = population[idx % len(population)]
            idx += 1
            
            target_text = target['text']
            score = target['score']
            reason = target.get('reason', 'N/A')
            
            # Step 1: 勾配(改善提案)の生成
            constraint = self._get_task_constraint(context)
            gradient_prompt = self.prompts["gradient_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target_text,
                score=score,
                gradient_max_words=self.config.get('gradient_max_words', 25)
            )
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: 勾配の適用 (Update)
            constraint = self._get_task_constraint(context)
            update_prompt = self.prompts["update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target_text,
                gradient=gradient
            )
            updated_text = llm.generate(update_prompt).strip()
            new_texts.append((updated_text, f"TextGrad Update:\nGradient: {gradient}\nMakePrompt: {update_prompt}"))
            
        return new_texts

    def _load_history(self, result_dir: str, current_iter: int) -> List[Tuple[str, float]]:
        """
        過去の全世代の実行結果をディスクから読み込むヘルパーメソッド。
        (テキストとそのスコアを返す)
        """
        history = []
        for i in range(current_iter):
            iter_dir = os.path.join(result_dir, f"iter{i}")
            metrics_path = os.path.join(iter_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
                
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # 対応するテキストファイルの読み込み
                for m in metrics:
                    try:
                        file_name = m["file"]
                        t_path = os.path.join(iter_dir, "texts", file_name)
                        if os.path.exists(t_path):
                             with open(t_path, 'r') as tf:
                                 t_text = tf.read().strip()
                             history.append((t_text, m["score"]))
                    except: pass
            except: pass
        return history
