import logging
import os
import random
import json
import yaml
from typing import List, Dict, Tuple, Any

import numpy as np # Need numpy for gumbel

from llm.interface import LLMInterface
from ..base import EvolutionStrategy

logger = logging.getLogger(__name__)

class TextGradV2Strategy(EvolutionStrategy):
    """
    TextGrad v2: 過去の入力/スコア履歴をサンプリングして補足情報として加える。

    [NOTE: EED v1.1時点では使用停止]
    EEDのExploitフェーズでは、他個体の参照によりスタンス矛盾などの不安定さが生じたため、
    本来のNormal TextGrad (V1) ロジックに戻されました。
    このクラスは将来的な研究・改善のためにコードベースに残されています。
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "textgradv2.yaml")
        default_config = {
            "ref_sampling_weight": 10.0,
            "ref_sampling_num": 3,
            "gradient_max_words": 25 # Default
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load textgradv2.yaml: {e}")
        return default_config

    def _load_history(self, result_dir: str, current_iter: int) -> List[Tuple[str, float]]:
         # TODO: Refactor to share this with TrajectoryStrategy or make it a Mixin
         # For now, duplicate logic or import if possible. 
         # Copy-pasting the logic from TrajectoryStrategy (or moving it to base class later)
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
                    except (IndexError, ValueError):
                        continue
            except Exception as e:
                logger.warning(f"Failed to load history from iter{i}: {e}")
        return history

    def _sample_history_gumbel(self, history: List[Tuple[str, float]], k: int, weight: float) -> List[Tuple[str, float]]:
        if not history: return []
        
        scores = [h[1] for h in history]
        logits = np.array(scores) * weight
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(history))))
        perturbed_logits = logits + gumbel_noise
        top_k_indices = np.argsort(perturbed_logits)[-k:][::-1]
        
        return [history[i] for i in top_k_indices]

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using TextGrad V2 Strategy...")
        
        self._ensure_prompts("textgradv2", context)
        
        result_dir = context.get('result_dir')
        current_iter = context.get('iteration', 1)
        
        # 1. Load History
        history = self._load_history(result_dir, current_iter)
        current_history = [(p['text'], p['score']) for p in population]
        history.extend(current_history)
        
        # 2. Sample References
        ref_num = self.config["ref_sampling_num"]
        ref_weight = self.config["ref_sampling_weight"]
        
        # Elitism
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        best_individual = sorted_pop[0]
        new_texts = [(best_individual['text'], "Elitism: Best individual preserved")]
        
        idx = 0
        while len(new_texts) < k:
            target = population[idx % len(population)]
            idx += 1
            
            target_text = target['text']
            score = target['score']
            
            # Sample History for this generation
            references = self._sample_history_gumbel(history, ref_num, ref_weight)
            ref_text = ""
            for i, (txt, scr) in enumerate(references):
                ref_text += f"- Ref {i+1} (Score: {scr}): {txt[:200]}...\n" # Truncate for prompt length
            
            # Step 1: Gradient
            constraint = self._get_task_constraint(context)
            gradient_prompt = self.prompts["gradient_prompt"].format(
                 constraint=constraint,
                 task_def=task_def,
                 target_text=target_text,
                 score=score,
                 ref_text=ref_text
            )
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: Update
            constraint = self._get_task_constraint(context)
            update_prompt = self.prompts["update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target_text,
                gradient=gradient
            )
            updated_text = llm.generate(update_prompt).strip()
            new_texts.append((updated_text, f"TextGrad V2 Update:\nGradient: {gradient}\nMakePrompt: {update_prompt}"))
            
        return new_texts
