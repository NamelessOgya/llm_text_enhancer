import logging
import os
import json
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from ..base import EvolutionStrategy

logger = logging.getLogger(__name__)

class TrajectoryStrategy(EvolutionStrategy):
    """
    3. Trajectory (Trajectory of Text Refinement)
    
    テキストの進化の軌跡(Trajectory)を利用する。
    過去のテキストをスコアの低い順に並べて、「改善の流れ」をLLMに提示し、
    「この流れに沿ってさらに良くなった次のバージョン」を生成させる。
    """

    def _load_history(self, result_dir: str, current_iter: int) -> List[Tuple[str, float]]:
         # TODO: Shared logic refactor
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

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Trajectory Strategy (Text Optimization)...")
        
        self._ensure_prompts("trajectory", context)
        
        result_dir = context.get('result_dir')
        current_iter = context.get('iteration', 1)
        
        # 1. 過去の全履歴を取得
        history = self._load_history(result_dir, current_iter)
        
        # 2. 現在の世代も履歴に追加
        current_history = [(p['text'], p['score']) for p in population]
        history.extend(current_history)
        
        # 3. スコアの昇順 (低い -> 高い) にソート
        history.sort(key=lambda x: x[1])
        
        # 4. サンプリング (悪い例 -> 良い例 の流れを作る)
        if len(history) > 10:
            selected_history = history[:3] + history[-7:]
        else:
            selected_history = history
            
        trajectory_text = ""
        for i, (txt, scr) in enumerate(selected_history):
            trajectory_text += f"\n[Text Version {i+1} | Score: {scr}]\n{txt}\n"

        new_texts = []
        while len(new_texts) < k:
            constraint = self._get_task_constraint(context)
            meta_prompt = self.prompts["meta_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                trajectory_text=trajectory_text
            )
            next_text = llm.generate(meta_prompt).strip()
            new_texts.append((next_text, meta_prompt))
            
        return new_texts
