import os
import glob
import logging
from typing import Any
from .filesystem import update_row_score_summary, save_metrics, get_target_content
from utils import setup_logging

logger = logging.getLogger(__name__)

def run_evaluation_loop(
    args: Any, 
    row_dirs: list, 
    dataset: list, 
    raw_target_pref: str, 
    ref_info: dict,
    rule_evaluator: Any
):
    """
    各Rowディレクトリに対して評価を実行するループ。
    """
    for row_dir in row_dirs:
        row_name = os.path.basename(row_dir)
        try:
            row_idx = int(row_name.split("_")[1])
        except (IndexError, ValueError):
            row_idx = 0
            
        iter_dir = os.path.join(row_dir, f"iter{args.iteration}")
        texts_dir = os.path.join(iter_dir, "texts")
        logs_dir = os.path.join(row_dir, "logs")
        
        if not os.path.exists(texts_dir):
            continue
            
        setup_logging(os.path.join(logs_dir, "execution.log"))
        
        # 評価基準(Target/Reference)の決定
        eval_target_content = get_target_content(
            args, row_idx, dataset, raw_target_pref, ref_info
        )

        logger.info(f"Evaluating texts in {iter_dir}")
        
        metrics = []
        text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
        
        for text_file in text_files:
            file_name = os.path.basename(text_file)
            try:
                with open(text_file, 'r') as f:
                    content = f.read().strip()
                
                result = {"file": file_name}
                score = 0.0
                reason = ""
                
                if rule_evaluator:
                    s, r = rule_evaluator.evaluate(content, eval_target_content)
                    score = s
                    reason = r
                
                result["score"] = score
                if reason:
                    result["reason"] = reason
                
                metrics.append(result)
            except Exception as e:
                logger.error(f"Error evaluating file {file_name}: {e}")
            
        save_metrics(iter_dir, metrics)
        update_row_score_summary(row_dir)
