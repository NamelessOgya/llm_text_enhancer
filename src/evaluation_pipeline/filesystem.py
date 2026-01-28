import os
import glob
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def update_row_score_summary(row_dir: str):
    """
    指定されたrowディレクトリ配下の全イテレーションのスコアを集計し、
    score_summary.json を生成・更新する。
    
    Args:
        row_dir (str): 行ディレクトリパス
    """
    iter_dirs = glob.glob(os.path.join(row_dir, "iter*"))
    summary = {}

    for i_dir in iter_dirs:
        iter_name = os.path.basename(i_dir)
        metrics_path = os.path.join(i_dir, "metrics.json")
        
        if not os.path.exists(metrics_path):
            continue
            
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            if not metrics:
                continue
                
            scores = [m.get("score", 0.0) for m in metrics]
            if not scores:
                continue
                
            summary[iter_name] = {
                "max": max(scores),
                "min": min(scores),
                "mean": sum(scores) / len(scores),
                "count": len(scores)
            }
        except Exception as e:
            logger.warning(f"Failed to summarize metrics for {iter_name}: {e}")

    # イテレーション番号でソートして保存する
    sorted_keys = sorted(summary.keys(), key=lambda x: int(x.replace("iter", "")))
    sorted_summary = {k: summary[k] for k in sorted_keys}
    
    summary_path = os.path.join(row_dir, "score_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(sorted_summary, f, indent=4)
        logger.info(f"Updated score summary at {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write score summary: {e}")

def save_metrics(iter_dir: str, metrics: list):
    """
    評価指標をJSONファイルに保存する。
    """
    try:
        with open(os.path.join(iter_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def get_target_content(
    args: Any, 
    row_idx: int, 
    dataset: list, 
    raw_target_pref: str, 
    ref_info: Dict[str, str]
) -> str:
    """
    評価対象の正解データ(Reference)または嗜好(Target Preference)を決定する。
    """
    current_target_pref = raw_target_pref
    reference_text_from_dataset = ""
    
    if dataset and row_idx < len(dataset):
        row_data = dataset[row_idx]
        
        # 変数置換
        for key, val in row_data.items():
            current_target_pref = current_target_pref.replace(f"{{{{ {key} }}}}", str(val))
            current_target_pref = current_target_pref.replace(f"{{{{{key}}}}}", str(val))
        
        # Reference Textの取得
        target_col = ref_info.get("target", "target")
        if target_col in row_data:
            reference_text_from_dataset = row_data[target_col]

    evaluation_content = current_target_pref
    # METEORやPerspectrum系など、正解データ（Reference）との比較が必要なEvaluatorの場合
    if args.evaluator_type in ["rule_meteor", "perspectrum_rule", "perspectrum_llm"]:
            if reference_text_from_dataset:
                evaluation_content = reference_text_from_dataset

    return evaluation_content
