import os
import json
import argparse
import glob
import logging
from typing import Dict, Any

from utils import setup_logging, load_content, parse_taml_ref, load_dataset
from llm.interface import LLMInterface
from llm.factory import get_llm_adapter

logger = logging.getLogger(__name__)

def evaluate_single_text_llm(llm: LLMInterface, text: str, target_pref: str) -> Dict[str, Any]:
    """
    LLMを使用して単一のテキストを評価する。
    """
    eval_prompt = f"""
Target Preference: "{target_pref}"

Text to Evaluate:
"{text}"

Task:
Evaluate the text based on the target preference.
Give a score from 0.0 to 1.0 (float).
Provide a brief reason for the score.

Output JSON format:
{{
    "score": <float>,
    "reason": "<string>"
}}
"""
    response = llm.generate(eval_prompt)
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.rfind("```") != -1:
                cleaned = cleaned[:cleaned.rfind("```")]
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Evaluation parsing failed: {e}. Defaulting to score 0.")
        return {"score": 0.0, "reason": "Failed to parse evaluator response."}

def update_row_score_summary(row_dir: str):
    """
    指定されたrowディレクトリ配下の全イテレーションのスコアを集計し、
    score_summary.json を生成・更新する。
    """
    iter_dirs = glob.glob(os.path.join(row_dir, "iter*"))
    summary = {}

    for i_dir in iter_dirs:
        iter_name = os.path.basename(i_dir) # "iter0", "iter1"
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

    # イテレーション番号でソートして保存したい場合（オプション）
    # JSONのキー順序は保証されないが、リスト化する手もある。
    # ここでは辞書のまま保存する。
    
    summary_path = os.path.join(row_dir, "score_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Updated score summary at {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write score summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="評価実行スクリプト")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--model-name", default="gpt-4o")
    parser.add_argument("--adapter-type", required=True)
    parser.add_argument("--evaluator-type", default="llm", help="llm, rule_keyword, rule_regex, rule_meteor")
    parser.add_argument("--target-preference", required=True)
    parser.add_argument("--evolution-method", default="ga") # ディレクトリ特定のため必要
    args = parser.parse_args()

    # 親ディレクトリ: result/exp/method
    # この下の row_* を全て処理する
    base_result_dir = os.path.join("result", args.experiment_id, args.evolution_method)
    
    # row_* ディレクトリを検索
    row_dirs = glob.glob(os.path.join(base_result_dir, "row_*"))
    if not row_dirs:
        # rowがない場合は古い構造またはデータセット未使用の可能性 -> base_result_dir 自体を対象にする(互換性)
        # ただし generate.py がデフォルトで row_0 を作るようになったので、基本は row_* があるはず
        # もし見つからなければ、iterディレクトリが直下にあるか確認
        if os.path.exists(os.path.join(base_result_dir, f"iter{args.iteration}")):
            row_dirs = [base_result_dir]
        else:
            logger.warning(f"No iteration directory found in {base_result_dir}")
            return

    # Evaluatorの準備
    llm = None
    rule_evaluator = None
    
    # import rule evaluator factory
    from evaluation.rule_evaluator import get_rule_evaluator
    
    if args.evaluator_type == "llm":
        llm = get_llm_adapter(args.adapter_type, args.model_name)
    else:
        rule_evaluator = get_rule_evaluator(args.evaluator_type)
        if not rule_evaluator:
            logger.warning(f"Unknown evaluator type: {args.evaluator_type}")
            return

    # Target Contentのロード
    raw_target_pref = load_content(args.target_preference)
    # [ref] 解析
    ref_info = parse_taml_ref(args.target_preference)
    
    dataset = []
    if "dataset" in ref_info: # タスク定義側と同じデータセットを参照していると仮定、あるいはターゲット定義で指定
        # 通常はタスクとターゲットでデータセット整合性が必要。
        # ここでは実装簡略化のため、ターゲット定義に書かれたデータセットを正解データとしてロードする
        d_path = ref_info["dataset"]
        dataset = load_dataset(d_path)
    # データセットがない場合は空リスト (row_0処理用)
    
    for row_dir in row_dirs:
        # row_N からインデックスを取得
        row_name = os.path.basename(row_dir) # "row_0"
        try:
            row_idx = int(row_name.split("_")[1])
        except (IndexError, ValueError):
            row_idx = 0
            
        iter_dir = os.path.join(row_dir, f"iter{args.iteration}")
        texts_dir = os.path.join(iter_dir, "texts")
        logs_dir = os.path.join(row_dir, "logs") # ログはRow単位
        
        if not os.path.exists(texts_dir):
            continue
            
        setup_logging(os.path.join(logs_dir, "execution.log"))
        
        # この行に対応する Target Preference と Reference Text を準備
        current_target_pref = raw_target_pref
        reference_text_from_dataset = ""
        
        if dataset and row_idx < len(dataset):
            row_data = dataset[row_idx]
            
            # ターゲット文の変数置換 (LLM評価用などで使う場合)
            for key, val in row_data.items():
                current_target_pref = current_target_pref.replace(f"{{{{ {key} }}}}", str(val))
                current_target_pref = current_target_pref.replace(f"{{{{{key}}}}}", str(val))
            
            # Reference Textの取得 (データセットから直接正解を取る場合)
            # target: column_name で指定されたカラムの値を取得
            target_col = ref_info.get("target", "target")
            if target_col in row_data:
                reference_text_from_dataset = row_data[target_col]
                    
        # RuleEvaluatorでMETEORなどの場合、target引数には Reference Text を渡す必要がある
        
        # 決定ロジック
        eval_target_content = current_target_pref
        if args.evaluator_type == "rule_meteor":
             # METEORの場合はデータセットからのReferenceを優先。なければTAMLの中身(もしあれば)
             if reference_text_from_dataset:
                 eval_target_content = reference_text_from_dataset

        logger.info(f"Evaluating texts in {iter_dir}")
        
        metrics = []
        text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
        
        for text_file in text_files:
            file_name = os.path.basename(text_file)
            with open(text_file, 'r') as f:
                content = f.read().strip()
            
            result = {"file": file_name}
            
            score = 0.0
            reason = ""
            
            if llm:
                eval_res = evaluate_single_text_llm(llm, content, eval_target_content)
                score = eval_res.get("score", 0.0)
                reason = eval_res.get("reason", "")
            elif rule_evaluator:
                s, r = rule_evaluator.evaluate(content, eval_target_content)
                score = s
                reason = r
            
            result["score"] = score
            if reason:
                result["reason"] = reason
            
            
            metrics.append(result)
            
        # 保存
        with open(os.path.join(iter_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Rowごとのスコア集計を更新
        update_row_score_summary(row_dir)

if __name__ == "__main__":
    main()
