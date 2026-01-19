import os
import json
import argparse
import glob
import logging
from typing import List, Dict

from llm.factory import get_llm_adapter
from utils import setup_logging, save_token_usage
from evaluation.llm_evaluator import LLMEvaluator
from evaluation.rule_evaluator import get_rule_evaluator

logger = logging.getLogger(__name__)

def evaluate_texts(evaluator, text_files: List[str], target: str) -> List[Dict]:
    """
    指定されたEvaluatorを用いて、複数のテキストファイルを一括評価する。
    
    Args:
        evaluator: 評価ロジックを持つオブジェクト (evaluateメソッドの実装が必須)
        text_files (List[str]): 評価対象となるテキストファイルの絶対パスリスト
        target (str): 評価の基準となるターゲット嗜好 (正解ラベル、またはキーワード群)
        
    Returns:
        List[Dict]: 各ファイルに対する評価結果のリスト。
                    [{"file": "text_0.txt", "score": 8.5, "reason": "..."}, ...] の形式。
    """
    metrics = []
    
    for t_file in text_files:
        with open(t_file, 'r') as f:
            content = f.read()
            
        score, reason = evaluator.evaluate(content, target)
        
        metrics.append({
            "file": os.path.basename(t_file),
            "score": score,
            "reason": reason
        })
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="テキスト評価を実行するメインスクリプト")
    parser.add_argument("--iteration", type=int, required=True, help="現在の世代数 (結果の出力先ディレクトリ特定に使用)")
    parser.add_argument("--model-name", default="gpt-4o", help="評価に使用するLLMモデル名 (LLM評価の場合のみ有効)")
    parser.add_argument("--adapter-type", required=True, help="LLMアダプターの種類 (openai, gemini, dummy)")
    parser.add_argument("--evaluator-type", default="llm", help="評価手法の種類 (llm, rule_keyword, rule_regex)")
    parser.add_argument("--target-preference", required=True, help="評価評価の基準となる正解情報 (Target Preference)")
    parser.add_argument("--result-dir", required=True, help="結果出力のルートディレクトリ")
    args = parser.parse_args()

    # ディレクトリパスの構築
    current_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration}")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(args.result_dir, "logs")
    
    setup_logging(os.path.join(logs_dir, "execution.log"))
    
    # TAMLファイルの読み込み (target_preferenceがファイルパスの場合は内容を展開)
    from utils import load_content
    target_content = load_content(args.target_preference)
    
    # LLMアダプターの初期化 (Factory Pattern)
    # ルールベース評価の場合でも、トークン計算や将来的な拡張(ハイブリッド評価など)を見越して初期化を行っているが、
    # 完全に不要な場合は最適化の余地がある。
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    evaluator = None
    if args.evaluator_type == "llm":
        # LLMを用いた評価
        evaluator = LLMEvaluator(llm)
    else:
        # ルールベース評価器の取得
        evaluator = get_rule_evaluator(args.evaluator_type)
        if evaluator is None:
            # 指定されたタイプが見つからない場合のフォールバック
            logger.error(f"Unknown evaluator type: {args.evaluator_type}. Defaulting to LLMEvaluator.")
            evaluator = LLMEvaluator(llm)
    
    # 生成されたテキストファイルを収集して評価を実行
    # ファイル名順不同による不整合を防ぐため、glob結果は必ずソートする。
    text_files = sorted(glob.glob(os.path.join(texts_dir, "text_*.txt")))
    metrics = evaluate_texts(evaluator, text_files, target_content)
    
    # 評価結果の保存 (JSON)
    # 次世代の生成ステップ(evolve_prompts)でこのファイルを読み込む。
    metrics_path = os.path.join(current_iter_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # トークン使用量の記録
    # LLM評価器を使用した場合のみ、APIコスト管理のために記録を行う。
    if args.evaluator_type == "llm" and llm:
        save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
