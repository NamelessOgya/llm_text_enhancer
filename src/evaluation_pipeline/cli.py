import os
import argparse
import glob
import logging
from typing import Optional

from utils import load_content, parse_taml_ref, load_dataset
from llm.factory import get_llm_adapter
from evaluator.rule_evaluator import get_rule_evaluator
from .runner import run_evaluation_loop

logger = logging.getLogger(__name__)

def get_evaluator(args: argparse.Namespace, context: dict):
    """
    Evaluatorインスタンスを生成・取得するFactoryラッパー
    """
    llm = None
    rule_evaluator = None

    if args.evaluator_type == "llm":
        from evaluator.llm_evaluator import LLMEvaluator
        base_llm = get_llm_adapter(args.adapter_type, args.model_name)
        rule_evaluator = LLMEvaluator(base_llm, context=context)
        
    elif args.evaluator_type == "perspectrum_llm":
        from llm.interface import LLMInterface
        base_llm = get_llm_adapter(args.adapter_type, args.model_name)
        from evaluator.perspectrum.judge import PerspectrumLLMEvaluator
        rule_evaluator = PerspectrumLLMEvaluator(base_llm)
    else:
        rule_evaluator = get_rule_evaluator(args.evaluator_type)
    
    return rule_evaluator

def main():
    parser = argparse.ArgumentParser(description="評価実行スクリプト")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--model-name", default="gpt-4o")
    parser.add_argument("--adapter-type", required=True)
    parser.add_argument("--evaluator-type", default="llm", help="llm, rule_keyword, rule_regex, rule_meteor")
    parser.add_argument("--target-preference", required=True)
    parser.add_argument("--evolution-method", default="ga")
    parser.add_argument("--population-name", default="default")
    parser.add_argument("--task-definition", help="タスク定義ファイルパス (Optional)")
    args = parser.parse_args()

    base_result_dir = os.path.join("result", args.experiment_id, args.population_name, args.evolution_method, args.evaluator_type)
    
    # row_* ディレクトリを検索
    row_dirs = glob.glob(os.path.join(base_result_dir, "row_*"))
    if not row_dirs:
        if os.path.exists(os.path.join(base_result_dir, f"iter{args.iteration}")):
            row_dirs = [base_result_dir]
        else:
            logger.warning(f"No iteration directory found in {base_result_dir}")
            return

    # Evaluator準備
    context = {}
    if args.task_definition:
        try:
            task_def_basename = os.path.basename(args.task_definition)
            if task_def_basename.startswith("task_") and task_def_basename.endswith(".taml"):
                context["task_name"] = task_def_basename[5:-5]
        except: pass

    rule_evaluator = get_evaluator(args, context)
    if not rule_evaluator:
        logger.warning(f"Unknown evaluator type: {args.evaluator_type}")
        return

    # データセット準備
    raw_target_pref = load_content(args.target_preference)
    ref_info = parse_taml_ref(args.target_preference)
    
    dataset = []
    if "dataset" in ref_info:
        d_path = ref_info["dataset"]
        dataset = load_dataset(d_path)
    
    # 実行
    run_evaluation_loop(
        args, row_dirs, dataset, raw_target_pref, ref_info, rule_evaluator
    )

if __name__ == "__main__":
    main()
