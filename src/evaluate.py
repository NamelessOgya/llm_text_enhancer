import os
import json
import argparse
import glob
import logging
from typing import List, Dict

from llm.openai_adapter import OpenAIAdapter
from llm.dummy_adapter import DummyAdapter
from utils import setup_logging, save_token_usage
from evaluation.llm_evaluator import LLMEvaluator
from evaluation.rule_evaluator import get_rule_evaluator

logger = logging.getLogger(__name__)

def evaluate_texts(evaluator, text_files: List[str], target: str) -> List[Dict]:
    metrics = []
    
    for t_file in text_files:
        with open(t_file, 'r') as f:
            content = f.read()
            
        score = evaluator.evaluate(content, target)
        
        metrics.append({
            "file": os.path.basename(t_file),
            "score": score
        })
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--model-name", default="gpt-4o")
    parser.add_argument("--evaluator-type", default="llm")
    parser.add_argument("--target-preference", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    current_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration}")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(args.result_dir, "logs")
    
    setup_logging(os.path.join(logs_dir, "execution.log"))
    
    # Initialize LLM only if needed or for token tracking if we want to track it somewhere?
    # Actually, RuleEvaluator doesn't need LLM.
    llm = None
    if args.model_name == "dummy":
        llm = DummyAdapter()
    else:
        llm = OpenAIAdapter(model_name=args.model_name)

    # Factory Pattern Selection
    evaluator = None
    if args.evaluator_type == "llm":
        evaluator = LLMEvaluator(llm)
    else:
        # Try to get rule-based evaluator
        evaluator = get_rule_evaluator(args.evaluator_type)
        if evaluator is None:
            # Fallback or Error?
            logger.error(f"Unknown evaluator type: {args.evaluator_type}. Defaulting to LLMEvaluator.")
            evaluator = LLMEvaluator(llm)
    
    text_files = sorted(glob.glob(os.path.join(texts_dir, "text_*.txt")))
    metrics = evaluate_texts(evaluator, text_files, args.target_preference)
    
    metrics_path = os.path.join(current_iter_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    if args.evaluator_type == "llm" and llm:
        save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
