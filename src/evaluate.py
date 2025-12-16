import os
import json
import argparse
import glob
import logging
from typing import List, Dict

from llm.openai_adapter import OpenAIAdapter
from utils import setup_logging, save_token_usage

logger = logging.getLogger(__name__)

def evaluate_texts(llm: OpenAIAdapter, text_files: List[str], target: str) -> List[Dict]:
    metrics = []
    
    for t_file in text_files:
        with open(t_file, 'r') as f:
            content = f.read()
            
        prompt = f"""
        Rate how well the following text matches the description/preference: "{target}".
        Text: "{content}"
        
        Provide a score from 0 to 10 (10 being perfect match). Return ONLY the numeric score.
        """
        
        try:
            score_str = llm.generate(prompt).strip()
            # extract number if extra text exists
            import re
            match = re.search(r'\d+(\.\d+)?', score_str)
            score = float(match.group()) if match else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating {t_file}: {e}")
            score = 0.0
            
        metrics.append({
            "file": os.path.basename(t_file),
            "score": score
        })
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--model-name", default="gpt-4o")
    parser.add_argument("--target-preference", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    current_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration}")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(args.result_dir, "logs")
    
    setup_logging(os.path.join(logs_dir, "execution.log"))
    
    if args.model_name == "dummy":
        from llm.dummy_adapter import DummyAdapter
        llm = DummyAdapter()
    else:
        llm = OpenAIAdapter(model_name=args.model_name)
    
    text_files = sorted(glob.glob(os.path.join(texts_dir, "text_*.txt")))
    metrics = evaluate_texts(llm, text_files, args.target_preference)
    
    metrics_path = os.path.join(current_iter_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
