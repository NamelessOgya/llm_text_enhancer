import os
import json
import argparse
import logging
import shutil
from typing import List, Dict, Tuple

from utils import setup_logging, save_token_usage, load_content, parse_taml_ref, load_dataset
from llm.interface import LLMInterface
from strategies import get_evolution_strategy
from generation.filesystem import get_initial_population_path

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="実験の初期集団生成スクリプト")
    parser.add_argument("--experiment-id", required=True, help="実験ID")
    parser.add_argument("--population-name", default="default", help="初期集団名 (例: run1)")
    parser.add_argument("--population-size", type=int, required=True, help="個体数 (k)")
    parser.add_argument("--model-name", default="gpt-4o", help="モデル名")
    parser.add_argument("--adapter-type", required=True, help="アダプター種別")
    parser.add_argument("--task-definition", required=True, help="タスク定義ファイル")
    parser.add_argument("--dataset", help="データセットパス (Optional: TAML参照を上書き)")
    
    args = parser.parse_args()

    # Setup LLM
    from llm.factory import get_llm_adapter
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    # Task Definition Content
    raw_task_def_content = load_content(args.task_definition)
    
    # Dataset Loading
    dataset = []
    if args.dataset:
        dataset_path = args.dataset
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        logger.info(f"Loading dataset from provided arg: {dataset_path}")
        dataset = load_dataset(dataset_path)
    else:
        # Load from TAML ref
        ref_info = parse_taml_ref(args.task_definition)
        if "dataset" in ref_info:
            dataset_path = ref_info["dataset"]
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.abspath(dataset_path)
            logger.info(f"Loading dataset from TAML ref: {dataset_path}")
            dataset = load_dataset(dataset_path)
        else:
             # Dummy single row
            dataset = [{"_dummy": "dummy"}]
            
    logger.info(f"Generating initial population '{args.population_name}' for {len(dataset)} items.")
    
    # Base output dir
    base_output_dir = get_initial_population_path(os.getcwd(), args.experiment_id, args.population_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save generic info if needed? 
    # Maybe save the task_def here for reference
    shutil.copy(args.task_definition, os.path.join(base_output_dir, "task_definition.taml"))
    
    strategy = get_evolution_strategy("ga") # Default strategy for initialization is typically uniform (random/ga)

    for row_idx, row_data in enumerate(dataset):
        # テンプレート変数の置換
        current_task_def = raw_task_def_content
        for key, val in row_data.items():
            current_task_def = current_task_def.replace(f"{{{{ {key} }}}}", str(val))
            current_task_def = current_task_def.replace(f"{{{{{key}}}}}", str(val))
            
        row_dir_name = f"row_{row_idx}"
        target_dir = os.path.join(base_output_dir, row_dir_name)
        
        texts_dir = os.path.join(target_dir, "texts")
        logic_dir = os.path.join(target_dir, "logic")
        logs_dir = os.path.join(target_dir, "logs") # Keep logs here for initialization
        
        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(logic_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        setup_logging(os.path.join(logs_dir, "initialization.log"))
        
        # Save input data
        with open(os.path.join(target_dir, "input_data.json"), 'w') as f:
            json.dump(row_data, f, indent=4)
            
        # Context
        context = {
            "result_dir": target_dir, # Might not be heavily used by generate_initial but good practice
            "task_name": "", # Can extract if needed
            "iteration": 0
        }
        
        # Determine Task Name
        try:
            task_def_basename = os.path.basename(args.task_definition)
            if task_def_basename.startswith("task_") and task_def_basename.endswith(".taml"):
                context["task_name"] = task_def_basename[5:-5]
        except: pass
        
        # Generate
        logger.info(f"Row {row_idx}: Generating {args.population_size} items...")
        generated_outputs = strategy.generate_initial(llm, args.population_size, current_task_def, context=context)
        
        # Save
        for i, (text, logic) in enumerate(generated_outputs):
             # 1. Logic (Meta-Prompt)
            prompt_file = os.path.join(logic_dir, f"prompt_{i}.txt")
            with open(prompt_file, 'w') as f:
                f.write(logic)
                
            # 2. Text
            text_file = os.path.join(texts_dir, f"text_{i}.txt")
            with open(text_file, 'w') as f:
                f.write(text)
                
        save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))
        
    print(f"Initialization completed. Output: {base_output_dir}")

if __name__ == "__main__":
    main()
