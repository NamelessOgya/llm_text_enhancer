import os
import logging
import argparse
import glob
from typing import List, Dict, Tuple, Any

from strategies import get_evolution_strategy
from llm.interface import LLMInterface
from .filesystem import load_metrics

logger = logging.getLogger(__name__)

def load_or_generate_initial_population(
    args: argparse.Namespace, 
    llm: LLMInterface, 
    row_idx: int, 
    row_dir_name: str, 
    task_def: str, 
    context: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    初期集団(Iter 0)の取得を行う。
    以下の順で試行する:
    1. `initial_population_dir` が指定されていれば、そこからロードする。
    2. ロードできなければ、Strategyを使用して新規生成する。
    
    Args:
        args (argparse.Namespace): 引数
        llm (LLMInterface): LLM
        row_idx (int): 行インデックス
        row_dir_name (str): 行ディレクトリ名
        task_def (str): タスク定義
        context (Dict): コンテキスト
        
    Returns:
        List[Tuple[str, str]]: 生成された (テキスト, ロジック) のリスト
    """
    generated_outputs = []
    loaded_from_shared = False
    
    if args.initial_population_dir:
        shared_row_dir = os.path.join(args.initial_population_dir, row_dir_name)
        if os.path.exists(shared_row_dir):
            try:
                logger.info(f"Loading initial population from {shared_row_dir}")
                shared_texts_dir = os.path.join(shared_row_dir, "texts")
                shared_logic_dir = os.path.join(shared_row_dir, "logic")
                
                text_files = glob.glob(os.path.join(shared_texts_dir, "text_*.txt"))
                text_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                
                for tf in text_files:
                    basename = os.path.basename(tf)
                    idx = basename.split('_')[1].split('.')[0]
                    with open(tf, 'r') as f:
                        txt_content = f.read()
                    
                    logic_content = "Loaded from initial population"
                    logic_file = os.path.join(shared_logic_dir, f"prompt_{idx}.txt")
                    if os.path.exists(logic_file):
                        with open(logic_file, 'r') as lf:
                            logic_content = lf.read()
                    
                    generated_outputs.append((txt_content, logic_content))
                
                if generated_outputs:
                    loaded_from_shared = True
                    if len(generated_outputs) > args.population_size:
                        generated_outputs = generated_outputs[:args.population_size]
            except Exception as e:
                logger.error(f"Failed to load initial population: {e}. Falling back to generation.")
                generated_outputs = []

    if not loaded_from_shared:
        strategy = get_evolution_strategy(args.evolution_method)
        generated_outputs = strategy.generate_initial(llm, args.population_size, task_def, context=context)
        
    return generated_outputs

def evolve_population(
    args: argparse.Namespace,
    llm: LLMInterface,
    row_dir: str,
    task_def: str,
    context: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    既存の集団を進化(Evolve)させる。
    前世代(iter N-1)の結果を読み込み、Strategyに従って次世代を生成する。
    
    Args:
        args (argparse.Namespace): 引数
        llm (LLMInterface): LLM
        row_dir (str): 行ディレクトリパス
        task_def (str): タスク定義
        context (Dict): コンテキスト
        
    Returns:
        List[Tuple[str, str]]: 生成された (テキスト, ロジック) のリスト
    """
    prev_iter_dir = os.path.join(row_dir, f"iter{args.iteration-1}")
    metrics_path = os.path.join(prev_iter_dir, "metrics.json")
    
    metrics = load_metrics(metrics_path)
    population = []
    
    for m in metrics:
        try:
            file_name = m["file"]
            text_path = os.path.join(prev_iter_dir, "texts", file_name)
            if not os.path.exists(text_path):
                continue
            with open(text_path, 'r') as f:
                t_text = f.read().strip()
                
            pop_item = {
                "text": t_text,
                "score": m.get("score", 0.0),
                "reason": m.get("reason", ""),
                "file": file_name,
                "prompt": "N/A"
            }
            
            # Logicロード
            try:
                idx_str = file_name.replace("text_", "").replace(".txt", "")
                logic_path = os.path.join(prev_iter_dir, "logic", f"creation_prompt_{idx_str}.txt")
                if os.path.exists(logic_path):
                        with open(logic_path, 'r') as lf:
                            pop_item["prompt"] = lf.read()
            except Exception: pass
            
            population.append(pop_item)

        except Exception as e:
            logger.warning(f"Skipping metric {m}: {e}")
    
    if not population:
        logger.warning(f"No valid population found in {prev_iter_dir}. Using initial generation fallback.")
        strategy = get_evolution_strategy(args.evolution_method)
        return strategy.generate_initial(llm, args.population_size, task_def, context=context)
    else:
        strategy = get_evolution_strategy(args.evolution_method)
        
        if args.evolution_method == "ensemble" and args.ensemble_ratios:
                try:
                    ratios = {}
                    for item in args.ensemble_ratios.split(','):
                        key, val = item.split(':')
                        ratios[key.strip()] = float(val)
                    context["ensemble_ratios"] = ratios
                except: pass

        return strategy.evolve(llm, population, args.population_size, task_def, context)
