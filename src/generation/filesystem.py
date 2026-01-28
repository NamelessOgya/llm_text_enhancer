import os
import shutil
import json
import logging
import argparse
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_metrics(metrics_path: str) -> list[dict]:
    """
    評価指標(metrics.json)を読み込む。
    
    Args:
        metrics_path (str): metrics.jsonのパス
        
    Returns:
        List[Dict]: 評価データのリスト
    """
    if not os.path.exists(metrics_path):
        return []
    with open(metrics_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode json: {metrics_path}")
            return []

def save_input_configs(args: argparse.Namespace, ref_info: Dict[str, str]):
    """
    実行時の設定ファイルやデータセットを保存する（再現性確保のため）。
    
    Args:
        args (argparse.Namespace): コマンドライン引数
        ref_info (Dict[str, str]): TAMLから解析された参照情報
    """
    if args.iteration != 0:
        return

    input_save_dir = os.path.join(args.result_dir, "input")
    os.makedirs(input_save_dir, exist_ok=True)
    
    # 1. Task Definition (TAML)
    if os.path.exists(args.task_definition):
        try:
            shutil.copy(args.task_definition, input_save_dir)
        except Exception as e:
            logger.warning(f"Failed to copy task definition: {e}")
            
    # 2. Referenced Dataset (CSV/JSON)
    if "dataset" in ref_info:
        d_path = ref_info["dataset"]
        if not os.path.isabs(d_path):
            d_path = os.path.abspath(d_path)
        
        if os.path.exists(d_path):
            try:
                shutil.copy(d_path, input_save_dir)
            except Exception as e:
                logger.warning(f"Failed to copy dataset: {e}")

    # 3. Reference Configs
    for config_file in ["config/data_generation_config.yaml", "config/experiments.csv"]:
        possible_config = os.path.abspath(config_file)
        if os.path.exists(possible_config):
             try:
                shutil.copy(possible_config, input_save_dir)
             except: pass

def setup_row_directories(base_result_dir: str, row_idx: int, iteration: int) -> Dict[str, str]:
    """
    行(row)とイテレーションごとのディレクトリ構造をセットアップする。
    
    Args:
        base_result_dir (str): 結果出力のルートディレクトリ
        row_idx (int): データセットの行インデックス
        iteration (int): 現在のイテレーション
        
    Returns:
        Dict[str, str]: 作成されたディレクトリパスのマッピング
            - current_iter_dir
            - prompts_dir
            - logic_dir
            - texts_dir
            - logs_dir
            - row_dir
    """
    row_dir_name = f"row_{row_idx}"
    row_dir = os.path.join(base_result_dir, row_dir_name)
    current_iter_dir = os.path.join(row_dir, f"iter{iteration}")
    
    prompts_dir = os.path.join(current_iter_dir, "input_prompts")
    logic_dir = os.path.join(current_iter_dir, "logic")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(row_dir, "logs")
    
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(logic_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return {
        "current_iter_dir": current_iter_dir,
        "prompts_dir": prompts_dir,
        "logic_dir": logic_dir,
        "texts_dir": texts_dir,
        "logs_dir": logs_dir,
        "row_dir": row_dir
    }

def save_generation_results(
    generated_outputs: list[tuple[str, str]],
    dirs: Dict[str, str]
):
    """
    生成結果をファイルに保存する。
    
    Args:
        generated_outputs: (text, logic) のリスト
        dirs: ディレクトリパス辞書 (setup_row_directoriesの戻り値)
    """
    prompts_dir = dirs["prompts_dir"]
    texts_dir = dirs["texts_dir"]
    logic_dir = dirs["logic_dir"]
    
    for i, (text, logic) in enumerate(generated_outputs):
        # 1. Logic (Meta-Prompt) -> input_prompts
        prompt_file = os.path.join(prompts_dir, f"prompt_{i}.txt")
        with open(prompt_file, 'w') as f:
            f.write(logic)
            
        # 2. Text -> texts
        text_file = os.path.join(texts_dir, f"text_{i}.txt")
        with open(text_file, 'w') as f:
            f.write(text)
            
        # 3. Logic -> creation_prompt (重複保存だが慣習)
        creation_file = os.path.join(logic_dir, f"creation_prompt_{i}.txt")
        with open(creation_file, 'w') as f:
            f.write(logic)

def get_experiment_result_path(
    project_root: str, 
    experiment_id: str, 
    population_name: str = None, 
    method: str = None, 
    evaluator: str = None
) -> str:
    """
    実験結果のディレクトリパスを一元管理して返す。
    
    Args:
        project_root (str): プロジェクトルートパス
        experiment_id (str): 実験ID
        population_name (str, optional): Population名
        method (str, optional): 手法名 (ga, gatd, textgrad, eed, etc.)
        evaluator (str, optional): 評価器名 (llm, rule, etc.)
        
    Returns:
        str: 構築された絶対パス
    """
    path = os.path.join(project_root, "result", experiment_id)
    
    if population_name:
        path = os.path.join(path, population_name)
    
    if method:
        path = os.path.join(path, method) 
        
    if evaluator:
        path = os.path.join(path, evaluator)
        
    return path

def get_initial_population_path(project_root: str, experiment_id: str, population_name: str) -> str:
    """
    初期集団のディレクトリパスを一元管理して返す。
    result/[experiment_id]/initial_population/[population_name]
    
    Args:
        project_root (str): プロジェクトルート
        experiment_id (str): 実験ID
        population_name (str): Population名
        
    Returns:
        str: 初期集団ディレクトリの絶対パス
    """
    return os.path.join(project_root, "result", experiment_id, "initial_population", population_name)
