import csv
import os
import argparse
from typing import List, Dict

def load_experiments(config_path: str) -> List[Dict[str, str]]:
    """
    実験設定CSVを読み込む。
    
    Args:
        config_path (str): CSVファイルのパス
        
    Returns:
        List[Dict[str, str]]: 各実験設定のリスト
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return []
        
    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        try:
            return list(reader)
        except csv.Error as e:
            print(f"Error reading CSV: {e}")
            return []

from generation.filesystem import get_experiment_result_path, get_initial_population_path

def generate_script_content(experiments: List[Dict[str, str]], project_root: str) -> str:
    """
    実行用シェルスクリプトの内容を生成する。
    
    Args:
        experiments (List[Dict]): 実験設定リスト
        project_root (str): プロジェクトルートパス
        
    Returns:
        str: 生成されたスクリプト文字列
    """
    lines = []
    lines.append("#!/bin/bash")
    lines.append("# Generated pipeline script")
    lines.append("")
    lines.append("set -e") # Stop on error
    lines.append("")
    lines.append(f"PROJECT_ROOT=\"{project_root}\"")
    lines.append(f"PYTHONpath=\"$PROJECT_ROOT/src\"")
    lines.append("")

    for exp in experiments:
        exp_id = exp['experiment_id']
        max_gen = int(exp['max_generations'])
        pop_size = exp['population_size']
        model = exp['model_name']
        adapter_type = exp['adapter_type']
        evaluator = exp.get('evaluator_type', 'llm')
        task_def = exp['task_definition']
        target = exp['target_preference']
        evolution_method = exp.get('evolution_method', 'ga')
        ensemble_ratios = exp.get('ensemble_ratios', '')
        population_name = exp.get('population_name', 'default')
        # Append hyperparameters to population_name to avoid overwriting results
        population_name = f"{population_name}_p{pop_size}_g{max_gen}"
        logic_config = exp.get('logic_config', '')
        
        lines.append(f"# Experiment: {exp_id} ({evolution_method}) [Pop: {population_name}]")
        lines.append(f"echo \"Starting Experiment: {exp_id} with {evolution_method} (Pop: {population_name})\"")

        # Initial Population Setup
        initial_pop_dir = get_initial_population_path(project_root, exp_id, population_name)
        
        lines.append(f"if [ ! -d \"{initial_pop_dir}\" ]; then")
        lines.append(f"    echo \"Initializing Population: {population_name} for {exp_id}\"")
        lines.append(f"    export PYTHONPATH=$PYTHONpath")
        lines.append(f"    python3 \"$PROJECT_ROOT/src/initialize_task.py\" \\")
        lines.append(f"        --experiment-id \"{exp_id}\" \\")
        lines.append(f"        --population-name \"{population_name}\" \\")
        lines.append(f"        --population-size \"{pop_size}\" \\")
        lines.append(f"        --model-name \"{model}\" \\")
        lines.append(f"        --adapter-type \"{adapter_type}\" \\")
        lines.append(f"        --task-definition \"{task_def}\"")
        lines.append("fi")
        lines.append("")
        
        # Iteration Loop
        for i in range(max_gen):
            # USE CENTRALIZED PATH GENERATION
            base_iter_dir = get_experiment_result_path(project_root, exp_id, population_name, evolution_method, evaluator, logic_config)
            
            check_file = os.path.join(base_iter_dir, f"row_0/iter{i}/metrics.json")
            
            lines.append(f"if [ ! -f \"{check_file}\" ]; then")
            lines.append(f"    echo \"Running Iteration {i}\"")
            
            # Generate Step - Pass FULL PATHS to avoid logic duplication in shell script
            lines.append(f"    ./cmd/generate_next_step.sh \"{exp_id}\" \"{i}\" \"{pop_size}\" \"{model}\" \"{adapter_type}\" \"{task_def}\" \"{evolution_method}\" \"{ensemble_ratios}\" \"{evaluator}\" \"{population_name}\" \"{base_iter_dir}\" \"{logic_config}\"")
            
            # Evaluate Step - Pass FULL PATHS
            lines.append(f"    ./cmd/evaluate_step.sh \"{exp_id}\" \"{i}\" \"{model}\" \"{adapter_type}\" \"{evaluator}\" \"{target}\" \"{evolution_method}\" \"{population_name}\" \"{task_def}\" \"{base_iter_dir}\"")
            
            lines.append("else")
            lines.append(f"    echo \"Skipping Iteration {i} (already completed)\"")
            lines.append("fi")
            lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="実験パイプライン生成スクリプト")
    parser.add_argument("--config", default="config/experiments.csv", help="実験設定CSVファイルへのパス")
    parser.add_argument("--output", default="run_pipeline.sh", help="生成される実行用シェルスクリプトのファイル名")
    args = parser.parse_args()

    # プロジェクトルートの解決
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    experiments = load_experiments(args.config)
    if not experiments:
        return

    script_content = generate_script_content(experiments, project_root)
    
    with open(args.output, 'w') as f:
        f.write(script_content)

    os.chmod(args.output, 0o755)
    print(f"Pipeline script generated at {args.output}")

if __name__ == "__main__":
    main()
