import os
import json
import argparse
import logging
from typing import Dict, Any

from utils import setup_logging, save_token_usage, load_content, parse_taml_ref, load_dataset
from llm.factory import get_llm_adapter
from .filesystem import save_input_configs, setup_row_directories, save_generation_results
from .executor import load_or_generate_initial_population, evolve_population

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="テキスト直接最適化を実行するメインスクリプト")
    parser.add_argument("--experiment-id", required=True, help="実験ID")
    parser.add_argument("--iteration", type=int, required=True, help="現在の世代数")
    parser.add_argument("--population-size", type=int, required=True, help="個体数 (k)")
    parser.add_argument("--model-name", default="gpt-4o", help="モデル名")
    parser.add_argument("--adapter-type", required=True, help="アダプター種別")
    parser.add_argument("--task-definition", required=True, help="タスク定義ファイル")
    parser.add_argument("--result-dir", required=True, help="結果出力ディレクトリ (親ディレクトリ: result/exp/method)")
    parser.add_argument("--evolution-method", default="ga", help="進化手法")
    parser.add_argument("--ensemble-ratios", help="アンサンブル比率")
    parser.add_argument("--initial-population-dir", help="初期集団をロードするディレクトリ")
    parser.add_argument("--logic-config", help="ロジック設定yamlファイルパス")
    args = parser.parse_args()

    # タスク定義とデータセットのロード
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    raw_task_def_content = load_content(args.task_definition)
    ref_info = parse_taml_ref(args.task_definition)
    
    dataset = []
    if "dataset" in ref_info:
        dataset_path = ref_info["dataset"]
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
    else:
        dataset = [{"_dummy": "dummy"}]
        
    logger.info(f"Processing {len(dataset)} items/rows.")
    
    # 入力設定の保存
    save_input_configs(args, ref_info)

    for row_idx, row_data in enumerate(dataset):
        # テンプレート変数の置換
        current_task_def = raw_task_def_content
        for key, val in row_data.items():
            current_task_def = current_task_def.replace(f"{{{{ {key} }}}}", str(val))
            current_task_def = current_task_def.replace(f"{{{{{key}}}}}", str(val))

        # ディレクトリ準備
        dirs = setup_row_directories(args.result_dir, row_idx, args.iteration)
        
        # 入力データの保存
        input_data_path = os.path.join(dirs["row_dir"], "input_data.json")
        if not os.path.exists(input_data_path):
            with open(input_data_path, 'w') as f:
                json.dump(row_data, f, indent=4)
        
        setup_logging(os.path.join(dirs["logs_dir"], "execution.log"))
        
        # Context初期化
        context = {
            "result_dir": dirs["row_dir"],
            "iteration": args.iteration
        }
        if args.logic_config:
            context["logic_config"] = args.logic_config
            
        try:
            task_def_basename = os.path.basename(args.task_definition)
            if task_def_basename.startswith("task_") and task_def_basename.endswith(".taml"):
                context["task_name"] = task_def_basename[5:-5]
        except: pass

        generated_outputs = []
        try:
            if args.iteration == 0:
                generated_outputs = load_or_generate_initial_population(
                    args, llm, row_idx, f"row_{row_idx}", current_task_def, context
                )
            else:
                generated_outputs = evolve_population(
                    args, llm, dirs["row_dir"], current_task_def, context
                )
        except Exception as e:
            logger.error(f"Generation failed for row {row_idx}: {e}", exc_info=True)
            continue
            
        # 保存
        save_generation_results(generated_outputs, dirs)
        save_token_usage(llm.get_token_usage(), os.path.join(dirs["logs_dir"], "token_usage.json"))

if __name__ == "__main__":
    main()
