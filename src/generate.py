import os
import json
import argparse
import glob
import logging
from typing import List, Dict, Tuple

from utils import setup_logging, save_token_usage, load_content, parse_taml_ref, load_dataset
from llm.interface import LLMInterface
from evolution_strategies import get_evolution_strategy

logger = logging.getLogger(__name__)

def load_metrics(metrics_path: str) -> List[Dict]:
    """
    評価指標(metrics.json)を読み込む。
    """
    if not os.path.exists(metrics_path):
        return []
    with open(metrics_path, 'r') as f:
        return json.load(f)

def generate_initial_texts(llm: LLMInterface, k: int, task_def: str) -> List[Tuple[str, str]]:
    """
    初期世代(Generation 0)のテキスト個体群を生成する。
    タスク定義に基づき、多様なテキスト案をゼロベースで生成する。
    
    Returns:
        List[Tuple[str, str]]: (生成されたテキスト, 生成に使用したメタプロンプト)
    """
    logger.info("Generating initial population (Texts)...")
    
    # 複数生成を一括で行うプロンプト
    creation_prompt = f"""
Task Definition: "{task_def}"

Goal: Generate {k} distinct, high-quality text outputs for the above task.

Requirements:
1. Generate exactly {k} different text variations.
2. The texts should vary in style, tone, or perspective.
3. OUTPUT MUST BE A VALID JSON LIST OF STRINGS. No other text.

Example Output:
["Text option 1 including details...", "Text option 2 with different focus...", ...]
"""
    response = llm.generate(creation_prompt)
    
    # JSONパース
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.split("\n", 1)[1]
            if cleaned_response.rfind("```") != -1:
                cleaned_response = cleaned_response[:cleaned_response.rfind("```")]
        
        text_list = json.loads(cleaned_response)
        
        if isinstance(text_list, list):
            # 文字列かつ空でないものだけ抽出
            texts = [(str(t).strip(), creation_prompt) for t in text_list if str(t).strip()]
        else:
            logger.error(f"LLM returned invalid JSON structure: {type(text_list)}")
            texts = []
            
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON response: {response}. Falling back to line splitting.")
        lines = response.split('\n')
        texts = []
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith("here are") or line.startswith("```"):
                continue
            if line[0].isdigit() and ". " in line[:5]:
                line = line.split(". ", 1)[1]
            elif line.startswith("- "):
                line = line[2:]
            
            texts.append((line, creation_prompt))

    # 数合わせ
    texts = texts[:k]
    while len(texts) < k:
        fallback_creation = f"Fallback padding for task: {task_def}"
        # フォールバックとして単発生成を行う
        single_prompt = f"Generate a high-quality text for the task: {task_def}\nOutput only the text."
        single_text = llm.generate(single_prompt).strip()
        texts.append((single_text, single_prompt))
        
    return texts

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
    args = parser.parse_args()

    # タスク定義のロードとデータセット参照解析
    from llm.factory import get_llm_adapter
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    # Task Definition Contentの読み込み (テンプレートの可能性あり)
    raw_task_def_content = load_content(args.task_definition)
    
    # [ref] セクションの解析 (データセットを使用するか判定)
    ref_info = parse_taml_ref(args.task_definition)
    dataset = []
    
    if "dataset" in ref_info:
        dataset_path = ref_info["dataset"]
        if not os.path.isabs(dataset_path):
            # プロジェクトルートからの相対パスとして解決を試みる
            # このスクリプトは src/generate.py なので、実行ディレクトリ(プロジェクトルート)を基準とする
            dataset_path = os.path.abspath(dataset_path)
            
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
    else:
        # データセットを使わない従来モード (単一の row_0 として扱う)
        dataset = [{"_dummy": "dummy"}]
        
    logger.info(f"Processing {len(dataset)} items/rows.")

    for row_idx, row_data in enumerate(dataset):
        # テンプレート変数の置換
        current_task_def = raw_task_def_content
        for key, val in row_data.items():
            current_task_def = current_task_def.replace(f"{{{{ {key} }}}}", str(val))
            current_task_def = current_task_def.replace(f"{{{{{key}}}}}", str(val))

        # ディレクトリ構造: [result_dir]/row_[idx]/iter[N]
        row_dir_name = f"row_{row_idx}"
        current_iter_dir = os.path.join(args.result_dir, row_dir_name, f"iter{args.iteration}")
        
        prompts_dir = os.path.join(current_iter_dir, "input_prompts")
        logic_dir = os.path.join(current_iter_dir, "logic")
        texts_dir = os.path.join(current_iter_dir, "texts")
        logs_dir = os.path.join(args.result_dir, row_dir_name, "logs")
        
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(logic_dir, exist_ok=True)
        os.makedirs(texts_dir, exist_ok=True)
        # logsはiterごとに分けないのが通例だが、rowごとに分ける
        os.makedirs(logs_dir, exist_ok=True) 

        # 入力データを保存 (トレーサビリティのため)
        input_data_path = os.path.join(args.result_dir, row_dir_name, "input_data.json")
        if not os.path.exists(input_data_path):
            with open(input_data_path, 'w') as f:
                json.dump(row_data, f, indent=4)

        setup_logging(os.path.join(logs_dir, "execution.log"))
        
        # --- 生成/進化フェーズ ---
        generated_outputs = [] # List[Tuple[text, logic]]

        if args.iteration == 0:
            # 初期生成: タスク定義から直接テキストを生成
            generated_outputs = generate_initial_texts(llm, args.population_size, current_task_def)
        else:
            # 進化: 前世代のテキストを変異/改善
            # 前イテレーションのディレクトリ: [result_dir]/row_[idx]/iter[N-1]
            prev_iter_dir = os.path.join(args.result_dir, row_dir_name, f"iter{args.iteration-1}")
            metrics_path = os.path.join(prev_iter_dir, "metrics.json")
            
            metrics = load_metrics(metrics_path)
            population = []
            
            for m in metrics:
                try:
                    file_name = m["file"]
                    # テキスト読み込み
                    text_path = os.path.join(prev_iter_dir, "texts", file_name)
                    if not os.path.exists(text_path):
                        continue
                    with open(text_path, 'r') as f:
                        t_text = f.read().strip()
                        
                    population.append({
                        "text": t_text,
                        "score": m["score"],
                        "reason": m.get("reason", ""),
                        "file": file_name,
                        "prompt": "N/A"
                    })
                except Exception as e:
                    logger.warning(f"Skipping metric {m}: {e}")
            
            if not population:
                logger.warning(f"No valid population found in {prev_iter_dir}. Using initial generation fallback.")
                generated_outputs = generate_initial_texts(llm, args.population_size, current_task_def)
            else:
                strategy = get_evolution_strategy(args.evolution_method)
                context = {
                    "result_dir": os.path.join(args.result_dir, row_dir_name), # Strategy内で過去履歴を読むためrowディレクトリを渡す
                    "iteration": args.iteration
                }
                if args.evolution_method == "ensemble" and args.ensemble_ratios:
                     # アンサンブル比率のパース (簡易実装)
                     try:
                        ratios = {}
                        for item in args.ensemble_ratios.split(','):
                            key, val = item.split(':')
                            ratios[key.strip()] = float(val)
                        context["ensemble_ratios"] = ratios
                     except: pass
        
                generated_outputs = strategy.evolve(llm, population, args.population_size, current_task_def, context)

        # --- 保存フェーズ ---
        for i, (text, logic) in enumerate(generated_outputs):
            # 1. Logic (Meta-Prompt) を input_prompts に保存
            prompt_file = os.path.join(prompts_dir, f"prompt_{i}.txt")
            with open(prompt_file, 'w') as f:
                f.write(logic)
                
            # 2. 生成テキストを texts に保存
            text_file = os.path.join(texts_dir, f"text_{i}.txt")
            with open(text_file, 'w') as f:
                f.write(text)
                
            # 3. creation_prompt も一応保存
            creation_file = os.path.join(logic_dir, f"creation_prompt_{i}.txt")
            with open(creation_file, 'w') as f:
                f.write(logic)

        save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
