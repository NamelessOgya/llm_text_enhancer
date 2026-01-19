import os
import json
import argparse
import glob
import logging
from typing import List, Dict, Tuple

from utils import setup_logging, save_token_usage
from llm.interface import LLMInterface
from evolution_strategies import get_evolution_strategy

logger = logging.getLogger(__name__)

def load_metrics(metrics_path: str) -> List[Dict]:
    """
    評価指標(metrics.json)を読み込む。
    """
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
    parser.add_argument("--result-dir", required=True, help="結果出力ディレクトリ")
    parser.add_argument("--evolution-method", default="ga", help="進化手法")
    parser.add_argument("--ensemble-ratios", help="アンサンブル比率")
    args = parser.parse_args()

    current_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration}")
    # input_prompts は "生成ロジック(Meta-Prompt)" を保存する場所として再定義する
    prompts_dir = os.path.join(current_iter_dir, "input_prompts")
    # logic ディレクトリも念のため維持 (同じ内容になるかもしれないが構造維持のため)
    logic_dir = os.path.join(current_iter_dir, "logic")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(args.result_dir, "logs")
    
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(logic_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    setup_logging(os.path.join(logs_dir, "execution.log"))
    
    from utils import load_content
    task_def_content = load_content(args.task_definition)
    
    from llm.factory import get_llm_adapter
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    # --- 生成/進化フェーズ ---
    generated_outputs = [] # List[Tuple[text, logic]]

    if args.iteration == 0:
        # 初期生成: タスク定義から直接テキストを生成
        generated_outputs = generate_initial_texts(llm, args.population_size, task_def_content)
    else:
        # 進化: 前世代のテキストを変異/改善
        prev_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration-1}")
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
                    
                # プロンプト(ロジック)読み込み - 必須ではないが文脈としてあれば
                # ここでの "prompt" は以前の「生成指示」ではなく、前回の「変異指示」などのメタ情報
                # 互換性のため "prompt" キーにはテキストを入れておくか、空にしておく
                # TextGradなどで「元の指示」が必要な場合があるが、Text Optimizationでは「元のテキスト」が重要。
                
                population.append({
                    "text": t_text,
                    "score": m["score"],
                    "reason": m.get("reason", ""),
                    "file": file_name,
                    "prompt": "N/A" # Prompt optimizationではないのでダミー
                })
            except Exception as e:
                logger.warning(f"Skipping metric {m}: {e}")
        
        strategy = get_evolution_strategy(args.evolution_method)
        context = {
            "result_dir": args.result_dir,
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

        generated_outputs = strategy.evolve(llm, population, args.population_size, task_def_content, context)

    # --- 保存フェーズ ---
    # ここでは "Inference" ステップは不要。generated_outputs に既にテキストが入っている。
    for i, (text, logic) in enumerate(generated_outputs):
        # 1. Logic (Meta-Prompt) を input_prompts に保存 (トレーサビリティ)
        # これが「このテキストを生み出した指示書」となる
        prompt_file = os.path.join(prompts_dir, f"prompt_{i}.txt")
        with open(prompt_file, 'w') as f:
            f.write(logic)
            
        # 2. 生成テキストを texts に保存
        text_file = os.path.join(texts_dir, f"text_{i}.txt")
        with open(text_file, 'w') as f:
            f.write(text)
            
        # 3. creation_prompt も一応保存 (input_promptsと同じ内容だが)
        creation_file = os.path.join(logic_dir, f"creation_prompt_{i}.txt")
        with open(creation_file, 'w') as f:
            f.write(logic)

    save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
