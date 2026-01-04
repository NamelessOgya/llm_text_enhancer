import os
import json
import argparse
import glob
import logging
from typing import List, Dict, Tuple


from utils import setup_logging, save_token_usage

from llm.interface import LLMInterface

logger = logging.getLogger(__name__)

def load_metrics(metrics_path: str) -> List[Dict]:
    """
    評価指標(metrics.json)を読み込む。
    
    Args:
        metrics_path (str): metrics.jsonへのパス
        
    Returns:
        List[Dict]: 読み込まれた指標データのリスト。各辞書は'file', 'score', 'reason'などのキーを持つ。
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)

def generate_initial_prompts(llm: LLMInterface, k: int, task_def: str) -> List[Tuple[str, str]]:
    """
    初期世代のプロンプト個体群を生成する。
    LLMに対して、指定されたタスク定義に基づき多様なプロンプトを提案させる。
    
    Args:
        llm (LLMInterface): 使用するLLMインターフェース
        k (int): 生成する個体数 (Population Size)
        task_def (str): タスク定義 (ドメイン定義。例: "ポケモンの説明文を生成せよ")
        
    Returns:
        List[Tuple[str, str]]: (生成されたプロンプト, 生成に使用したメタプロンプト) のタプルリスト
    """
    # LLMを使用して初期プロンプトを生成
    logger.info("Generating initial population...")
    creation_prompt = f"Generate {k} distinct prompts that would lead an AI to generate text for the following task: '{task_def}'. Return only the prompts, one per line."
    response = llm.generate(creation_prompt)
    
    # プロンプトと生成元メタプロンプトのペアを保存
    prompts = [(p.strip(), creation_prompt) for p in response.split('\n') if p.strip()]
    prompts = prompts[:k]
    
    # 生成数が不足した場合のフォールバック処理 (パディング)
    # LLMが指定数未満しか返さなかった場合の安全策
    while len(prompts) < k:
        fallback_creation = f"Fallback padding for task: {task_def}"
        prompts.append((f"Generate text for: {task_def}", fallback_creation))
    return prompts

def evolve_prompts(llm: LLMInterface, prev_prompts: List[str], metrics: List[Dict], k: int, task_def: str) -> List[Tuple[str, str]]:
    """
    遺伝的アルゴリズム(Genetic Algorithm)に基づいてプロンプトを進化させる。
    評価スコアに基づいて上位個体を選択(エリート保存)し、残りを変異(Mutation)によって生成する。
    
    【重要】データリーク防止の観点:
    変異プロンプトを生成する際、Evaluatorが持つ「正解の嗜好(target_preference)」はGeneratorに与えない。
    あくまで「タスク定義(task_def)」と「高評価だった親プロンプト」のみをヒントに改善を行わせることで、
    Generatorが正解を直接知ることなく最適化を進める構成としている。
    
    Args:
        llm (LLMInterface): LLMインターフェース
        prev_prompts (List[str]): 前世代のプロンプトリスト
        metrics (List[Dict]): 前世代の評価指標
        k (int): 目標とする個体数
        task_def (str): タスク定義 (Mutation時のコンテキストとして使用)
        
    Returns:
        List[Tuple[str, str]]: (新世代プロンプト, 生成に使用したメタプロンプト) のタプルリスト
    """
    logger.info("Evolving population...")
    
    # スコアとプロンプトの紐付け処理
    scored_prompts = []
    for m in metrics:
        # metricsのfile名からインデックスを抽出する (例: "text_0.txt" -> 0)
        # 注意: ファイル名の命名規則(text_{index}.txt)に強く依存しているため、変更時は同期が必要。
        try:
            idx = int(m["file"].split('_')[1].split('.')[0]) 
            if 0 <= idx < len(prev_prompts):
                scored_prompts.append((prev_prompts[idx], m["score"]))
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse index from filename '{m.get('file')}': {e}. Skipping this metric.")
    
    # スコア降順でソート (評価が高い順)
    scored_prompts.sort(key=lambda x: x[1], reverse=True)
    
    # エリート保存戦略 (Elitism): 
    # 上位20%の優秀なプロンプトは変異させずにそのまま次世代に継承する。
    # これにより、良質な解が破壊されるのを防ぐ。
    num_elite = max(1, int(k * 0.2))
    elites = [p for p, s in scored_prompts[:num_elite]]
    
    # エリート個体を次世代リストに追加
    new_prompts = [(p, "Elitism: Preserved from previous generation") for p in elites]
    
    # 残りの個体を生成 (Mutation / Crossover的な処理)
    # エリート個体を親として選び、LLMに書き換え(Rewrite)させることで変異を行う。
    while len(new_prompts) < k:
        # エリート個体から親を選択 (ラウンドロビン方式で均等に選出)
        parent = new_prompts[len(new_prompts) % len(elites)][0] # tupleの0番目(プロンプト)を取得
        
        # Mutation用プロンプトの構築
        # "Generate better prompt" という指示を与える。
        mutation_prompt = f"The following prompt generated good content for the task '{task_def}': \"{parent}\". Rewrite it to be even more effective. Return only the new prompt."
        
        mutated = llm.generate(mutation_prompt).strip()
        new_prompts.append((mutated, mutation_prompt))
        
    return new_prompts

def main():
    parser = argparse.ArgumentParser(description="テキスト生成とプロンプト進化を実行するメインスクリプト")
    parser.add_argument("--experiment-id", required=True, help="実験ID (結果の保存先ディレクトリ名に使用)")
    parser.add_argument("--iteration", type=int, required=True, help="現在の世代数 (0: 初期生成, 1以降: 進化)")
    parser.add_argument("--population-size", type=int, required=True, help="個体数 (k)")
    parser.add_argument("--model-name", default="gpt-4o", help="使用するLLMモデル名")
    parser.add_argument("--adapter-type", required=True, help="LLMアダプターの種類 (openai, gemini, dummy)")
    parser.add_argument("--task-definition", required=True, help="タスク定義 (Generatorへの指示・ドメイン制約)")
    parser.add_argument("--result-dir", required=True, help="結果出力のルートディレクトリ")
    args = parser.parse_args()

    # ディレクトリパスの設定
    current_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration}")
    prompts_dir = os.path.join(current_iter_dir, "input_prompts")
    # ロジック/生成用プロンプトの保存用ディレクトリ (input_promptsの兄弟ディレクトリ)
    # 実験の再現性確保やデバッグのために、生成に使用したメタプロンプトも保存する。
    logic_dir = os.path.join(current_iter_dir, "logic")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(args.result_dir, "logs")
    
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(logic_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    # ロギングの設定
    setup_logging(os.path.join(logs_dir, "execution.log"))
    
    # LLMアダプターの初期化
    from llm.factory import get_llm_adapter
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    prompts = []
    
    if args.iteration == 0:
        # 第0世代: 初期プロンプト生成
        # タスク定義のみからゼロベースでプロンプトを生成する。
        prompts = generate_initial_prompts(llm, args.population_size, args.task_definition)
    else:
        # 第n>0世代: 前世代の評価に基づく進化
        # 前世代のプロンプトと評価結果を読み込み、GAにより次世代プロンプトを生成する。
        prev_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration-1}")
        metrics_path = os.path.join(prev_iter_dir, "metrics.json")
        prev_prompts = []
        
        # 前世代のプロンプト読み込み 
        # file_system上の順序とmetricsのインデックスを一致させるため、ファイル名ソートが必須。
        prev_prompt_files = sorted(glob.glob(os.path.join(prev_iter_dir, "input_prompts", "prompt_*.txt")))
        for p_file in prev_prompt_files:
            with open(p_file, 'r') as f:
                prev_prompts.append(f.read().strip())
                
        metrics = load_metrics(metrics_path)
        prompts = evolve_prompts(llm, prev_prompts, metrics, args.population_size, args.task_definition)

    # プロンプトの保存とテキスト生成
    for i, (p, creation_p) in enumerate(prompts):
        # プロンプトの保存
        prompt_file = os.path.join(prompts_dir, f"prompt_{i}.txt")
        with open(prompt_file, 'w') as f:
            f.write(p)

        # 生成に使用したメタプロンプトの保存 (ロジックディレクトリ)
        creation_file = os.path.join(logic_dir, f"creation_prompt_{i}.txt")
        with open(creation_file, 'w') as f:
            f.write(creation_p)
            
        # テキスト生成の実行
        logger.info(f"Generating text for prompt {i}...")
        
        # タスク定義とプロンプトを組み合わせて最終的な生成指示を作成
        # Generatorには常にタスク定義(ドメイン)と、最適化されたインストラクション(プロンプト)の両方が渡される。
        final_prompt = f"Task: {args.task_definition}\nInstruction: {p}"
        text = llm.generate(final_prompt)
        
        # 生成テキストの保存
        text_file = os.path.join(texts_dir, f"text_{i}.txt")
        with open(text_file, 'w') as f:
            f.write(text)

    # トークン使用量の保存
    save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
