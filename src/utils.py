import os
import json
import logging
from typing import Dict, List

def setup_logging(log_file_path: str):
    """
    ファイルとコンソールへのロギング設定を初期化する。
    
    Args:
        log_file_path (str): ログファイルの出力先パス (ディレクトリが存在しない場合は自動作成される)
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

def save_token_usage(token_usage: Dict[str, int], file_path: str):
    """
    トークン使用量をJSONファイルに累積して保存する。
    Experiment全体でのコスト計算などに使用する。
    
    【仕様】
    - 指定されたパスにファイルが存在しない場合は新規作成する。
    - 存在する場合は読み込み、今回の使用量を加算して上書き保存する。
    
    Args:
        token_usage (Dict[str, int]): 今回の実行で消費したトークン量。
            期待されるキー: 'prompt_tokens', 'completion_tokens', 'total_tokens' など。
        file_path (str): トークン使用量ファイルのパス
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # デフォルトの構造定義
    current_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                current_usage = json.load(f)
        except json.JSONDecodeError:
            pass # ファイル破損時は初期化して再開する (ロギングは省略しているが、厳密にはWarningが望ましい)
            
    # 使用量の累積計算
    for key in current_usage:
        current_usage[key] += token_usage.get(key, 0)

    with open(file_path, 'w') as f:
        json.dump(current_usage, f, indent=4)

import yaml

def load_yaml(file_path: str) -> dict:
    """
    YAML ファイル内の全セクションを辞書としてロードする。
    
    Args:
        file_path (str): .yamlファイルのパス
        
    Returns:
        dict: セクション名をキーとする辞書。
              例: {'background': '...', 'content': '...'}
    """
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            return {}

def load_content(input_value: str) -> str:
    """
    入力がファイルパス(.yamlなど)であればその内容(contentキー)を読み込み、
    そうでなければそのまま文字列として返す。
    
    Args:
        input_value (str): ファイルパス または 直接のテキスト内容
        
    Returns:
        str: 解決されたコンテンツテキスト
        
    Raises:
        FileNotFoundError: パスとして認識されたがファイルが存在しない場合
    """
    stripped_val = input_value.strip()
    
    # パスとして扱うかどうかの判定 (拡張子やスラッシュの存在)
    is_path_like = (
        stripped_val.endswith(".yaml") or 
        stripped_val.endswith(".yml") or 
        stripped_val.endswith(".txt") or 
        "/" in stripped_val or
        "\\" in stripped_val
    )

    if is_path_like:
        if os.path.exists(stripped_val):
            if stripped_val.endswith(".yaml") or stripped_val.endswith(".yml"):
                data = load_yaml(stripped_val)
                # content キーがあれば返す。なければ judge_prompt などを優先するか、
                # あるいは全体を文字列化して返すなどのフォールバックを行う
                if "content" in data:
                    return str(data["content"])
                elif "judge_prompt" in data:
                    return str(data["judge_prompt"])
                else:
                    return ""
            else:
                try:
                    with open(stripped_val, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    raise IOError(f"Failed to read file: {stripped_val}. Error: {e}")
        else:
            raise FileNotFoundError(f"Content file not found at: {stripped_val}. "
                                   f"If this was intended as raw text, avoid using path indicators like '.' or '/'.")
            
    return input_value

def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    CSVまたはJSONLファイルをロードし、辞書のリストとして返す。
    Args:
        file_path (str): ファイルパス
    Returns:
        List[Dict[str, str]]: 行データのリスト
    """
    data = []
    if file_path.endswith('.csv'):
        import csv
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        # 簡易的にCSVとしてトライ
        import csv
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    return data

def parse_yaml_ref(file_path: str) -> Dict[str, str]:
    """
    YAMLファイルの 'ref' セクション（辞書）をパースする。
    Returns:
        Dict: キーバリューペア (dataset, target_column 等)
    """
    if not os.path.exists(file_path):
        return {}
        
    data = load_yaml(file_path)
    if "ref" in data and isinstance(data["ref"], dict):
        return {str(k): str(v) for k, v in data["ref"].items()}
    return {}
