import os
import json
import logging
from typing import Dict

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
