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

def load_taml(file_path: str) -> str:
    """
    TAML (Task/Target Augmented Markup Language) ファイルを読み込み、[content]セクションの内容を返す。
    TAMLは以下の形式を想定する:
    
    [background]
    (変更履歴やコンテキストなど)
    
    [content]
    (実際のプロンプト定義やターゲット定義)
    
    Args:
        file_path (str): .tamlファイルのパス
        
    Returns:
        str: [content]セクションのテキスト。セクションが見つからない場合はファイル全体を返すか、エラーログを出力する。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TAML file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    content_lines = []
    in_content = False
    
    for line in lines:
        stripped = line.strip()
        if stripped == "[content]":
            in_content = True
            continue
        elif stripped.startswith("[") and stripped.endswith("]"):
            in_content = False
            continue
            
        if in_content:
            content_lines.append(line)
            
    # [content]セクションが見つからなかった場合、ファイル全体を返すフォールバックを行うか、
    # あるいは空文字を返すか。ここでは利便性のため、もし[content]タグが全くなければ
    # ファイル全体をcontentとして扱う (後方互換性あるいは簡易記述用)。
    if not content_lines:
        # [content]タグ自体が存在したか確認
        has_content_tag = any(l.strip() == "[content]" for l in lines)
        if has_content_tag:
            return "" # タグはあるが中身がない
        else:
            return "".join(lines).strip() # タグがないので全体を返す
            
    return "".join(content_lines).strip()

def load_content(input_value: str) -> str:
    """
    入力がファイルパス(.tamlなど)であればその内容を読み込み、
    そうでなければそのまま文字列として返す。
    
    Args:
        input_value (str): ファイルパス または 直接のテキスト内容
        
    Returns:
        str: 解決されたコンテンツテキスト
    """
    # 入力が .taml ファイルパスと思われる場合
    if input_value.strip().endswith(".taml") and os.path.exists(input_value):
        return load_taml(input_value)
    
    # 通常のテキストファイルの場合もサポート (汎用性)
    if os.path.exists(input_value) and os.path.isfile(input_value):
        try:
            with open(input_value, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            # パスっぽいが読めない、あるいは長すぎてパス制限に引っかかる文字列など
            pass
            
    # ファイルでない、または見つからない場合は生のテキストとして扱う
    return input_value
