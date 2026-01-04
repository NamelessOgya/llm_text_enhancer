import re
from typing import Optional, Tuple
from .interface import Evaluator

class KeywordRuleEvaluator(Evaluator):
    """
    キーワードの一致率に基づくシンプルなルールベース評価器。
    LLMを使用せず、決定論的な評価を行う。
    """
    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        """
        テキストとターゲット嗜好の間で単語の重複率(Jaccard係数ライクな指標)を計算して評価する。
        
        Args:
            text (str): 評価対象テキスト
            target (str): ターゲット嗜好 (スペース区切りのキーワードリストとみなして処理される)
            
        Returns:
            Tuple[float, str]: (重複率に基づく0-10のスコア, 理由(空文字))
                完全に一致すれば10.0、全く一致しなければ0.0となる。
        """
        if not target:
            return 0.0, ""
            
        target_words = set(target.lower().split())
        text_words = set(text.lower().split())
        
        if not target_words:
            return 0.0, ""
            
        matches = target_words.intersection(text_words)
        score = (len(matches) / len(target_words)) * 10.0
        
        return min(max(score, 0.0), 10.0), ""

class RegexRuleEvaluator(Evaluator):
    """
    正規表現マッチングに基づくバイナリ(0 or 10)のルールベース評価器。
    特定のパターンが含まれているか否かのみを判定する。
    """
    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        """
        テキストがターゲット嗜好で指定された正規表現パターンにマッチするかを評価する。
        
        Args:
            text (str): 評価対象テキスト
            target (str): 正規表現パターン文字列
            
        Returns:
            Tuple[float, str]: (マッチすれば10.0, しなければ0.0, 理由(空文字))
        """
        if not target:
            return 0.0, ""
            
        try:
            if re.search(target, text, re.IGNORECASE):
                return 10.0, ""
        except re.error:
            pass # 正規表現エラー時はマッチしなかったものとして扱う
            
        return 0.0, ""

def get_rule_evaluator(rule_type: str) -> Optional[Evaluator]:
    """
    指定されたタイプに基づいてルールベース評価器のインスタンスを生成して返す (Factory関数)。
    
    Args:
        rule_type (str): 評価器のタイプ識別子 ('rule_keyword', 'rule_regex')
        
    Returns:
        Optional[Evaluator]: 評価器インスタンス。該当するタイプがない場合はNoneを返す。
    """
    if rule_type == "rule_keyword" or rule_type == "rule_based": # Backward compatibility (旧設定ファイル互換用)
        return KeywordRuleEvaluator()
    elif rule_type == "rule_regex":
        return RegexRuleEvaluator()
    return None
