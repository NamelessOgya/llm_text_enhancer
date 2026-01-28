from abc import ABC, abstractmethod
from typing import Tuple

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        """
        生成されたテキストを評価し、スコアと理由を返す。
        全ての評価ロジック(LLMベース、ルールベース等)はこのインターフェースに従う必要がある。
        
        Args:
            text (str): 評価対象の生成テキスト
            target (str): 評価の基準となるターゲット嗜好 (正解ラベル、または期待される特性の記述)
            
        Returns:
            Tuple[float, str]:
                - score (float): 評価スコア。0.0(不適)から10.0(最適)の範囲。
                - reason (str): スコアの根拠となる説明。機械的な判定の場合は空文字でも可。
        """
        pass
