import logging
import re
from typing import Tuple
from .interface import Evaluator
from llm.interface import LLMInterface

logger = logging.getLogger(__name__)

class LLMEvaluator(Evaluator):
    """
    LLM (Large Language Model) を使用してテキストの品質を定性的・定量的に評価するクラス。
    指定されたLLMアダプターを経由して評価用プロンプトを送信し、その応答をパースしてスコア化する。
    """
    def __init__(self, llm: LLMInterface):
        """
        Args:
            llm (LLMInterface): 評価に使用するLLMアダプター (OpenAIやGeminiなど)
        """
        self.llm = llm
        
        # Initialize Cache
        from .cache import EvaluationCache
        self.cache = EvaluationCache()

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        """
        LLMにプロンプトを投げ、テキストがターゲット嗜好に合致しているかを評価させる。
        
        【仕組み】
        1. ターゲット嗜好(target)と評価対象テキスト(text)を含む評価指示プロンプトを構築。
        2. LLMに対して0〜10点のスコアと、その理由(Reason)を出力するように指示。
        3. LLMの応答を正規表現でパースし、数値を抽出する。
        
        Args:
            text (str): 評価対象テキスト
            target (str): ターゲット嗜好
            
        Returns:
            Tuple[float, str]: (スコア, 理由)
                パース失敗時やエラー時はスコア0.0、理由にエラー内容をセットして返す。
        """
        # 0. Check Cache
        cached_result = self.cache.get("LLMEvaluator", text, target)
        if cached_result:
            return cached_result

        # 評価用プロンプトの構築
        # "Score:" と "Reason:" という特定のフォーマットでの出力を強制する。
        prompt = f"""
        Rate how well the following text matches the description/preference: "{target}".
        Text: "{text}"
        
        Provide a score from 0 to 10 (10 being perfect match) and a brief reason.
        
        Format your response exactly as follows:
        Score: <score>
        Reason: <reason>
        """
        
        score = 0.0
        reason = ""
        
        try:
            response_text = self.llm.generate(prompt).strip()
            
            # 正規表現によるスコアと理由のパース
            # 形式: "Score: 8.5" や "Reason: Because..."
            score_match = re.search(r'Score:\s*(\d+(\.\d+)?)', response_text, re.IGNORECASE)
            # Reasonは改行を含む可能性があるため DOTALL フラグを使用
            reason_match = re.search(r'Reason:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
            
            if score_match:
                score = float(score_match.group(1))
            else:
                 # フォールバック処理: 明示的な "Score:" ラベルがない場合、
                 # レスポンスに含まれる最初の数値をスコアとして解釈を試みる。
                match = re.search(r'\d+(\.\d+)?', response_text)
                score = float(match.group()) if match else 0.0
                
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # 理由のフォーマットが見つからない場合、
                # スコアも見つからなければレスポンス全体をエラー理由として扱うことでデバッグを容易にする。
                if not score_match: 
                     reason = response_text
            
            # Save to Cache
            # Only save if we got a valid score (or if we want to cache 0.0 failures too? usually better to retry failures)
            # But here failures return 0.0 with reason.
            # Let's cache everything that didn't raise Exception.
            self.cache.set("LLMEvaluator", text, target, score, reason)
            
        except Exception as e:
            logger.error(f"Error evaluating text: {e}")
            score = 0.0
            reason = f"Error: {e}"
            # Do NOT cache DB/API errors so we can retry
            
        return score, reason
