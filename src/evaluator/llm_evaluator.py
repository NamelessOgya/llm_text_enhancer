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
    def __init__(self, llm: LLMInterface, context: dict = None):
        """
        Args:
            llm (LLMInterface): 評価に使用するLLMアダプター (OpenAIやGeminiなど)
            context (dict): タスク情報などを含むコンテキスト
        """
        self.llm = llm
        self.context = context or {}
        self.prompts = {}
        
        # Initialize Cache
        from .cache import EvaluationCache
        self.cache = EvaluationCache()

    def _ensure_prompts(self):
        if not self.prompts:
            import os
            from utils import load_yaml
            
            strategy_name = "judge"
            # Default path (root of prompts)
            path = os.path.join(os.getcwd(), "config", "definitions", "prompts", f"{strategy_name}.yaml")
            
            # Task-specific path if available
            if self.context and "task_name" in self.context:
                task_name = self.context["task_name"]
                task_path = os.path.join(os.getcwd(), "config", "definitions", "prompts", task_name, f"{strategy_name}.yaml")
                if os.path.exists(task_path):
                    path = task_path
            
            self.prompts = load_yaml(path)

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
        self._ensure_prompts()
        prompt = self.prompts["judge_prompt"].format(
            target=target,
            text=text
        )
        
        score = 0.0
        reason = ""
        
        try:
            response_text = self.llm.generate(prompt).strip()
            score, reason = self._parse_response(response_text)
            
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

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        import json
        import re

        clean_text = response_text.strip()
        # Remove Markdown formatting if it wrapped the output
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        score = 0.0
        reason = ""

        try:
            # 1. 優先: JSONとしてパース
            data = json.loads(clean_text)
            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", data.get("analysis", "")))
            if not reason:
                 reason = response_text
            return score, reason
        except json.JSONDecodeError:
            # 2. JSONパース失敗時: 正規表現によるフォールバック
            pass

        # 柔軟な正規表現を使ったフォールバック
        # フォーマット例: "score": 8.5 または Score: 8.5 
        score_match = re.search(r'["\']?score["\']?\s*[:=]\s*(\d+(\.\d+)?)', clean_text, re.IGNORECASE)
        # Reasonの抽出（後続の } や " などの余計な文字をなるべく省く）
        reason_match = re.search(r'["\']?reason["\']?\s*[:=]\s*["\']?([^"}\n]*)["\']?', clean_text, re.IGNORECASE)
        
        if score_match:
            score = float(score_match.group(1))
        else:
            match = re.search(r'\d+(\.\d+)?', clean_text)
            score = float(match.group()) if match else 0.0
            
        if reason_match and reason_match.group(1).strip():
            reason = reason_match.group(1).strip()
        else:
            if not score_match: 
                 reason = response_text

        return score, reason
