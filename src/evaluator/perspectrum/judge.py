import os
import yaml
import json
import logging
from typing import Tuple
from ..interface import Evaluator
from llm.interface import LLMInterface
from .rules import PerspectrumRuleEvaluator
from ..cache import EvaluationCache

logger = logging.getLogger(__name__)

class PerspectrumLLMEvaluator(Evaluator):
    """
    LLM-as-a-judge for Perspectrum.
    """
    def __init__(self, llm: LLMInterface, config_path: str = "config/task/perspectrum.yaml"):
        self.llm = llm
        self.llm_eval_repeat = 3
        # Use helper from rules.py via instantiation
        self._rule_checker = PerspectrumRuleEvaluator(config_path)
        self.cache = EvaluationCache()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and "llm_eval_repeat" in config:
                        self.llm_eval_repeat = int(config["llm_eval_repeat"])
            except: pass

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        cached_result = self.cache.get("PerspectrumLLMEvaluator", text, target)
        if cached_result:
            return cached_result

        # Rule check call directly to instance method
        score, reason = self._rule_checker.evaluate(text, target)
        # Note: rule evaluator returns meteor score on success, but here if it returns non-zero/empty reason,
        # we treat it as passing filters. wait, rule evaluator returns (score, "") on success?
        # Actually PerspectrumRuleEvaluator returns METEOR score.
        # But LLMEvaluator wants to filter out 0.0s coming from violations.
        
        # If score is 0.0 and reason is set, it's a violation.
        if score == 0.0 and reason != "":
             self.cache.set("PerspectrumLLMEvaluator", text, target, 0.0, reason)
             return 0.0, reason

        # LLM Evaluation
        raw_prompt = """
You are an objective "Logical Consistency Checker".
Your task is to determine if the Generated Perspective uses the **exact same logical basis and set of evidence** as the Reference Perspective.

Reference Perspective:
"{target}"

Generated Perspective:
"{text}"

Evaluation Task:
1. Extract the set of logical points/evidence used in the Reference.
2. Extract the set of logical points/evidence used in the Generated Perspective.
3. Compare the two sets.

Strict Criteria:
- **High Scores (8-10)**: The set of logical points matches. Differentiate based on **Nuance** and **Granularity**.
...
Output JSON format:
{{
    "score": <int 1-10>,
    "reason": "<string>"
}}
"""
        valid_results = []
        for i in range(self.llm_eval_repeat):
            try:
                response = self.llm.generate(raw_prompt, temperature=0.0)
                cleaned = response.strip()
                if cleaned.startswith("```"):
                     cleaned = cleaned.split("\n", 1)[1]
                     if cleaned.rfind("```") != -1:
                        cleaned = cleaned[:cleaned.rfind("```")]
                
                data = json.loads(cleaned)
                raw_score = float(data.get("score", 1.0))
                if raw_score < 1.0: raw_score = 1.0
                normalized_score = raw_score / 10.0
                reason = data.get("reason", "")
                
                valid_results.append((normalized_score, reason))
            except Exception as e:
                logger.warning(f"Attempt {i+1} failed: {e}")

        if not valid_results:
             return 0.0, "All attempts failed."

        best_score, best_reason = max(valid_results, key=lambda x: x[0])
        self.cache.set("PerspectrumLLMEvaluator", text, target, best_score, best_reason)
        return best_score, best_reason
