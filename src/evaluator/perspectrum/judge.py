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

from utils import load_taml_sections

class PerspectrumLLMEvaluator(Evaluator):
    """
    LLM-as-a-judge for Perspectrum.
    """
    def __init__(self, llm: LLMInterface, config_path: str = "config/task/perspectrum.yaml", prompt_path: str = None):
        self.llm = llm
        self.llm_eval_repeat = 3
        # Use helper from rules.py via instantiation
        self._rule_checker = PerspectrumRuleEvaluator(config_path)
        self.cache = EvaluationCache()
        
        # Determine prompt path
        self.prompt_path = prompt_path
        if not self.prompt_path or not os.path.exists(self.prompt_path):
             # Default
             self.prompt_path = "config/definitions/prompts/judge.taml"

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
        
        # If score is 0.0 and reason is set, it's a violation.
        if score == 0.0 and reason != "":
             self.cache.set("PerspectrumLLMEvaluator", text, target, 0.0, reason)
             return 0.0, reason
        
        # Load Prompt from TAML
        raw_prompt = ""
        sections = load_taml_sections(self.prompt_path)
        if "judge_prompt" in sections:
            raw_prompt = sections["judge_prompt"]
        elif "content" in sections:
            raw_prompt = sections["content"]
        
        if not raw_prompt:
             # Fallback to hardcoded if really missing, but warn
             logger.warning(f"No prompt found in {self.prompt_path}, using hardcoded fallback.")
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

Output JSON format:
{{
    "score": <int 1-10>,
    "reason": "<string>"
}}
"""
        
        # Format Prompt
        # Handle simple replace or format safely
        formatted_prompt = raw_prompt.replace("{target}", target).replace("{text}", text)

        valid_results = []
        for i in range(self.llm_eval_repeat):
            try:
                response = self.llm.generate(formatted_prompt, temperature=0.0)
                cleaned = response.strip()
                
                # Attempt to extract JSON from code blocks first
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                             part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                             cleaned = part
                             break
                
                # Fallback: Extract from first { to last }
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    cleaned = cleaned[start_idx : end_idx + 1]
                
                data = json.loads(cleaned)
                raw_score = float(data.get("score", 1.0))
                if raw_score < 1.0: raw_score = 1.0
                normalized_score = raw_score / 10.0
                reason = data.get("reason", "")
                
                valid_results.append((normalized_score, reason))
            except Exception as e:
                logger.warning(f"Attempt {i+1} failed: {e}. Raw response: {response}")

        if not valid_results:
             return 0.0, "All attempts failed."

        best_score, best_reason = max(valid_results, key=lambda x: x[0])
        self.cache.set("PerspectrumLLMEvaluator", text, target, best_score, best_reason)
        return best_score, best_reason
