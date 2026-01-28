import os
import yaml
import logging
from typing import Tuple, Optional
from ..interface import Evaluator
from ..rule_evaluator import MeteorRuleEvaluator
from .utils import parse_stance_reason

logger = logging.getLogger(__name__)

class PerspectrumRuleEvaluator(Evaluator):
    """
    Perspectrum v1.1 Rule Evaluator.
    Checks:
    1. Word count limit (max_words)
    2. Stance consistency (SUPPORT/OPPOSE)
    3. METEOR score for valid responses
    """
    def __init__(self, config_path: str = "config/task/perspectrum.yaml"):
        self.max_words = 50 # Default
        self.meteor_evaluator = MeteorRuleEvaluator()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and "max_words" in config:
                        self.max_words = int(config["max_words"])
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        try:
            # 1. Parse Candidate
            c_stance, c_reason = parse_stance_reason(text)
            if not c_stance:
                return 0.0, "Invalid format (No STANCE: prefix)"
                
            # 2. Parse Reference
            r_stance, r_reason = parse_stance_reason(target)
            
            # 3. Check Stance
            if r_stance and c_stance != r_stance:
                return 0.0, f"Stance mismatch. Expected {r_stance}, got {c_stance}."

            # 4. Check Word Count
            word_count = len(c_reason.split())
            if word_count > self.max_words:
                return 0.0, f"Word count limit exceeded. {word_count} > {self.max_words}"

            # 5. METEOR Score
            ref_text = r_reason if r_stance else target
            return self.meteor_evaluator.evaluate(c_reason, ref_text)
            
        except Exception as e:
            logger.error(f"PerspectrumRuleEvaluator error: {e}", exc_info=True)
            return 0.0, f"[EXCEPTION HANDLED] Evaluator Error: {e}"
