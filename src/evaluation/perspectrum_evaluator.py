import os
import yaml
import logging
from typing import Tuple, Dict, Any, Optional
from .interface import Evaluator
from .rule_evaluator import MeteorRuleEvaluator
from llm.interface import LLMInterface

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
        
        # Load config if exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and "max_words" in config:
                        self.max_words = int(config["max_words"])
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        else:
             # Try absolute path resolution if relative path fails (e.g. running from deeper dir)
            try:
                # Assuming standard project structure: [root]/config/task/perspectrum.yaml
                # But allowing for flexibility if env vars or other means are used.
                # For now just warn.
                logger.warning(f"Config file not found at {config_path}. Using default max_words={self.max_words}")
            except: pass

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        """
        Args:
            text (str): Candidate text (Format: "STANCE: Reason")
            target (str): Reference text (Format: "STANCE: Reason")
            
        Returns:
            Tuple[float, str]: (Score, Reason)
        """
        try:
            # 1. Parse Candidate
            c_stance, c_reason = self._parse_stance_reason(text)
            if not c_stance:
                return 0.0, "Invalid format (No STANCE: prefix)"
                
            # 2. Parse Reference
            # Reference might be just text if it comes from raw dataset without formatting, 
            # but our prepare_perspectrum_data.py formats it as "STANCE: text"
            r_stance, r_reason = self._parse_stance_reason(target)
            if not r_stance:
                # Fallback: if reference doesn't have stance prefix, we can't strict check stance
                logger.warning(f"Reference does not have stance prefix: {target[:20]}...")
                pass

            # 3. Check Stance
            if r_stance and c_stance != r_stance:
                return 0.0, f"Stance mismatch. Expected {r_stance}, got {c_stance}."

            # 4. Check Word Count (on reason part)
            word_count = len(c_reason.split())
            if word_count > self.max_words:
                return 0.0, f"Word count limit exceeded. {word_count} > {self.max_words}"

            # 5. METEOR Score on Reason part
            # If reference has no stance, treat whole target as reason
            ref_text = r_reason if r_stance else target
            
            score, reason = self.meteor_evaluator.evaluate(c_reason, ref_text)
            return score, reason
        except Exception as e:
            logger.error(f"PerspectrumRuleEvaluator error: {e}", exc_info=True)
            return 0.0, f"[EXCEPTION HANDLED] Evaluator Error: {e}"

    def _parse_stance_reason(self, text: str) -> Tuple[Optional[str], str]:
        """
        Parses "STANCE: Reason" string.
        Returns upper-case STANCE and trimmed Reason.
        """
        # Supported stances: SUPPORT, OPPOSE
        text = text.strip()
        parts = text.split(":", 1)
        if len(parts) < 2:
            return None, text # Cannot parse
            
        potential_stance = parts[0].strip().upper()
        content = parts[1].strip()
        
        if potential_stance in ["SUPPORT", "OPPOSE"]:
            return potential_stance, content
            
        return None, text

class PerspectrumLLMEvaluator(Evaluator):
    """
    LLM-as-a-judge for Perspectrum.
    Evaluates:
    1. Rule-based Constraints (Hard Filter -> 0.0):
       - Stance Mismatch
       - Word Count Violation
    2. Logical Basis Match (LLM -> 0.0 - 1.0):
       - Precision/Recall of logical points (Evidence).
       - Ignoring style/ambiguity.
    """
    def __init__(self, llm: LLMInterface, config_path: str = "config/task/perspectrum.yaml"):
        self.llm = llm
        self.max_words = 50 # Default
        
        # Reuse parsing logic from RuleEvaluator
        # We can instantiate it or copy logic. Instantiating is cleaner if we want to share config loading.
        # But PerspectrumRuleEvaluator takes config_path.
        self._rule_checker = PerspectrumRuleEvaluator(config_path)
        self.max_words = self._rule_checker.max_words

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        # 1. Rule-based Pre-check
        # We use the rule evaluator's helper or logic
        score, reason = self._check_rules(text, target)
        if score == 0.0 and reason != "":
             return 0.0, reason

        # 2. LLM Evaluation (Logical Basis Match)
        prompt = f"""
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
- **Mid/Low Scores (1-7)**: Differentiate based on **Missing** or **Extra** logical points (Evidence recall/precision).
- **IGNORE**: Fluency or writing style (unless it obscures logic).

Scoring Scale (1-10):
(Note: Minimum score for a valid text is 1. Score 0 is reserved for rule violations.)

**High Tier (Logically Correct)**:
- 10: **Perfect Match**. Evidence, Granularity (level of detail), and Nuance (tone/emphasis) are identical to the reference.
- 9: **Near Perfect**. Evidence is identical. Slight difference in **Nuance** (e.g. stronger/weaker emphasis) or **Granularity** (very slightly more/less detailed).
- 8: **Excellent**. Evidence is identical. Clear difference in **Granularity** (e.g. reference is specific vs generated is abstract, or vice versa).

**Mid Tier (Minor Logical Issues)**:
- 7: **Good**. Main points match. Contains one **minor extra point** (harmless context/noise) or misses one **minor detail**.
- 6: **Fair**. Main points match. Contains **clear extra points** (distracting) or misses a **secondary logical point**.
- 5: **Mediocre**. Partial match (~50%). Mix of correct points and unrelated points/interpretations.

**Low Tier (Major Logical Issues)**:
- 4: **Weak**. Misses the **Core Argument** (matches only side points). Or dominated by unrelated arguments.
- 3: **Poor**. Stance matches, but the argument is largely unrelated to the reference.
- 2: **Very Poor**. Stance matches, but no logical connection to reference found.
- 1: **Mismatch**. Passed rule checks but logic is effectively completely different.

Output Format:
1. "Reference Points": [List of points]
2. "Generated Points": [List of points]
3. "Analysis": Compare Evidence, then Nuance/Granularity.
4. "Final Score": 1-10 integer.
5. "Reason": Concise summary.

Output JSON format:
{{
    "reference_points": [...],
    "generated_points": [...],
    "analysis": "<string>",
    "score": <int 1-10>,
    "reason": "<string>"
}}
"""
        response = self.llm.generate(prompt, temperature=0.0)
        try:
            import json
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.rfind("```") != -1:
                    cleaned = cleaned[:cleaned.rfind("```")]
            data = json.loads(cleaned)
            
            raw_score = float(data.get("score", 1.0))
            
            # Clamp to min 1.0 just in case LLM hallucinates 0
            if raw_score < 1.0:
                raw_score = 1.0
            
            # Normalize 1-10 to 0.1-1.0
            normalized_score = raw_score / 10.0
            
            reason = data.get("reason", "")
            
            return normalized_score, reason

        except Exception as e:
            logger.warning(f"PerspectrumLLMEvaluator LLM parse failed: {e}", exc_info=True)
            return 0.0, f"[Exception] LLM Parse Error: {e}"

    def _check_rules(self, text: str, target: str) -> Tuple[float, str]:
        """
        Reuse PerspectrumRuleEvaluator logic for parsing and constraints.
        Returns (1.0, "") if pass, (0.0, reason) if fail.
        """
        # Parse
        c_stance, c_reason = self._rule_checker._parse_stance_reason(text)
        r_stance, r_reason = self._rule_checker._parse_stance_reason(target)
        
        if not c_stance:
            return 0.0, "Invalid format (No STANCE: prefix)"
        
        # Check Stance
        # If reference has no stance, we skip strict stance check (or assume support? strict match usually requires ref to have stance)
        if r_stance and c_stance != r_stance:
            return 0.0, f"Stance mismatch. Expected {r_stance}, got {c_stance}."
            
        # Check Word Count
        word_count = len(c_reason.split())
        if word_count > self.max_words:
             return 0.0, f"Word count limit exceeded. {word_count} > {self.max_words}"
             
        return 1.0, ""
