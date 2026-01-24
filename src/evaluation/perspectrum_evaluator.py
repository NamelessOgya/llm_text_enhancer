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
    Evaluates semantic closeness of argument.
    """
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        # target in evaluate() usually comes from `target_preference` argument or dataset reference.
        # In this task, target is effectively the "User's Reference Perspective"
        
        prompt = f"""
You are an objective judge evaluating how well a generated perspective matches a reference perspective.

Reference Perspective:
"{target}"

Generated Perspective:
"{text}"

Task:
Compare the Generated Perspective against the Reference Perspective.
Determine if the Generated Perspective shares the same core argument and stance as the Reference Perspective.

Evaluation Criteria:
1. **Stance Match**: Must strictly match the reference stance (SUPPORT vs OPPOSE).
2. **Key Points Coverage**: The generated text must cover the main logical points present in the reference.
3. **No Added Claims**: The generated text must NOT contain new arguments or claims that are absent in the reference (hallucinations/extra info).
4. **Semantic Accuracy**: Focus on meaning, not just word overlap.

Scoring Scale (1-5):
1: **Stance Mismatch / Irrelevant**. The text opposes the reference or argues about a completely different topic.
2: **Weak Match**. Stance matches, but misses the core argument or consists mostly of irrelevant info.
3: **Partial Match**. Captures part of the core argument but misses key nuance OR includes significant extra claims/noise not in reference.
4: **Good Match**. Captures the main argument well. May miss minor details or have very slight extra info.
5: **Perfect Match**. Perfectly captures the core argument and nuance of the reference. Contains NO extra/unsupported claims.

Output Format:
First, provide a step-by-step "Review" comparing the two texts based on the criteria.
Then, output the final "score" (1-5) and a short "reason".

Output JSON format:
{{
    "review": "<detailed comparison>",
    "score": <int 1-5>,
    "reason": "<short summary>"
}}
"""
        response = self.llm.generate(prompt)
        try:
            import json
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.rfind("```") != -1:
                    cleaned = cleaned[:cleaned.rfind("```")]
            data = json.loads(cleaned)
            data = json.loads(cleaned)
            raw_score = float(data.get("score", 0.0))
            # Normalize 1-5 scale to 0.0-1.0
            # 1->0.0, 2->0.25, 3->0.5, 4->0.75, 5->1.0 ? 
            # Or simply raw_score / 5.0? 
            # Let's map 1 (lowest valid) to significantly low, but 0 is usually reserved for failure/mismatch.
            # If 1 is "Stance Mismatch", it should probably be 0.0.
            if raw_score <= 1.0:
                normalized_score = 0.0
            else:
                # Map 2..5 to 0.25..1.0
                # (x - 1) / 4
                normalized_score = (raw_score - 1) / 4.0
            
            return normalized_score, data.get("reason", "")
        except Exception as e:
            logger.warning(f"PerspectrumLLMEvaluator parse failed: {e}", exc_info=True)
            return 0.0, f"[EXCEPTION HANDLED] Parse Error: {e}"
