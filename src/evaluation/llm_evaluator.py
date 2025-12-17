import logging
import re
from .interface import Evaluator
from llm.interface import LLMInterface

logger = logging.getLogger(__name__)

class LLMEvaluator(Evaluator):
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def evaluate(self, text: str, target: str) -> float:
        prompt = f"""
        Rate how well the following text matches the description/preference: "{target}".
        Text: "{text}"
        
        Provide a score from 0 to 10 (10 being perfect match). Return ONLY the numeric score.
        """
        
        try:
            score_str = self.llm.generate(prompt).strip()
            # extract number if extra text exists
            match = re.search(r'\d+(\.\d+)?', score_str)
            score = float(match.group()) if match else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating text: {e}")
            score = 0.0
            
        return score
