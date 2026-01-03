import logging
import re
from typing import Tuple
from .interface import Evaluator
from llm.interface import LLMInterface

logger = logging.getLogger(__name__)

class LLMEvaluator(Evaluator):
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
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
            
            # Simple parsing
            score_match = re.search(r'Score:\s*(\d+(\.\d+)?)', response_text, re.IGNORECASE)
            reason_match = re.search(r'Reason:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
            
            if score_match:
                score = float(score_match.group(1))
            else:
                 # Fallback: try to find just a number if format failed
                match = re.search(r'\d+(\.\d+)?', response_text)
                score = float(match.group()) if match else 0.0
                
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # If reason format missed, use the whole text if it's not just a number
                if not score_match: 
                     reason = response_text
            
        except Exception as e:
            logger.error(f"Error evaluating text: {e}")
            score = 0.0
            reason = f"Error: {e}"
            
        return score, reason
