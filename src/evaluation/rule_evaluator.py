import re
from typing import Optional
from .interface import Evaluator

class KeywordRuleEvaluator(Evaluator):
    def evaluate(self, text: str, target: str) -> float:
        """
        Evaluates by checking overlap of words between text and target preference.
        """
        if not target:
            return 0.0
            
        target_words = set(target.lower().split())
        text_words = set(text.lower().split())
        
        if not target_words:
            return 0.0
            
        matches = target_words.intersection(text_words)
        score = (len(matches) / len(target_words)) * 10.0
        
        return min(max(score, 0.0), 10.0)

class RegexRuleEvaluator(Evaluator):
    def evaluate(self, text: str, target: str) -> float:
        """
        Evaluates if the text matches the given regex pattern in target.
        Score is 10.0 if match found, 0.0 otherwise.
        """
        if not target:
            return 0.0
            
        try:
            if re.search(target, text, re.IGNORECASE):
                return 10.0
        except re.error:
            pass
            
        return 0.0

def get_rule_evaluator(rule_type: str) -> Optional[Evaluator]:
    if rule_type == "rule_keyword" or rule_type == "rule_based": # Backward compatibility
        return KeywordRuleEvaluator()
    elif rule_type == "rule_regex":
        return RegexRuleEvaluator()
    return None
