from abc import ABC, abstractmethod
from typing import Tuple

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, text: str, target: str) -> Tuple[float, str]:
        """
        Evaluates the text against the target preference.
        Returns:
            - score (float): from 0.0 to 10.0
            - reason (str): Explanation for the score (optional)
        """
        pass
