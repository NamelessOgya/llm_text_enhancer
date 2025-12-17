from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, text: str, target: str) -> float:
        """
        Evaluates the text against the target preference.
        Returns a score from 0.0 to 10.0.
        """
        pass
