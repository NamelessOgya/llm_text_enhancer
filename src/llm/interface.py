from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """
    Abstract base class for LLM interactions.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generates text based on the given prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional model parameters (e.g., temperature, max_tokens).

        Returns:
            The generated text response.
        """
        pass

    @abstractmethod
    def get_token_usage(self) -> Dict[str, int]:
        """
        Returns the accumulated token usage statistics.

        Returns:
            A dictionary containing token usage (e.g., {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}).
        """
        pass
