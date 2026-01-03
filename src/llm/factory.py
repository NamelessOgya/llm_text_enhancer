from typing import Optional
from .interface import LLMInterface
from .openai_adapter import OpenAIAdapter
from .dummy_adapter import DummyAdapter
from .gemini_adapter import GeminiAdapter

def get_llm_adapter(adapter_type: str, model_name: str) -> LLMInterface:
    """
    Factory function to get the appropriate LLM adapter based on the adapter type.
    
    Args:
        adapter_type: The type of adapter (e.g., 'openai', 'gemini', 'dummy').
        model_name: The name of the model to be passed to the adapter.
        
    Returns:
        An instance of LLMInterface.
    """
    if adapter_type == "dummy":
        return DummyAdapter()
    elif adapter_type == "gemini":
        return GeminiAdapter(model_name=model_name)
    elif adapter_type == "openai":
        return OpenAIAdapter(model_name=model_name)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
