from typing import Optional
from .interface import LLMInterface
from .openai_adapter import OpenAIAdapter
from .dummy_adapter import DummyAdapter
from .gemini_adapter import GeminiAdapter
from .azure_openai_adapter import AzureOpenAIAdapter

def get_llm_adapter(adapter_type: str, model_name: str) -> LLMInterface:
    """
    指定されたアダプタータイプに基づいて、適切なLLMアダプターインスタンスを生成して返すFactory関数。
    
    Args:
        adapter_type (str): アダプターの種類識別子 ('openai', 'gemini', 'dummy')
        model_name (str): インスタンス化するモデル名 (各アダプターのコンストラクタに渡される)
        
    Returns:
        LLMInterface: 初期化されたLLMアダプターインスタンス
        
    Raises:
        ValueError: サポートされていない adapter_type が指定された場合に発生する
    """
    if adapter_type == "dummy":
        return DummyAdapter()
    elif adapter_type == "gemini":
        return GeminiAdapter(model_name=model_name)
    elif adapter_type == "openai":
        return OpenAIAdapter(model_name=model_name)
    elif adapter_type == "azure_openai":
        return AzureOpenAIAdapter(model_name=model_name)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
