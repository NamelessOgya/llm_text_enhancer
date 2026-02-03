import os
import logging
import openai
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .interface import LLMInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureOpenAIAdapter(LLMInterface):
    """
    Azure OpenAI Service を使用するためのアダプタークラス。
    `openai.AzureOpenAI` クライアントを利用して実装されている。
    """
    def __init__(self, model_name: str):
        """
        Args:
            model_name (str): Azure OpenAI のデプロイメント名 (deployment_name)
        """
        load_dotenv()
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not api_key or not endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in environment variables.")

        self.model_name = model_name
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError, openai.APIError))
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Azure OpenAI Chat Completion APIを使用してテキストを生成する。
        
        Args:
            prompt (str): ユーザーメッセージ
            system_prompt (Optional[str]): システムメッセージ
            **kwargs: その他のAPIパラメータ (temperature, max_tokensなど)
            
        Returns:
            str: 生成された応答テキスト
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            usage = response.usage
            if usage:
                self.token_usage["prompt_tokens"] += usage.prompt_tokens
                self.token_usage["completion_tokens"] += usage.completion_tokens
                self.token_usage["total_tokens"] += usage.total_tokens

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error extracting response from Azure OpenAI: {e}")
            raise e

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage
