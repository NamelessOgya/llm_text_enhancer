import os
import time
import logging
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .interface import LLMInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIAdapter(LLMInterface):
    """
    OpenAI API (GPT-4など) を使用するためのアダプタークラス。
    公式の `openai` Python SDKを利用して実装されている。
    """
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Args:
            model_name (str): 使用するモデル名 (デフォルト: gpt-4o)
        """
        load_dotenv()
        self.model_name = model_name
        # APIキーは環境変数 OPENAI_API_KEY から自動的に読み込まれる
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # エラーハンドリング:
        # OpenAI固有の一時的なネットワークエラーやレート制限エラーを捕捉してリトライする。
        # これにより、APIの不安定性に対する耐性を高めている。
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError, openai.APIError))
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        OpenAI Chat Completion APIを使用してテキストを生成する。
        
        Args:
            prompt (str): ユーザーメッセージ
            system_prompt (Optional[str]): システムメッセージ (指示など)
            **kwargs: その他のAPIパラメータ (temperature, max_tokensなど。API呼び出しに直接渡される)
            
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
            
            # 使用トークン数の集計
            # レスポンスオブジェクトの usage フィールドから正確な消費量を取得する。
            usage = response.usage
            if usage:
                self.token_usage["prompt_tokens"] += usage.prompt_tokens
                self.token_usage["completion_tokens"] += usage.completion_tokens
                self.token_usage["total_tokens"] += usage.total_tokens

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error extracting response from OpenAI: {e}")
            raise e

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage
