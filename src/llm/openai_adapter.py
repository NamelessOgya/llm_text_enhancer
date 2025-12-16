import os
import time
import logging
import openai
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .interface import LLMInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIAdapter(LLMInterface):
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            logger.error(f"Error extracting response from OpenAI: {e}")
            raise e

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage
