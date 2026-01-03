import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .interface import LLMInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAdapter(LLMInterface):
    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("model_name must be specified.")
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Token usage tracking for Gemini is per-request, so we accumulate it.
        # Note: accurate tokenizer count might differ slightly from billable, 
        # but we use usage metadata from response if available.
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # Detect relevant exceptions for Gemini. 
        # Typically grpc errors or GoogleAPIError, but for simplicity catching broad Exception
        # inside the method or specific ones if known. 
        # google.api_core.exceptions.ResourceExhausted is common.
        retry=retry_if_exception_type(Exception) 
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # Gemini handling of system prompt:
        # "system_instruction" can be passed during GenerativeModel init or as part of chat history?
        # For simplicity in this interface, we can prepend it or use the model's system_instruction capability if we re-init.
        # However, re-init is expensive. 
        # current google-generativeai allows system_instruction in GenerativeModel constructor.
        # If we need dynamic system prompt per call, we might need to prepend it to the user prompt 
        # or use a chat session with system instruction if supported effectively.
        # A simple robust way for single-turn is prepending.
        
        full_content = prompt
        if system_prompt:
             # Prepending system prompt as Gemini doesn't always support per-call system instruction easily without re-init
             # or using the beta API. 
             full_content = f"System: {system_prompt}\n\nUser: {prompt}"

        try:
            # Set generation config from kwargs if needed
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                # map common kwargs if needed, e.g. temperature
                temperature=kwargs.get("temperature", 0.7) 
            )

            response = self.model.generate_content(
                full_content, 
                generation_config=generation_config
            )
            
            # Response validation
            if not response.parts:
                logger.warning("Gemini returned no parts.")
                return ""

            # text extraction
            text = response.text
            
            # Usage metadata (if available in the response object)
            # Currently the python SDK object `response` has `usage_metadata`
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                self.token_usage["prompt_tokens"] += usage.prompt_token_count
                self.token_usage["completion_tokens"] += usage.candidates_token_count
                self.token_usage["total_tokens"] += usage.total_token_count
            
            return text.strip()

        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            raise e

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage
