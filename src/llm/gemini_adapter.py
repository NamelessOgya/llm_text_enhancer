import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .interface import LLMInterface

logging.basicConfig(level=logging.INFO)
# クライアントライブラリからの冗長なログを抑制
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class GeminiAdapter(LLMInterface):
    """
    Google Gemini APIを使用するためのアダプタークラス。
    `google-genai` 公式SDK (v1.0以降) を使用して実装されている。
    """
    def __init__(self, model_name: str):
        """
        Args:
            model_name (str): 使用するGeminiモデル名 (例: "gemini-1.5-pro")
        
        Raises:
            ValueError: model_nameが空、またはAPIキーが環境変数に設定されていない場合
        """
        if not model_name:
            raise ValueError("model_name must be specified.")
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Geminiのトークン使用量管理:
        # レスポンスに含まれる usage_metadata を使用して、リクエストごとの消費量を累積する。
        # 注意: SDKが返すカウントは課金上のトークン数と完全に一致しない場合があるが、目安として使用する。
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # エラーハンドリング:
        # 通信エラーや一時的なサーバーエラー(5xx)に対してリトライを行う。
        # 本来は google.api_core.exceptions 等の具体的な例外クラスを指定すべきだが、
        # 依存関係の複雑さを避けるため、ここでは広範な Exception を捕捉している点に留意。
        retry=retry_if_exception_type(Exception) 
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Geminiモデルを使用してテキストを生成する。
        
        システムプロンプトの実装について:
        GenerativeModelの初期化時に system_instruction を設定する方法もあるが、
        呼び出しごとに動的に変更する柔軟性を持たせるため、ここではユーザープロンプトの
        前段に "System: ..." として埋め込む疑似的な手法を採用している。
        これはステートレスな運用においてシンプルかつ堅牢である。

        Args:
            prompt (str): 入力プロンプト
            system_prompt (Optional[str]): システムプロンプト (オプション)
            **kwargs: 生成パラメータ (temperatureなど)
            
        Returns:
            str: 生成されたテキスト
        """
        
        full_content = prompt
        if system_prompt:
             # システムプロンプトの埋め込み処理
             full_content = f"System: {system_prompt}\n\nUser: {prompt}"

        try:
            # GenerateContentConfigを用いたパラメータ設定
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_content,
                config=types.GenerateContentConfig(
                    candidate_count=1,
                    temperature=kwargs.get("temperature", 0.7),
                    response_mime_type=kwargs.get("response_mime_type"),
                    response_schema=kwargs.get("response_schema")
                )
            )
            
            # 生成候補(candidates)の有無を確認
            if not response.candidates:
                logger.warning("Gemini returned no candidates (blocked info logic likely triggered).")
                return ""

            # テキスト抽出
            text = response.text
            
            # 使用量メタデータの取得と集計
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
