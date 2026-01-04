from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """
    LLM (Large Language Model) 操作のための抽象基底クラス。
    全てのLLMアダプター(OpenAI, Gemini, Dummy等)はこのインターフェースを継承・実装しなければならない。
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        指定されたプロンプトに基づいてテキスト生成を行う。

        Args:
            prompt (str): ユーザーからの入力プロンプト
            system_prompt (Optional[str]): システムプロンプト (役割設定や前提条件の指示など)
            **kwargs: その他のモデル固有パラメータ (例: temperature, max_tokens, stop_sequences)

        Returns:
            str: 生成されたテキスト文字列 (前後の空白除去や正規化は実装側に委ねられる)
        """
        pass

    @abstractmethod
    def get_token_usage(self) -> Dict[str, int]:
        """
        インスタンス生成から現在までの累積トークン使用量を取得する。

        Returns:
            Dict[str, int]: トークン使用量の辞書
            期待されるキー:
                - "prompt_tokens": 入力トークン数
                - "completion_tokens": 出力トークン数
                - "total_tokens": 合計トークン数
        """
        pass
