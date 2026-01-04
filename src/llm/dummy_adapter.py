import random
from typing import Dict, Optional
from .interface import LLMInterface

class DummyAdapter(LLMInterface):
    """
    テスト用のダミーアダプタークラス。
    LLM APIを呼び出さずに、ランダムなポケモン名やスコアを返すことで、
    パイプラインの動作確認やデバッグを低コストで行うために使用する。
    
    APIコールが発生しないため、ネットワーク遮断環境でのテストや、
    ロジック部分の単体テストにおけるモックとしても機能する。
    """
    def __init__(self):
        self.pokemon_names = [
            "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Eevee", 
            "Snorlax", "Gengar", "Jigglypuff", "Meowth", "Psyduck",
            "Magikarp", "Gyarados", "Lapras", "Ditto", "Mewtwo"
        ]
        # ダミーのトークン使用量 (固定値)
        self.token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        プロンプトの内容に応じて、それらしいダミーレスポンスを返す。
        
        Args:
            prompt (str): 入力プロンプト
            system_prompt (Optional[str]): システムプロンプト (無視される)
            **kwargs: その他 (無視される)
            
        Returns:
            str: ダミー生成テキスト
                 - スコア要求("Score from 0 to 10")が含まれる場合: ランダムな数字
                 - Mutation要求("Rewrite")が含まれる場合: "Mutated <Pokemon>"
                 - 初期生成("Generate X prompts")の場合: X行のテキスト
                 - その他: ランダムなポケモン名を含む文
        """
        # 評価プロンプトの場合 (スコア要求を検知)
        if "Provide a score from 0 to 10" in prompt:
            return str(random.randint(0, 10))
        
        # 変異プロンプトの場合 (Mutation)
        if "Rewrite it to be even more effective" in prompt:
             return f"Mutated {random.choice(self.pokemon_names)}"

        # デフォルト生成 (初期生成時など)
        # 複数行の出力を要求されているか簡易判定
        if "Generate" in prompt and "prompts" in prompt:
            # 正規表現で要求数Kを取得
            import re
            match = re.search(r'Generate (\d+) distinct prompts', prompt)
            k = int(match.group(1)) if match else 3
            return "\n".join([f"Generate text for {random.choice(self.pokemon_names)}" for _ in range(k)])

        # 単一生成
        return f"This is a {random.choice(self.pokemon_names)}."

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage
