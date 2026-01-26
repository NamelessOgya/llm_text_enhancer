import os
import json
import hashlib
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

class EvaluationCache:
    """
    Evaluatorの結果をファイルにキャッシュするためのシンプルなクラス。
    tmp/evaluator_cache.json に保存する。
    """
    def __init__(self, cache_file: str = "tmp/evaluator_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
                self.cache = {}
        else:
            self.cache = {}
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_file}: {e}")

    def _generate_key(self, evaluator_name: str, text: str, target: str) -> str:
        """
        ユニークなキャッシュキーを生成する
        """
        # 内容のハッシュを取ってキーにする（長い文字列をそのままキーにしないため）
        content = f"{evaluator_name}|{text}|{target}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, evaluator_name: str, text: str, target: str) -> Optional[Tuple[float, str]]:
        """
        キャッシュから値を取得する。
        """
        key = self._generate_key(evaluator_name, text, target)
        if key in self.cache:
            data = self.cache[key]
            # data structure: {"score": float, "reason": str}
            return data.get("score", 0.0), data.get("reason", "")
        return None

    def set(self, evaluator_name: str, text: str, target: str, score: float, reason: str):
        """
        キャッシュに値を保存する。
        """
        key = self._generate_key(evaluator_name, text, target)
        self.cache[key] = {
            "score": score,
            "reason": reason,
            # metadata for debugging if needed
            "evaluator": evaluator_name,
            # "text_preview": text[:50], 
            # "target_preview": target[:50]
        }
        self._save_cache()
