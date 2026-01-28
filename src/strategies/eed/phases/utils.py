import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_logic_json(
    phase: str,
    parent_text: str,
    instruction: str,
    meta: Dict[str, Any] = None
) -> str:
    """
    生成ロジック(メタプロンプト)のJSON文字列表現を作成するヘルパー
    """
    data = {
        "EED_Phase": phase,
        "Parent_Excerpt": parent_text,
        "Instruction": instruction,
        "Meta": meta or {}
    }
    return json.dumps(data, indent=2, ensure_ascii=False)
