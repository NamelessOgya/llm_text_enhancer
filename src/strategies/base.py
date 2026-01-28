import logging
import os
import random
import glob
import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

import yaml

from llm.interface import LLMInterface
from utils import load_taml_sections

logger = logging.getLogger(__name__)

class EvolutionStrategy(ABC):
    """
    プロンプト進化戦略の基底クラス (Abstract Base Class)。
    全ての手法はこのクラスを継承し、evolveメソッドを実装する必要がある。
    """
    def __init__(self):
        self.prompts = {}
    
    def _ensure_prompts(self, strategy_name: str, context: Dict[str, Any] = None):
        if not self.prompts:
            # Default path (root of prompts)
            path = os.path.join(os.getcwd(), "config", "definitions", "prompts", f"{strategy_name}.taml")
            
            # Task-specific path if available
            if context and "task_name" in context:
                task_name = context["task_name"]
                task_path = os.path.join(os.getcwd(), "config", "definitions", "prompts", task_name, f"{strategy_name}.taml")
                if os.path.exists(task_path):
                    path = task_path
            
            self.prompts = load_taml_sections(path)
            
    @abstractmethod
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        個体群を進化させ、次世代のテキストを生成する。

        Args:
            llm (LLMInterface): テキスト生成に使用するLLMインターフェイス。
            population (List[Dict[str, Any]]): 現在の世代の全個体データ。
                                               各要素は {"prompt": str, "text": str, "score": float, "reason": str, ...} の形式。
                                               ※ "prompt" キーは互換性のため残るが、本質的には "text" が主役となる。
            k (int): 生成すべき次世代の個体数 (Population Size)。
            task_def (str): タスクの定義文 (最適化対象のドメインや制約)。
            context (Dict[str, Any]): その他の環境情報 (result_dir: 結果出力先, iteration: 現在の世代数など)。

        Returns:
            List[Tuple[str, str]]: 次世代のテキストとその生成ロジック(メタプロンプト)のペアのリスト。
                                   [(new_text, creation_meta_prompt), ...]
        """
        pass

    def _get_max_words(self, context: Dict[str, Any]) -> int:
        """
        config/task/{task_name}.yaml から max_words 設定を取得する。
        設定がない場合、または読み込み失敗時は 0 (制限なし) を返す。
        """
        task_name = context.get("task_name")
        if not task_name:
            return 0
            
        config_path = os.path.join(os.getcwd(), "config", "task", f"{task_name}.yaml")
        if not os.path.exists(config_path):
            return 0
            
        try:
            with open(config_path, 'r') as f:
                task_config = yaml.safe_load(f)
                max_words = task_config.get("max_words")
                if max_words and isinstance(max_words, int):
                    return max_words
        except:
            pass
            
        return 0

    def _get_task_constraint(self, context: Dict[str, Any]) -> str:
        """
        プロンプトに追加する制約文字列を返す。
        """
        max_words = self._get_max_words(context)
        if max_words > 0:
            # Buffer Strategy: Subtract 2 to account for labels like "SUPPORT:"
            safe_max = max(0, max_words - 2)
            return f"CRITICAL CONSTRAINT: The text MUST be strictly {safe_max} words or less. Violations receive 0 score."
        return ""

    def generate_initial(self, llm: LLMInterface, k: int, task_def: str, context: Dict[str, Any] = None) -> List[Tuple[str, str]]:
        """
        初期世代(Generation 0)のテキスト個体群を生成する。
        デフォルト実装(GeneticAlgorithmなど)ではタスク定義に基づき、多様なテキスト案をゼロベースで生成する。
        """
        logger.info("Generating initial population (Texts) using Default Strategy...")
        
        # Load Prompts for Default/Base
        self._ensure_prompts("default", context)

        constraint = self._get_task_constraint(context)
        
        # Format Creation Prompt
        try:
            creation_prompt = self.prompts["creation_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                k=k
            )
        except KeyError as e:
            logger.error(f"Missing prompt key 'creation_prompt' in default strategy: {e}")
            # Fallback simple prompt
            creation_prompt = f"{constraint}\nTask: {task_def}\nGenerate {k} items."

        response = llm.generate(creation_prompt)
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.split("\n", 1)[1]
                if cleaned_response.rfind("```") != -1:
                    cleaned_response = cleaned_response[:cleaned_response.rfind("```")]
            
            text_list = json.loads(cleaned_response)
            
            if isinstance(text_list, list):
                texts = [(str(t).strip(), creation_prompt) for t in text_list if str(t).strip()]
            else:
                logger.error(f"LLM returned invalid JSON structure: {type(text_list)}")
                texts = []
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response}. Falling back to line splitting.")
            lines = response.split('\n')
            texts = []
            for line in lines:
                line = line.strip()
                if not line or line.lower().startswith("here are") or line.startswith("```"):
                    continue
                if line[0].isdigit() and ". " in line[:5]:
                    line = line.split(". ", 1)[1]
                elif line.startswith("- "):
                    line = line[2:]
                
                texts.append((line, creation_prompt))

        # 数合わせ
        texts = texts[:k]
        while len(texts) < k:
            constraint = self._get_task_constraint(context)
            try:
                single_prompt = self.prompts["single_prompt"].format(
                    constraint=constraint,
                    task_def=task_def
                )
            except KeyError:
                single_prompt = f"{constraint}\nTask: {task_def}\nGenerate 1 item."
                
            single_text = llm.generate(single_prompt).strip()
            texts.append((single_text, single_prompt))
            
        return texts

    def _generate_with_retry(self, llm: LLMInterface, prompt: str, context: Dict[str, Any], max_retries: int = 3) -> str:
        """
        Generates text with retry logic if validation fails (e.g. word count).
        This is a helper to enforce constraints during evolution.
        """
        max_words = self._get_max_words(context)
        if max_words == 0:
            return llm.generate(prompt).strip()
            
        # Buffer Strategy
        safe_max = max(0, max_words - 2)
        
        for _ in range(max_retries):
            text = llm.generate(prompt).strip()
            # Simple word count check
            if len(text.split()) <= safe_max:
                return text
            # If fail, maybe append constraint again? (Not implemented here to keep simple)
            
        return text # Return best effort
