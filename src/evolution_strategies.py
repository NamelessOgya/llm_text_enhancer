import logging
import random
import os
import glob
import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    CountVectorizer = None
    cosine_similarity = None

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

    def _generate_with_retry(self, llm: LLMInterface, prompt: str, context: Dict[str, Any], max_retries: int = 3) -> str:
        """
        制約(max_words)を満たすまでリトライを行う生成メソッド。
        """
        max_words = self._get_max_words(context)
        
        # 制限がない場合は即座に生成して終了
        if max_words <= 0:
            return llm.generate(prompt).strip()
            
        current_prompt = prompt
        
        for i in range(max_retries):
            text = llm.generate(current_prompt).strip()
            
            # Simple word count (splitting by whitespace)
            word_count = len(text.split())
            
            if word_count <= max_words:
                return text
            
            # Constraint Violated
            logger.warning(f"Constraint Violation (Attempt {i+1}/{max_retries}): {word_count} words > {max_words}. Retrying...")
            
            if i < max_retries - 1:
                # Add explicit warning to prompt
                warning = f"\n\nConstraint Violation: The generated text was {word_count} words long.\nThe constraint is STRICTLY {max_words} words or less.\nRewrite the text to be shorter."
                current_prompt += warning
                
        # Final attempt failed, return the last result anyway (or empty?)
        # Returning logic usually expects a string.
        logger.error(f"Failed to satisfy constraint after {max_retries} retries. returning last output.")
        return text

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
        creation_prompt = self.prompts["creation_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            k=k
        )
        
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
            single_prompt = self.prompts["single_prompt"].format(
                constraint=constraint,
                task_def=task_def
            )
            single_text = llm.generate(single_prompt).strip()
            texts.append((single_text, single_prompt))
            
        return texts

class GeneticAlgorithmStrategy(EvolutionStrategy):
    """
    1. 遺伝的アルゴリズム (Genetic Algorithm)
    
    テキストそのものを変異(Mutation)させて進化させる。
    エリート(高スコアのテキスト)を保存し、それを親として「言い換え」「拡張」「要約」「トーン変更」などの操作を行い、
    新しいテキスト変形を生成する。
    """
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Genetic Algorithm Strategy (Text Optimization)...")
        
        self._ensure_prompts("ga", context)
        
        # スコア順にソート (降順)
        # スコア順にソート (降順) - 同点時はランダム
        sorted_pop = sorted(population, key=lambda x: (x['score'], random.random()), reverse=True)
        
        # エリート選択 (上位20%, 最低1個)
        num_elite = max(1, int(k * 0.2))
        elites = [p for p in sorted_pop[:num_elite]]
        
        new_texts = []
        
        # エリートはそのまま次世代に残す (Elitism)
        for elite in elites:
            new_texts.append((elite['text'], "Elitism: Preserved from previous generation"))
            
        # 残りの枠を変異で埋める
        while len(new_texts) < k:
            # 親をエリートからランダム選択
            parent = random.choice(elites)
            parent_text = parent['text']
            
            # 変異タイプをランダム選択
            mutation_types = [
                "paraphrase (言い換え)", 
                "expand (詳しく・具体的にする)", 
                "shorten (簡潔にする)", 
                "change tone (より魅力的なトーンにする)"
            ]
            m_type = random.choice(mutation_types)
            
            constraint = self._get_task_constraint(context)
            mutation_prompt = self.prompts["mutation_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                parent_text=parent_text,
                m_type=m_type
            )
            mutated_text = llm.generate(mutation_prompt).strip()
            new_texts.append((mutated_text, mutation_prompt))
            
        return new_texts

class TextGradStrategy(EvolutionStrategy):
    """
    2. TextGrad (Pseudo-Gradient)
    
    テキストに対する「改善の方向性(勾配)」を言語化し、それに基づいてテキストを修正する。
    1. 現状のテキストとスコアから「なぜスコアがこの値なのか」「どうすれば上がるか」を分析(Gradient Generation)。
    2. その分析に基づいてテキストをリライトする(Update)。
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "textgrad.yaml")
        default_config = {
            "elite_sample_num": 1,
            "gradient_max_words": 25 # Default
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load textgrad.yaml: {e}")
        return default_config

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using TextGrad Strategy (Text Optimization)...")
        
        self._ensure_prompts("textgrad", context)
        
        elite_num = self.config.get("elite_sample_num", 1)

        # エリートは保存
        sorted_pop = sorted(population, key=lambda x: (x['score'], random.random()), reverse=True)
        new_texts = []
        for i in range(min(elite_num, len(sorted_pop))):
             best_individual = sorted_pop[i]
             new_texts.append((best_individual['text'], f"Elitism: Rank {i+1} preserved (Score: {best_individual['score']})"))
        
        # 残りの枠を勾配更新で生成
        idx = 0
        while len(new_texts) < k:
            target = population[idx % len(population)]
            idx += 1
            
            target_text = target['text']
            score = target['score']
            reason = target.get('reason', 'N/A')
            
            # Step 1: 勾配(改善提案)の生成
            # Step 1: 勾配(改善提案)の生成
            constraint = self._get_task_constraint(context)
            gradient_prompt = self.prompts["gradient_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target_text,
                score=score,
                gradient_max_words=self.config.get('gradient_max_words', 25)
            )
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: 勾配の適用 (Update)
            # Step 2: 勾配の適用 (Update)
            constraint = self._get_task_constraint(context)
            update_prompt = self.prompts["update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target_text,
                gradient=gradient
            )
            updated_text = llm.generate(update_prompt).strip()
            new_texts.append((updated_text, f"TextGrad Update:\nGradient: {gradient}\nMakePrompt: {update_prompt}"))
            
        return new_texts


    def _load_history(self, result_dir: str, current_iter: int) -> List[Tuple[str, float]]:
        """
        過去の全世代の実行結果をディスクから読み込むヘルパーメソッド。
        (テキストとそのスコアを返す)
        """
        history = []
        for i in range(current_iter):
            iter_dir = os.path.join(result_dir, f"iter{i}")
            metrics_path = os.path.join(iter_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
                
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # 対応するテキストファイルの読み込み
                for m in metrics:
                    try:
                        file_name = m["file"]
                        t_path = os.path.join(iter_dir, "texts", file_name)
                        if os.path.exists(t_path):
                            with open(t_path, 'r') as tf:
                                t_text = tf.read().strip()
                            history.append((t_text, m["score"]))
                    except (IndexError, ValueError):
                        continue
            except Exception as e:
                logger.warning(f"Failed to load history from iter{i}: {e}")
                
        return history

    def _sample_history_gumbel(self, history: List[Tuple[str, float]], k: int, weight: float) -> List[Tuple[str, float]]:
        """
        履歴から Gumbel-Max Trick を用いて重み付きサンプリングを行う。
        score * weight をロジットとして使用。
        """
        if not history:
            return []
            
        import numpy as np
        
        scores = [h[1] for h in history]
        logits = np.array(scores) * weight
        
        # Gumbelノイズを加える
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(history))))
        perturbed_logits = logits + gumbel_noise
        
        # Top-k indices
        top_k_indices = np.argsort(perturbed_logits)[-k:][::-1]
        
        selected = [history[i] for i in top_k_indices]
        return selected

class TextGradV2Strategy(EvolutionStrategy):
    """
    TextGrad v2: 過去の入力/スコア履歴をサンプリングして補足情報として加える。

    [NOTE: EED v1.1時点では使用停止]
    EEDのExploitフェーズでは、他個体の参照によりスタンス矛盾などの不安定さが生じたため、
    本来のNormal TextGrad (V1) ロジックに戻されました。
    このクラスは将来的な研究・改善のためにコードベースに残されています。
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "textgradv2.yaml")
        default_config = {
            "ref_sampling_weight": 10.0,
            "ref_sampling_weight": 10.0,
            "ref_sampling_num": 3,
            "gradient_max_words": 25 # Default
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load textgradv2.yaml: {e}")
        return default_config

    def _load_history(self, result_dir: str, current_iter: int) -> List[Tuple[str, float]]:
         # TODO: Refactor to share this with TrajectoryStrategy or make it a Mixin
         # For now, duplicate logic or import if possible. 
         # Copy-pasting the logic from TrajectoryStrategy (or moving it to base class later)
        history = []
        for i in range(current_iter):
            iter_dir = os.path.join(result_dir, f"iter{i}")
            metrics_path = os.path.join(iter_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
                
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # 対応するテキストファイルの読み込み
                for m in metrics:
                    try:
                        file_name = m["file"]
                        t_path = os.path.join(iter_dir, "texts", file_name)
                        if os.path.exists(t_path):
                            with open(t_path, 'r') as tf:
                                t_text = tf.read().strip()
                            history.append((t_text, m["score"]))
                    except (IndexError, ValueError):
                        continue
            except Exception as e:
                logger.warning(f"Failed to load history from iter{i}: {e}")
        return history

    def _sample_history_gumbel(self, history: List[Tuple[str, float]], k: int, weight: float) -> List[Tuple[str, float]]:
        if not history: return []
        import numpy as np
        
        scores = [h[1] for h in history]
        logits = np.array(scores) * weight
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(history))))
        perturbed_logits = logits + gumbel_noise
        top_k_indices = np.argsort(perturbed_logits)[-k:][::-1]
        
        return [history[i] for i in top_k_indices]

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using TextGrad V2 Strategy...")
        
        self._ensure_prompts("textgradv2", context)
        
        result_dir = context.get('result_dir')
        current_iter = context.get('iteration', 1)
        
        # 1. Load History
        history = self._load_history(result_dir, current_iter)
        current_history = [(p['text'], p['score']) for p in population]
        history.extend(current_history)
        
        # 2. Sample References
        ref_num = self.config["ref_sampling_num"]
        ref_weight = self.config["ref_sampling_weight"]
        
        # Elitism
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        best_individual = sorted_pop[0]
        new_texts = [(best_individual['text'], "Elitism: Best individual preserved")]
        
        idx = 0
        while len(new_texts) < k:
            target = population[idx % len(population)]
            idx += 1
            
            target_text = target['text']
            score = target['score']
            
            # Sample History for this generation
            references = self._sample_history_gumbel(history, ref_num, ref_weight)
            ref_text = ""
            for i, (txt, scr) in enumerate(references):
                ref_text += f"- Ref {i+1} (Score: {scr}): {txt[:200]}...\n" # Truncate for prompt length
            
            # Step 1: Gradient
            constraint = self._get_task_constraint(context)
            gradient_prompt = self.prompts["gradient_prompt"].format(
                 constraint=constraint,
                 task_def=task_def,
                 target_text=target_text,
                 score=score,
                 ref_text=ref_text
            )
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: Update
            constraint = self._get_task_constraint(context)
            update_prompt = self.prompts["update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target_text,
                gradient=gradient
            )
            updated_text = llm.generate(update_prompt).strip()
            new_texts.append((updated_text, f"TextGrad V2 Update:\nGradient: {gradient}\nMakePrompt: {update_prompt}"))
            
        return new_texts

class TrajectoryStrategy(EvolutionStrategy):
    """
    3. Trajectory (Trajectory of Text Refinement)
    
    テキストの進化の軌跡(Trajectory)を利用する。
    過去のテキストをスコアの低い順に並べて、「改善の流れ」をLLMに提示し、
    「この流れに沿ってさらに良くなった次のバージョン」を生成させる。
    """


    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Trajectory Strategy (Text Optimization)...")
        
        self._ensure_prompts("trajectory", context)
        
        result_dir = context.get('result_dir')
        current_iter = context.get('iteration', 1)
        
        # 1. 過去の全履歴を取得
        history = self._load_history(result_dir, current_iter)
        
        # 2. 現在の世代も履歴に追加
        current_history = [(p['text'], p['score']) for p in population]
        history.extend(current_history)
        
        # 3. スコアの昇順 (低い -> 高い) にソート
        history.sort(key=lambda x: x[1])
        
        # 4. サンプリング (悪い例 -> 良い例 の流れを作る)
        if len(history) > 10:
            selected_history = history[:3] + history[-7:]
        else:
            selected_history = history
            
        trajectory_text = ""
        for i, (txt, scr) in enumerate(selected_history):
            trajectory_text += f"\n[Text Version {i+1} | Score: {scr}]\n{txt}\n"

        new_texts = []
        while len(new_texts) < k:
            constraint = self._get_task_constraint(context)
            meta_prompt = self.prompts["meta_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                trajectory_text=trajectory_text
            )
            next_text = llm.generate(meta_prompt).strip()
            new_texts.append((next_text, meta_prompt))
            
        return new_texts

class DemonstrationStrategy(EvolutionStrategy):
    """
    4. Demonstration (Few-Shot Examples)
    
    高スコアのテキストを「成功事例 (Demonstration)」としてFew-shotプロンプトに含め、
    「これらと同じくらい高品質な新しいテキスト」を生成させる。
    """
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Demonstration Strategy (Text Optimization)...")
        
        self._ensure_prompts("demonstration", context)
        
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        top_examples = sorted_pop[:5] # Top 5を事例として使う
        
        examples_text = ""
        for i, ex in enumerate(top_examples):
            examples_text += f"\nExample {i+1} (High Quality):\n{ex['text']}\n"
            
        new_texts = []
        while len(new_texts) < k:
            constraint = self._get_task_constraint(context)
            meta_prompt = self.prompts["meta_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                examples_text=examples_text
            )
            generated_text = llm.generate(meta_prompt).strip()
            new_texts.append((generated_text, meta_prompt))
            
        return new_texts

class EnsembleStrategy(EvolutionStrategy):
    """
    5. アンサンブル型 (Ensemble Strategy)
    
    複数の進化戦略を組み合わせて次世代テキストを生成する。
    """
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using Ensemble Strategy...")
        
        ratios = context.get("ensemble_ratios", {"ga": 1.0})
        
        counts = {}
        remaining = k
        keys = list(ratios.keys())
        
        for idx, method in enumerate(keys):
            ratio = ratios[method]
            if idx == len(keys) - 1:
                count = remaining
            else:
                count = int(k * ratio)
            counts[method] = count
            remaining -= count
            
        new_texts = []
        
        for method, count in counts.items():
            if count <= 0:
                continue
                
            logger.info(f"Ensemble sub-strategy: {method}, count: {count}")
            strategy = get_evolution_strategy(method)
            sub_results = strategy.evolve(llm, population, count, task_def, context)
            
            for text, meta in sub_results:
                new_meta = f"[Ensemble: {method}] {meta}"
                new_texts.append((text, new_meta))
                
        return new_texts[:k]

class HGStrategy(EvolutionStrategy):
    """
    6. 仮説生成 (Hypothesis Generation)
    
    仮説に基づきテキストを生成・改善する。
    knowledge.json に有効な仮説を蓄積し、それを利用して次世代を生成する。
    """
    def _create_logic_json(self, meta_prompt: str, hypothesis_id: str, hypothesis_text: str, type_: str) -> str:
        data = {
            "hypothesis_id": hypothesis_id,
            "hypothesis_text": hypothesis_text,
            "meta_prompt": meta_prompt,
            "type": type_
        }
        return json.dumps(data, ensure_ascii=False)

    def _clean_hypotheses(self, hypotheses: List[str]) -> List[str]:
        """
        生成された仮説リストから、不適切（コンテンツそのものなど）なものを除去する。
        """
        valid = []
        for h in hypotheses:
            h = h.strip()
            # 明らかにコンテンツ生成と思われるものを弾く
            if h.upper().startswith("SUPPORT:") or h.upper().startswith("OPPOSE:") or h.upper().startswith("SUPPORT ") or h.upper().startswith("OPPOSE "):
                continue
            # 長すぎるものも弾く (仮説は簡潔な指示であるべき)
            if len(h) > 300: 
                continue
            if not h:
                continue
            valid.append(h)
        return valid



    def _update_knowledge(self, knowledge_path: str, valid_hypotheses: List[Dict]) -> List[Dict]:
        knowledge = []
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r') as f:
                    knowledge = json.load(f)
            except: pass
            
        # Add new valid hypotheses
        # Simple deduplication could be added here
        knowledge.extend(valid_hypotheses)
        
        # Save
        with open(knowledge_path, 'w') as f:
            json.dump(knowledge, f, indent=4, ensure_ascii=False)
            
        return knowledge

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using HG Strategy...")
        
        result_dir = context.get('result_dir')
        iteration = context.get('iteration')
        knowledge_path = os.path.join(result_dir, "knowledge.json")
        
        # Step 1: Verify Hypotheses
        # Find baseline score (from previous generation)
        # We need to look at population logic to find "baseline" or the best score from previous?
        # The plan says: "Compare hypothesis individual score vs baseline score".
        # Assuming there was a baseline in the population.
        
        baseline_score = 0.0
        # Try to find explicit baseline
        for p in population:
            try:
                logic = json.loads(p['prompt'])
                if logic.get('type') in ['baseline', 'theoretical_best', 'inherited']:
                    # If multiple, take max?
                    baseline_score = max(baseline_score, p['score'])
            except: pass
            
        # If no explicit baseline found, use the max score of the previous generation as the bar
        if baseline_score == 0.0 and population:
             baseline_score = max([p['score'] for p in population])
             
        valid_hypotheses = []
        for p in population:
            try:
                logic = json.loads(p['prompt'])
                h_text = logic.get('hypothesis_text')
                h_id = logic.get('hypothesis_id')
                if h_text and h_text != "N/A (Baseline)" and p['score'] >= baseline_score:
                    # It's a valid hypothesis (equal or better than baseline)
                    valid_hypotheses.append({
                        "hypothesis_id": h_id,
                        "hypothesis_text": h_text,
                        "score": p['score'],
                        "baseline_score": baseline_score,
                        "verified_at_iter": iteration - 1
                    })
            except: pass
            
        # Step 2: Update Knowledge
        knowledge = self._update_knowledge(knowledge_path, valid_hypotheses)
        
        # Step 3: Generate New Hypotheses
        t = 2 # Exploration count (tunable)
        exploit_count = max(0, k - 1 - 1 - t) # k - inherited - theoretical - exploration
        
        knowledge_text = "\n".join([f"- {item['hypothesis_text']}" for item in knowledge])
        
        new_hypotheses_exploratory = []
        if t > 0:
            explore_prompt = f"""
Task: "{task_def}"

Existing Knowledge (Validated Hypotheses):
{knowledge_text}

Goal: Generate {t} NEW, distinct hypotheses (strategies) that are DIFFERENT from existing knowledge.
Explore new angles (e.g. format, psychological appeal, logical structure).

CRITICAL:
- The output must be an INSTRUCTION or STRATEGY, NOT an example of the text.
- Do NOT generate the actual content.

Output JSON list of strings.
"""
            resp = llm.generate(explore_prompt).strip()
            try:
                if resp.startswith("```"): resp = resp.split("\n",1)[1].rsplit("```",1)[0]
                new_hypotheses_exploratory = json.loads(resp)
            except: new_hypotheses_exploratory = []
            
            # Cleaning
            new_hypotheses_exploratory = self._clean_hypotheses(new_hypotheses_exploratory)
            
            # Fallback
            if len(new_hypotheses_exploratory) < t:
                 new_hypotheses_exploratory.extend([f"Exploratory Hypothesis {i}" for i in range(t - len(new_hypotheses_exploratory))])
            
        new_hypotheses_exploitative = []
        if exploit_count > 0:
            exploit_prompt = f"""
Task: "{task_def}"

Existing Knowledge:
{knowledge_text}

Goal: Generate {exploit_count} NEW hypotheses that COMBINE or REFINE the existing knowledge.

CRITICAL:
- The output must be an INSTRUCTION or STRATEGY, NOT an example of the text.
- Do NOT generate the actual content.

Output JSON list of strings.
"""
            resp = llm.generate(exploit_prompt).strip()
            try:
                if resp.startswith("```"): resp = resp.split("\n",1)[1].rsplit("```",1)[0]
                new_hypotheses_exploitative = json.loads(resp)
            except: new_hypotheses_exploitative = []

            # Cleaning
            new_hypotheses_exploitative = self._clean_hypotheses(new_hypotheses_exploitative)

            if len(new_hypotheses_exploitative) < exploit_count:
                 new_hypotheses_exploitative.extend([f"Exploitative Hypothesis {i}" for i in range(exploit_count - len(new_hypotheses_exploitative))])

        # Step 4: Generate Population
        new_texts = []
        
        # 4.1 Inherited (Best)
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        if sorted_pop:
            best = sorted_pop[0]
            logic = self._create_logic_json("Inherited", "N/A", "N/A", "inherited")
            new_texts.append((best['text'], logic))
            
        # 4.2 Theoretical Best
        constraint = self._get_task_constraint(context)
        theo_prompt = f"""
{constraint}

Task: "{task_def}"

Knowledge:
{knowledge_text}

Directive:
Generate the THEORETICALLY OPTIMAL text by applying ALL the knowledge above.
Output ONLY the text.
"""
        theo_text = llm.generate(theo_prompt).strip()
        theo_logic = self._create_logic_json(theo_prompt, "theoretical", "Combined Knowledge", "theoretical_best")
        new_texts.append((theo_text, theo_logic))
        
        # 4.3 Exploratory
        for hyp in new_hypotheses_exploratory[:t]:
             constraint = self._get_task_constraint(context)
             p_text = f"{constraint}\nGenerate text based on hypothesis: {hyp}"
             gen = llm.generate(f"{p_text}\nOutput only text.").strip()
             new_texts.append((gen, self._create_logic_json(p_text, str(uuid.uuid4())[:8], hyp, "exploration")))

        # 4.4 Exploitative
        for hyp in new_hypotheses_exploitative[:exploit_count]:
             constraint = self._get_task_constraint(context)
             p_text = f"{constraint}\nGenerate text based on hypothesis: {hyp}"
             gen = llm.generate(f"{p_text}\nOutput only text.").strip()
             new_texts.append((gen, self._create_logic_json(p_text, str(uuid.uuid4())[:8], hyp, "exploitation")))
             
        # Fill rest if needed (e.g. if k is large)
        current_len = len(new_texts)
        if current_len < k:
             # Fill with mutations of best or random
             # For simplicity, just more theoretical bests or baseline
             for i in range(k - current_len):
                  new_texts.append((theo_text, theo_logic)) # Duplicate theoretical best
                  
        return new_texts[:k]


class HEStrategy(EvolutionStrategy):
    """
    7. 仮説探索 (Hypothesis Exploration) - "HE"
    
    - Exploitation: Batch TextGrad (前世代の全結果を参照して改善)
    - Exploration: Hypothesis Generation based on accumulated Knowledge (KV)
    """
    
    def _create_logic_json(self, meta_prompt: str, strategy_type: str, source: str) -> str:
        data = {
            "meta_prompt": meta_prompt,
            "type": strategy_type,
            "source": source
        }
        return json.dumps(data, ensure_ascii=False)

    def _update_he_knowledge(self, knowledge_path: str, population: List[Dict], result_dir: str, current_iter: int, llm: LLMInterface, task_def: str, context: Dict[str, Any]) -> Dict[str, str]:
        """
        直近N世代の結果を分析し、KV形式の知識を抽出・更新する。
        Nは config/logic/hc.yaml で指定(デフォルト3)。
        """
        current_knowledge = {}
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r') as f:
                    current_knowledge = json.load(f)
            except: pass

        # Load config from config/logic/hc.yaml
        lookback = 3
        try:
            import yaml
            config_path = os.path.join(os.getcwd(), "config", "logic", "hc.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    hc_config = yaml.safe_load(f)
                    lookback = hc_config.get("he_lookback", 3)
        except Exception as e:
            logger.warning(f"Failed to load config/logic/hc.yaml: {e}. Using default lookback=3.")

        # 1. Recent History Construction
        combined_population = []
        combined_population.extend(population)
        
        base_iter = current_iter - 1
        
        for i in range(1, lookback):
            target_iter = base_iter - i
            if target_iter < 0: break
            
            iter_dir = os.path.join(result_dir, f"iter{target_iter}")
            metrics_path = os.path.join(iter_dir, "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    for m in metrics:
                         t_path = os.path.join(iter_dir, "texts", m["file"])
                         if os.path.exists(t_path):
                             with open(t_path, 'r') as tf:
                                 t_text = tf.read().strip()
                             combined_population.append({
                                 "text": t_text,
                                 "score": m["score"],
                                 "reason": m.get("reason", "")
                             })
                except: pass

        # 分析用コンテキスト
        sorted_pop = sorted(combined_population, key=lambda x: x['score'], reverse=True)
        summary_context = ""
        
        summary_context += f"### High Scoring Examples (Recent {lookback} Generations):\n"
        for p in sorted_pop[:3]:
            summary_context += f"- Score {p['score']}: {p['text'][:200]}...\n"
            
        summary_context += f"\n### Low Scoring Examples (Recent {lookback} Generations):\n"
        for p in sorted_pop[-3:]:
             summary_context += f"- Score {p['score']}: {p['text'][:200]}...\n"

        if not self.prompts:
             self._ensure_prompts("he", context)
             
        prompt = self.prompts["knowledge_update_prompt"].format(
            task_def=task_def,
            current_knowledge_json=json.dumps(current_knowledge, indent=2),
            summary_context=summary_context
        )
        response = llm.generate(prompt).strip()
        try:
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                if response.rfind("```") != -1:
                    response = response[:response.rfind("```")]
            new_knowledge = json.loads(response)
            if isinstance(new_knowledge, dict):
                current_knowledge = new_knowledge
        except:
            logger.warning("Failed to parse Knowledge Update JSON.")
            
        with open(knowledge_path, 'w') as f:
            json.dump(current_knowledge, f, indent=4, ensure_ascii=False)
            
        return current_knowledge

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using HE Strategy...")
        
        self._ensure_prompts("he", context)
        
        result_dir = context.get('result_dir')
        iteration = context.get('iteration')
        knowledge_path = os.path.join(result_dir, "knowledge.json")
        
        # 1. Update Knowledge (KV Extraction)
        knowledge = self._update_he_knowledge(knowledge_path, population, result_dir, iteration, llm, task_def, context)
        
        # Determine counts
        t = max(1, int(k * 0.3)) 
        num_batch = k - t        
        
        new_texts = []
        
        # --- Phase 1: Exploration (Hypothesis based on KV Knowledge) ---
        if t > 0:
            explore_prompt = self.prompts["explore_prompt"].format(
                task_def=task_def,
                knowledge_json=json.dumps(knowledge, indent=2),
                t=t
            )
            resp = llm.generate(explore_prompt).strip()
            hypotheses = []
            try:
                if resp.startswith("```"): resp = resp.split("\n",1)[1].rsplit("```",1)[0]
                hypotheses = json.loads(resp)
            except: pass
            
            while len(hypotheses) < t:
                hypotheses.append("Explore a unique perspective not yet covered.")
            
            for hyp in hypotheses[:t]:
                constraint = self._get_task_constraint(context)
                gen_prompt = self.prompts["gen_prompt"].format(
                    constraint=constraint,
                    task_def=task_def,
                    hyp=hyp
                )
                text = llm.generate(gen_prompt).strip()
                logic = self._create_logic_json(gen_prompt, "exploration_he", hyp)
                new_texts.append((text, logic))

        # --- Phase 2: Exploitation (Batch TextGrad) ---
        if num_batch > 0:
            batch_context = ""
            for i, p in enumerate(population):
                batch_context += f"Item {i}: Score={p['score']}, Text='{p['text']}'\n"
                
            sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
            targets = []
            for i in range(num_batch):
                targets.append(sorted_pop[i % len(sorted_pop)])
                
            for target in targets:
                grad_prompt = self.prompts["grad_prompt"].format(
                    task_def=task_def,
                    batch_context=batch_context,
                    target_text=target['text']
                )
                gradient = llm.generate(grad_prompt).strip()
                
                constraint = self._get_task_constraint(context)
                update_prompt = self.prompts["update_prompt"].format(
                    constraint=constraint,
                    task_def=task_def,
                    target_text=target['text'],
                    gradient=gradient
                )
                text = llm.generate(update_prompt).strip()
                logic = self._create_logic_json(update_prompt, "batch_textgrad", gradient)
                new_texts.append((text, logic))
                
        return new_texts[:k]


class EEDStrategy(EvolutionStrategy):
    """
    8. EED (Expert-driven Evolution and Diagnosis)
    
    Exploit, Diversity, Explore の3つのフェーズを組み合わせて進化させる。
    Config: config/logic/eed.yaml
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "eed.yaml")
        default_config = {
            "exploit_sample_num": 6,
            "diversity_sample_num": 2,
            "explore_sample_num": 2,
            "grad_temperature": 1.0,
            "sample_pass_rate": 3,
            "elite_sample_num": 1,
            "gradient_max_words": 25 # Default
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load eed.yaml: {e}")
        return default_config

    def _calculate_novelty(self, candidates: List[str], distinct_set: List[str]) -> List[float]:
        """
        Novelty(x, K) = 1 - max_{k in K}(similarity(x, k))
        similarity: 3-gram cosine similarity
        """
        if not distinct_set or not candidates or CountVectorizer is None:
            return [1.0] * len(candidates)

        all_texts = candidates + distinct_set
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            candidate_vectors = tfidf_matrix[:len(candidates)]
            set_vectors = tfidf_matrix[len(candidates):]
            
            similarities = cosine_similarity(candidate_vectors, set_vectors)
            # similarities shape: (len(candidates), len(distinct_set))
            
            max_similarities = similarities.max(axis=1) # Max sim for each candidate
            novelty_scores = 1.0 - max_similarities
            return novelty_scores.tolist()
            
        except ValueError:
            # Empty vocabulary or other issue
            return [0.0] * len(candidates)

    def _generate_contrastive_gradient(self, llm: LLMInterface, parent_text: str, score: float, task_def: str, existing_gradient: str, context: Dict[str, Any]) -> str:
        """
        Generates a "Contrastive Gradient" that explicitly avoids the direction of the existing gradient.
        """
        constraint = self._get_task_constraint(context)
        constraint = self._get_task_constraint(context)
        prompt = self.prompts["contrastive_gradient_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            parent_text=parent_text,
            score=score,
            existing_gradient=existing_gradient
        )
        return llm.generate(prompt).strip()

    def _create_logic_json(self, meta_prompt: str, strategy_type: str, source: str) -> str:
        data = {
            "meta_prompt": meta_prompt,
            "type": strategy_type,
            "source": source
        }
        return json.dumps(data, ensure_ascii=False)

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using EED Strategy...")
        
        self._ensure_prompts("eed", context)
        
        # Load Config
        exploit_num = self.config["exploit_sample_num"]
        diversity_num = self.config["diversity_sample_num"]
        explore_num = self.config["explore_sample_num"]
        grad_temp = self.config["grad_temperature"]
        pass_rate = self.config["sample_pass_rate"]
        elite_num = self.config.get("elite_sample_num", 1)
        persona_reuse = self.config.get("persona_reuse_rate", 0.5)

        # Adjust k if config sums don't match (Prioritize config ratios, but fit to k)
        total_config = exploit_num + diversity_num + explore_num

        if k != total_config:
             logger.warning(f"Population size {k} != Config Sum {total_config}. Scaling config.")
             ratio_ex = exploit_num / total_config
             ratio_div = diversity_num / total_config
             
             exploit_num = int(k * ratio_ex)
             diversity_num = int(k * ratio_div)
             explore_num = k - exploit_num - diversity_num

        sorted_pop = sorted(population, key=lambda x: (x['score'], random.random()), reverse=True)
        
        new_texts = [] # List[Tuple[text, logic]]

        # --- 0. Elitism ---
        # Explicitly preserve top elite_num individuals
        for i in range(min(elite_num, len(sorted_pop))):
            elite = sorted_pop[i]
            new_texts.append((elite['text'], f"Elitism: Rank {i+1} preserved (Score: {elite['score']})"))
        
        # Adjust exploit_num to account for elites
        # We reduce the *newly generated* exploit samples so that Total = k
        # Actually, we should just fill up to exploit_num + elite_num?
        # Let's subtract from exploit_num primarily.
        exploit_num = max(0, exploit_num - len(new_texts)) 
        

        # Dictionary to store standard gradients for each parent text (key: text_hash or text)
        parent_gradients = {} 

        # --- 1. Exploit Phase (Normal TextGrad - No History Reference) ---
        # Top 3 parents -> Generate variations
        parents_exploit = sorted_pop[:3]

        for parent in parents_exploit:
            # Generate 2 variations per parent
            for i in range(2):
                
                grad_prompt = self.prompts["grad_prompt"].format(
                    constraint=self._get_task_constraint(context),
                    task_def=task_def,
                    parent_text=parent['text'],
                    parent_score=parent['score'],
                    gradient_max_words=self.config.get('gradient_max_words', 25)
                )
                gradient = llm.generate(grad_prompt).strip()
                
                # Store the PRIMARY gradient for this parent
                # We only store the FIRST one generated if multiple passes (i=0)
                if i == 0:
                    parent_gradients[parent['text']] = gradient

                constraint = self._get_task_constraint(context)
                constraint = self._get_task_constraint(context)
                update_prompt = self.prompts["update_prompt"].format(
                    constraint=constraint,
                    task_def=task_def,
                    parent_text=parent['text'],
                    gradient=gradient
                )
                text = self._generate_with_retry(llm, update_prompt, context).strip()
                logic = self._create_logic_json(update_prompt, "eed_exploit_v2", f"Parent Score:{parent['score']}")
                new_texts.append((text, logic))
                
                if len(new_texts) >= exploit_num: break
            if len(new_texts) >= exploit_num: break
            
        # Fill if short
        while len(new_texts) < exploit_num:
             if new_texts:
                new_texts.append(new_texts[0]) # Duplicate best
             else:
                new_texts.append(("Fallback", self._create_logic_json("N/A", "fallback", "error")))

        # --- 2. Diversity Phase (Contrastive Gradient) ---
        # Generate 'diversity_num' samples by applying CONTRASTIVE gradients to the exploit parents.
        
        diversity_new_texts = []
        
        # We iterate through the same parents used in Exploit
        # If diversity_num is large, we circle back.
        
        div_idx = 0
        while len(diversity_new_texts) < diversity_num:
            parent = parents_exploit[div_idx % len(parents_exploit)]
            div_idx += 1
            
            # Retrieve the standard gradient used in Exploit phase
            existing_gradient = parent_gradients.get(parent['text'], "General improvement")
            
            # Generate Contrastive Gradient
            contrastive_gradient = self._generate_contrastive_gradient(
                llm, parent['text'], parent['score'], task_def, existing_gradient, context
            )
            
            # Apply Contrastive Update
            constraint = self._get_task_constraint(context)
            update_prompt = self.prompts["contrastive_update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                parent_text=parent['text'],
                contrastive_gradient=contrastive_gradient
            )
            text = self._generate_with_retry(llm, update_prompt, context).strip()
            logic = self._create_logic_json(update_prompt, "eed_diversity_contrastive", f"Contrastive to: {existing_gradient[:50]}...")
            
            diversity_new_texts.append((text, logic))

        new_texts.extend(diversity_new_texts)        
        # --- 3. Explore Phase ---
        # Top half samples -> Verbalize "Norm"
        top_half = sorted_pop[:len(sorted_pop)//2]
        top_texts = "\n".join([f"- {p['text'][:100]}..." for p in top_half])
        
        norm_prompt = self.prompts["norm_prompt"].format(
            task_def=task_def,
            top_texts=top_texts
        )
        norms = llm.generate(norm_prompt).strip()
        
        # Logic to save/load explored items (Persona + Argument)
        def _get_knowledge_path(context):
            res_dir = context.get('result_dir')
            if res_dir:
                 return os.path.join(res_dir, "eed_knowledge.json")
            return None

        def _load_knowledge(path):
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except: pass
            return {"explored_items": []} # changed from explored_arguments to explored_items

        def _save_knowledge(path, data):
            if path:
                try:
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                except: pass

        # Extract Common Stance
        common_stance = "Unknown"
        try:
            if "Common Stance:" in norms:
                common_stance = norms.split("Common Stance:", 1)[1].split('\n')[0].strip()
        except: pass

        # Load Knowledge
        k_path = _get_knowledge_path(context)
        knowledge_data = _load_knowledge(k_path)
        
        # Migration check: if old format exists, convert or ignore? Just ignore/overwrite for now as per user instruction impl plan implied fresh or compat
        # Let's support loading old format if "explored_arguments" exists and "explored_items" is empty
        if "explored_arguments" in knowledge_data and not knowledge_data.get("explored_items"):
            # Convert old args to items with "Legacy" persona
            knowledge_data["explored_items"] = [{"persona": "Legacy", "argument": arg} for arg in knowledge_data["explored_arguments"]]
            
        explored_items = knowledge_data.get("explored_items", [])
        
        # --- Persona & Argument Generation ---
        
        # Generate explore_sample_num * sample_pass_rate candidates
        explore_candidates = []
        gen_count = explore_num * pass_rate
        
        new_items_to_save = []
        
        # Track personas generated in this session to prevent immediate duplicates
        current_session_personas = [item.get('persona', 'General') for item in explored_items]

        for i in range(gen_count):
             # 1. Decide Persona Strategy (Rate from config, default 0.5)
             use_existing_persona = False
             if explored_items and random.random() < persona_reuse:
                 use_existing_persona = True
                 
             target_persona = ""
             
             if use_existing_persona:
                 # Sample from existing personas
                 # Get unique personas from history, favoring those NOT yet used in this session if possible?
                 # Actually, reuse means we WANT to reuse. But maybe not the same one 5 times in a row?
                 # Let's just pick random.
                 existing_unique = list(set([item.get('persona', 'General') for item in explored_items]))
                 if existing_unique:
                     target_persona = random.choice(existing_unique)
                 else:
                     use_existing_persona = False # Fallback
             
             if not use_existing_persona:
                 # Generate NEW Persona
                 # Context: Use the DYNAMIC list including items generated in this loop
                 # Take last 50 unique ones
                 unique_session_personas = list(set(current_session_personas))
                 # Sort to keep some order or just take last added? set is unordered. 
                 # Let's use the list reversed and unique-ified preserving order
                 seen = set()
                 unique_ordered = []
                 for p in reversed(current_session_personas):
                     if p not in seen:
                         allowed = True
                         # Filter out empty or generic
                         if not p or p == "General" or p == "Legacy": allowed = False
                         if allowed:
                            unique_ordered.append(p)
                            seen.add(p)
                 
                 # unique_ordered is now newest first. Take top 50, then reverse back to chronological for prompt
                 recent_personas = unique_ordered[:50][::-1]
                 existing_personas_str = ", ".join(recent_personas)
                 
                 # unique_ordered is now newest first. Take top 50, then reverse back to chronological for prompt
                 recent_personas = unique_ordered[:50][::-1]
                 existing_personas_str = ", ".join(recent_personas)
                 
                 persona_gen_prompt = self.prompts["persona_gen_prompt"].format(
                     task_def=task_def,
                     common_stance=common_stance,
                     existing_personas_str=existing_personas_str
                 )
                 # Retry loop for uniqueness
                 max_retries = 3
                 for retry in range(max_retries):
                     try:
                         target_json_str = llm.generate(
                             persona_gen_prompt, 
                             response_mime_type="application/json"
                         ).strip()
                         target_data = json.loads(target_json_str)
                         if isinstance(target_data, list):
                             if target_data and isinstance(target_data[0], dict):
                                 target_persona = target_data[0].get("persona", "").strip()
                             elif target_data and isinstance(target_data[0], str):
                                 target_persona = target_data[0].strip()
                             else:
                                 target_persona = ""
                         elif isinstance(target_data, dict):
                             target_persona = target_data.get("persona", "").strip()
                         else:
                             target_persona = ""
                     except Exception as e:
                         # Fallback or retry on parse error
                         logger.warning(f"Persona generation JSON parse error: {e}")
                         target_persona = ""
                     
                     # Clean up basics (just in case)
                     target_persona = target_persona.replace('"', '').replace("'", "").strip()
                     
                     # Check uniqueness (case-insensitive)
                     is_duplicate = False
                     norm_target = target_persona.lower()
                     for existing in current_session_personas:
                         if existing.lower() == norm_target:
                             is_duplicate = True
                             break
                    
                     if not is_duplicate and target_persona:
                         break # Found unique
                     elif retry < max_retries - 1:
                         # Append previous attempt to prompt to explicitly avoid it?
                         # Or just retry. Let's just retry for now, maybe with a small tweak if possible but simple retry often works for temperature>0
                         pass
                 
                 # Add to session tracking immediately
                 if target_persona:
                     current_session_personas.append(target_persona)
             
             # 2. Generate Argument based on Persona
             # Filter explored arguments for this persona if any? No, just list global explored args to avoid repetition?
             # Let's list global arguments to avoid repetition generally
             global_args = [item.get('argument', '') for item in explored_items[-20:]]
             avoid_args_str = "\n".join([f"- {a}" for a in global_args if a])
             
             arg_gen_prompt = self.prompts["arg_gen_prompt"].format(
                 task_def=task_def,
                 common_stance=common_stance,
                 target_persona=target_persona,
                 avoid_args_str=avoid_args_str
             )
             target_argument = llm.generate(arg_gen_prompt).strip()
             
             # Save this exploration attempt to knowledge later if selected, 
             # but here we just need to generate text.
             # Note: We should probably only save to knowledge if it's actually selected? 
             # Implementation plan says "store Persona with Argument". 
             # Let's generate candidates first, then save selected ones?
             # Or save all generated valid ones? EED usually saves "explored" things.
             # Let's track them temp and save successful candidates.
             
             # 3. Generate Text
             constraint = self._get_task_constraint(context)
             gen_prompt = self.prompts["gen_prompt"].format(
                 constraint=constraint,
                 task_def=task_def,
                 target_persona=target_persona,
                 common_stance=common_stance,
                 target_argument=target_argument
             )
             text = self._generate_with_retry(llm, gen_prompt, context).strip()
             
             logic_dict = {
                 "meta_prompt": gen_prompt,
                 "type": "eed_explore_persona",
                 "explanation": f"Persona: {target_persona}, Arg: {target_argument}",
                 "persona": target_persona,
                 "argument": target_argument
             }
             logic = json.dumps(logic_dict, ensure_ascii=False)
             
             explore_candidates.append((text, logic, target_persona, target_argument))
             
        # Selection (Iterative Novelty)
        selected_explore = []
        current_set = [t[0] for t in new_texts]
        
        for _ in range(explore_num):
             if not explore_candidates: break
             
             cand_texts = [c[0] for c in explore_candidates]
             novelties = self._calculate_novelty(cand_texts, current_set)
             
             best_idx = novelties.index(max(novelties))
             selected = explore_candidates.pop(best_idx)
             
             # Add to results
             selected_explore.append((selected[0], selected[1]))
             current_set.append(selected[0])
             
             # Add to items to save
             new_items_to_save.append({
                 "persona": selected[2],
                 "argument": selected[3]
             })
             
        new_texts.extend(selected_explore)
        
        # Update Knowledge with SELECTED items
        if new_items_to_save:
            explored_items.extend(new_items_to_save)
            knowledge_data["explored_items"] = explored_items
            _save_knowledge(k_path, knowledge_data)
        
        # Final Fill
        while len(new_texts) < k:
             if new_texts:
                new_texts.append(new_texts[0])
             else:
                new_texts.append(("Fallback", self._create_logic_json("N/A", "fallback", "error")))

        return new_texts[:k]


def get_evolution_strategy(strategy_type: str) -> EvolutionStrategy:
    """
    ストラテジーファクトリ関数。
    """
    strategies = {
        "ga": GeneticAlgorithmStrategy(),
        "textgrad": TextGradStrategy(),
        "textgradv2": TextGradV2Strategy(),
        "trajectory": TrajectoryStrategy(),
        "demonstration": DemonstrationStrategy(),
        "ensemble": EnsembleStrategy(),
        "he": HEStrategy(),
        "eed": EEDStrategy(),
        "gatd": GATDStrategy(),
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown evolution strategy: {strategy_type}")
        
    return strategies[strategy_type]

class GATDStrategy(EvolutionStrategy):
    """
    7. GATD (Genetic Algorithm with TextGrad and Diversity) Strategy
    
    A hybrid strategy combining Elitism, Genetic Algorithm (Crossover), TextGrad, and Persona-based Mutation.
    Designed for stability and diversity in optimization.
    """
    def __init__(self):
        super().__init__()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        # Load from gatd.yaml
        config_path = os.path.join(os.getcwd(), "config", "logic", "gatd.yaml")
        default_config = {
            "elite_sample_num": 2,
            "gd_sample_num": 4,
            "grad_sample_num": 2,
            "mutation_sample_num": 2,
            "gradient_max_words": 25
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load gatd.yaml: {e}")
        return default_config

    def _rank_selection(self, population: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Rank-based selection.
        Probability P_i = (N - i) / Sum(N - j)
        """
        n = len(population)
        if n == 0:
            return []
            
        # Sort by score descending
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        
        # Calculate weights: N, N-1, ..., 1
        weights = [n - i for i in range(n)]
        total_weight = sum(weights)
        
        if total_weight == 0:
             return random.choices(sorted_pop, k=k)
             
        # Select k individuals with replacement
        selected = random.choices(sorted_pop, weights=weights, k=k)
        return selected

    def _get_persona_history(self, context: Dict[str, Any]) -> List[str]:
        result_dir = context.get('result_dir')
        if not result_dir:
            return []
        history_path = os.path.join(result_dir, "persona_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _update_persona_history(self, context: Dict[str, Any], new_personas: List[str]):
        result_dir = context.get('result_dir')
        if not result_dir:
            return
        history_path = os.path.join(result_dir, "persona_history.json")
        
        history = self._get_persona_history(context)
        history.extend(new_personas)
        
        # Deduplicate
        history = list(set(history))
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using GATD Strategy...")
        
        # Load config params
        elite_num = self.config.get("elite_sample_num", 2)
        gd_num = self.config.get("gd_sample_num", 4)
        grad_num = self.config.get("grad_sample_num", 2)
        mutation_num = self.config.get("mutation_sample_num", 2)
        
        # Adjust numbers if k is different from sum (scale proportionally or prioritized)
        # Priority: Elite > GD > Grad > Mutation
        
        new_texts = []
        
        self._ensure_prompts("gatd", context)
        
        # 1. Elitism
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        for i in range(min(elite_num, len(sorted_pop))):
             best_individual = sorted_pop[i]
             new_texts.append((best_individual['text'], f"Elitism: Rank {i+1} preserved (Score: {best_individual['score']})"))
        
        # 2. Crossover (GD)
        gd_count = 0
        while len(new_texts) < k and gd_count < gd_num:
            parents = self._rank_selection(population, 2)
            if len(parents) < 2:
                # Fallback if population is too small, duplicate parent
                parents = parents * 2 
            
            p1 = parents[0]
            p2 = parents[1]
            
            constraint = self._get_task_constraint(context)
            
            crossover_prompt = self.prompts["crossover_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                p1_score=p1['score'],
                p1_text=p1['text'],
                p2_score=p2['score'],
                p2_text=p2['text']
            )
            child_text = self._generate_with_retry(llm, crossover_prompt, context)
            new_texts.append((child_text, f"GATD Crossover (Score {p1['score']} & {p2['score']})"))
            gd_count += 1
            
        # 3. TextGrad
        grad_count = 0
        while len(new_texts) < k and grad_count < grad_num:
            target_list = self._rank_selection(population, 1)
            if not target_list: break
            target = target_list[0]
            
            constraint = self._get_task_constraint(context)
            grad_max_words = self.config.get("gradient_max_words", 25)
            
            # Gradient Step
            gradient_prompt = self.prompts["gradient_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target['text'],
                target_score=target['score'],
                grad_max_words=grad_max_words
            )
            gradient = llm.generate(gradient_prompt).strip()
            
            # Update Step
            update_prompt = self.prompts["update_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_text=target['text'],
                gradient=gradient
            )
            updated_text = self._generate_with_retry(llm, update_prompt, context)
            new_texts.append((updated_text, f"GATD TextGrad:\nGradient: {gradient}"))
            grad_count += 1

        # 4. Mutation (Persona)
        mutation_count = 0
        # Load history
        persona_history = self._get_persona_history(context)
        new_personas = []
        
        while len(new_texts) < k and mutation_count < mutation_num:
             # Select parent to inherit stance from
             parent_list = self._rank_selection(population, 1)
             if not parent_list: break
             parent = parent_list[0]

             constraint = self._get_task_constraint(context)
             
             # Use recent history for prompt context to avoid immediate repeats
             history_sample = persona_history[-10:] if len(persona_history) > 10 else persona_history
             history_str = "\n".join([f"- {p}" for p in history_sample])
             
             persona_prompt = self.prompts["persona_prompt"].format(
                 task_def=task_def,
                 history_str=history_str
             )
             persona = llm.generate(persona_prompt).strip()
             # Simple validation
             if len(persona) > 200: persona = persona[:200]
             
             new_personas.append(persona)
             
             generation_prompt = self.prompts["generation_prompt"].format(
                 constraint=constraint,
                 task_def=task_def,
                 parent_text=parent['text'],
                 persona=persona
             )
             mutated_text = self._generate_with_retry(llm, generation_prompt, context)
             new_texts.append((mutated_text, f"GATD Persona Mutation (Parent Score: {parent['score']}): {persona}"))
             mutation_count += 1
             
        # Save new personas
        if new_personas:
            self._update_persona_history(context, new_personas)
        
        # Fill remaining if any (due to exact matching issues)
        while len(new_texts) < k:
             # Fallback to simple mutation of best
             if sorted_pop:
                 best = sorted_pop[0]
                 constraint = self._get_task_constraint(context)
                 fallback_prompt = self.prompts["fallback_prompt"].format(
                     constraint=constraint,
                     task_def=task_def,
                     best_text=best['text']
                 )
                 t = self._generate_with_retry(llm, fallback_prompt, context)
                 new_texts.append((t, "GATD Fallback Fill"))
             else:
                 # Should not happen if population > 0
                 constraint = self._get_task_constraint(context)
                 fallback_prompt = self.prompts["fallback_empty_prompt"].format(
                     constraint=constraint,
                     task_def=task_def
                 )
                 t = self._generate_with_retry(llm, fallback_prompt, context)
                 new_texts.append((t, "GATD Fallback (Empty Pop)"))

        return new_texts[:k]
