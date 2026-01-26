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

logger = logging.getLogger(__name__)

class EvolutionStrategy(ABC):
    """
    プロンプト進化戦略の基底クラス (Abstract Base Class)。
    全ての手法はこのクラスを継承し、evolveメソッドを実装する必要がある。
    """
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

    def _get_task_constraint(self, context: Dict[str, Any]) -> str:
        """
        config/task/{task_name}.yaml から制約(max_words)を読み込み、
        プロンプトに追加する制約文字列を返す。
        """
        task_name = context.get("task_name")
        if not task_name:
            return ""
            
        config_path = os.path.join(os.getcwd(), "config", "task", f"{task_name}.yaml")
        if not os.path.exists(config_path):
            return ""
            
        try:
            with open(config_path, 'r') as f:
                task_config = yaml.safe_load(f)
                max_words = task_config.get("max_words")
                if max_words:
                    # User request: Unify to "words" for English tasks.
                    return f"Constraint: The text must be strictly {max_words} words or less."
        except:
            pass
            
        return ""

    def generate_initial(self, llm: LLMInterface, k: int, task_def: str, context: Dict[str, Any] = None) -> List[Tuple[str, str]]:
        """
        初期世代(Generation 0)のテキスト個体群を生成する。
        デフォルト実装(GeneticAlgorithmなど)ではタスク定義に基づき、多様なテキスト案をゼロベースで生成する。
        """
        logger.info("Generating initial population (Texts) using Default Strategy...")
        
        constraint = self._get_task_constraint(context)
        creation_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Goal: Generate {k} distinct, high-quality text outputs for the above task.

Requirements:
1. Generate exactly {k} different text variations.
2. The texts should vary in style, tone, or perspective.
3. OUTPUT MUST BE A VALID JSON LIST OF STRINGS. No other text.

Example Output:
["Text option 1 including details...", "Text option 2 with different focus...", ...]
"""
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
            single_prompt = f"{constraint}\nGenerate a high-quality text for the task: {task_def}\nOutput only the text."
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
        
        # スコア順にソート (降順)
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        
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
            mutation_prompt = f"""
{constraint}

Your goal is to improve the following text for the task: "{task_def}".

Original Text:
"{parent_text}"

Directive:
Apply the following mutation to the text: {m_type}.
Ensure the result is a valid output for the task.
Output ONLY the improved text.
"""
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
    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using TextGrad Strategy (Text Optimization)...")
        
        # エリートは保存
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        best_individual = sorted_pop[0]
        new_texts = [(best_individual['text'], "Elitism: Best individual preserved")]
        
        # 残りの枠を勾配更新で生成
        idx = 0
        while len(new_texts) < k:
            target = population[idx % len(population)]
            idx += 1
            
            target_text = target['text']
            score = target['score']
            reason = target.get('reason', 'N/A')
            
            # Step 1: 勾配(改善提案)の生成
            constraint = self._get_task_constraint(context)
            gradient_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Current Text:
"{target_text}"

Score: {score}

Goal: Critique the Current Text strictly. Identify specific weaknesses that prevent it from achieving a higher score.
Provide a clear, actionable "Gradient" (instruction) on how to modify the text to improve it.

Note: The score is on a scale of 0.0 to 1.0.

Important: Any improvement suggestion MUST be achievable within the task constraints (e.g. word count).
"""
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: 勾配の適用 (Update)
            constraint = self._get_task_constraint(context)
            update_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Current Text:
"{target_text}"

Feedback (Gradient) for Improvement:
{gradient}

Directive:
Rewrite the Current Text incorporating the feedback above to maximize the score.
Output ONLY the rewritten text.
"""
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
    """
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "textgradv2.yaml")
        default_config = {
            "ref_sampling_weight": 10.0,
            "ref_sampling_num": 3
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
            gradient_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Current Text:
"{target_text}"

Score: {score}

Reference Information (Past Attempts):
{ref_text}

Goal: Critique the Current Text. Use the Reference Information to identify what high-scoring texts have in common or what low-scoring texts missed.
Provide a clear, actionable "Gradient" (instruction) on how to modify the text to improve it.

Note: The score is on a scale of 0.0 to 1.0.

Important: Any improvement suggestion MUST be achievable within the task constraints (e.g. word count).
"""
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: Update
            constraint = self._get_task_constraint(context)
            update_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Current Text:
"{target_text}"

Feedback (Gradient):
{gradient}

Directive:
Rewrite the Current Text incorporating the feedback above to maximize the score.
Output ONLY the rewritten text.
"""
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
            meta_prompt = f"""
{constraint}

Task Definition: "{task_def}"

I have a sequence of texts generated for this task, sorted by their quality score (low to high).
Your goal is to identify the pattern of improvement and generate the NEXT, SUPERIOR version of the text.

Optimization Trajectory:
{trajectory_text}

Note: The score is on a scale of 0.0 to 1.0.

Directive:
Analyze the improvements made in the trajectory.
Generate a new text that is strictly better than the last version in the trajectory.
Output ONLY the new text.
"""
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
        
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        top_examples = sorted_pop[:5] # Top 5を事例として使う
        
        examples_text = ""
        for i, ex in enumerate(top_examples):
            examples_text += f"\nExample {i+1} (High Quality):\n{ex['text']}\n"
            
        new_texts = []
        while len(new_texts) < k:
            constraint = self._get_task_constraint(context)
            meta_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Here are some examples of high-quality texts generated for this task:

{examples_text}

Directive:
Generate a NEW text that achieves the same high quality as the examples above, but is distinct/creative.
Output ONLY the new text.
"""
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

        prompt = f"""
Task: "{task_def}"

Current Knowledge (JSON):
{json.dumps(current_knowledge, indent=2)}

Recent Results:
{summary_context}

Goal: Update the Knowledge JSON based on the Recent Results.
- Extract insights about what makes a text good or bad (e.g. length, tone, structure).
- If a new insight contradicts or refines existing knowledge, update the value.
- If it's a new insight, add a new key.
- Keep values concise.

Output ONLY the updated JSON object (Key-Value pairs).
"""
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
            explore_prompt = f"""
Task: "{task_def}"

Current Knowledge:
{json.dumps(knowledge, indent=2)}

Goal: Generate {t} NEW hypotheses (strategies) for writing a high-quality text.
- Look for gaps in the Current Knowledge or try a different angle.
- The hypothesis must be an INSTRUCTION.

Output JSON list of strings.
"""
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
                gen_prompt = f"""
{constraint}

Task: "{task_def}"
Hypothesis: "{hyp}"

Directive: Generate text based on the hypothesis. Output only text.
"""
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
                grad_prompt = f"""
Task: "{task_def}"

Previous Generation Results (Context):
{batch_context}

Target Text:
"{target['text']}"

Goal: Improve the Target Text.
Using the insights from the Previous Generation Results (what works vs what fails), provide a Gradient (instruction) to optimize the Target Text.

Note: The score is on a scale of 0.0 to 1.0.
"""
                gradient = llm.generate(grad_prompt).strip()
                
                constraint = self._get_task_constraint(context)
                update_prompt = f"""
{constraint}

Task: "{task_def}"
Original Text: "{target['text']}"
Feedback: {gradient}

Directive: Rewrite the text to maximize score. Output only text.
"""
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
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.getcwd(), "config", "logic", "eed.yaml")
        default_config = {
            "exploit_sample_num": 6,
            "diversity_sample_num": 2,
            "explore_sample_num": 2,
            "grad_temperature": 1.0,
            "sample_pass_rate": 3,
            "elite_sample_num": 1
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

    def _create_logic_json(self, meta_prompt: str, strategy_type: str, source: str) -> str:
        data = {
            "meta_prompt": meta_prompt,
            "type": strategy_type,
            "source": source
        }
        return json.dumps(data, ensure_ascii=False)

    def evolve(self, llm: LLMInterface, population: List[Dict[str, Any]], k: int, task_def: str, context: Dict[str, Any]) -> List[Tuple[str, str]]:
        logger.info("Evolving population using EED Strategy...")
        
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

        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        
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
        

        # Instantiate TextGradV2 for usage
        tgv2 = TextGradV2Strategy()
        
        # --- 1. Exploit Phase (Modified to use TextGrad V2) ---
        # Top 3 parents -> Generate variations
        parents_exploit = sorted_pop[:3]

        # Load History once for Efficiency
        history = tgv2._load_history(result_dir=context.get('result_dir'), current_iter=context.get('iteration', 1))
        current_history = [(p['text'], p['score']) for p in population]
        history.extend(current_history)
        
        for parent in parents_exploit:
            # Generate 2 variations per parent
            for i in range(2):
                
                # Sample references for V2
                references = tgv2._sample_history_gumbel(history, tgv2.config["ref_sampling_num"], tgv2.config["ref_sampling_weight"])
                ref_text = ""
                for ii, (txt, scr) in enumerate(references):
                    ref_text += f"- Ref {ii+1} (Score: {scr}): {txt[:200]}...\n"

                grad_prompt = f"""
{self._get_task_constraint(context)}

Task Definition: "{task_def}"

Current Text:
"{parent['text']}"

Score: {parent['score']}

Reference Information (Past Attempts):
{ref_text}

Goal: Critique the Current Text strictly. Identify specific weaknesses that prevent it from achieving a higher score.
Provide a clear, actionable "Gradient" (instruction) on how to modify the text to improve it.

Note: The score is on a scale of 0.0 to 1.0.

Important: Any improvement suggestion MUST be achievable within the task constraints (e.g. word count).
"""
                gradient = llm.generate(grad_prompt).strip()
                
                constraint = self._get_task_constraint(context)
                update_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Current Text:
"{parent['text']}"

Feedback (Gradient) for Improvement:
{gradient}

Directive:
Rewrite the Current Text incorporating the feedback above to maximize the score.
Output ONLY the rewritten text.
"""
                text = llm.generate(update_prompt).strip()
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

        exploit_texts_only = [t[0] for t in new_texts]

        # --- 2. Diversity Phase (Novelty Refinement) ---
        # Instead of mutating elites, we find the "rough diamonds" (novel but potentially low score) 
        # from the population and refine them using TextGrad.
        
        # Step 1: Identify the "Dominant Norm" (Top 1 Text)
        dominant_text = sorted_pop[0]['text']

        # Step 2: Select the "Most Novel" candidates
        # We look at the lower half or non-elites to find something distinct.
        # Let's consider candidates from rank 2 onwards to avoid just picking the runner-up.
        candidates = sorted_pop[1:]
        
        diversity_new_texts = []
        
        if candidates and diversity_num > 0:
            # Format candidates for LLM selection
            cand_str = ""
            for i, c in enumerate(candidates):
                cand_str += f"Candidate {i}: \"{c['text'][:200]}...\"\n"

            # Select N distinct candidates (iteratively or batch? Batch is easier for now)
            # Actually, let's select iteratively to ensure we pick distinct ones.
            # For simplicity in this implementation, we pick the Top-K distinct ones based on LLM judgement.
            
            novelty_prompt = f"""
Task Definition: "{task_def}"

Dominant Perspective (Highest Score):
"{dominant_text}"

Candidate Perspectives:
{cand_str}

Goal: We want to diversify the gene pool.
Identify the top {diversity_num} candidates that represent the MOST DISTINCT perspectives compared to the Dominant Perspective.
We are looking for "Hidden Gems" - arguments that are semantically different, even if their current phrasing is imperfect.

Output ONLY the indices of the selected candidates (e.g. "0, 3").
"""
            novelty_indices_str = llm.generate(novelty_prompt).strip()
            
            # Parse indices
            selected_indices = []
            try:
                import re
                nums = re.findall(r'\d+', novelty_indices_str)
                selected_indices = [int(n) for n in nums[:diversity_num]]
            except:
                pass
                
            # Fallback if selection failed
            if not selected_indices:
                selected_indices = list(range(min(len(candidates), diversity_num)))

            selected_candidates = [candidates[i] for i in selected_indices if i < len(candidates)]

            # Step 3: Apply TextGrad Refinement to selected candidates
            for parent in selected_candidates:
                # TextGrad Logic (Same as Exploit, but with "Preserve Perspective" instruction)
                
                # Sample references (optional, but good for context)
                history = tgv2._load_history(result_dir=context.get('result_dir'), current_iter=context.get('iteration', 1))
                references = tgv2._sample_history_gumbel(history, tgv2.config["ref_sampling_num"], tgv2.config["ref_sampling_weight"])
                ref_text = ""
                for ii, (txt, scr) in enumerate(references):
                    ref_text += f"- Ref {ii+1} (Score: {scr}): {txt[:200]}...\n"

                grad_prompt = f"""
{self._get_task_constraint(context)}

Task Definition: "{task_def}"

Current Text (Selected for Novelty):
"{parent['text']}"

Dominant Perspective (For Context - DO NOT COPY):
"{dominant_text}"

Score: {parent['score']}

Goal: Improving this text's persuasiveness and clarity while MAINTAINING its unique perspective.
The goal is NOT to make it look like the Dominant Perspective. The goal is to make THIS specific argument as strong as possible.

Critique the text and provide a specific instruction (Gradient) to improve it.
"""
                gradient = llm.generate(grad_prompt).strip()
                
                constraint = self._get_task_constraint(context)
                update_prompt = f"""
{constraint}

Task Definition: "{task_def}"

Original Text:
"{parent['text']}"

Improvement Instruction:
{gradient}

Directive:
Rewrite the text to improve it based on the instruction.
Ensure you KEEP the original unique angle/argument, just make it better.
Output ONLY the rewritten text.
"""
                text = llm.generate(update_prompt).strip()
                logic = self._create_logic_json(update_prompt, "eed_novelty_refining", f"Refining Novelty (Parent Score:{parent['score']})")
                diversity_new_texts.append((text, logic))

        new_texts.extend(diversity_new_texts)
        
        # --- 3. Explore Phase ---
        # Top half samples -> Verbalize "Norm"
        top_half = sorted_pop[:len(sorted_pop)//2]
        top_texts = "\n".join([f"- {p['text'][:100]}..." for p in top_half])
        
        norm_prompt = f"""
Task: "{task_def}"

High Scoring Texts (Excerpt):
{top_texts}

Goal: Analyze these texts to identify the successful pattern.
1. Identify the "Common Stance" (e.g. SUPPORT or OPPOSE).
2. Summarize the "Core Arguments" used in these texts.

Output format:
Common Stance: [Stance]
Core Arguments:
- [Argument 1]
...
"""
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
                 
                 persona_gen_prompt = f"""
Task: "{task_def}"
Common Stance: {common_stance}

Forbidden Personas (DO NOT USE): {existing_personas_str}

Goal: Generate a NEW, DISTINCT Persona who supports the Common Stance but from a different background/perspective.
Constraint 1: You must NOT use any persona listed in the "Forbidden Personas" list above.
Constraint 2: DO NOT output any reasoning, explanation, or "SUPPORT:" prefix.
Constraint 3: Output MUST be a single line containing ONLY the Persona Name.

Examples: "Economist", "Historian", "Single Parent", "Software Engineer", "Ethicist", "Local Business Owner".

Output ONLY the Persona Name.
"""
                 # Retry loop for uniqueness
                 max_retries = 3
                 for retry in range(max_retries):
                     target_persona = llm.generate(persona_gen_prompt).strip()
                     # Clean up basics only
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
             
             arg_gen_prompt = f"""
Task: "{task_def}"
Common Stance: {common_stance}
Target Persona: {target_persona}

Explored Arguments (Avoid these):
{avoid_args_str}

Goal: As the "{target_persona}", identify a specific ARGUMENT or REASON that supports the "{common_stance}".
This argument should reflect the unique professional or personal experience of a {target_persona}.

Output ONLY the Argument text (1 sentence).
"""
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
             gen_prompt = f"""
{constraint}

Task: "{task_def}"

Role: {target_persona}
Common Stance: {common_stance}
Target Argument: "{target_argument}"

Requirement:
1. Write a text supporting the "{common_stance}" from the perspective of a {target_persona}.
2. Focus specifically on the reason: "{target_argument}".
3. Adopt the tone and vocabulary appropriate for this persona.

Goal: Create a high-quality text that introduces this specific angle.
CRITICAL: Output ONLY the text content.
"""
             text = llm.generate(gen_prompt).strip()
             
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
        "hg": HGStrategy(),
        "he": HEStrategy(),
        "eed": EEDStrategy(),
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown evolution strategy: {strategy_type}")
        
    return strategies[strategy_type]
