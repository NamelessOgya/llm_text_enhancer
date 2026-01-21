import logging
import random
import os
import glob
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

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
            
            mutation_prompt = f"""
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
            gradient_prompt = f"""
Task Definition: "{task_def}"

Current Text:
"{target_text}"

Score: {score}
Evaluator Logic/Reasoning: {reason}

Goal: Critique the Current Text strictly. Identify specific weaknesses that prevent it from achieving a higher score.
Provide a clear, actionable "Gradient" (instruction) on how to modify the text to improve it.

Note: The score is on a scale of 0.0 to 1.0.
"""
            gradient = llm.generate(gradient_prompt).strip()
            
            # Step 2: 勾配の適用 (Update)
            update_prompt = f"""
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

class TrajectoryStrategy(EvolutionStrategy):
    """
    3. Trajectory (Trajectory of Text Refinement)
    
    テキストの進化の軌跡(Trajectory)を利用する。
    過去のテキストをスコアの低い順に並べて、「改善の流れ」をLLMに提示し、
    「この流れに沿ってさらに良くなった次のバージョン」を生成させる。
    """
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
            meta_prompt = f"""
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
            meta_prompt = f"""
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

def get_evolution_strategy(strategy_type: str) -> EvolutionStrategy:
    """
    ストラテジーファクトリ関数。
    """
    strategies = {
        "ga": GeneticAlgorithmStrategy(),
        "textgrad": TextGradStrategy(),
        "trajectory": TrajectoryStrategy(),
        "demonstration": DemonstrationStrategy(),
        "ensemble": EnsembleStrategy(),
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown evolution strategy: {strategy_type}")
        
    return strategies[strategy_type]
