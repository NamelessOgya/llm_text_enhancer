import logging
import random
import os
import glob
import json
import uuid
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

    def generate_initial(self, llm: LLMInterface, k: int, task_def: str, context: Dict[str, Any] = None) -> List[Tuple[str, str]]:
        """
        初期世代(Generation 0)のテキスト個体群を生成する。
        デフォルト実装(GeneticAlgorithmなど)ではタスク定義に基づき、多様なテキスト案をゼロベースで生成する。
        """
        logger.info("Generating initial population (Texts) using Default Strategy...")
        
        creation_prompt = f"""
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
            single_prompt = f"Generate a high-quality text for the task: {task_def}\nOutput only the text."
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

    def generate_initial(self, llm: LLMInterface, k: int, task_def: str, context: Dict[str, Any] = None) -> List[Tuple[str, str]]:
        logger.info("Generating initial population using HG Strategy...")
        
        # p=1 (Baseline), others (Hypotheses)
        p = 1
        num_hypotheses = max(0, k - p)
        
        new_texts = []
        
        # 1. Baseline Generation
        baseline_prompt = f"Generate a high-quality text for the task: {task_def}\nOutput only the text."
        baseline_text = llm.generate(baseline_prompt).strip()
        
        # JSON Logic
        baseline_logic = self._create_logic_json(baseline_prompt, "baseline", "N/A (Baseline)", "baseline")
        new_texts.append((baseline_text, baseline_logic))
        
        if num_hypotheses > 0:
            # 2. Hypothesis Generation
            hyp_gen_prompt = f"""
Task Definition: "{task_def}"

Goal: Generate {num_hypotheses} distinct hypotheses (strategies) about how to write a text that achieves a high score for this task.
Focus on structure, tone, length, or content strategy.

CRITICAL:
- The output must be an INSTRUCTION or STRATEGY, NOT an example of the text.
- Do NOT generate the actual content (e.g. do not start with "SUPPORT:" or "OPPOSE:").
- Address the "writer" of the text.

Output ONLY a JSON list of strings.
Example: ["Text should be concise.", "Include specific examples.", ...]
"""
            hyp_response = llm.generate(hyp_gen_prompt).strip()
            # Parse JSON
            try:
                if hyp_response.startswith("```"):
                    hyp_response = hyp_response.split("\n", 1)[1]
                    if hyp_response.rfind("```") != -1:
                        hyp_response = hyp_response[:hyp_response.rfind("```")]
                hypotheses = json.loads(hyp_response)
            except:
                logger.warning("Failed to parse hypotheses JSON. Using default.")
                hypotheses = [f"Hypothesis {i}" for i in range(num_hypotheses)]
            
            # Cleaning
            hypotheses = self._clean_hypotheses(hypotheses)

            # Ensure count
            hypotheses = hypotheses[:num_hypotheses]
            while len(hypotheses) < num_hypotheses:
                hypotheses.append("The text should be clear and directly address the prompt.")
                
            # 3. Generate Texts from Hypotheses
            for i, hyp in enumerate(hypotheses):
                gen_prompt = f"""
Task Definition: "{task_def}"

Hypothesis: "{hyp}"

Directive:
Generate a text based strictly on the above hypothesis.
Output ONLY the text.
"""
                text = llm.generate(gen_prompt).strip()
                hid = str(uuid.uuid4())[:8]
                logic = self._create_logic_json(gen_prompt, hid, hyp, "exploration_init")
                new_texts.append((text, logic))
                
        return new_texts

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
        theo_prompt = f"""
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
             p_text = f"Generate text based on hypothesis: {hyp}"
             gen = llm.generate(f"{p_text}\nOutput only text.").strip()
             new_texts.append((gen, self._create_logic_json(p_text, str(uuid.uuid4())[:8], hyp, "exploration")))

        # 4.4 Exploitative
        for hyp in new_hypotheses_exploitative[:exploit_count]:
             p_text = f"Generate text based on hypothesis: {hyp}"
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
                gen_prompt = f"""
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
"""
                gradient = llm.generate(grad_prompt).strip()
                
                update_prompt = f"""
Task: "{task_def}"
Original Text: "{target['text']}"
Feedback: {gradient}

Directive: Rewrite the text to maximize score. Output only text.
"""
                text = llm.generate(update_prompt).strip()
                logic = self._create_logic_json(update_prompt, "batch_textgrad", gradient)
                new_texts.append((text, logic))
                
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
        "hg": HGStrategy(),
        "he": HEStrategy(),
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown evolution strategy: {strategy_type}")
        
    return strategies[strategy_type]
