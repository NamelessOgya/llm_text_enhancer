import logging
import random
import json
from typing import List, Tuple, Dict, Any
from .utils import create_logic_json

logger = logging.getLogger(__name__)

def run_explore_phase(
    strategy: Any,
    llm: Any,
    k_explore: int,
    task_def: str,
    context: Dict[str, Any],
    top_texts: List[str]
) -> List[Tuple[str, str]]:
    """
    Exploreフェーズ: 新規視点の開拓 (Norm Analysis -> Persona -> Argument)
    
    Args:
        strategy: EEDStrategy
        llm: LLM
        k_explore: 生成数
        task_def: タスク定義
        context: コンテキスト
        top_texts: 分析対象となる高スコアテキスト群
        
    Returns:
        List[Tuple[str, str]]: 生成結果
    """
    logger.info(f"--- Explore Phase (k={k_explore}) ---")
    
    constraint = strategy._get_task_constraint(context)
    generated_texts = []
    
    try:
        # 1. Norm Analysis (Top textsから共通項を見つける)
        # Random sample if too many
        sample_texts = top_texts[:5] 
        texts_str = "\n\n".join([f"Text {i+1}:\n{t}" for i, t in enumerate(sample_texts)])
        
        norm_prompt = strategy.prompts["norm_prompt"].format(
            task_def=task_def,
            top_texts=texts_str
        )
        norms = strategy._generate_with_retry(llm, norm_prompt, context).strip()
        
        # Parse Norms (naive parsing)
        common_stance = "Unknown"
        if "Common Stance:" in norms:
            common_stance = norms.split("Common Stance:")[1].split("\n")[0].strip()
            
        logger.info(f"Identified Norms: {common_stance}")
        
        # 2. Persona & Argument Generation Loop
        existing_personas = []
        avoid_args = []
        
        for i in range(k_explore):
            # A. Persona Generation
            persona_prompt = strategy.prompts["persona_gen_prompt"].format(
                task_def=task_def,
                common_stance=common_stance,
                existing_personas_str=", ".join(existing_personas)
            )
            persona_json_str = strategy._generate_with_retry(llm, persona_prompt, context).strip()
            
            target_persona = "General Expert"
            try:
                # JSON extract
                if "```json" in persona_json_str:
                    persona_json_str = persona_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in persona_json_str:
                     persona_json_str = persona_json_str.split("```")[1].split("```")[0].strip()
                
                p_data = json.loads(persona_json_str)
                target_persona = p_data.get("persona", target_persona)
            except:
                logger.warning(f"Failed to parse persona json: {persona_json_str}")
            
            existing_personas.append(target_persona)
            
            # B. Argument Generation (Skipped for simplicity or merged? Code below merges it into Generation or separate?)
            # TAML has arg_gen_prompt, let's use it if we want detailed control, 
            # But normally Explore creates a text based on Persona + Norm deviation.
            # Let's verify TAML usage. The `explore_gen_prompt` depends on `target_argument`.
            
            arg_prompt = strategy.prompts["arg_gen_prompt"].format(
                task_def=task_def,
                common_stance=common_stance,
                target_persona=target_persona,
                avoid_args_str="; ".join(avoid_args)
            )
            target_argument = strategy._generate_with_retry(llm, arg_prompt, context).strip()
            avoid_args.append(target_argument)
            
            # C. Integration (Generation)
            gen_prompt = strategy.prompts["explore_gen_prompt"].format(
                constraint=constraint,
                task_def=task_def,
                target_persona=target_persona,
                norms=norms
            )
            text = strategy._generate_with_retry(llm, gen_prompt, context).strip()
            
            logic = create_logic_json(
                phase="Explore",
                parent_text="N/A (New Generation)",
                instruction=f"Persona: {target_persona}, Arg: {target_argument}",
                meta={
                    "norms": norms[:100] + "...",
                    "persona": target_persona
                }
            )
            generated_texts.append((text, logic))
            
    except Exception as e:
        logger.error(f"Error in Explore Phase: {e}", exc_info=True)
        # Fallback: simple generation if explore fails
        
    return generated_texts
