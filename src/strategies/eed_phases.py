import logging
import random
import json
import os
from typing import List, Dict, Tuple, Any

from llm.interface import LLMInterface
from .metrics import calculate_novelty

logger = logging.getLogger(__name__)

def create_logic_json(meta_prompt: str, strategy_type: str, source: str) -> str:
    data = {
        "meta_prompt": meta_prompt,
        "type": strategy_type,
        "source": source
    }
    return json.dumps(data, ensure_ascii=False)

def run_exploit_phase(
    strategy: Any, 
    llm: LLMInterface, 
    sorted_pop: List[Dict[str, Any]], 
    exploit_num: int, 
    task_def: str, 
    context: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict[str, str], List[Dict[str, Any]]]:
    
    new_texts = []
    parent_gradients = {}
    
    # Refine top samples using Gradient, referencing Top-1 as specific 'good example'
    parents_exploit = sorted_pop[:exploit_num]
    
    # Reuse parents if we don't have enough
    if len(parents_exploit) < exploit_num and len(parents_exploit) > 0:
        while len(parents_exploit) < exploit_num:
            parents_exploit.append(random.choice(parents_exploit))
            
    best_ref = sorted_pop[0]['text']
    
    grad_temp = strategy.config.get("grad_temperature", 1.0)
    grad_max_words = strategy.config.get('gradient_max_words', 25)

    for parent in parents_exploit:
        # Gradient Generation
        constraint = strategy._get_task_constraint(context)
        grad_prompt = strategy.prompts["exploit_gradient_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            parent_text=parent['text'],
            score=parent['score'],
            best_ref_text=best_ref,
            grad_temperature=grad_temp,
            gradient_max_words=grad_max_words
        )
        gradient = llm.generate(grad_prompt).strip()
        parent_gradients[parent['text']] = gradient
        
        # Update
        constraint = strategy._get_task_constraint(context)
        update_prompt = strategy.prompts["update_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            parent_text=parent['text'],
            gradient=gradient
        )
        
        text = strategy._generate_with_retry(llm, update_prompt, context).strip()
        
        if text:
            logic = create_logic_json(update_prompt, "eed_exploit_v2", f"Parent Score:{parent['score']}")
            new_texts.append((text, logic))
            
            if len(new_texts) >= exploit_num: break
        if len(new_texts) >= exploit_num: break
        
    # Fill if short
    while len(new_texts) < exploit_num:
         if new_texts:
            new_texts.append(new_texts[0]) # Duplicate best
         else:
            new_texts.append(("Fallback", create_logic_json("N/A", "fallback", "error")))
            
    return new_texts, parent_gradients, parents_exploit

def _generate_contrastive_gradient(strategy: Any, llm: LLMInterface, parent_text: str, score: float, task_def: str, existing_gradient: str, context: Dict[str, Any]) -> str:
    constraint = strategy._get_task_constraint(context)
    prompt = strategy.prompts["contrastive_gradient_prompt"].format(
        constraint=constraint,
        task_def=task_def,
        parent_text=parent_text,
        score=score,
        existing_gradient=existing_gradient
    )
    return llm.generate(prompt).strip()

def run_diversity_phase(
    strategy: Any,
    llm: LLMInterface,
    sorted_pop: List[Dict[str, Any]], # Not strictly needed if we reuse parents from exploit, but logic assumes same parents?
    parents_exploit: List[Dict[str, Any]], # Need to reconstruct or pass parents used in exploit
    parent_gradients: Dict[str, str],
    diversity_num: int,
    task_def: str,
    context: Dict[str, Any]
) -> List[Tuple[str, str]]:
    
    diversity_new_texts = []
    
    if not parents_exploit:
        parents_exploit = sorted_pop[:diversity_num] # Fallback
        if not parents_exploit: return []

    div_idx = 0
    while len(diversity_new_texts) < diversity_num:
        parent = parents_exploit[div_idx % len(parents_exploit)]
        div_idx += 1
        
        # Retrieve the standard gradient used in Exploit phase
        existing_gradient = parent_gradients.get(parent['text'], "General improvement")
        
        # Generate Contrastive Gradient
        contrastive_gradient = _generate_contrastive_gradient(
            strategy, llm, parent['text'], parent['score'], task_def, existing_gradient, context
        )
        
        # Apply Contrastive Update
        constraint = strategy._get_task_constraint(context)
        update_prompt = strategy.prompts["contrastive_update_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            parent_text=parent['text'],
            contrastive_gradient=contrastive_gradient
        )
        text = strategy._generate_with_retry(llm, update_prompt, context).strip()
        logic = create_logic_json(update_prompt, "eed_diversity_contrastive", f"Contrastive to: {existing_gradient[:50]}...")
        
        diversity_new_texts.append((text, logic))
        
    return diversity_new_texts

def run_explore_phase(
    strategy: Any,
    llm: LLMInterface,
    sorted_pop: List[Dict[str, Any]],
    explore_num: int,
    task_def: str,
    context: Dict[str, Any]
) -> List[Tuple[str, str]]:
    
    pass_rate = strategy.config.get("sample_pass_rate", 3)
    persona_reuse = strategy.config.get("persona_reuse_rate", 0.5)

    # Top half samples -> Verbalize "Norm"
    top_half = sorted_pop[:max(1, len(sorted_pop)//2)]
    top_texts = "\n".join([f"- {p['text'][:100]}..." for p in top_half])
    
    norm_prompt = strategy.prompts["norm_prompt"].format(
        task_def=task_def,
        top_texts=top_texts
    )
    norms = llm.generate(norm_prompt).strip()
    
    # Internal helpers for Knowledge
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
        return {"explored_items": []}

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
    
    if "explored_arguments" in knowledge_data and not knowledge_data.get("explored_items"):
        knowledge_data["explored_items"] = [{"persona": "Legacy", "argument": arg} for arg in knowledge_data["explored_arguments"]]
        
    explored_items = knowledge_data.get("explored_items", [])
    
    # Generate candidates
    explore_candidates = []
    gen_count = explore_num * pass_rate
    
    new_items_to_save = []
    current_session_personas = [item.get('persona', 'General') for item in explored_items]

    for i in range(gen_count):
         # Decide Persona Strategy
         use_existing_persona = False
         if explored_items and random.random() < persona_reuse:
             use_existing_persona = True
             
         target_persona = ""
         
         if use_existing_persona:
             existing_unique = list(set([item.get('persona', 'General') for item in explored_items]))
             if existing_unique:
                 target_persona = random.choice(existing_unique)
             else:
                 use_existing_persona = False
         
         if not use_existing_persona:
             # Generate NEW Persona
             # Logic to avoid duplicates logic simplified for brevity but functional
             unique_session_personas = list(set(current_session_personas))
             
             # Context of recent personas
             recent_personas = unique_session_personas[-50:] # approximate last 50
             existing_personas_str = ", ".join(recent_personas)
             
             persona_gen_prompt = strategy.prompts["persona_gen_prompt"].format(
                 task_def=task_def,
                 common_stance=common_stance,
                 existing_personas_str=existing_personas_str
             )
             
             # Retry logic for generation
             max_retries = 3
             for retry in range(max_retries):
                 try:
                     target_json_str = llm.generate(persona_gen_prompt).strip()
                     if target_json_str.startswith("```"):
                         target_json_str = target_json_str.split("\n", 1)[1]
                         if target_json_str.rfind("```") != -1:
                             target_json_str = target_json_str[:target_json_str.rfind("```")]

                     target_data = json.loads(target_json_str)
                     if isinstance(target_data, list) and target_data:
                          target_persona = target_data[0].get("persona", "") if isinstance(target_data[0], dict) else str(target_data[0])
                     elif isinstance(target_data, dict):
                          target_persona = target_data.get("persona", "")
                     else:
                          target_persona = ""
                 except: target_persona = ""
                 
                 target_persona = str(target_persona).replace('"', '').replace("'", "").strip()
                 
                 if target_persona and target_persona.lower() not in [p.lower() for p in current_session_personas]:
                     break
            
             if not target_persona:
                 target_persona = "Creative Thinker"

             current_session_personas.append(target_persona)

         # Generate Argument
         constraint = strategy._get_task_constraint(context)
         gen_prompt = strategy.prompts["explore_gen_prompt"].format(
             constraint=constraint,
             task_def=task_def,
             persona=target_persona,
             norms=norms
         )
         text = strategy._generate_with_retry(llm, gen_prompt, context).strip()
         
         explore_candidates.append({
             "text": text,
             "persona": target_persona,
             "prompt": gen_prompt
         })

    # Calculate Novelty
    distinct_set = [p['text'] for p in sorted_pop] + [item.get('argument', '') for item in explored_items]
    distinct_set = [d for d in distinct_set if isinstance(d, str) and d]
    
    candidates_text = [c['text'] for c in explore_candidates]
    novelty_scores = calculate_novelty(candidates_text, distinct_set)
    
    # Select Top-k by Novelty
    novelty_pairs = zip(explore_candidates, novelty_scores)
    sorted_candidates = sorted(novelty_pairs, key=lambda x: x[1], reverse=True)
    
    new_texts = []
    for i in range(min(explore_num, len(sorted_candidates))):
        cand, nov_score = sorted_candidates[i]
        logic = create_logic_json(
            cand['prompt'], 
            "eed_explore_persona", 
            f"Persona: {cand['persona']}, Novelty: {nov_score:.4f}"
        )
        new_texts.append((cand['text'], logic))
        new_items_to_save.append({
            "persona": cand['persona'],
            "argument": cand['text'],
            "novelty": nov_score
        })
        
    # Save Explore History
    if new_items_to_save:
        explored_items.extend(new_items_to_save)
        knowledge_data["explored_items"] = explored_items
        _save_knowledge(k_path, knowledge_data)

    # Fill final gaps
    while len(new_texts) < explore_num:
        if new_texts:
            new_texts.append(new_texts[0])
        else:
            new_texts.append(("Fallback", "Error"))
            
    return new_texts
