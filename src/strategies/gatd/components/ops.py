from typing import List, Dict, Tuple, Any
from .utils import rank_selection, get_persona_history, update_persona_history

def op_elitism(
    sorted_pop: List[Dict[str, Any]], 
    count: int
) -> List[Tuple[str, str]]:
    results = []
    for i in range(min(count, len(sorted_pop))):
        best = sorted_pop[i]
        results.append((best['text'], f"Elitism: Rank {i+1} preserved (Score: {best['score']})"))
    return results

def op_crossover(
    strategy: Any,
    llm: Any,
    population: List[Dict[str, Any]],
    count: int,
    task_def: str,
    context: Dict[str, Any]
) -> List[Tuple[str, str]]:
    results = []
    for _ in range(count):
        parents = rank_selection(population, 2)
        if len(parents) < 2: parents = parents * 2
        
        p1, p2 = parents[0], parents[1]
        constraint = strategy._get_task_constraint(context)
        
        prompt = strategy.prompts["crossover_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            p1_score=p1['score'], p1_text=p1['text'],
            p2_score=p2['score'], p2_text=p2['text']
        )
        child = strategy._generate_with_retry(llm, prompt, context)
        results.append((child, f"GATD Crossover (Score {p1['score']} & {p2['score']})"))
    return results

def op_textgrad(
    strategy: Any,
    llm: Any,
    population: List[Dict[str, Any]],
    count: int,
    task_def: str,
    context: Dict[str, Any],
    max_words: int
) -> List[Tuple[str, str]]:
    results = []
    for _ in range(count):
        target_list = rank_selection(population, 1)
        if not target_list: break
        target = target_list[0]
        
        constraint = strategy._get_task_constraint(context)
        
        grad_prompt = strategy.prompts["gradient_prompt"].format(
             constraint=constraint,
             task_def=task_def,
             target_text=target['text'],
             target_score=target['score'],
             grad_max_words=max_words
        )
        gradient = llm.generate(grad_prompt).strip()
        
        update_prompt = strategy.prompts["update_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            target_text=target['text'],
            gradient=gradient
        )
        updated = strategy._generate_with_retry(llm, update_prompt, context)
        results.append((updated, f"GATD TextGrad:\nGradient: {gradient}"))
    return results

def op_mutation(
    strategy: Any,
    llm: Any,
    population: List[Dict[str, Any]],
    count: int,
    task_def: str,
    context: Dict[str, Any]
) -> List[Tuple[str, str]]:
    results = []
    persona_history = get_persona_history(context)
    new_personas = []
    
    for _ in range(count):
        parent_list = rank_selection(population, 1)
        if not parent_list: break
        parent = parent_list[0]
        
        history_sample = persona_history[-10:] if len(persona_history) > 10 else persona_history
        history_str = "\n".join([f"- {p}" for p in history_sample])
        
        persona_prompt = strategy.prompts["persona_prompt"].format(
             task_def=task_def,
             history_str=history_str
        )
        persona = llm.generate(persona_prompt).strip()[:200]
        new_personas.append(persona)
        
        constraint = strategy._get_task_constraint(context)
        gen_prompt = strategy.prompts["generation_prompt"].format(
            constraint=constraint,
            task_def=task_def,
            parent_text=parent['text'],
            persona=persona
        )
        mutated = strategy._generate_with_retry(llm, gen_prompt, context)
        results.append((mutated, f"GATD Persona Mutation (Parent Score: {parent['score']}): {persona}"))
        
    if new_personas:
        update_persona_history(context, new_personas)
        
    return results
