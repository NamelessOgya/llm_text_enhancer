import os
import json
import random
from typing import List, Dict, Any

def rank_selection(population: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Rank-based selection.
    """
    n = len(population)
    if n == 0:
        return []
        
    sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
    weights = [n - i for i in range(n)]
    total_weight = sum(weights)
    
    if total_weight == 0:
         return random.choices(sorted_pop, k=k)
         
    return random.choices(sorted_pop, weights=weights, k=k)

def get_persona_history(context: Dict[str, Any]) -> List[str]:
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

def update_persona_history(context: Dict[str, Any], new_personas: List[str]):
    result_dir = context.get('result_dir')
    if not result_dir:
        return
    history_path = os.path.join(result_dir, "persona_history.json")
    
    history = get_persona_history(context)
    history.extend(new_personas)
    history = list(set(history))
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
