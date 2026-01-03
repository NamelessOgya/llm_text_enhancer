import os
import json
import argparse
import glob
import logging
from typing import List, Dict


from utils import setup_logging, save_token_usage

logger = logging.getLogger(__name__)

def load_metrics(metrics_path: str) -> List[Dict]:
    with open(metrics_path, 'r') as f:
        return json.load(f)

def generate_initial_prompts(llm: OpenAIAdapter, k: int, task_def: str) -> List[str]:
    # Simple initialization or ask LLM
    logger.info("Generating initial population...")
    prompt = f"Generate {k} distinct prompts that would lead an AI to generate text for the following task: '{task_def}'. Return only the prompts, one per line."
    response = llm.generate(prompt)
    prompts = [p.strip() for p in response.split('\n') if p.strip()][:k]
    # Fallback padding
    while len(prompts) < k:
        prompts.append(f"Generate text for: {task_def}")
    return prompts

def evolve_prompts(llm: OpenAIAdapter, prev_prompts: List[str], metrics: List[Dict], k: int, task_def: str) -> List[str]:
    logger.info("Evolving population...")
    
    scored_prompts = []
    for m in metrics:
        idx = int(m["file"].split('_')[1].split('.')[0]) 
        if 0 <= idx < len(prev_prompts):
            scored_prompts.append((prev_prompts[idx], m["score"]))
    
    scored_prompts.sort(key=lambda x: x[1], reverse=True)
    
    # Elitism: keep top 20%
    num_elite = max(1, int(k * 0.2))
    elites = [p for p, s in scored_prompts[:num_elite]]
    
    new_prompts = elites[:]
    
    # Mutation for rest
    while len(new_prompts) < k:
        parent = new_prompts[len(new_prompts) % len(elites)]
        # LEAK PREVENTION: Do not reveal target preference. Use task definition.
        mutation_prompt = f"The following prompt generated good content for the task '{task_def}': \"{parent}\". Rewrite it to be even more effective. Return only the new prompt."
        mutated = llm.generate(mutation_prompt).strip()
        new_prompts.append(mutated)
        
    return new_prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--model-name", default="gpt-4o")
    parser.add_argument("--adapter-type", required=True, help="openai, gemini, or dummy")
    parser.add_argument("--task-definition", required=True) # Changed from target-preference
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    # Setup paths
    current_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration}")
    prompts_dir = os.path.join(current_iter_dir, "input_prompts")
    texts_dir = os.path.join(current_iter_dir, "texts")
    logs_dir = os.path.join(args.result_dir, "logs")
    
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    setup_logging(os.path.join(logs_dir, "execution.log"))
    
    from llm.factory import get_llm_adapter
    llm = get_llm_adapter(args.adapter_type, args.model_name)
    
    prompts = []
    
    if args.iteration == 0:
        prompts = generate_initial_prompts(llm, args.population_size, args.task_definition)
    else:
        prev_iter_dir = os.path.join(args.result_dir, f"iter{args.iteration-1}")
        metrics_path = os.path.join(prev_iter_dir, "metrics.json")
        prev_prompts = []
        
        prev_prompt_files = sorted(glob.glob(os.path.join(prev_iter_dir, "input_prompts", "prompt_*.txt")))
        for p_file in prev_prompt_files:
            with open(p_file, 'r') as f:
                prev_prompts.append(f.read().strip())
                
        metrics = load_metrics(metrics_path)
        prompts = evolve_prompts(llm, prev_prompts, metrics, args.population_size, args.task_definition)

    # Save Prompts and Generate Texts
    for i, p in enumerate(prompts):
        # Save prompt
        prompt_file = os.path.join(prompts_dir, f"prompt_{i}.txt")
        with open(prompt_file, 'w') as f:
            f.write(p)
            
        # Generate Text
        logger.info(f"Generating text for prompt {i}...")
        
        # Enforce task definition context in generation
        final_prompt = f"Task: {args.task_definition}\nInstruction: {p}"
        text = llm.generate(final_prompt)
        
        text_file = os.path.join(texts_dir, f"text_{i}.txt")
        with open(text_file, 'w') as f:
            f.write(text)

    save_token_usage(llm.get_token_usage(), os.path.join(logs_dir, "token_usage.json"))

if __name__ == "__main__":
    main()
