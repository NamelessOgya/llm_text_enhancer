
import os
import json
import glob
import re
import numpy as np

def analyze_mutation_performance(run_dir):
    print(f"Analyzing run: {run_dir}")
    
    iters = sorted([d for d in os.listdir(run_dir) if d.startswith("iter") and os.path.isdir(os.path.join(run_dir, d))], key=lambda x: int(x.replace("iter", "")))
    
    mutation_stats = []
    
    for i, iter_name in enumerate(iters):
        iter_path = os.path.join(run_dir, iter_name)
        metrics_path = os.path.join(iter_path, "metrics.json")
        logic_dir = os.path.join(iter_path, "logic")
        
        if not os.path.exists(metrics_path):
            continue
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Map file to score
        file_scores = {m['file']: m['score'] for m in metrics}
        
        # Identify Mutation individuals
        mutation_individuals = []
        all_scores = []
        
        prompt_files = glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt"))
        for p_path in prompt_files:
            idx = p_path.split("_")[-1].replace(".txt", "")
            text_file = f"text_{idx}.txt"
            score = file_scores.get(text_file, 0.0)
            all_scores.append(score)
            
            with open(p_path, 'r') as f:
                content = f.read()
                
            if "GATD Persona Mutation" in content:
                persona = content.split("GATD Persona Mutation:")[-1].strip()[:50] + "..."
                mutation_individuals.append({
                    "file": text_file,
                    "score": score,
                    "persona": persona,
                    "iter": iter_name
                })
        
        # Check survival in NEXT iteration (if exists)
        survived_count = 0
        if i < len(iters) - 1:
            next_iter_name = iters[i+1]
            next_logic_dir = os.path.join(run_dir, next_iter_name, "logic")
            next_prompts = glob.glob(os.path.join(next_logic_dir, "creation_prompt_*.txt"))
            
            # Check if any Elite in next gen originated from this mutation
            # Logic: If next gen Elite content matches mutation text? 
            # Easier: Check if "Elitism" prompt mentions score that matches mutation score? (Not unique enough)
            # Best: Compare text hashes or check if next gen prompts say "Elitism" and refer to same content.
            # Only exact text match works reliably without IDs.
            
            # Load current mutation texts
            mut_texts = {}
            for mut in mutation_individuals:
                t_path = os.path.join(iter_path, "texts", mut['file'])
                if os.path.exists(t_path):
                    with open(t_path, 'r') as f:
                        mut_texts[mut['file']] = f.read().strip()
            
            # Check next gen elites
            for np_path in next_prompts:
                with open(np_path, 'r') as f:
                    n_content = f.read()
                if "Elitism" in n_content:
                    # Find the text file for this prompt
                    n_idx = np_path.split("_")[-1].replace(".txt", "")
                    n_text_path = os.path.join(run_dir, next_iter_name, "texts", f"text_{n_idx}.txt")
                    if os.path.exists(n_text_path):
                        with open(n_text_path, 'r') as f:
                            n_text = f.read().strip()
                        
                        # Check against mutation texts
                        for m_file, m_text in mut_texts.items():
                            if m_text == n_text:
                                survived_count += 1
                                # print(f"  -> Mutation {m_file} survived to {next_iter_name} as Elite!")
                                break

        # Stats for this iter
        if mutation_individuals:
            avg_mut = np.mean([m['score'] for m in mutation_individuals])
            max_mut = np.max([m['score'] for m in mutation_individuals])
            avg_all = np.mean(all_scores)
            
            print(f"[{iter_name}] Mutation Count: {len(mutation_individuals)}")
            print(f"  Scores: {[m['score'] for m in mutation_individuals]}")
            print(f"  Avg Score: {avg_mut:.2f} (vs Generation Avg: {avg_all:.2f})")
            print(f"  Survived to Next Gen: {survived_count}")
            for m in mutation_individuals:
                print(f"    - {m['score']}: {m['persona']}")
        else:
            print(f"[{iter_name}] No Mutation individuals found.")
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze GATD Mutation Performance")
    parser.add_argument("--target-dir", required=True, help="Path to row directory, e.g., result/exp/pop/method/eval/row_0")
    args = parser.parse_args()
    
    if not os.path.exists(args.target_dir):
        print(f"Directory not found: {args.target_dir}")
    else:
        analyze_mutation_performance(args.target_dir)
