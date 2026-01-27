import os
import json
import glob
import numpy as np
import re

def analyze_crossover_performance(base_dir):
    """
    Analyzes the performance of Crossover individuals in GATD.
    """
    
    # Store scores by method
    scores_by_method = {
        "Elitism": [],
        "Crossover": [],
        "TextGrad": [],
        "Mutation": [],
        "Other": []
    }
    
    # Track progression of Crossover counts/scores per iteration
    crossover_stats_per_iter = {} # iter -> {scores: [], count: 0}

    # Iterate through all rows
    rows = glob.glob(os.path.join(base_dir, "row_*"))
    
    total_texts = 0
    
    print(f"Analyzing rows in {base_dir}...")

    for row_path in rows:
        row_name = os.path.basename(row_path)
        
        # Iterate through iterations (1 to 9 typically)
        iter_dirs = sorted(glob.glob(os.path.join(row_path, "iter*")))
        
        for iter_dir in iter_dirs:
            iter_name = os.path.basename(iter_dir)
            iter_num = int(iter_name.replace("iter", ""))
            
            metrics_path = os.path.join(iter_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
                
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            logic_dir = os.path.join(iter_dir, "logic")
            if not os.path.exists(logic_dir):
                continue
                
            # metrics is a list of dicts: [{"file": "text_2.txt", "score": 0.0, ...}, ...]
            for metric_data in metrics:
                score = metric_data.get("score", 0.0)
                file_name = metric_data.get("file", "")
                
                # Extract index from "text_2.txt"
                match = re.search(r"text_(\d+)\.txt", file_name)
                if not match:
                    continue
                text_idx = match.group(1)
                
                prompt_file = os.path.join(logic_dir, f"creation_prompt_{text_idx}.txt")
                
                method = "Other"
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        content = f.read()
                        if "Elitism:" in content:
                            method = "Elitism"
                        elif "GATD Crossover" in content:
                            method = "Crossover"
                        elif "GATD TextGrad" in content:
                            method = "TextGrad"
                        elif "GATD Mutation" in content or "Persona" in content:
                             if "Adopt the following persona" in content or "GATD Mutation" in content:
                                 method = "Mutation"
                        
                scores_by_method[method].append(score)
                total_texts += 1
                
                if method == "Crossover":
                    if iter_num not in crossover_stats_per_iter:
                        crossover_stats_per_iter[iter_num] = []
                    crossover_stats_per_iter[iter_num].append(score)

    # Report Results
    print("\n--- Overall Performance by Method ---")
    print(f"{'Method':<15} | {'Count':<6} | {'Mean':<6} | {'Max':<6} | {'Min':<6} | {'Std':<6}")
    print("-" * 60)
    for method, scores in scores_by_method.items():
        if scores:
            scores_np = np.array(scores)
            print(f"{method:<15} | {len(scores):<6} | {scores_np.mean():.3f} | {scores_np.max():.3f} | {scores_np.min():.3f} | {scores_np.std():.3f}")
        else:
            print(f"{method:<15} | {0:<6} | {'-':<6} | {'-':<6} | {'-':<6} | {'-':<6}")

    print("\n--- Crossover Performance per Iteration ---")
    print(f"{'Iter':<5} | {'Count':<5} | {'Mean':<6}")
    for iter_num in sorted(crossover_stats_per_iter.keys()):
        scores = crossover_stats_per_iter[iter_num]
        print(f"{iter_num:<5} | {len(scores):<5} | {np.mean(scores):.3f}")

if __name__ == "__main__":
    base_dir = "result/perspectrum_v1.1/run1/gatd/perspectrum_llm"
    analyze_crossover_performance(base_dir)
