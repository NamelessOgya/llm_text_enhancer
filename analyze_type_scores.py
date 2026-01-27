
import os
import json
import glob
import re
import pandas as pd
import numpy as np

def get_candidate_type(iter_dir, filename):
    try:
        idx_match = re.search(r'text_(\d+)\.txt', filename)
        if not idx_match:
            return "Unknown"
        idx = idx_match.group(1)
        
        logic_path = os.path.join(iter_dir, "logic", f"creation_prompt_{idx}.txt")
        if not os.path.exists(logic_path):
            return "Unknown"
            
        with open(logic_path, 'r') as f:
            content = f.read().strip()
            
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                t = data.get("type", "Unknown")
                if "exploit" in t: return "Exploit"
                if "diversity" in t or "novelty" in t: return "Diversity"
                if "explore" in t: return "Explore"
                if t == "inherited" or t == "eed_inherited" or "Elitism" in content: return "Elitism"
                return t 
        except:
            if "Elitism:" in content: return "Elitism"
            pass
            
        return "Other"
    except:
        return "Error"

def analyze_type_scores(result_dir):
    print(f"Analyzing scores by type in: {result_dir}")
    
    # Recursively find metrics.json or logic files?
    # Better to iterate row_* dirs
    
    rows = []
    for root, dirs, files in os.walk(result_dir):
        for d in dirs:
            if d.startswith("row_"):
                rows.append(os.path.join(root, d))
                
    if not rows:
        # Maybe it's a single row dir
        if os.path.exists(os.path.join(result_dir, "iter0")):
            rows = [result_dir]
            
    all_data = []
    
    for row_path in rows:
        row_name = os.path.basename(row_path)
        iter_dirs = sorted(glob.glob(os.path.join(row_path, "iter*")))
        
        for iter_path in iter_dirs:
            iteration_str = os.path.basename(iter_path).replace("iter", "")
            if not iteration_str.isdigit(): continue
            iteration = int(iteration_str)
            
            # Load metrics
            metrics_path = os.path.join(iter_path, "metrics.json")
            if not os.path.exists(metrics_path): continue
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            for m in metrics:
                score = m.get("score", 0.0)
                fname = m.get("file", "")
                ctype = get_candidate_type(iter_path, fname)
                
                # Check for Contrastive Type specifically
                is_contrastive = False
                logic_path = os.path.join(iter_path, "logic", f"creation_prompt_{fname.split('_')[-1].split('.')[0]}.txt")
                if os.path.exists(logic_path):
                    with open(logic_path,'r') as f: l=f.read()
                    if "contrastive" in l.lower(): is_contrastive = True
                
                final_type = ctype
                if is_contrastive and ctype == "Diversity":
                    final_type = "Contrastive Diversity"
                
                all_data.append({
                    "Row": row_name,
                    "Iteration": iteration,
                    "Type": final_type,
                    "Score": score
                })

    if not all_data:
        print("No data found.")
        return

    df = pd.DataFrame(all_data)
    
    # Filter Iteration > 0 (Iter 0 is Initial)
    df_running = df[df["Iteration"] > 0]
    
    print("\n--- Summary Statistics by Type (Iter > 0) ---")
    stats = df_running.groupby("Type")["Score"].agg(['count', 'mean', 'max', 'min', 'std'])
    print(stats)
    
    # Pivot for report
    print("\n--- Max Score by Type vs Iteration ---")
    pivot = df_running.pivot_table(index="Iteration", columns="Type", values="Score", aggfunc='mean')
    print(pivot)

if __name__ == "__main__":
    import sys
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_type_scores(res_dir)
