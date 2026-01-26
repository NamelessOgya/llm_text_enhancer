
import os
import json
import glob
import re
import pandas as pd
import numpy as np

base_dir = "result/perspectrum_v1.1/run1/eed/perspectrum_llm/"

def get_candidate_type(iter_dir, filename):
    try:
        idx_match = re.search(r'text_(\d+)\.txt', filename)
        if not idx_match: return "Unknown"
        idx = idx_match.group(1)
        
        logic_path = os.path.join(iter_dir, "logic", f"creation_prompt_{idx}.txt")
        if not os.path.exists(logic_path): return "Unknown"
            
        with open(logic_path, 'r') as f:
            content = f.read().strip()
            
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                t = data.get("type", "Unknown")
                if "exploit" in t: return "Exploit"
                if "diversity" in t or "novelty" in t: return "Diversity"
                if "explore" in t: return "Explore"
                if t == "inherited" or t == "eed_inherited": return "Elitism"
                if "fallback" in t: return "Fallback"
                return t 
        except:
             if "Elitism:" in content: return "Elitism"
    except:
        pass
    return "Other"

stats = []

for row_dir in glob.glob(os.path.join(base_dir, "row_*")):
    row_name = os.path.basename(row_dir)
    for iter_path in glob.glob(os.path.join(row_dir, "iter*")):
        metrics_path = os.path.join(iter_path, "metrics.json")
        if not os.path.exists(metrics_path): continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        for m in metrics:
            score = m.get("score", 0.0)
            fname = m.get("file", "")
            ctype = get_candidate_type(iter_path, fname)
            
            stats.append({
                "Row": row_name,
                "Type": ctype,
                "Score": score
            })

df = pd.DataFrame(stats)
print(df.groupby("Type")["Score"].describe())
