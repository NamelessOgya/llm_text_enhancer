import os
import json
import glob
import re

base_dir = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_0"

print(f"Checking Elitism Stability in {base_dir}...")

for iter_dir in sorted(glob.glob(os.path.join(base_dir, "iter*"))):
    logic_dir = os.path.join(iter_dir, "logic")
    metrics_path = os.path.join(iter_dir, "metrics.json")
    
    if not os.path.exists(logic_dir) or not os.path.exists(metrics_path):
        continue
        
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            metric_map = {m['file']: m for m in metrics}
    except: continue
    
    for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
        try:
            with open(logic_file, 'r') as f:
                content = f.read()

            filename = os.path.basename(logic_file)
            idx = filename.replace("creation_prompt_", "").replace(".txt", "")
            text_filename = f"text_{idx}.txt"

            metric = metric_map.get(text_filename)
            if not metric: continue
            
            current_score = metric['score']
            
            # Check for Elitism logic
            # "Elitism: Rank {i+1} preserved (Score: {elite['score']})"
            # Logic File might be simple string not JSON structure for Elitism (based on previous view)
            
            match = re.search(r"Score: ([\d\.]+)", content)
            
            if "Elitism:" in content and match:
                parent_score = float(match.group(1))
                diff = parent_score - current_score
                
                print(f"[ELITE] {iter_dir}/{text_filename} | Parent: {parent_score} -> Current: {current_score} (Diff: {diff:.2f})")
                
                    
        except Exception as e:
            # print(f"Error: {e}")
            pass

print("Done.")
