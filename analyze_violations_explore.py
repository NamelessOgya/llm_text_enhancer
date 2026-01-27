import os
import json
import glob

base_dir = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_0"

explore_count = 0
violation_count = 0

print(f"Analyzing Explore Phase in {base_dir}...")

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
            
            if '"type": "eed_explore_persona"' in content:
                filename = os.path.basename(logic_file)
                idx = filename.replace("creation_prompt_", "").replace(".txt", "")
                text_filename = f"text_{idx}.txt"
                
                metric = metric_map.get(text_filename)
                if metric:
                    explore_count += 1
                    if metric['score'] == 0.0:
                        reason = metric.get('reason', '').lower()
                        if "word count" in reason or "limit exceeded" in reason or "too long" in reason:
                            violation_count += 1
                            print(f"[Violation] {iter_dir}/{text_filename}: {reason}")
        except: pass

print("-" * 30)
print(f"Total Explore Items: {explore_count}")
print(f"Constraint Violations: {violation_count}")
if explore_count > 0:
    print(f"Violation Rate: {violation_count / explore_count * 100:.2f}%")
