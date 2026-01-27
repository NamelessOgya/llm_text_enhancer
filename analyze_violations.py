import os
import json
import glob

base_dir = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_0"

exploit_count = 0
violation_count = 0
zero_score_count = 0

print(f"Analyzing {base_dir}...")

for iter_dir in sorted(glob.glob(os.path.join(base_dir, "iter*"))):
    logic_dir = os.path.join(iter_dir, "logic")
    metrics_path = os.path.join(iter_dir, "metrics.json")
    
    if not os.path.exists(logic_dir) or not os.path.exists(metrics_path):
        continue
        
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            # Create a map from filename to metric
            metric_map = {m['file']: m for m in metrics}
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}")
        continue
        
    for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
        try:
            with open(logic_file, 'r') as f:
                content = f.read()
                
            # Check if it's an exploit type
            if '"type": "eed_exploit_v2"' in content or '"type": "eed_diversity_contrastive"' in content:
                # Identify index
                filename = os.path.basename(logic_file)
                idx = filename.replace("creation_prompt_", "").replace(".txt", "")
                text_filename = f"text_{idx}.txt"
                
                metric = metric_map.get(text_filename)
                if metric:
                    exploit_count += 1
                    if metric['score'] == 0.0:
                        zero_score_count += 1
                        reason = metric.get('reason', '').lower()
                        if "word count" in reason or "limit exceeded" in reason or "too long" in reason:
                            violation_count += 1
                            print(f"[Violation] {iter_dir}/{text_filename}: {reason}")
        except Exception as e:
            print(f"Error processing {logic_file}: {e}")

print("-" * 30)
print(f"Total Exploit/Diversity Items: {exploit_count}")
print(f"Total Zero Scores: {zero_score_count}")
print(f"Constraint Violations: {violation_count}")
if exploit_count > 0:
    print(f"Violation Rate: {violation_count / exploit_count * 100:.2f}%")
