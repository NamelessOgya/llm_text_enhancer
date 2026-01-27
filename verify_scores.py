import json
import os
import glob
import numpy as np

base_path = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1"
methods = ["textgrad", "eed"]
rows = range(10)

print(f"Checking data in: {base_path}")

for method in methods:
    method_maxs = {}
    absolute_max = -1.0
    row_count = 0
    
    for i in rows:
        file_path = os.path.join(base_path, method, "perspectrum_llm", f"row_{i}", "score_summary.json")
        if not os.path.exists(file_path):
            print(f"[{method}] Missing: row_{i}")
            continue
        
        row_count += 1
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for iter_key, stats in data.items():
                    if iter_key not in method_maxs:
                        method_maxs[iter_key] = []
                    
                    val = stats['max']
                    method_maxs[iter_key].append(val)
                    if val > absolute_max:
                        absolute_max = val
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"\nMethod: {method}")
    print(f"Processed Rows: {row_count}")
    print(f"Absolute Max Score found in any row: {absolute_max}")
    
    # Calculate average max
    print(f"{'Iteration':<10} | {'Avg Max':<10} | {'Raw Maxes (iter 8)'}")
    sorted_iters = sorted(method_maxs.keys(), key=lambda x: int(x.replace('iter', '')))
    for iter_key in sorted_iters:
        avg = np.mean(method_maxs[iter_key])
        raw_values = method_maxs[iter_key] if iter_key == 'iter8' else ""
        print(f"{iter_key:<10} | {avg:<10.4f} | {raw_values}")
