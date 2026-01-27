import json
import os
import glob
import numpy as np

base_path = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1"
methods = ["textgrad", "eed"]
rows = range(10)

results_mean = {}
results_max = {}

for method in methods:
    method_means = {}
    method_maxs = {}
    for i in rows:
        file_path = os.path.join(base_path, method, "perspectrum_llm", f"row_{i}", "score_summary.json")
        if not os.path.exists(file_path):
            # print(f"Warning: {file_path} not found")
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for iter_key, stats in data.items():
                    if iter_key not in method_means:
                        method_means[iter_key] = []
                    method_means[iter_key].append(stats['mean'])
                    
                    if iter_key not in method_maxs:
                        method_maxs[iter_key] = []
                    method_maxs[iter_key].append(stats['max'])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Calculate average across rows for each iteration
    avg_means = {}
    for iter_key, scores in method_means.items():
        avg_means[iter_key] = np.mean(scores)
    results_mean[method] = avg_means
    
    avg_maxs = {}
    for iter_key, scores in method_maxs.items():
        avg_maxs[iter_key] = np.mean(scores)
    results_max[method] = avg_maxs

# Print comparison
print("=== Average MAX Scores ===")
print(f"{'Iteration':<10} | {'TextGrad Max':<15} | {'EED Max':<15}")
print("-" * 45)

all_iters = sorted(set(list(results_max['textgrad'].keys()) + list(results_max['eed'].keys())), key=lambda x: int(x.replace('iter', '')))

for iter_key in all_iters:
    tg_score = results_max['textgrad'].get(iter_key, "N/A")
    eed_score = results_max['eed'].get(iter_key, "N/A")
    
    tg_str = f"{tg_score:.4f}" if isinstance(tg_score, float) else str(tg_score)
    eed_str = f"{eed_score:.4f}" if isinstance(eed_score, float) else str(eed_str)
    
    print(f"{iter_key:<10} | {tg_str:<15} | {eed_str:<15}")
