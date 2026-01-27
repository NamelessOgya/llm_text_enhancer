import json
import os
import numpy as np

def load_scores(base_path, method_name, num_rows=10):
    all_scores = {} # iter -> {'max': [], 'mean': []}
    
    for i in range(num_rows):
        path = os.path.join(base_path, method_name, f'perspectrum_llm/row_{i}/score_summary.json')
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        for iter_key, stats in data.items():
            if iter_key not in all_scores:
                all_scores[iter_key] = {'max': [], 'mean': [], 'min': []}
            all_scores[iter_key]['max'].append(stats['max'])
            all_scores[iter_key]['mean'].append(stats['mean'])
            all_scores[iter_key]['min'].append(stats['min'])
            
    aggregated = {}
    sorted_iters = sorted(all_scores.keys(), key=lambda x: int(x.replace('iter', '')))
    
    for iter_key in sorted_iters:
        aggregated[iter_key] = {
            'avg_max': np.mean(all_scores[iter_key]['max']),
            'avg_mean': np.mean(all_scores[iter_key]['mean']),
            'avg_min': np.mean(all_scores[iter_key]['min']),
            'global_max': np.max(all_scores[iter_key]['max'])
        }
    return aggregated

base_path = '/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1'

print("--- EED (Perspectrum v1.1) ---")
eed_stats = load_scores(base_path, 'eed')
for k, v in eed_stats.items():
    print(f"{k}: Avg Max: {v['avg_max']:.3f}, Avg Mean: {v['avg_mean']:.3f}, Global Max: {v['global_max']:.3f}")

print("\n--- TextGrad ---")
tg_stats = load_scores(base_path, 'textgrad')
for k, v in tg_stats.items():
    print(f"{k}: Avg Max: {v['avg_max']:.3f}, Avg Mean: {v['avg_mean']:.3f}, Global Max: {v['global_max']:.3f}")
