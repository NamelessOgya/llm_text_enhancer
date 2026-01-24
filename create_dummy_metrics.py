import os
import json
import random

result_base = "result/test_he/he"

for row_idx in range(10):
    row_dir = os.path.join(result_base, f"row_{row_idx}")
    iter_dir = os.path.join(row_dir, "iter0")
    texts_dir = os.path.join(iter_dir, "texts")
    metrics_path = os.path.join(iter_dir, "metrics.json")
    
    if not os.path.exists(texts_dir):
        print(f"Skipping {row_dir} (no texts)")
        continue
        
    metrics = []
    for f_name in os.listdir(texts_dir):
        if not f_name.endswith(".txt"): continue
        
        metrics.append({
            "file": f_name,
            "score": random.random(), # Random score 0.0-1.0
            "reason": "Dummy evaluation"
        })
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Created {metrics_path}")
