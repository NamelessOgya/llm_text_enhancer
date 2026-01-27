import json
import os
import glob

base_path = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1"
row_id = "row_7"
iter_id = "iter8"
methods = ["textgrad", "eed"]

print(f"=== Analysis for {row_id}  ({iter_id}) ===")

for method in methods:
    print(f"\nMethod: {method}")
    iter_path = os.path.join(base_path, method, "perspectrum_llm", row_id, iter_id)
    metrics_path = os.path.join(iter_path, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print("Metrics file not found.")
        continue

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            # metrics is likely a list of scores or a dictionary mapping id to score
            # Let's inspect the structure by assuming it's a dict or list
            # If it's a list, index corresponds to text_i.txt
            pass
    except Exception as e:
        print(f"Error reading metrics: {e}")
        metrics = []

    # Read all texts
    texts = []
    text_dir = os.path.join(iter_path, "texts")
    if os.path.exists(text_dir):
        files = sorted(glob.glob(os.path.join(text_dir, "text_*.txt")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        for fp in files:
            with open(fp, 'r') as f:
                texts.append(f.read().strip())
    
    # Combine (assuming list order for metrics matches file order)
    # The metrics.json format in previous views was not fully shown, but often it's a list of results.
    # Let's print raw metrics first to be safe, then try to zip.
    print(f"Metrics Raw: {metrics}")
    
    # Try to pair if possible
    params = metrics.get('parameters', []) if isinstance(metrics, dict) else metrics
    # If parameters is a list of objects with 'score'
    
    if isinstance(params, list) and len(params) == len(texts):
        zipper = zip(texts, params)
        sorted_pairs = sorted(zipper, key=lambda x: x[1].get('score', 0) if isinstance(x[1], dict) else 0, reverse=True)
        
        print(f"{'Score':<6} | Text")
        print("-" * 60)
        for text, metric in sorted_pairs:
            score = metric.get('score', 'N/A') if isinstance(metric, dict) else 'N/A'
            print(f"{score:<6} | {text}")
    else:
        print("Could not align metrics and texts automatically. Printing texts:")
        for i, t in enumerate(texts):
            print(f"[{i}] {t}")
