import json
import os
import glob

base_path = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1"
row_id = "row_1" # Changed to row_1
iter_id = "iter8"
methods = ["textgrad", "eed"]

print(f"=== Analysis for {row_id}  ({iter_id}) - CORRECTED MAPPING ===")

for method in methods:
    print(f"\nMethod: {method}")
    iter_path = os.path.join(base_path, method, "perspectrum_llm", row_id, iter_id)
    metrics_path = os.path.join(iter_path, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print("Metrics file not found.")
        continue

    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            # Handle if metrics is dict with 'parameters' key or direct list
            if isinstance(metrics_data, dict) and 'parameters' in metrics_data:
                metrics_list = metrics_data['parameters']
            elif isinstance(metrics_data, list):
                metrics_list = metrics_data
            else:
                metrics_list = []
    except Exception as e:
        print(f"Error reading metrics: {e}")
        metrics_list = []

    # Map filename -> metric object
    metric_map = {}
    for m in metrics_list:
        fname = m.get('file')
        if fname:
            metric_map[fname] = m

    # Read all texts
    text_dir = os.path.join(iter_path, "texts")
    results = []
    
    if os.path.exists(text_dir):
        files = sorted(glob.glob(os.path.join(text_dir, "text_*.txt")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        for fp in files:
            fname = os.path.basename(fp)
            with open(fp, 'r') as f:
                content = f.read().strip()
            
            metric = metric_map.get(fname, {})
            score = metric.get('score', 'N/A')
            reason = metric.get('reason', 'N/A')
            results.append({
                'score': score,
                'text': content,
                'reason': reason,
                'file': fname
            })

    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x['score'] if isinstance(x['score'], (int, float)) else -1, reverse=True)
        
    print(f"{'Score':<6} | {'File':<10} | Text")
    print("-" * 80)
    for res in sorted_results:
        text_preview = res['text'][:100] + "..." if len(res['text']) > 100 else res['text']
        print(f"{res['score']:<6} | {res['file']:<10} | {text_preview}")
