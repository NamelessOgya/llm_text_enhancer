
import os
import json
import numpy as np

def get_final_stats(method_dir):
    max_scores = []
    mean_scores = []
    
    # Rows 0 to 9
    for i in range(10):
        row_dir = os.path.join(method_dir, f"row_{i}")
        summary_path = os.path.join(row_dir, "score_summary.json")
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
                
            # Get last iteration (iter9)
            last_iter = "iter9"
            if last_iter in data:
                max_scores.append(data[last_iter].get("max", 0))
                mean_scores.append(data[last_iter].get("mean", 0))
            else:
                # Fallback to last available
                last_key = sorted(data.keys(), key=lambda x: int(x.replace("iter", "")))[-1]
                max_scores.append(data[last_key].get("max", 0))
                mean_scores.append(data[last_key].get("mean", 0))
                
    return max_scores, mean_scores

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare GATD and TextGrad")
    parser.add_argument("--base-dir", required=True, help="Base directory containing method subdirectories, e.g. result/exp_id/pop_name")
    args = parser.parse_args()

    base_dir = args.base_dir
    gatd_dir = os.path.join(base_dir, "gatd", "perspectrum_llm")
    tg_dir = os.path.join(base_dir, "textgrad", "perspectrum_llm")
    
    if not os.path.exists(gatd_dir):
        print(f"Error: GATD directory not found: {gatd_dir}")
        return
    if not os.path.exists(tg_dir):
        print(f"Error: TextGrad directory not found: {tg_dir}")
        return

    g_max, g_mean = get_final_stats(gatd_dir)
    t_max, t_mean = get_final_stats(tg_dir)
    
    print("=== Comparison (Iter 9, 10 Rows) ===")
    print(f"{'Metric':<10} {'GATD':<10} {'TextGrad':<10} {'Diff':<10}")
    print("-" * 40)
    print(f"Avg Max   {np.mean(g_max):.4f}     {np.mean(t_max):.4f}     {np.mean(g_max) - np.mean(t_max):+.4f}")
    print(f"Avg Mean  {np.mean(g_mean):.4f}     {np.mean(t_mean):.4f}     {np.mean(g_mean) - np.mean(t_mean):+.4f}")
    print("-" * 40)
    print("Raw Max Scores:")
    print(f"GATD: {g_max}")
    print(f"TextGrad: {t_max}")

if __name__ == "__main__":
    main()
