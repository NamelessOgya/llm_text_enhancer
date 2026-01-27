import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re

def load_metrics(iter_dir):
    metrics_path = os.path.join(iter_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return []
    with open(metrics_path, 'r') as f:
        return json.load(f)

def get_candidate_type(iter_dir, filename):
    # logic/creation_prompt_{idx}.txt
    # filename is text_{idx}.txt
    try:
        idx_match = re.search(r'text_(\d+)\.txt', filename)
        if not idx_match:
            return "Unknown"
        idx = idx_match.group(1)
        
        logic_path = os.path.join(iter_dir, "logic", f"creation_prompt_{idx}.txt")
        if not os.path.exists(logic_path):
            return "Unknown"
            
        with open(logic_path, 'r') as f:
            content = f.read().strip()
            
        # GATD Type Mapping
        if "Elitism:" in content: return "Elitism"
        if "GATD Crossover" in content: return "Crossover"
        if "GATD TextGrad" in content: return "TextGrad"
        if "GATD Mutation" in content or "GATD Persona Mutation" in content or "Adopt the following persona" in content: return "Mutation"
        
        return "Other"
    except Exception as e:
        return "Error"

def analyze_row(row_dir, output_dir):
    # Sort iterations
    iter_dirs = sorted(glob.glob(os.path.join(row_dir, "iter*")), 
                       key=lambda x: int(os.path.basename(x).replace("iter", "")))
    
    data = []

    for iter_path in iter_dirs:
        iteration = int(os.path.basename(iter_path).replace("iter", ""))
        metrics = load_metrics(iter_path)
        
        for m in metrics:
            score = m.get("score", 0.0)
            fname = m.get("file", "")
            ctype = get_candidate_type(iter_path, fname)
            
            # Map iter0 to Initial
            if iteration == 0:
                ctype = "Initial"
                
            data.append({
                "Iteration": iteration,
                "Score": score,
                "Type": ctype
            })

    if not data:
        print(f"No data found in {row_dir}.")
        return

    df = pd.DataFrame(data)
    
    print(f"\n=== Analysis for {os.path.basename(row_dir)} ===")
    
    # Calculate Max Score per Type per Iteration
    pivot = df.pivot_table(index="Iteration", columns="Type", values="Score", aggfunc="max")
    
    # Identify Global Winners
    for iter_num in sorted(df["Iteration"].unique()):
        iter_data = df[df["Iteration"] == iter_num]
        if iter_data.empty: continue
        
        max_score = iter_data["Score"].max()
        winners = iter_data[iter_data["Score"] == max_score]["Type"].unique()
        print(f"Iter {iter_num}: Max {max_score:.4f} by {', '.join(winners)}")

    # Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    # robustly extract metadata from path for filename
    parts = row_dir.split(os.sep)
    try:
        if "result" in parts:
            res_idx = parts.index("result")
            exp_name = parts[res_idx+1]
            remaining = parts[res_idx+2:]
            mid_parts = remaining[:-1]
            mid_name = "_".join(mid_parts)
            
            row_name = parts[-1]
            filename_base = f"{exp_name}_{mid_name}_{row_name}"
        else:
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(row_dir))) 
            row_name = os.path.basename(row_dir)
            filename_base = f"{exp_name}_{row_name}"
    except:
        dataset_name = "unknown"
        row_name = os.path.basename(row_dir)
        filename_base = f"analysis_{row_name}"
    
    plt.figure(figsize=(12, 6))
    
    # Define colors for GATD
    colors = {
        "Initial": "gray",
        "Elitism": "black",
        "Crossover": "blue",
        "TextGrad": "red",
        "Mutation": "darkturquoise", # Visually distinct
        "Other": "purple"
    }

    present_types = df["Type"].unique()
    
    for t in present_types:
        if t not in pivot.columns: continue
        subset = pivot[t]
        
        color = colors.get(t, "orange")
        linestyle = "--" if t == "Elitism" else "-"
        marker = "o"
        linewidth = 2 if t in ["Mutation", "TextGrad"] else 1.5
        
        plt.plot(subset.index, subset.values, label=t, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth)

    plt.title(f"GATD Effectiveness Analysis: {exp_name} / {row_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Max Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"{filename_base}_gatd_analysis.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze GATD Effectiveness (Elitism vs Crossover vs TextGrad vs Mutation)")
    parser.add_argument("--result-dir", required=True, help="Path to 'row_N' directory OR parent directory containing multiple 'row_*'")
    parser.add_argument("--output-dir", default="analysis/plots/gatd_effectiveness", help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Directory {args.result_dir} does not exist.")
        return

    target_dirs = []
    if os.path.exists(os.path.join(args.result_dir, "iter0")):
        target_dirs.append(args.result_dir)
    else:
        for root, dirs, files in os.walk(args.result_dir):
            for d in dirs:
                if d.startswith("row_"):
                   target_dirs.append(os.path.join(root, d))
        
        target_dirs.sort(key=lambda x: int(re.search(r'row_(\d+)', x).group(1)) if re.search(r'row_(\d+)', x) else 9999)
        
    if not target_dirs:
        print(f"No valid row directories found in {args.result_dir}")
        return
        
    print(f"Found {len(target_dirs)} row directories to analyze.")
    
    for r_dir in target_dirs:
        try:
            analyze_row(r_dir, args.output_dir)
        except Exception as e:
            print(f"Failed to analyze {r_dir}: {e}")

if __name__ == "__main__":
    main()
