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
    # logic/creation_prompt_{idx}.txt or logic metadata
    # filename is text_{idx}.txt
    try:
        idx_match = re.search(r'text_(\d+)\.txt', filename)
        if not idx_match:
            return "Unknown"
        idx = idx_match.group(1)
        
        # Try prompt file first (contains JSON logic in newer versions)
        # In src/generate.py: 
        # prompt_file = prompts/prompt_{i}.txt -> raw logic
        # creation_file = logic/creation_prompt_{i}.txt -> raw logic
        
        logic_path = os.path.join(iter_dir, "logic", f"creation_prompt_{idx}.txt")
        if not os.path.exists(logic_path):
            return "Unknown"
            
        with open(logic_path, 'r') as f:
            content = f.read().strip()
            
        # Check if it's JSON
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                t = data.get("type", "Unknown")
                
                # EED types mapping (Robust substring matching)
                if "exploit" in t: return "Exploit"
                if "diversity" in t or "novelty" in t: return "Diversity"
                if "explore" in t: return "Explore"
                if t == "inherited" or t == "eed_inherited": return "Elitism"
                if "fallback" in t: return "Fallback"
                return t # Other JSON types
        except json.JSONDecodeError:
            # Fallback for text-based logic or older versions
            if "Elitism:" in content: return "Elitism"
            pass
            
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
    # print(pivot) # Less verbose for bulk run
    
    # Identify Global Winners
    for iter_num in sorted(df["Iteration"].unique()):
        iter_data = df[df["Iteration"] == iter_num]
        if iter_data.empty: continue
        
        max_score = iter_data["Score"].max()
        winners = iter_data[iter_data["Score"] == max_score]["Type"].unique()
        print(f"Iter {iter_num}: Max {max_score:.4f} by {', '.join(winners)}")

    # Plotting
    # Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    # robustly extract metadata from path for filename
    # row_dir: .../result/exp/pop/method/eval/row_N
    parts = row_dir.split(os.sep)
    try:
        # Backward search for 'result' or know structure
        if "result" in parts:
            res_idx = parts.index("result")
            # exp is res_idx + 1
            exp_name = parts[res_idx+1]
            # remaining parts until row_dir base
            remaining = parts[res_idx+2:]
            # handle method/pop extraction crudely by joining usually safe parts
            # except the last one which is row_name
            mid_parts = remaining[:-1]
            mid_name = "_".join(mid_parts)
            
            row_name = parts[-1]
            filename_base = f"{exp_name}_{mid_name}_{row_name}"
        else:
             # Fallback
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(row_dir))) 
            row_name = os.path.basename(row_dir)
            filename_base = f"{exp_name}_{row_name}"
    except:
        dataset_name = "unknown"
        row_name = os.path.basename(row_dir)
        filename_base = f"analysis_{row_name}"
    
    plt.figure(figsize=(12, 6))
    
    # Define colors
    colors = {
        "Initial": "gray",
        "Elitism": "black",
        "Exploit": "blue",
        "Diversity": "green",
        "Explore": "red",
        "Other": "purple",
        "Fallback": "orange"
    }

    present_types = df["Type"].unique()
    
    for t in present_types:
        if t not in pivot.columns: continue
        subset = pivot[t]
        
        color = colors.get(t, "orange")
        linestyle = "--" if t == "Elitism" else "-"
        marker = "o"
        linewidth = 2 if t in ["Diversity", "Explore"] else 1.5
        
        plt.plot(subset.index, subset.values, label=t, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth)

    plt.title(f"EED Effectiveness Analysis: {exp_name} / {row_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Max Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"{filename_base}_eed_analysis.png")
    plt.savefig(out_file, dpi=150)
    plt.close() # Close figure to free memory
    print(f"Plot saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze EED Effectiveness (Exploit vs Diversity vs Explore)")
    parser.add_argument("--result-dir", required=True, help="Path to 'row_N' directory OR parent directory containing multiple 'row_*'")
    parser.add_argument("--output-dir", default="analysis/plots/eed_effectiveness", help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Directory {args.result_dir} does not exist.")
        return

    # Check if input is a specific row dir or parent dir
    # If it has "iter0", it's a row dir.
    # If it has "row_*", it's a parent dir.
    
    target_dirs = []
    if os.path.exists(os.path.join(args.result_dir, "iter0")):
        target_dirs.append(args.result_dir)
    else:
        # Recursively find row_* directories
        for root, dirs, files in os.walk(args.result_dir):
            for d in dirs:
                if d.startswith("row_"):
                   target_dirs.append(os.path.join(root, d))
        
        # Sort reasonably (try to extract numeric part)
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
