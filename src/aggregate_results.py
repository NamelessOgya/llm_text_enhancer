import os
import json
import glob
import pandas as pd
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results across tasks and methods.")
    parser.add_argument("--result-dir", default="result", help="Root directory of results")
    parser.add_argument("--output", default="result/aggregation_report.csv", help="Output CSV file path")
    return parser.parse_args()

def aggregate_scores(result_dir):
    data = []
    
    # Structure: result/{task_id}/{method}/row_{N}/iter{M}/...
    # We look for result/{task_id}/{method}/row_{N}/score_summary.json
    
    # glob pattern to find score_summary.json
    # We assume 3 levels of depth for task/method/row
    pattern = os.path.join(result_dir, "*", "*", "row_*", "score_summary.json")
    summary_files = glob.glob(pattern)
    
    print(f"Found {len(summary_files)} summary files.")
    
    for summary_file in summary_files:
        try:
            # path parts: .../result/{task_id}/{method}/{row_dir}/score_summary.json
            parts = summary_file.split(os.sep)
            # Find index of result_dir to identify relative parts
            # simplistic approach: take last 4 parts: task, method, row, filename
            if len(parts) < 4:
                continue
                
            task_id = parts[-4]
            method = parts[-3]
            row_name = parts[-2]
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            for iter_name, stats in summary.items():
                # iter_name example: "iter0"
                iteration = int(iter_name.replace("iter", ""))
                
                data.append({
                    "Task": task_id,
                    "Method": method,
                    "Row": row_name,
                    "Iteration": iteration,
                    "Max_Score": stats.get("max", 0.0),
                    "Min_Score": stats.get("min", 0.0),
                    "Mean_Score": stats.get("mean", 0.0),
                    "Count": stats.get("count", 0)
                })
                
        except Exception as e:
            print(f"Error processing {summary_file}: {e}")
            continue

    return pd.DataFrame(data)

def main():
    args = parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"Result directory '{args.result_dir}' does not exist.")
        sys.exit(1)
        
    df = aggregate_scores(args.result_dir)
    
    if df.empty:
        print("No data found.")
        sys.exit(0)
        
    # Validation/Sanity check
    print(f"Collected {len(df)} data points.")
    
    # Group by Task, Method, Iteration to calculate global stats
    # We want: Mean of Max Scores (across rows), Global Max (across rows), etc.
    grouped = df.groupby(["Task", "Method", "Iteration"]).agg({
        "Max_Score": ["mean", "max", "min", "std"],
        "Min_Score": ["mean", "min"]
    })
    
    # Flatten columns
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
    grouped = grouped.reset_index()
    
    # Rename for clarity
    grouped = grouped.rename(columns={
        "Max_Score_mean": "Avg_Max_Score",
        "Max_Score_max": "Global_Max_Score",
        "Max_Score_min": "Worst_Max_Score",
        "Max_Score_std": "Std_Max_Score",
        "Min_Score_mean": "Avg_Min_Score",
        "Min_Score_min": "Global_Min_Score"
    })
    
    # Sort
    grouped = grouped.sort_values(by=["Task", "Method", "Iteration"])
    
    # Output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    grouped.to_csv(args.output, index=False)
    print(f"Aggregation report saved to: {args.output}")
    
    # Also print a preview
    print("\nPreview of Aggregated Results:")
    print(grouped.head(20).to_string())

if __name__ == "__main__":
    main()
