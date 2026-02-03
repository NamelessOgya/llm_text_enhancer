
import os
import json
import pandas as pd
import glob
import numpy as np

def analyze_results():
    base_dir = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer_edit/result/perspectrum_v1.1"
    
    data = []
    
    # Simple pattern matching might be tricky with the strategy/strategy duplication
    # Let's walk the directory manually to be safe or use specific glob
    
    # Pattern: run*/**/score_summary.json
    # Use recursive glob to handle different depths (ga/ vs gatd/gatd/)
    search_pattern = os.path.join(base_dir, "run*", "**", "score_summary.json")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} score summary files.")
    
    for file_path in files:
        parts = file_path.split(os.sep)
        # file_path example: .../result/perspectrum_v1.1/run1_p10_g15/gatd/gatd/perspectrum_llm/row_0/score_summary.json
        
        # Extract metadata from path
        try:
            # Assuming standard structure, find the index of "result" or "perspectrum_v1.1"
            idx_v1 = parts.index("perspectrum_v1.1")
            run_id = parts[idx_v1 + 1]
            
            # Strategy dir is the one immediately following run_id
            strategy_dir = parts[idx_v1 + 2] 
            strategy = strategy_dir
            
            # Row id is usually near the end, parent of the file
            # path .../row_X/score_summary.json
            row_id = parts[-2]
            
            # Validation: ensure row_id looks like "row_"
            if not row_id.startswith("row_"):
                # fallback or skip
                continue
                
        except ValueError:
            pass
            
        try:
            with open(file_path, 'r') as f:
                scores = json.load(f)
                
            for iter_key, metrics in scores.items():
                # iter_key is "iter0", "iter1", ...
                iteration = int(iter_key.replace("iter", ""))
                
                data.append({
                    "Run": run_id,
                    "Strategy": strategy,
                    "Row": row_id,
                    "Iteration": iteration,
                    "Max_Score": metrics.get("max", 0),
                    "Mean_Score": metrics.get("mean", 0),
                    "Min_Score": metrics.get("min", 0),
                    "Count": metrics.get("count", 0)
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data found!")
        return

    # Pivot / Group to get averages per strategy per iteration
    # Aggregating across runs and rows
    
    # Custom aggregation for failure rate
    def success_rate_08(x):
        return (x >= 0.8).mean()

    summary = df.groupby(['Strategy', 'Iteration']).agg({
        'Max_Score': ['mean', 'std', 'max', 'min', success_rate_08],
        'Mean_Score': ['mean', 'std']
    }).reset_index()
    
    # Flatten columns
    summary.columns = ['Strategy', 'Iteration', 
                       'Avg_Max_Score', 'Std_Max_Score', 'Global_Max_Score', 'Global_Min_Max_Score', 'Success_Rate_>=0.8',
                       'Avg_Mean_Score', 'Std_Mean_Score']
    
    print("\n=== Comparative Analysis: TextGrad vs GATD vs GA ===")
    
    strategies = summary['Strategy'].unique()
    for strat in strategies:
        strat_data = summary[summary['Strategy'] == strat]
        print(f"\n--- Strategy: {strat} ---")
        # Format output for easier reading
        print(strat_data.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
        
    # stability check: Average std deviation of max scores across rows
    # We want to see if the scores are consistent across different samples/runs
    
    print("\n=== Final Iteration Comparison ===")
    max_iter = summary['Iteration'].max()
    final_stats = summary[summary['Iteration'] == max_iter]
    print(final_stats.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Statistical Significance Test
    print("\n=== Statistical Significance Test (Paired Samples) ===")
    # Removed scipy dependency, calculating manually with numpy
    
    # Filter for final iteration
    final_data = df[df['Iteration'] == max_iter].copy()
    
    # We need to pair data by Run and Row
    pivot_data = final_data.pivot_table(index=['Run', 'Row'], columns='Strategy', values='Max_Score').dropna()
    
    if 'gatd' in pivot_data.columns and 'textgrad' in pivot_data.columns:
        gatd = pivot_data['gatd'].values
        textgrad = pivot_data['textgrad'].values
        
        n = len(gatd)
        diffs = gatd - textgrad
        mean_diff = np.mean(diffs)
        # ddof=1 for sample standard deviation
        std_diff = np.std(diffs, ddof=1)
        
        # Standard Error
        se = std_diff / np.sqrt(n)
        
        # T-statistic
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = mean_diff / se
            
        print(f"Number of paired samples (N): {n}")
        print(f"Mean Difference (GATD - TextGrad): {mean_diff:.4f}")
        print(f"Std Dev of Difference: {std_diff:.4f}")
        print(f"Standard Error: {se:.4f}")
        print(f"T-statistic: {t_stat:.4f}")
        
        # Critical values for df = n-1 (approx 29 for 30 samples)
        # Two-tailed test
        # p < 0.05  => t > 2.045
        # p < 0.01  => t > 2.756
        # p < 0.001 => t > 3.659
        
        df_dof = n - 1
        print(f"Degrees of Freedom: {df_dof}")
        
        # Approximate critical values (hardcoded for common N)
        # For N=30
        crit_05 = 2.045
        crit_01 = 2.756
        
        # Simple lookup logic for display
        if abs(t_stat) > crit_01:
             print(f"Result is Highly Significant (p < 0.01). (Critical Value: {crit_01})")
        elif abs(t_stat) > crit_05:
             print(f"Result is Significant (p < 0.05). (Critical Value: {crit_05})")
        else:
             print(f"Result is NOT Statistically Significant (p >= 0.05). (Critical Value: {crit_05})")

    else:
        print("Could not find both 'gatd' and 'textgrad' data for pairing.")

if __name__ == "__main__":
    analyze_results()
