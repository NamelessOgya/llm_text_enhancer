
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set style for publication (Matching analysis/generate_figs.py)
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.family'] = 'Hiragino Sans' # Match analysis/generate_figs.py
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

BASE_DIR_LLM = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer_edit/result/perspectrum_v1.1"
BASE_DIR_RULE = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer_edit/result/perspectrum_v1.1_rule"
OUTPUT_DIR = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer_edit/analysis/figs"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_logic_type(content):
    if "Elitism" in content:
        return "Elitism"
    elif "GATD TextGrad" in content or "Gradient" in content:
        return "TextGrad"
    elif "Crossover" in content:
        return "Crossover"
    elif "Mutation" in content:
        if "Persona Mutation" in content:
            return "Persona Mutation"
        return "Mutation" 
    return "Unknown"

def load_data(base_dir, dataset_name):
    print(f"Loading data from {base_dir} for {dataset_name}...")
    data = []
    
    # We walk: run* -> strategy -> strategy -> dataset -> row_*
    row_dirs = glob.glob(os.path.join(base_dir, "run*", "**", dataset_name, "row_*"), recursive=True)
    
    for row_dir in row_dirs:
        try:
            parts = row_dir.split(os.sep)
            
            # Find run id index
            run_idx = -1
            for i, p in enumerate(parts):
                if p.startswith('run') and 'p10' in p:
                    run_idx = i
                    break
            
            if run_idx == -1:
                continue
            
            run_id = parts[run_idx]
            try:
                p_idx = parts.index(dataset_name)
                strategy = parts[p_idx - 1]
            except ValueError:
                strategy = parts[run_idx + 1]
            
            # Filter specifically for gatd_4td as requested
            if strategy != 'gatd_4td':
                continue

            row_id = parts[-1]
            iter_dirs = glob.glob(os.path.join(row_dir, "iter*"))
            
            for iter_dir in iter_dirs:
                iter_name = os.path.basename(iter_dir)
                try:
                    iteration = int(iter_name.replace("iter", ""))
                except ValueError:
                    continue

                metrics_path = os.path.join(iter_dir, "metrics.json")
                if not os.path.exists(metrics_path):
                    continue
                
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                logic_dir = os.path.join(iter_dir, "logic")
                
                for item in metrics:
                    score = item.get('score', 0.0)
                    filename = item.get('file', '')
                    
                    logic_type = "Unknown"
                    if iteration == 0:
                        logic_type = "Initialization"
                    elif os.path.exists(logic_dir) and filename:
                        match = re.search(r"(\d+)", filename)
                        if match:
                            idx_str = match.group(1)
                            prompt_file = os.path.join(logic_dir, f"creation_prompt_{idx_str}.txt")
                            if os.path.exists(prompt_file):
                                with open(prompt_file, 'r') as pf:
                                    content = pf.read()
                                    logic_type = parse_logic_type(content)
                    
                    data.append({
                        "Run": run_id,
                        "Row": row_id,
                        "Iteration": iteration,
                        "Score": score,
                        "Logic": logic_type
                    })
                    
        except Exception as e:
            print(f"Error processing {row_dir}: {e}")
            continue

    return pd.DataFrame(data)

def plot_stagnation_breakers(df, title, filename):
    print(f"Generating {filename}...")
    
    if df.empty:
        print(f"No data for {filename}")
        return

    # 1. Max Score per Iter
    iter_stats = df.groupby(['Run', 'Row', 'Iteration'])['Score'].max().reset_index()
    iter_stats = iter_stats.sort_values(['Run', 'Row', 'Iteration'])
    
    # 2. Shift for previous score
    iter_stats['PrevScore'] = iter_stats.groupby(['Run', 'Row'])['Score'].shift(1)
    
    # Filter for Iteration > 0 (Iter 0 has no prev)
    improvement_steps = iter_stats[(iter_stats['Iteration'] > 0) & (iter_stats['Score'] > iter_stats['PrevScore'])]
    
    breaker_logics = []
    
    for _, row in improvement_steps.iterrows():
        run, r_id, iter_num, max_score = row['Run'], row['Row'], row['Iteration'], row['Score']
        
        winners = df[
            (df['Run'] == run) & 
            (df['Row'] == r_id) & 
            (df['Iteration'] == iter_num) & 
            (df['Score'] == max_score)
        ]
        
        for logic in winners['Logic'].unique():
            breaker_logics.append(logic)
            
    if not breaker_logics:
        print(f"No stagnation breaking events found for {filename}.")
        return

    breaker_counts = pd.Series(breaker_logics).value_counts().reset_index()
    breaker_counts.columns = ['Logic', 'Count']
    
    # Calculate Total Applications per Logic (excluding Initialization which is not a breaker)
    total_apps = df[df['Logic'] != 'Initialization']['Logic'].value_counts().reset_index()
    total_apps.columns = ['Logic', 'Total']
    
    # Merge and calculate Contribution Rate
    results = breaker_counts.merge(total_apps, on='Logic', how='left')
    results['ContributionRate'] = (results['Count'] / results['Total']) * 100
    
    # Define Plot Order
    plot_order = ['TextGrad', 'Crossover', 'Persona Mutation']
    # Add any other logic types found in results to the end of the order
    found_logics = results['Logic'].unique()
    for logic in found_logics:
        if logic not in plot_order:
            plot_order.append(logic)

    # 1. Plot Count of Breakout Events
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x='Logic', y='Count', palette='viridis', order=plot_order)
    plt.title(title)
    plt.ylabel('Count of Breakout Events')
    plt.xlabel('Logic Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

    # 2. Plot Contribution Rate (%)
    contribution_filename = filename.replace('breakers', 'contribution')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x='Logic', y='ContributionRate', palette='viridis', order=plot_order)
    plt.title(title.replace('Number of Times', 'Improvement Contribution Rate (%)'))
    plt.ylabel('Contribution Rate (%)')
    plt.xlabel('Logic Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, contribution_filename))
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    # LLM
    df_llm = load_data(BASE_DIR_LLM, "perspectrum_llm")
    plot_stagnation_breakers(
        df_llm, 
        title='Number of Times Logic Broke Stagnation (LLM-as-a-judge)',
        filename='tagd_stagnation_breakers_llm.png'
    )
    
    # RULE
    df_rule = load_data(BASE_DIR_RULE, "perspectrum_rule")
    plot_stagnation_breakers(
        df_rule, 
        title='Number of Times Logic Broke Stagnation (METEOR)',
        filename='tagd_stagnation_breakers_meteor.png'
    )
    
    print(f"All stagnation plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
