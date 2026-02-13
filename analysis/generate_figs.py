
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.family'] = 'Hiragino Sans'
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

def load_data(base_dir, dataset_name):
    print(f"Loading data from {base_dir}...")
    data = []
    
    # We walk: run* -> strategy -> strategy -> dataset -> row_*
    row_dirs = glob.glob(os.path.join(base_dir, "run*", "**", dataset_name, "row_*"), recursive=True)
    
    print(f"Found {len(row_dirs)} row directories in {dataset_name}.")
    
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
            # Strategy is the folder right before dataset_name
            try:
                p_idx = parts.index(dataset_name)
                strategy = parts[p_idx - 1]
            except ValueError:
                strategy = parts[run_idx + 1]
            
            # Filter strategies
            if strategy == 'ga':
                display_strategy = 'GA'
            elif strategy == 'textgrad':
                display_strategy = 'TD'
            elif strategy == 'gatd_4td':
                display_strategy = 'GATD'
            else:
                continue # Skip other strategies

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
                
                for item in metrics:
                    score = item.get('score', 0.0)
                    data.append({
                        "Run": run_id,
                        "Strategy": display_strategy,
                        "Row": row_id,
                        "Iteration": iteration,
                        "Score": score
                    })
                    
        except Exception as e:
            print(f"Error processing {row_dir}: {e}")
            continue

    df = pd.DataFrame(data)
    return df

def plot_avg_max_score(df, ylabel, title, filename):
    print(f"Generating {filename}...")
    
    # Calculate Max Score per Run/Row/Iter
    run_max_scores = df.groupby(['Run', 'Strategy', 'Row', 'Iteration'])['Score'].max().reset_index()
    
    plt.figure(figsize=(8, 5))
    # Filter for GA, TD, GATD to ensure correct order/selection
    strategies = ['GA', 'TD', 'GATD']
    filtered_scores = run_max_scores[run_max_scores['Strategy'].isin(strategies)]
    
    # Sort strategies for consistent coloring
    filtered_scores['Strategy'] = pd.Categorical(filtered_scores['Strategy'], categories=strategies, ordered=True)
    
    sns.lineplot(data=filtered_scores, x='Iteration', y='Score', hue='Strategy', style='Strategy', markers=True, dashes=False, errorbar=None)
    
    plt.ylabel(ylabel)
    plt.xlabel('Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if "METEOR" in ylabel:
        plt.ylim(0.0, 0.4)
    else:
        plt.ylim(0.5, 1.05)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    # LLM-as-a-judge (perspectrum_llm)
    df_llm = load_data(BASE_DIR_LLM, "perspectrum_llm")
    if not df_llm.empty:
        plot_avg_max_score(
            df_llm, 
            ylabel='Score(LLM as a judge)', 
            title='図 1: 手法ごとのスコア推移の比較 (LLM-as-a-judge)',
            filename='avg_max_score_llm.png'
        )
    
    # METEOR (perspectrum_rule)
    df_rule = load_data(BASE_DIR_RULE, "perspectrum_rule")
    if not df_rule.empty:
        plot_avg_max_score(
            df_rule, 
            ylabel='Score(METEOR)', 
            title='図 2: 手法ごとのスコア推移の比較 (METEOR)',
            filename='avg_max_score_meteor.png'
        )
    
    print(f"All plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
