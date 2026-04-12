
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.ticker as mtick

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

# Define consistent colors for logics (Hardcoded HEX for absolute consistency)
LOGIC_COLORS = {
    'TextGrad': '#440154',
    'Crossover': '#31688e',
    'Persona Mutation': '#35b779',
    'Mutation': '#e66101'
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_logic_type(content):
    content_lower = content.lower()
    if "elitism" in content_lower:
        return "Elitism"
    elif "gatd textgrad" in content_lower or "gradient" in content_lower:
        return "TextGrad"
    elif "crossover" in content_lower:
        return "Crossover"
    elif "mutation" in content_lower:
        if "persona mutation" in content_lower:
            return "Persona Mutation"
        return "Mutation" 
    return "Unknown"

def load_data(base_dir, dataset_name):
    print(f"Loading data from {base_dir} for {dataset_name}...")
    data = []
    
    row_dirs = glob.glob(os.path.join(base_dir, "run*", "**", dataset_name, "row_*"), recursive=True)
    
    for row_dir in row_dirs:
        try:
            parts = row_dir.split(os.sep)
            
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

                if iteration > 15:
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

def get_score_bin(score, is_meteor=False):
    if is_meteor:
        # Scale to 0-10 for comparison if it's 0-1 standard
        # But let's check the actual data first or use relative bins
        # METEOR is usually 0-1. Let's use 0-0.3, 0.3-0.6, 0.6+ if it's 0-1.
        # However, for consistency with the prompt, I'll try to find if it's already 0-10.
        # If max score > 1, assume 0-10.
        if score < 3.0: return "0-3"
        elif score < 6.0: return "3-6"
        else: return "6-10"
    else:
        if score < 3.0: return "0-3"
        elif score < 6.0: return "3-6"
        else: return "6-10"

def plot_stagnation_score_split(df, title, filename, is_meteor=False):
    print(f"Generating {filename}...")
    
    if df.empty:
        print(f"No data for {filename}")
        return

    # Check score range
    max_score_seen = df['Score'].max()
    print(f"Max score in data: {max_score_seen}")
    
    # Scale to 0-10 for binning if it's 0-1 scale
    # This aligns 0.7 score into "6-10" bin per user request "0-3, 3-6, 6+"
    scale_factor = 1.0
    if max_score_seen <= 1.1:
        print("Detected 0-1 scale. Scaling scores by 10 for binning (0-3, 3-6, 6-10).")
        scale_factor = 10.0
    
    def get_scaled_bin(s):
        s_scaled = s * scale_factor
        if s_scaled < 3.0: return "0-3"
        elif s_scaled < 6.0: return "3-6"
        else: return "6-10"
        
    bin_order = ["0-3", "3-6", "6-10"]
    x_label = 'Score Range (Scaled to 0-10)' if scale_factor == 10.0 else 'Score Range'

    # 1. Max Score per Iter
    iter_stats = df.groupby(['Run', 'Row', 'Iteration'])['Score'].max().reset_index()
    iter_stats = iter_stats.sort_values(['Run', 'Row', 'Iteration'])
    
    # 2. Previous score shift
    iter_stats['PrevScore'] = iter_stats.groupby(['Run', 'Row'])['Score'].shift(1)
    
    # Improvement steps
    improvement_steps = iter_stats[(iter_stats['Iteration'] > 0) & (iter_stats['Score'] > iter_stats['PrevScore'])]
    
    breaker_data = []
    
    for _, row in improvement_steps.iterrows():
        run, r_id, iter_num, max_score, prev_score = row['Run'], row['Row'], row['Iteration'], row['Score'], row['PrevScore']
        
        winners = df[
            (df['Run'] == run) & 
            (df['Row'] == r_id) & 
            (df['Iteration'] == iter_num) & 
            (df['Score'] == max_score)
        ]
        
        score_bin = get_scaled_bin(prev_score)

        for logic in winners['Logic'].unique():
            breaker_data.append({"ScoreBin": score_bin, "Logic": logic, "Event": 1})
            
    if not breaker_data:
        print(f"No breakout events for {filename}")
        return

    breaker_df = pd.DataFrame(breaker_data)
    breaker_counts = breaker_df.groupby(['ScoreBin', 'Logic']).size().reset_index(name='Count')

    # Total applications per ScoreBin and Logic
    app_stats = iter_stats.copy()
    app_stats['ScoreBin'] = app_stats['PrevScore'].apply(lambda x: get_scaled_bin(x) if pd.notnull(x) else None)
    
    # Merge apps with the original df to get Logic
    df_with_bin = df.merge(app_stats[['Run', 'Row', 'Iteration', 'ScoreBin']], on=['Run', 'Row', 'Iteration'], how='left')
    
    total_apps = df_with_bin[(df_with_bin['Logic'] != 'Initialization') & (df_with_bin['ScoreBin'].notnull())].groupby(['ScoreBin', 'Logic']).size().reset_index(name='Total')

    # Merge
    results = total_apps.merge(breaker_counts, on=['ScoreBin', 'Logic'], how='left').fillna(0)
    results['ContributionRate'] = (results['Count'] / results['Total']) * 100

    # Define Plot Order
    logic_order = ['TextGrad', 'Crossover', 'Persona Mutation']

    # Filter for interesting logics
    results = results[results['Logic'].isin(logic_order)]

    # Plot
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=results, x='ScoreBin', y='ContributionRate', hue='Logic', hue_order=logic_order, palette=LOGIC_COLORS, order=bin_order)

    plt.title(title)
    plt.ylabel('Improvement Contribution Rate (%)')
    plt.xlabel(x_label)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    # LLM
    df_llm = load_data(BASE_DIR_LLM, "perspectrum_llm")
    plot_stagnation_score_split(
        df_llm, 
        title='Logic Contribution Rate by Score Range (LLM-as-a-judge)',
        filename='tagd_stagnation_contribution_score_llm.png',
        is_meteor=False
    )
    
    # RULE
    df_rule = load_data(BASE_DIR_RULE, "perspectrum_rule")
    plot_stagnation_score_split(
        df_rule, 
        title='Logic Contribution Rate by Score Range (METEOR)',
        filename='tagd_stagnation_contribution_score_meteor.png',
        is_meteor=True
    )
    
    print(f"Score contribution plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
