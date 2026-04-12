
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
    elif "crossover" in content_lower:
        return "Crossover"
    elif "mutation" in content_lower:
        if "persona mutation" in content_lower:
            return "Persona Mutation"
        return "Mutation" 
    return "Unknown"

def load_data(base_dir, dataset_name):
    print(f"Loading GA data from {base_dir} for {dataset_name}...")
    data = []
    
    # Path pattern for GA results
    row_dirs = glob.glob(os.path.join(base_dir, "run*", "ga", dataset_name, "row_*"), recursive=True)
    
    for row_dir in row_dirs:
        try:
            parts = row_dir.split(os.sep)
            
            # Find run id
            run_idx = -1
            for i, p in enumerate(parts):
                if p.startswith('run') and 'p10' in p:
                    run_idx = i
                    break
            if run_idx == -1: continue
            
            run_id = parts[run_idx]
            row_id = parts[-1]
            iter_dirs = glob.glob(os.path.join(row_dir, "iter*"))
            
            for iter_dir in iter_dirs:
                iter_name = os.path.basename(iter_dir)
                try:
                    iteration = int(iter_name.replace("iter", ""))
                except ValueError: continue

                if iteration > 15: continue

                metrics_path = os.path.join(iter_dir, "metrics.json")
                if not os.path.exists(metrics_path): continue
                
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

def get_phase(iteration):
    if 1 <= iteration <= 5: return "1-5"
    elif 6 <= iteration <= 10: return "6-10"
    elif 11 <= iteration <= 15: return "11-15"
    return None

def get_scaled_bin(score, scale_factor):
    s_scaled = score * scale_factor
    if s_scaled < 3.0: return "0-3"
    elif s_scaled < 6.0: return "3-6"
    else: return "6-10"

def generate_plots(df, base_title, file_prefix):
    if df.empty:
        print(f"No data for {file_prefix}")
        return

    # 1. Prep stats
    iter_stats = df.groupby(['Run', 'Row', 'Iteration'])['Score'].max().reset_index()
    iter_stats = iter_stats.sort_values(['Run', 'Row', 'Iteration'])
    iter_stats['PrevScore'] = iter_stats.groupby(['Run', 'Row'])['Score'].shift(1)
    improvement_steps = iter_stats[(iter_stats['Iteration'] > 0) & (iter_stats['Score'] > iter_stats['PrevScore'])]

    # Logic filter
    logic_order = ['Crossover', 'Mutation']
    
    # --- Data for Overall ---
    breaker_logics = []
    for _, row in improvement_steps.iterrows():
        winners = df[(df['Run'] == row['Run']) & (df['Row'] == row['Row']) & (df['Iteration'] == row['Iteration']) & (df['Score'] == row['Score'])]
        for logic in winners['Logic'].unique():
            if logic in logic_order:
                breaker_logics.append(logic)
    
    if breaker_logics:
        breaker_counts = pd.Series(breaker_logics).value_counts().reset_index()
        breaker_counts.columns = ['Logic', 'Count']
        total_apps = df[df['Logic'].isin(logic_order)].groupby('Logic').size().reset_index(name='Total')
        results = total_apps.merge(breaker_counts, on='Logic', how='left').fillna(0)
        results['ContributionRate'] = (results['Count'] / results['Total']) * 100
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=results, x='Logic', y='ContributionRate', palette=LOGIC_COLORS, order=logic_order)
        
        plt.title(f"{base_title} (Overall)")
        plt.ylabel('Improvement Contribution Rate (%)')
        plt.xlabel('Logic Type')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{file_prefix}_contribution.png"))
        plt.close()

    # --- Data for Phase Split ---
    phase_data = []
    for _, row in improvement_steps.iterrows():
        phase = get_phase(row['Iteration'])
        if phase:
            winners = df[(df['Run'] == row['Run']) & (df['Row'] == row['Row']) & (df['Iteration'] == row['Iteration']) & (df['Score'] == row['Score'])]
            for logic in winners['Logic'].unique():
                if logic in logic_order:
                    phase_data.append({"Phase": phase, "Logic": logic, "Event": 1})
    
    if phase_data:
        df['Phase'] = df['Iteration'].apply(get_phase)
        total_apps = df[df['Logic'].isin(logic_order) & df['Phase'].notnull()].groupby(['Phase', 'Logic']).size().reset_index(name='Total')
        breaker_counts = pd.DataFrame(phase_data).groupby(['Phase', 'Logic']).size().reset_index(name='Count')
        results = total_apps.merge(breaker_counts, on=['Phase', 'Logic'], how='left').fillna(0)
        results['ContributionRate'] = (results['Count'] / results['Total']) * 100
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=results, x='Phase', y='ContributionRate', hue='Logic', hue_order=logic_order, palette=LOGIC_COLORS, order=["1-5", "6-10", "11-15"])

        plt.title(f"{base_title} by Phase")
        plt.ylabel('Improvement Contribution Rate (%)')
        plt.xlabel('Iterations')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{file_prefix}_contribution_split.png"))
        plt.close()

    # --- Data for Score Split ---
    max_score_seen = df['Score'].max()
    scale_factor = 10.0 if max_score_seen <= 1.1 else 1.0
    
    score_data = []
    for _, row in improvement_steps.iterrows():
        s_bin = get_scaled_bin(row['PrevScore'], scale_factor)
        winners = df[(df['Run'] == row['Run']) & (df['Row'] == row['Row']) & (df['Iteration'] == row['Iteration']) & (df['Score'] == row['Score'])]
        for logic in winners['Logic'].unique():
            if logic in logic_order:
                score_data.append({"ScoreBin": s_bin, "Logic": logic})
    
    if score_data:
        app_stats = iter_stats.copy()
        app_stats['ScoreBin'] = app_stats['PrevScore'].apply(lambda x: get_scaled_bin(x, scale_factor) if pd.notnull(x) else None)
        df_with_bin = df.merge(app_stats[['Run', 'Row', 'Iteration', 'ScoreBin']], on=['Run', 'Row', 'Iteration'], how='left')
        total_apps = df_with_bin[df_with_bin['Logic'].isin(logic_order) & df_with_bin['ScoreBin'].notnull()].groupby(['ScoreBin', 'Logic']).size().reset_index(name='Total')
        breaker_counts = pd.DataFrame(score_data).groupby(['ScoreBin', 'Logic']).size().reset_index(name='Count')
        results = total_apps.merge(breaker_counts, on=['ScoreBin', 'Logic'], how='left').fillna(0)
        results['ContributionRate'] = (results['Count'] / results['Total']) * 100
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=results, x='ScoreBin', y='ContributionRate', hue='Logic', hue_order=logic_order, palette=LOGIC_COLORS, order=["0-3", "3-6", "6-10"])

        plt.title(f"{base_title} by Score Range")
        plt.ylabel('Improvement Contribution Rate (%)')
        plt.xlabel('Score Range (Scaled to 0-10)' if scale_factor == 10.0 else 'Score Range')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{file_prefix}_contribution_score.png"))
        plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    # LLM
    df_llm = load_data(BASE_DIR_LLM, "perspectrum_llm")
    generate_plots(df_llm, "GA Logic Contribution (LLM)", "ga_stagnation")
    
    # RULE
    df_rule = load_data(BASE_DIR_RULE, "perspectrum_rule")
    generate_plots(df_rule, "GA Logic Contribution (METEOR)", "ga_stagnation_meteor")
    
    print(f"GA stagnation plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
