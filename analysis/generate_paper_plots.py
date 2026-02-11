
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
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

BASE_DIR = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer_edit/result/perspectrum_v1.1"
OUTPUT_DIR = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer_edit/analysis/plots/paper"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_logic_type(content):
    """
    Parse the logic type from creation_prompt content.
    """
    if "Elitism" in content:
        return "Elitism"
    elif "GATD TextGrad" in content or "Gradient" in content:
        return "TextGrad"
    elif "Crossover" in content:
        return "Crossover"
    elif "Mutation" in content: # Fallback for specific mutations might be needed if user wants "Stance Mutation" etc. specifically
        # Check specific mutation types if present in logs, otherwise generic Mutation
        if "Persona Mutation" in content:
            return "Persona Mutation"
        return "Mutation" 
    return "Unknown"

def load_data():
    print(f"Loading data from {BASE_DIR}...")
    data = []
    
    # We walk: run* -> strategy -> strategy -> dataset -> row_*
    # Strategy pattern: 
    # run1.../ga/ga/perspectrum_llm/row_X
    # run1.../gatd/gatd/perspectrum_llm/row_X
    # run1.../textgrad/textgrad/perspectrum_llm/row_X
    
    # Using recursive glob to find row directories regardless of depth
    row_dirs = glob.glob(os.path.join(BASE_DIR, "run*", "**", "perspectrum_llm", "row_*"), recursive=True)
    
    print(f"Found {len(row_dirs)} row directories.")
    
    for row_dir in row_dirs:
        try:
            parts = row_dir.split(os.sep)
            
            # Find run id index (starts with 'run')
            run_idx = -1
            for i, p in enumerate(parts):
                if p.startswith('run') and 'p10' in p: # Heuristic to find run folder
                    run_idx = i
                    break
            
            if run_idx == -1:
                # Fallback: assume standard relative position from 'perspectrum_llm'
                # row_dir ends with .../perspectrum_llm/row_X
                # parts[-3] is perspectrum_llm
                # search backwards
                try:
                    p_idx = parts.index("perspectrum_llm")
                    # strategy is usually p_idx - 1 (ga) or p_idx - 2 (gatd/gatd)?
                    # Strategy name is better derived from the folder *immediately inside* the Run folder.
                    # Run folder is likely p_idx - X.
                    pass
                except ValueError:
                    continue
            
            if run_idx != -1:
                run_id = parts[run_idx]
                # Strategy is the folder right before perspectrum_llm
                try:
                    p_idx = parts.index("perspectrum_llm")
                    strategy = parts[p_idx - 1]
                except ValueError:
                    strategy = parts[run_idx + 1]
            else:
                 # Last resort fallback if path prevents finding run somehow
                 continue
                 
            row_id = parts[-1]
            
            # Map strategy names to display names if needed
            if strategy == 'ga':
                display_strategy = 'GA'
            elif strategy == 'gatd':
                display_strategy = 'TAGD'
            elif strategy == 'gatd_4td':
                display_strategy = 'TAGD (4TD)'
            elif strategy == 'textgrad':
                display_strategy = 'TD'
            elif strategy == 'tagd':
                display_strategy = 'TAGD'
            else:
                display_strategy = strategy.upper()

            # Load metrics.json for each iteration
            # Iterations are subdirs: iter0, iter1, ...
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
                
                # Load logic prompts if available
                logic_dir = os.path.join(iter_dir, "logic")
                
                for item in metrics:
                    # Item keys: file, score, reason
                    score = item.get('score', 0.0)
                    filename = item.get('file', '')
                    
                    logic_type = "Unknown"
                    if display_strategy == 'GA':
                        logic_type = "GA" 
                    else:
                        # For TAGD and TD, check logic files
                        if iteration == 0:
                            logic_type = "Initialization"
                        elif os.path.exists(logic_dir) and filename:
                            # Use numeric index from filename to find logic prompt
                            # text_0.txt -> creation_prompt_0.txt
                            match = re.search(r"(\d+)", filename)
                            if match:
                                idx_str = match.group(1)
                                prompt_file = os.path.join(logic_dir, f"creation_prompt_{idx_str}.txt")
                                if os.path.exists(prompt_file):
                                    with open(prompt_file, 'r') as pf:
                                        content = pf.read()
                                        logic_type = parse_logic_type(content)
                                else:
                                    logic_type = "Unknown (Missing File)"
                            else:
                                logic_type = "Unknown (No naming convention)"
                        else:
                            logic_type = "Unknown (No logic dir)"
                    
                    data.append({
                        "Run": run_id,
                        "Strategy": display_strategy,
                        "Row": row_id,
                        "Iteration": iteration,
                        "File": filename,
                        "Score": score,
                        "Logic": logic_type
                    })
                    
        except Exception as e:
            print(f"Error processing {row_dir}: {e}")
            continue

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} data points.")
    return df

def plot_avg_score(df, show_ci=True):
    """
    1. GA/ TAGD/ TDの各iterの平均スコア
    """
    suffix = "_ci" if show_ci else "_noci"
    errorbar = ("ci", 95) if show_ci else None
    
    print(f"Generating Average Score Plot ({suffix})...")
    
    # Calculate Max Score per Run/Row/Iter
    run_max_scores = df.groupby(['Run', 'Strategy', 'Row', 'Iteration'])['Score'].max().reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=run_max_scores, x='Iteration', y='Score', hue='Strategy', style='Strategy', markers=True, dashes=False, errorbar=errorbar)
    
    plt.title(f'Average Max Score per Iteration {"(with 95% CI)" if show_ci else ""}')
    plt.ylabel('Score')
    plt.xlabel('Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"avg_max_score{suffix}.png"))
    plt.close()
    
    # Also plot Average Mean Score (Population Quality)
    run_mean_scores = df.groupby(['Run', 'Strategy', 'Row', 'Iteration'])['Score'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=run_mean_scores, x='Iteration', y='Score', hue='Strategy', style='Strategy', markers=True, dashes=False, errorbar=errorbar)
    
    plt.title(f'Average Population Mean Score per Iteration {"(with 95% CI)" if show_ci else ""}')
    plt.ylabel('Mean Score')
    plt.xlabel('Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"avg_population_mean_score{suffix}.png"))
    plt.close()

def plot_reach_rate(df, show_ci=True):
    """
    2. GA/ TAGD/ TDの各iterの80%到達率
    """
    suffix = "_ci" if show_ci else "_noci"
    errorbar = ("ci", 95) if show_ci else None
    
    print(f"Generating Reach Rate Plot ({suffix})...")
    
    # 1. Calculate Max Score per Run/Row/Iter
    sample_max = df.groupby(['Run', 'Strategy', 'Row', 'Iteration'])['Score'].max().reset_index()
    
    # 2. Check if Max Score >= 0.8
    sample_max['Reached_08'] = sample_max['Score'] >= 0.8
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=sample_max, x='Iteration', y='Reached_08', hue='Strategy', style='Strategy', markers=True, dashes=False, errorbar=errorbar)
    
    plt.title(f'80% Reach Rate per Iteration {"(with 95% CI)" if show_ci else ""}')
    plt.ylabel('Ratio of Samples with Max Score >= 0.8')
    plt.xlabel('Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"reach_rate_08{suffix}.png"))
    plt.close()

def plot_logic_contribution(df):
    """
    3. TAGDの各ロジックがどれだけ貢献したかがわかるグラフ
    Interpretation: How many individuals were produced by each logic type per iteration?
    Or better: How many *Top Performing* individuals came from each logic?
    "Contribution" usually implies quality. Let's plot:
    a) Population Composition (Distribution of Logic types)
    b) Source of Best Individual (Who produced the winner?)
    """
    print("Generating Logic Contribution Plot...")
    
    tagd_df = df[df['Strategy'] == 'TAGD'].copy()
    
    if tagd_df.empty:
        print("No TAGD data found for logic contribution.")
        return

    # a) Population Composition
    # Group by Iteration, Logic -> Count
    # Normalize by total population per iteration to get percentage
    
    logic_counts = tagd_df.groupby(['Iteration', 'Logic']).size().reset_index(name='Count')
    # total_per_iter line removed
    
    pivot_counts = logic_counts.pivot(index='Iteration', columns='Logic', values='Count').fillna(0)
    # Normalize
    pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
    
    # Remove 'Initialization' from plot if desired, usually Iter 0 is all Init
    # Just plot stacked bar
    
    pivot_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('TAGD Population Composition by Logic Type')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Iteration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tagd_logic_composition.png"))
    plt.close()
    
    # b) Source of Best Score (Contribution to Performance)
    # Find max score per (Run, Row, Iter)
    # Then find which logic produced that score.
    # Note: If multiple have same max score, we count them all or fractionally?
    # Let's count all that achieved the max score.
    
    # Get Max Score per Group
    max_scores = tagd_df.groupby(['Run', 'Row', 'Iteration'])['Score'].transform('max')
    best_individuals = tagd_df[tagd_df['Score'] == max_scores]
    
    best_logic_counts = best_individuals.groupby(['Iteration', 'Logic']).size().reset_index(name='Count')
    best_pivot = best_logic_counts.pivot(index='Iteration', columns='Logic', values='Count').fillna(0)
    best_pct = best_pivot.div(best_pivot.sum(axis=1), axis=0) * 100
    
    best_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Source of Best Individual (TAGD)')
    plt.ylabel('Percentage of Max Score Contributors')
    plt.xlabel('Iteration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tagd_best_score_source.png"))
    plt.close()

def plot_stagnation_breaking(df):
    """
    4. TAGDのmutationによってスコアの停滞が打開された回数
    Interpretation: 
    - Track Max Score Trajectory for each (Run, Row).
    - Identify Iterations where Max Score[i] > Max Score[i-1].
    - For those "Improvement Steps", check the Logic of the individual(s) that achieved the new Max Score.
    - If Logic == Mutation (or Persona Mutation), it is a Stagnation Break by Mutation.
    """
    print("Generating Stagnation Analysis...")
    
    tagd_df = df[df['Strategy'] == 'TAGD'].copy()
    
    if tagd_df.empty:
        print("No TAGD data for stagnation analysis.")
        return

    # Get max score per iteration per run/row
    # We need to preserve Logic information.
    # Approach:
    # 1. Get DataFrame of [Run, Row, Iteration, MaxScore]
    # 2. Shift to get PrevMaxScore
    # 3. Filter where MaxScore > PrevMaxScore (Improvement)
    # 4. Join back to get Logic of individuals with Score == MaxScore at that Iteration
    
    # 1. Max Score per Iter
    iter_stats = tagd_df.groupby(['Run', 'Row', 'Iteration'])['Score'].max().reset_index()
    iter_stats = iter_stats.sort_values(['Run', 'Row', 'Iteration'])
    
    # 2. Shift for previous score
    iter_stats['PrevScore'] = iter_stats.groupby(['Run', 'Row'])['Score'].shift(1)
    
    # Filter for Iteration > 0 (Iter 0 has no prev)
    improvement_steps = iter_stats[(iter_stats['Iteration'] > 0) & (iter_stats['Score'] > iter_stats['PrevScore'])]
    
    # Now we know which (Run, Row, Iter) had improvements.
    # We need to find the Logic responsible.
    
    # Create key for join
    tagd_df['key'] = list(zip(tagd_df.Run, tagd_df.Row, tagd_df.Iteration))
    improvement_steps['key'] = list(zip(improvement_steps.Run, improvement_steps.Row, improvement_steps.Iteration))
    
    # Keys of improvement steps
    imp_keys = set(improvement_steps['key'])
    
    # Filter original DF for only individuals in Improvement Steps who achieved the Max Score
    # We merge improvement_steps back to tagd_df on (Run, Row, Iter) AND (Score == MaxScore)
    # Actually simpler: Iterate improvements and lookup.
    
    breaker_logics = []
    
    for _, row in improvement_steps.iterrows():
        run, r_id, iter_num, max_score = row['Run'], row['Row'], row['Iteration'], row['Score']
        
        # Find individuals in this Run/Row/Iter with this Score
        winners = tagd_df[
            (tagd_df['Run'] == run) & 
            (tagd_df['Row'] == r_id) & 
            (tagd_df['Iteration'] == iter_num) & 
            (tagd_df['Score'] == max_score)
        ]
        
        # There might be multiple winners (e.g. 2 individuals got 0.95).
        # We count each logic type found. 
        # Or better: "Event Count". If Mutation is present in winners, we say "Mutation Broke Stagnation".
        # If both TextGrad and Mutation got the new high score, both contributed? Usually rare.
        # Let's count Logic Occurrences.
        
        for logic in winners['Logic'].unique():
            breaker_logics.append(logic)
            
    # Count breakers
    breaker_counts = pd.Series(breaker_logics).value_counts().reset_index()
    breaker_counts.columns = ['Logic', 'Count']
    
    print("Stagnation Breakers by Logic:")
    print(breaker_counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=breaker_counts, x='Logic', y='Count', palette='viridis')
    plt.title('Number of Times Logic Broke Stagnation (New Global Max for Run)')
    plt.ylabel('Count of Breakout Events')
    plt.xlabel('Logic Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tagd_stagnation_breakers.png"))
    plt.close()

def plot_stagnation_breaking_split(df):
    """
    世代を半分に分け、それぞれの期間における停滞打開回数を可視化する。
    """
    print("Generating Stagnation Analysis (Split by Generations)...")
    
    tagd_df = df[df['Strategy'] == 'TAGD'].copy()
    
    if tagd_df.empty:
        print("No TAGD data for stagnation analysis.")
        return

    # 1. Max Score per Iter
    iter_stats = tagd_df.groupby(['Run', 'Row', 'Iteration'])['Score'].max().reset_index()
    iter_stats = iter_stats.sort_values(['Run', 'Row', 'Iteration'])
    
    # 2. Shift for previous score
    iter_stats['PrevScore'] = iter_stats.groupby(['Run', 'Row'])['Score'].shift(1)
    
    # Filter for Iteration > 0 (Iter 0 has no prev)
    improvement_steps = iter_stats[(iter_stats['Iteration'] > 0) & (iter_stats['Score'] > iter_stats['PrevScore'])]
    
    max_iter = tagd_df['Iteration'].max()
    mid_point = max_iter // 2
    
    def get_phase(iter_num):
        return "First Half" if iter_num <= mid_point else "Second Half"
        
    breaker_data = []
    
    for _, row in improvement_steps.iterrows():
        run, r_id, iter_num, max_score = row['Run'], row['Row'], row['Iteration'], row['Score']
        phase = get_phase(iter_num)
        
        winners = tagd_df[
            (tagd_df['Run'] == run) & 
            (tagd_df['Row'] == r_id) & 
            (tagd_df['Iteration'] == iter_num) & 
            (tagd_df['Score'] == max_score)
        ]
        
        for logic in winners['Logic'].unique():
            breaker_data.append({"Logic": logic, "Phase": phase})
            
    if not breaker_data:
        print("No stagnation breaking events found.")
        return

    breaker_df = pd.DataFrame(breaker_data)
    
    # Count occurrences by Logic and Phase
    count_df = breaker_df.groupby(['Logic', 'Phase']).size().reset_index(name='Count')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=count_df, x='Logic', y='Count', hue='Phase', hue_order=["First Half", "Second Half"], palette='viridis')
    plt.title(f'Stagnation Breaking Events: First Half (iter 1-{mid_point}) vs Second Half (iter {mid_point+1}-{max_iter})')
    plt.ylabel('Count of Breakout Events')
    plt.xlabel('Logic Type')
    plt.legend(title="Generation Phase")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tagd_stagnation_breakers_split.png"))
    plt.close()

def plot_stagnation_breaking_by_score(df):
    """
    改善前のスコア帯ごとに、どのロジックが停滞を打破したか。
    """
    print("Generating Stagnation Analysis (by Score bands)...")
    
    tagd_df = df[df['Strategy'] == 'TAGD'].copy()
    
    if tagd_df.empty:
        print("No TAGD data for stagnation analysis.")
        return

    # 1. Max Score per Iter
    iter_stats = tagd_df.groupby(['Run', 'Row', 'Iteration'])['Score'].max().reset_index()
    iter_stats = iter_stats.sort_values(['Run', 'Row', 'Iteration'])
    
    # 2. Shift for previous score
    iter_stats['PrevScore'] = iter_stats.groupby(['Run', 'Row'])['Score'].shift(1)
    
    # Filter for Iteration > 0
    improvement_steps = iter_stats[(iter_stats['Iteration'] > 0) & (iter_stats['Score'] > iter_stats['PrevScore'])]
    
    # Define score bins
    # For paper, scores are typically 0.5-1.0. 
    # Let's use 0.1 intervals.
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    
    improvement_steps['ScoreBand'] = pd.cut(improvement_steps['PrevScore'], bins=bins, labels=labels, include_lowest=True)
        
    breaker_data = []
    
    for _, row in improvement_steps.iterrows():
        run, r_id, iter_num, max_score, band = row['Run'], row['Row'], row['Iteration'], row['Score'], row['ScoreBand']
        
        winners = tagd_df[
            (tagd_df['Run'] == run) & 
            (tagd_df['Row'] == r_id) & 
            (tagd_df['Iteration'] == iter_num) & 
            (tagd_df['Score'] == max_score)
        ]
        
        for logic in winners['Logic'].unique():
            breaker_data.append({"Logic": logic, "ScoreBand": str(band)})
            
    if not breaker_data:
        print("No stagnation breaking events found.")
        return

    breaker_df = pd.DataFrame(breaker_data)
    
    # Count occurrences by Logic and ScoreBand
    count_df = breaker_df.groupby(['Logic', 'ScoreBand']).size().reset_index(name='Count')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=count_df, x='ScoreBand', y='Count', hue='Logic', order=labels, palette='viridis')
    plt.title('Stagnation Breaking Events by Improvement Source Score')
    plt.ylabel('Count of Breakout Events')
    plt.xlabel('Score Band (Before Improvement)')
    plt.legend(title="Logic Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tagd_stagnation_breakers_by_score.png"))
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    df = load_data()
    
    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(df)} data points.")
    print(f"Strategies found: {df['Strategy'].unique()}")

    # Generate both versions
    for ci in [True, False]:
        plot_avg_score(df, show_ci=ci)
        plot_reach_rate(df, show_ci=ci)
    
    plot_logic_contribution(df)
    plot_stagnation_breaking(df)
    plot_stagnation_breaking_split(df)
    plot_stagnation_breaking_by_score(df)
    
    print(f"All plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
