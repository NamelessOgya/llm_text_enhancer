import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import matplotlib as mpl
from datasets import load_dataset

def load_config(config_path="analysis/agent/visualization_config.yaml"):
    """Load visualization configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_style(config):
    """Apply matplotlib style settings."""
    mpl.rcParams['font.family'] = config['font']['family']
    mpl.rcParams['font.serif'] = config['font'].get('serif', [])
    mpl.rcParams['font.sans-serif'] = config['font'].get('sans_serif', [])
    mpl.rcParams['font.size'] = config['font']['size_tick']
    mpl.rcParams['axes.titlesize'] = config['font']['size_title']
    mpl.rcParams['axes.labelsize'] = config['font']['size_label']
    mpl.rcParams['figure.dpi'] = config['figure']['dpi']

def analyze_full_dataset(output_dir="analysis/plots/prism_alignment", bin_width=0.5, config_path="analysis/agent/visualization_config.yaml"):
    # Load and setup style
    if os.path.exists(config_path):
        config = load_config(config_path)
        setup_style(config)
    else:
        print("Config not found, using defaults")
        config = {'output': {'formats': ['png'], 'save_kwargs': {}}, 'figure': {'figsize_single': [10, 6]}}

    print("Loading full dataset: HannahRoseKirk/prism-alignment (conversations)...")
    try:
        ds_conv = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loaded {len(ds_conv)} conversations.")
    
    all_scores = []
    
    for row in ds_conv:
        history = row.get("conversation_history", [])
        for turn in history:
            if turn.get("role") == "model":
                score = turn.get("score")
                if score is not None:
                    # Assuming score is clean (it was in the small sample)
                    try:
                        all_scores.append(float(score))
                    except:
                        pass
                        
    print(f"Total scores extracted: {len(all_scores)}")
    if not all_scores:
        print("No scores found.")
        return

    # Statistics
    scores = np.array(all_scores)
    print("Statistics:")
    print(f"  Count: {len(scores)}")
    print(f"  Mean:  {np.mean(scores):.2f}")
    print(f"  Std:   {np.std(scores):.2f}")
    print(f"  Min:   {np.min(scores)}")
    print(f"  Max:   {np.max(scores)}")
    print(f"  25%:   {np.percentile(scores, 25)}")
    print(f"  50%:   {np.median(scores)}")
    print(f"  75%:   {np.percentile(scores, 75)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot histogram
    plt.figure(figsize=tuple(config['figure']['figsize_single']))
    
    # Bins
    min_val = 0 # floor
    max_val = 100 # ceiling
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    sns.histplot(scores, bins=bins, kde=False, color='salmon', edgecolor='black')
    
    plt.title(f"Distribution of All PRISM Scores (Bin Width: {bin_width})")
    plt.xlabel("Original Score (0-100)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=config.get('figure', {}).get('grid_alpha', 0.5))
    
    for fmt in config['output']['formats']:
        output_file = os.path.join(output_dir, f"prism_full_score_dist.{fmt}")
        plt.savefig(output_file, format=fmt, **config['output']['save_kwargs'])
        print(f"Histogram saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-width", type=float, default=0.5, help="Bin width for histogram")
    args = parser.parse_args()
    
    analyze_full_dataset(bin_width=args.bin_width)
