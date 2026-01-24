import os
import requests
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_visualization_config(config_path="analysis/agent/visualization_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_json(url):
    print(f"Downloading {url}...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

def get_word_count_stats(texts):
    word_counts = [len(text.split()) for text in texts]
    return {
        "max": np.max(word_counts),
        "min": np.min(word_counts),
        "mean": np.mean(word_counts),
        "median": np.median(word_counts),
        "p25": np.percentile(word_counts, 25),
        "p75": np.percentile(word_counts, 75),
        "p95": np.percentile(word_counts, 95),
        "counts": word_counts
    }

def visualize_histogram(stats, config, output_path):
    # Apply font settings
    font_config = config.get('font', {})
    plt.rcParams['font.family'] = font_config.get('family', 'serif')
    if font_config.get('family') == 'serif':
        plt.rcParams['font.serif'] = font_config.get('serif', ['DejaVu Serif'])
    
    # Figure settings
    fig_config = config.get('figure', {})
    figsize = fig_config.get('figsize_single', [10, 6])
    dpi = fig_config.get('dpi', 300)
    
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Plot histogram
    counts = stats['counts']
    colors = config.get('colors', {})
    color = colors.get('methods', {}).get('ga', 'blue') # Use a default color

    plt.hist(counts, bins=50, color=color, alpha=0.7, edgecolor='black')
    
    plt.title("Distribution of Perspective Word Counts (Perspectrum Full Dataset)", fontsize=font_config.get('size_title', 16))
    plt.xlabel("Word Count", fontsize=font_config.get('size_label', 14))
    plt.ylabel("Frequency", fontsize=font_config.get('size_label', 14))
    plt.tick_params(axis='both', labelsize=font_config.get('size_tick', 12))
    
    # Add statistics text
    stats_text = (
        f"Max: {stats['max']}\n"
        f"Min: {stats['min']}\n"
        f"Mean: {stats['mean']:.2f}\n"
        f"Median: {stats['median']:.2f}\n"
        f"25%: {stats['p25']:.2f}\n"
        f"75%: {stats['p75']:.2f}\n"
        f"95%: {stats['p95']:.2f}"
    )
    
    # Position text box in upper right
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Grid
    plt.grid(alpha=fig_config.get('grid_alpha', 0.3))
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def main():
    base_url = "https://raw.githubusercontent.com/CogComp/perspectrum/master/data/dataset"
    pool_url = f"{base_url}/perspective_pool_v1.0.json"
    
    # We essentially only need perspective_pool_v1.0.json to get all unique perspectives and their lengths.
    # The 'perspectrum_with_answers_v1.0.json' maps claims to these pids, but the pool contains the text.
    # To be thorough and match the "User responses" idea, we should consider all valid perspectives that appear in the dataset.
    # However, 'perspective_pool' contains ALL perspectives.
    # Let's inspect all perspectives in the pool as they represent the universe of "user responses".
    
    pool_data = download_json(pool_url)
    
    # Extract all texts
    texts = [item['text'] for item in pool_data if 'text' in item]
    
    print(f"Total perspectives analyzed: {len(texts)}")
    
    stats = get_word_count_stats(texts)
    
    print("Statistics:")
    print(f"Max: {stats['max']}")
    print(f"Min: {stats['min']}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"25%: {stats['p25']:.2f}")
    print(f"75%: {stats['p75']:.2f}")
    print(f"95%: {stats['p95']:.2f}")
    
    config = load_visualization_config()
    output_path = "analysis/plots/perspectrum_full_length_distribution.png"
    
    visualize_histogram(stats, config, output_path)

if __name__ == "__main__":
    main()
