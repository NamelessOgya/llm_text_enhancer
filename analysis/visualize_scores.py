import os
import json
import glob
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import textwrap

def load_config(config_path="analysis/agent/visualization_config.yaml"):
    """Load visualization configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_style(config):
    """Apply matplotlib style settings from config."""
    # Font settings
    mpl.rcParams['font.family'] = config['font']['family']
    mpl.rcParams['font.serif'] = config['font'].get('serif', [])
    mpl.rcParams['font.sans-serif'] = config['font'].get('sans_serif', [])
    mpl.rcParams['font.size'] = config['font']['size_tick']
    mpl.rcParams['axes.titlesize'] = config['font']['size_title']
    mpl.rcParams['axes.labelsize'] = config['font']['size_label']
    mpl.rcParams['xtick.labelsize'] = config['font']['size_tick']
    mpl.rcParams['ytick.labelsize'] = config['font']['size_tick']
    mpl.rcParams['legend.fontsize'] = config['font']['size_legend']
    
    # Figure settings
    mpl.rcParams['figure.dpi'] = config['figure']['dpi']
    mpl.rcParams['figure.facecolor'] = config['figure']['facecolor']
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = config['figure']['grid_alpha']
    
    # Line settings
    mpl.rcParams['lines.linewidth'] = config['style']['linewidth']
    mpl.rcParams['lines.markersize'] = config['style']['markersize']

def parse_results(result_dir="result"):
    """
    Parse result directory to extract score summaries and metadata.
    Returns: extracted_data[task][row] = {
        "input": input_data_dict,
        "methods": {
            method_name: {
                "stats": [...],
                "best_text": str,
                "best_score": float
            }
        }
    }
    """
    extracted_data = defaultdict(lambda: defaultdict(lambda: {"methods": {}, "input": None}))
    
    # Find all score_summary.json files
    search_pattern = os.path.join(result_dir, "**", "score_summary.json")
    files = glob.glob(search_pattern, recursive=True)
    
    for file_path in files:
        # result/task/method/row/score_summary.json
        parts = os.path.relpath(file_path, result_dir).split(os.sep)
        if len(parts) < 4:
            continue
            
        task_base = parts[0]
        method = parts[1]
        # row is always the parent directory of the file (second to last in path parts)
        row = parts[-2]
        
        # Check for evaluator layer (depth 5: task/method/evaluator/row/file)
        if len(parts) > 4:
            # Assume everything between method and row is evaluator info
            evaluator_parts = parts[2:-2]
            evaluator_str = "/".join(evaluator_parts)
            # Treat different evaluators as different tasks
            task = f"{task_base} ({evaluator_str})"
        else:
            task = task_base
        
        method_dir = os.path.dirname(file_path)
        
        # Load input_data.json if not already loaded
        if extracted_data[task][row]["input"] is None:
            input_path = os.path.join(method_dir, "input_data.json")
            if os.path.exists(input_path):
                try:
                    with open(input_path, "r") as f:
                        extracted_data[task][row]["input"] = json.load(f)
                except Exception as e:
                    print(f"Error reading input_data.json at {input_path}: {e}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            # data is {iter0: {...}, iter1: {...}}
            sorted_iters = sorted(data.keys(), key=lambda x: int(x.replace("iter", "")))
            
            processed_data = []
            for it in sorted_iters:
                iter_idx = int(it.replace("iter", ""))
                stats = data[it]
                processed_data.append({
                    "iteration": iter_idx,
                    "max": stats.get("max", 0),
                    "mean": stats.get("mean", 0),
                    "min": stats.get("min", 0),
                    "count": stats.get("count", 0)
                })
            
            # Find Best Text (Highest Score across all iterations)
            best_score = -1.0
            best_text = "N/A"
            
            # Iterate through all iter directories
            iter_dirs = glob.glob(os.path.join(method_dir, "iter*"))
            for iter_dir in iter_dirs:
                metrics_path = os.path.join(iter_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                        # metrics is list of {file, score}
                        for m in metrics:
                            # score can be None if failed? assume float
                            score = float(m.get("score", -1))
                            if score > best_score:
                                best_score = score
                                text_filename = m.get("file")
                                if text_filename:
                                    text_path = os.path.join(iter_dir, "texts", text_filename)
                                    if os.path.exists(text_path):
                                        with open(text_path, "r") as tf:
                                            best_text = tf.read()
                    except Exception as e:
                        # print(f"Error reading metrics/text in {iter_dir}: {e}")
                        pass

            extracted_data[task][row]["methods"][method] = {
                "stats": processed_data,
                "best_text": best_text,
                "best_score": best_score
            }
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    return extracted_data

def plot_scores(data, task, row, config, output_dir):
    """Generate plots for a specific task and row."""
    row_data = data[task][row]
    methods_data = row_data["methods"]
    input_data = row_data["input"]
    
    if not methods_data:
        return

    # Ensure output directory exists
    metrics = ["max", "mean"]
    
    for metric in metrics:
        # Create output directory: analysis/plots/score_{metric}/{task}
        task_plot_dir = os.path.join(output_dir, f"score_{metric}", task)
        os.makedirs(task_plot_dir, exist_ok=True)

        # Configurable figure size - adjust for text area
        figsize = tuple(config['figure'].get('figsize_tall', [10, 12])) # Increased height for text
        fig = plt.figure(figsize=figsize)
        
        # Grid spec for Layout: Plot on top, Text on bottom
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2])
        ax = fig.add_subplot(gs[0])
        
        # Plot lines
        for method_name, content in methods_data.items():
            iterations = content["stats"]
            x = [it['iteration'] for it in iterations]
            y = [it[metric] for it in iterations]
            
            color = config['colors']['methods'].get(method_name, config['colors'].get('default', 'black'))
            ax.plot(x, y, label=method_name, color=color, marker='o')
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{metric.capitalize()} Score")
        ax.set_title(f"Task: {task}, Sample: {row} - {metric.capitalize()} Score")
        ax.legend()
        
        # Prepare Text Area
        text_ax = fig.add_subplot(gs[1])
        text_ax.axis('off')
        
        # Construct Text Content
        full_text = ""
        
        # 1. Input Data
        full_text += "=== Input Data ===\n"
        if input_data:
            if isinstance(input_data, dict):
                for k, v in input_data.items():
                    val_str = str(v)
                    wrapped_val = textwrap.fill(val_str, width=90, subsequent_indent='    ')
                    full_text += f"{k}: {wrapped_val}\n"
            else:
                full_text += str(input_data) + "\n"
        else:
            full_text += "No input data found.\n"
        
        full_text += "\n"
        
        # 2. Best Outputs (Only for Max Score Plot or if requested)
        # Even for mean plot, might be useful, but logic says max plot is most relevant for "best" output
        if metric == "max":
            full_text += "=== Best Outputs by Method (Max Score) ===\n"
            for method_name, content in methods_data.items():
                best_score = content.get("best_score", -1)
                best_text = content.get("best_text", "N/A")
                
                full_text += f"[{method_name}] (Score: {best_score:.4f})\n"
                
                # Truncate text if too long
                trunc_text = (best_text[:300] + '...') if len(best_text) > 300 else best_text
                wrapped_text = textwrap.fill(trunc_text, width=90)
                full_text += f"{wrapped_text}\n\n"
        
        # Draw Text
        # Escape dollar signs to avoid matplotlib math parsing errors
        full_text = full_text.replace("$", "\\$")
        text_ax.text(0, 1, full_text, fontsize=10, va='top', fontfamily='monospace')

        # Save plot
        base_filename = f"{task}_{row}_{metric}_score"
        for fmt in config['output']['formats']:
            save_path = os.path.join(task_plot_dir, f"{base_filename}.{fmt}")
            plt.savefig(save_path, format=fmt, **config['output']['save_kwargs'])
            print(f"Saved {save_path}")
            
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize experiment scores.")
    parser.add_argument("--result_dir", default="result", help="Path to result directory")
    parser.add_argument("--output_dir", default="analysis/plots", help="Path to save plots")
    parser.add_argument("--config", default="analysis/agent/visualization_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config and setup style
    if os.path.exists(args.config):
        config = load_config(args.config)
        setup_style(config)
    else:
        print(f"Config file not found at {args.config}, using defaults.")
        # Fallback config
        config = {
            'font': {'family': 'sans-serif', 'size_tick': 10, 'size_title': 12, 'size_label': 10, 'size_legend': 10},
            'figure': {'dpi': 100, 'facecolor': 'white', 'grid_alpha': 0.5, 'figsize_single': [10, 6]},
            'style': {'linewidth': 2, 'markersize': 6},
            'colors': {'methods': {}, 'default': 'black'},
            'output': {'formats': ['png'], 'save_kwargs': {}}
        }

    # Parse data
    print("Parsing results...")
    data = parse_results(args.result_dir)
    
    if not data:
        print("No data found.")
        return

    # Generate plots
    print("Generating plots...")
    for task in data:
        for row in data[task]:
            plot_scores(data, task, row, config, args.output_dir)
            
    print("Done.")

if __name__ == "__main__":
    main()
