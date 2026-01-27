import os
import json
import glob
import argparse
import random
from collections import defaultdict

def load_row_data(result_dir):
    """
    Load data organized by row.
    Returns a dict: row_id -> {
        "input": {...},
        "samples": [ {score, text, ...}, ... ]
    }
    """
    rows = {}
    
    # Iterate over row directories
    row_pattern = os.path.join(result_dir, "row_*")
    row_dirs = glob.glob(row_pattern)
    
    for row_dir in row_dirs:
        row_id = os.path.basename(row_dir)
        
        # Load Input Data
        input_file = os.path.join(row_dir, "input_data.json")
        if not os.path.exists(input_file):
            continue
            
        with open(input_file, "r") as f:
            input_data = json.load(f)
            
        samples = []
        
        # Iterate over iterations
        iter_dirs = glob.glob(os.path.join(row_dir, "iter*"))
        
        for iter_dir in iter_dirs:
            metrics_file = os.path.join(iter_dir, "metrics.json")
            if not os.path.exists(metrics_file):
                continue
                
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                
            for m in metrics:
                score = m.get("score")
                text_file = m.get("file")
                
                if score is None or not text_file:
                    continue
                
                # Filter out score < 0.1
                try:
                    score_val = float(score)
                except:
                    continue
                    
                if score_val < 0.1:
                    continue
                    
                text_path = os.path.join(iter_dir, "texts", text_file)
                if not os.path.exists(text_path):
                    continue
                    
                with open(text_path, "r") as f:
                    generated_text = f.read()
                    
                samples.append({
                    "score": score_val,
                    "text": generated_text,
                    "iter": os.path.basename(iter_dir)
                })
        
        if samples:
            rows[row_id] = {
                "input": input_data,
                "samples": samples
            }
                
    return rows

def generate_report(rows, output_path, num_rows=3):
    """Generate the row-based analysis report."""
    
    # Filter rows that have samples in ALL categories (High, Mid, Low)
    valid_rows = []
    for row_id in list(rows.keys()):
        row_data = rows[row_id]
        samples = row_data["samples"]
        
        high = [s for s in samples if s["score"] >= 0.8]
        mid  = [s for s in samples if 0.4 <= s["score"] < 0.8]
        low  = [s for s in samples if 0.1 <= s["score"] < 0.4]
        
        if high and mid and low:
            valid_rows.append(row_id)
            
    print(f"Rows with complete coverage (High/Mid/Low): {len(valid_rows)} / {len(rows)}")
    
    if not valid_rows:
        print("Warning: No rows have samples in ALL categories. Relaxing filter to ANY samples.")
        valid_rows = [r for r in rows if rows[r]["samples"]]
        
    # Select random rows
    selected_rows = random.sample(valid_rows, min(num_rows, len(valid_rows)))
    
    with open(output_path, "w") as f:
        f.write("# TextGrad Judge Row-based Analysis\n\n")
        f.write("ランダムに選択された3つのRowについて、定義、サマリー、およびスコア帯別の出力例を示します。\n")
        f.write("(High >= 0.8, Mid 0.4-0.8, Low 0.1-0.4, Score < 0.1 Excluded)\n\n")
        
        for row_id in selected_rows:
            row_data = rows[row_id]
            input_data = row_data["input"]
            samples = row_data["samples"]
            
            # Extract Definition/Summary
            content = input_data.get("content", "N/A")
            summary = input_data.get("summary", "N/A")
            stance = input_data.get("stance_label", "N/A")
            
            f.write(f"## Row: {row_id}\n")
            f.write(f"- **Topic (Content)**: {content}\n")
            f.write(f"- **Stance**: {stance}\n")
            f.write(f"- **Target Summary**: {summary}\n\n")
            
            # Categorize samples
            high = [s for s in samples if s["score"] >= 0.8]
            mid  = [s for s in samples if 0.4 <= s["score"] < 0.8]
            low  = [s for s in samples if 0.1 <= s["score"] < 0.4]
            
            def get_one(lst):
                return random.choice(lst) if lst else None
                
            sample_high = get_one(high)
            sample_mid = get_one(mid)
            sample_low = get_one(low)
            
            f.write("### Output Examples\n\n")
            
            # High
            f.write("**High Score Example** (Score >= 0.8)\n")
            if sample_high:
                f.write(f"- Score: {sample_high['score']} ({sample_high['iter']})\n")
                f.write(f"```\n{sample_high['text']}\n```\n")
            else:
                f.write("No samples found.\n")
            f.write("\n")
            
            # Mid
            f.write("**Mid Score Example** (0.4 <= Score < 0.8)\n")
            if sample_mid:
                f.write(f"- Score: {sample_mid['score']} ({sample_mid['iter']})\n")
                f.write(f"```\n{sample_mid['text']}\n```\n")
            else:
                f.write("No samples found.\n")
            f.write("\n")

            # Low
            f.write("**Low Score Example** (0.1 <= Score < 0.4)\n")
            if sample_low:
                f.write(f"- Score: {sample_low['score']} ({sample_low['iter']})\n")
                f.write(f"```\n{sample_low['text']}\n```\n")
            else:
                f.write("No samples found.\n")
            f.write("\n---\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_file", default="analysis/judge_row_samples_report.md")
    args = parser.parse_args()
    
    print(f"Loading data from {args.result_dir}...")
    rows = load_row_data(args.result_dir)
    
    print(f"Generating report for {len(rows)} rows...")
    generate_report(rows, args.output_file)
    print("Done.")

if __name__ == "__main__":
    main()
