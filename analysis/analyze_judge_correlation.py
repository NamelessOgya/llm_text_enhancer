import os
import json
import glob
import argparse
import difflib
from collections import defaultdict
import numpy as np

def jaccard_similarity(str1, str2):
    """Calculate Jaccard similarity between two strings."""
    if not str1 or not str2:
        return 0.0
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def difflib_similarity(str1, str2):
    """Calculate sequence matching ratio."""
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def load_data(result_dir):
    """
    Load data from the result directory.
    Structure: result_dir/row_X/iterY/metrics.json
               result_dir/row_X/iterY/texts/filename
               result_dir/row_X/input_data.json
    """
    data = []
    
    # Iterate over row directories
    row_pattern = os.path.join(result_dir, "row_*")
    row_dirs = glob.glob(row_pattern)
    
    for row_dir in row_dirs:
        row_id = os.path.basename(row_dir)
        
        # Load Input Data
        input_file = os.path.join(row_dir, "input_data.json")
        if not os.path.exists(input_file):
            print(f"Skipping {row_id}: No input_data.json")
            continue
            
        with open(input_file, "r") as f:
            input_data = json.load(f)
            
        # Target Summary/Claim from input
        # Adjust key based on actual schema. Assuming 'summary' or 'claim' or 'text'
        target_text = input_data.get("summary") or input_data.get("claim") or input_data.get("text", "")
        
        # Question / Topic
        question_text = input_data.get("content") or input_data.get("question", "")
        
        if not target_text:
             print(f"Warning: No explicit 'summary' or 'claim' found in input for {row_id}. Keys: {list(input_data.keys())}")
        
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
                    
                text_path = os.path.join(iter_dir, "texts", text_file)
                if not os.path.exists(text_path):
                    continue
                    
                with open(text_path, "r") as f:
                    generated_text = f.read()
                    
                data.append({
                    "row_id": row_id,
                    "score": float(score),
                    "generated_text": generated_text,
                    "target_text": target_text,
                    "question_text": question_text,
                    "iteration": os.path.basename(iter_dir)
                })
                
    return data

def analyze_correlation(data):
    """Compute correlations and stats."""
    
    # Filter: Keep only score >= 0.1 (User request: 0 is rule-based)
    original_count = len(data)
    data = [d for d in data if d["score"] >= 0.1]
    filtered_count = len(data)
    print(f"Filtered out {original_count - filtered_count} samples (Score < 0.1). Remaining: {filtered_count}")

    scores = []
    jaccard_sims = []
    difflib_sims = []
    lengths = []
    
    results = []
    
    for item in data:
        score = item["score"]
        gen_text = item["generated_text"]
        target = item["target_text"] or ""
        
        j_sim = jaccard_similarity(gen_text, target)
        d_sim = difflib_similarity(gen_text, target)
        length = len(gen_text)
        
        scores.append(score)
        jaccard_sims.append(j_sim)
        difflib_sims.append(d_sim)
        lengths.append(length)
        
        item["jaccard"] = j_sim
        item["difflib"] = d_sim
        item["length"] = length
        results.append(item)
        
    # Correlations
    def safe_corr(x, y):
        if len(x) < 2: return 0.0
        # Check if arrays are constant
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1]

    # Convert to numpy arrays for easier stats if needed, but lists work for corrcoef if converted
    
    corr_jaccard = safe_corr(scores, jaccard_sims)
    corr_difflib = safe_corr(scores, difflib_sims)
    corr_length = safe_corr(scores, lengths)
    
    stats = {
        "correlation_jaccard": corr_jaccard,
        "correlation_difflib": corr_difflib,
        "correlation_length": corr_length,
        "count": len(scores),
        "avg_score": np.mean(scores) if scores else 0,
        "std_score": np.std(scores) if scores else 0
    }
    
    return results, stats

def generate_report(results, stats, output_path):
    """Generate a markdown report."""
    
    # Sort by Score for Top/Bottom analysis
    sorted_by_score = sorted(results, key=lambda x: x["score"], reverse=True)
    top_5 = sorted_by_score[:5]
    bottom_5 = sorted_by_score[-5:]
    
    # Sort by Similarity (Jaccard) to see if High Similarity -> High Score
    sorted_by_sim = sorted(results, key=lambda x: x["jaccard"], reverse=True)
    top_5_sim = sorted_by_sim[:5]
    
    # Random Sampling for Qualitative Analysis
    import random
    
    # Define categories (heuristic based on typical 0-1 scoring)
    # High: >= 0.8
    # Mid:  0.4 <= score < 0.8
    # Low:  0.1 <= score < 0.4  (since we filtered < 0.1 already)
    
    high_scores = [d for d in results if d["score"] >= 0.8]
    mid_scores  = [d for d in results if 0.4 <= d["score"] < 0.8]
    low_scores  = [d for d in results if 0.1 <= d["score"] < 0.4]
    
    # Sampling 5 from each (or less if not enough data)
    sample_high = random.sample(high_scores, min(5, len(high_scores)))
    sample_mid  = random.sample(mid_scores, min(5, len(mid_scores)))
    sample_low  = random.sample(low_scores, min(5, len(low_scores)))

    with open(output_path, "w") as f:
        f.write("# TextGrad Judge 分析レポート\n\n")
        f.write("## 概要\n")
        f.write(f"- **分析サンプル総数 (Score >= 0.1)**: {stats['count']}\n")
        f.write(f"- **平均スコア**: {stats['avg_score']:.4f}\n")
        f.write(f"- **スコア標準偏差**: {stats['std_score']:.4f}\n\n")
        
        f.write("## 相関分析\n")
        f.write("Judgeのスコアは、入力サマリーとの類似度と相関があるか？\n\n")
        f.write(f"- **Jaccard類似度との相関**: {stats['correlation_jaccard']:.4f}\n")
        f.write(f"- **Sequence Match Ratioとの相関**: {stats['correlation_difflib']:.4f}\n")
        f.write(f"- **テキスト長との相関**: {stats['correlation_length']:.4f}\n\n")
        
        if stats['correlation_jaccard'] > 0.3:
            f.write("> **考察**: 中程度〜強い正の相関があります。Judgeはサマリーに近い出力を高く評価する傾向にあるようです。\n\n")
        elif stats['correlation_jaccard'] < -0.3:
            f.write("> **考察**: 負の相関があります。類似度が高いほどペナルティを受けている可能性があります。\n\n")
        else:
            f.write("> **考察**: 相関は弱いか、ありません。Judgeは単なる語彙の重複以外の基準を用いている可能性があります。\n\n")

        # Sampling Section
        f.write("## ランダムサンプリング分析 (Score 0 除外)\n")
        
        def write_samples(title, samples):
            f.write(f"### {title}\n")
            if not samples:
                f.write("該当なし\n\n")
                return
            for i, item in enumerate(samples):
                f.write(f"#### {i+1}. Score: {item['score']} (Row: {item['row_id']}, Iter: {item['iteration']})\n")
                f.write(f"- **Question**: {item['question_text']}\n")
                f.write(f"- **Target**: {item['target_text']}\n")
                f.write(f"- **Jaccard**: {item['jaccard']:.4f}\n")
                f.write(f"- **Generated**: \n```\n{item['generated_text']}\n```\n\n")

        write_samples(f"高スコア (Score >= 0.8, n={len(high_scores)})", sample_high)
        write_samples(f"中スコア (0.4 <= Score < 0.8, n={len(mid_scores)})", sample_mid)
        write_samples(f"低スコア (0.1 <= Score < 0.4, n={len(low_scores)})", sample_low)

        f.write("## 類似度が高い例 (サマリーとの重複が大)\n")
        f.write("これらは高いスコアを得ているか？\n\n")
        for i, item in enumerate(top_5_sim):
             f.write(f"### {i+1}. Score: {item['score']} (Jaccard: {item['jaccard']:.4f})\n")
             f.write(f"- **Generated**: \n```\n{item['generated_text'][:200]}...\n```\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True, help="Path to row directories")
    parser.add_argument("--output_file", default="analysis/judge_analysis_report.md")
    args = parser.parse_args()
    
    print(f"Loading data from {args.result_dir}...")
    data = load_data(args.result_dir)
    
    if not data:
        print("No data found!")
        return
        
    print(f"Analyzing {len(data)} samples...")
    results, stats = analyze_correlation(data)
    
    print(f"Generating report at {args.output_file}...")
    generate_report(results, stats, args.output_file)
    print("Done.")

if __name__ == "__main__":
    main()
