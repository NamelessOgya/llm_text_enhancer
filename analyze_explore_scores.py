import os
import glob
import json
import re
from collections import defaultdict

BASE_DIR_ROOT = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm"

def analyze_explore():
    print(f"Scanning Explore results in {BASE_DIR_ROOT}...")
    
    total_explore = 0
    failed_explore = 0 # Score < 0.2 (likely violation)
    high_explore = 0 # Score > 0.6
    
    scores = []
    
    for row_idx in range(10):
        row_dir = os.path.join(BASE_DIR_ROOT, f"row_{row_idx}")
        if not os.path.exists(row_dir): continue
        
        # Iterates
        for iter_dir in sorted(glob.glob(os.path.join(row_dir, "iter*"))):
            iter_idx = int(os.path.basename(iter_dir).replace("iter", ""))
            if iter_idx == 0: continue
            
            logic_dir = os.path.join(iter_dir, "logic")
            metrics_path = os.path.join(iter_dir, "metrics.json")
            if not os.path.exists(metrics_path): continue
            
            with open(metrics_path, 'r') as f: metrics = json.load(f)
            # Map file -> score
            score_map = {m['file']: m.get('score', 0) for m in metrics}
            
            for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
                with open(logic_file, 'r') as f: content = f.read()
                
                # Check Type
                is_explore = False
                if '"type": "eed_explore' in content:
                    is_explore = True
                
                if not is_explore: continue
                
                # Get Score
                fname = os.path.basename(logic_file).replace("creation_prompt_", "text_")
                score = score_map.get(fname, 0)
                
                total_explore += 1
                scores.append(score)
                
                if score < 0.2:
                    failed_explore += 1
                if score >= 0.6:
                    high_explore += 1

    print(f"Total Explore Attempts: {total_explore}")
    if total_explore > 0:
        print(f"Avg Score: {sum(scores)/len(scores):.3f}")
        
        # Breakdown
        score_zero = sum(1 for s in scores if s == 0.0)
        score_low = sum(1 for s in scores if 0.0 < s < 0.4)
        score_mid = sum(1 for s in scores if 0.4 <= s < 0.7)
        score_high = sum(1 for s in scores if s >= 0.7)
        
        print(f"DEATH (Score 0.0 / Violation): {score_zero} ({score_zero/total_explore*100:.1f}%)")
        print(f"LOW QUALITY (0.0 < Score < 0.4): {score_low} ({score_low/total_explore*100:.1f}%)")
        print(f"MID QUALITY (0.4 <= Score < 0.7): {score_mid} ({score_mid/total_explore*100:.1f}%)")
        print(f"HIGH QUALITY (Score >= 0.7): {score_high} ({score_high/total_explore*100:.1f}%)")
    else:
        print("No Explore attempts found.")

if __name__ == "__main__":
    analyze_explore()
