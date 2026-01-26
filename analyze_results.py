
import json
import os
import glob
from collections import defaultdict

base_dir = '/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1'
methods = ['eed', 'ga', 'textgrad', 'textgradv2']
rows = range(10)

print(f"{'Row':<5} | {'EED':<8} | {'GA':<8} | {'TextGrad':<8} | {'TextGradV2':<10} | {'Winner':<10}")
print("-" * 65)

wins = defaultdict(int)
total_rows = 0

failure_rows = []

for r in rows:
    scores = {}
    for m in methods:
        path = f"{base_dir}/{m}/perspectrum_llm/row_{r}/score_summary.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                # Get max score from the last available iteration
                last_iter = sorted(data.keys(), key=lambda x: int(x.replace('iter', '')))[-1]
                scores[m] = data[last_iter]['max']
        else:
            scores[m] = -1.0 # Missing

    # Comparison
    # Assuming we want to maximize score
    # Check if EED is strictly better than others? Or >=?
    # User says "winning", so likely > or >= best of others.
    
    eed_score = scores.get('eed', -1)
    others_max = max([v for k, v in scores.items() if k != 'eed'], default=-1)
    
    winner = "EED" if eed_score >= others_max else "Other"
    
    # Identify actual winner(s)
    best_score = max(scores.values())
    winners = [k for k, v in scores.items() if v == best_score]
    winner_str = ", ".join(winners)

    if 'eed' in winners:
        wins['eed'] += 1
    else:
        failure_rows.append(r)
        
    print(f"{r:<5} | {scores.get('eed', -1):<8} | {scores.get('ga', -1):<8} | {scores.get('textgrad', -1):<8} | {scores.get('textgradv2', -1):<10} | {winner_str:<10}")
    total_rows += 1

print("-" * 65)
print(f"EED Win Rate: {wins['eed']}/{total_rows}")
print(f"Failure Rows: {failure_rows}")
