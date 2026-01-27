import json
import os
import numpy as np

def analyze_rows(base_path, method_name, num_rows=10):
    print(f"--- Analyzing {method_name} rows ---")
    for i in range(num_rows):
        path = os.path.join(base_path, method_name, f'perspectrum_llm/row_{i}/score_summary.json')
        if not os.path.exists(path):
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        initial_max = data['iter0']['max']
        final_max = data['iter8']['max']
        improvement = final_max - initial_max
        
        # Check for stagnation or regression
        print(f"Row {i}: Start={initial_max:.2f}, End={final_max:.2f}, Diff={improvement:+.2f}")
        
        # Print trajectory for interesting rows (low improvement)
        if improvement <= 0.1:
            traj = [data[f'iter{j}']['max'] for j in range(9)]
            print(f"  -> STAGNANT/LOW GROWTH: {traj}")

base_path = '/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1'
analyze_rows(base_path, 'eed')
