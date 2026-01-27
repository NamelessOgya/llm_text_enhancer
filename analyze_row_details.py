import json
import os

def print_file_content(path, label):
    if not os.path.exists(path):
        print(f"[{label}] FILE NOT FOUND: {path}")
        return
    
    print(f"\n--- [{label}] {path} ---")
    if path.endswith('.json'):
        with open(path, 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        with open(path, 'r') as f:
            print(f.read().strip())

def inspect_row(row_idx, iter_idx):
    base_path = f'/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_{row_idx}/iter{iter_idx}'
    
    print(f"\n\n================ Analyzing Row {row_idx} Iter {iter_idx} ================")
    
    # Check 1st prompt/text as sample
    print_file_content(os.path.join(base_path, 'input_prompts/prompt_0.txt'), "PROMPT (Sample 0)")
    print_file_content(os.path.join(base_path, 'texts/text_0.txt'), "GENERATED TEXT (Sample 0)")
    
    # Check metrics
    print_file_content(os.path.join(base_path, 'metrics.json'), "METRICS")

# Inspect Row 1 (stuck at 0.4)
inspect_row(1, 0)
inspect_row(1, 8)

# Inspect Row 8 (stuck at 0.6)
inspect_row(8, 0)
inspect_row(8, 8)
