import os
import glob
import numpy as np

base_path = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm"
rows = range(10)
iters = range(10)

print(f"Checking duplications in: {base_path}")

total_generations = 0
fallback_duplication_count = 0
any_duplication_count = 0

for r in rows:
    for i in iters:
        if i == 0: continue # Skip initial generation (usually unique or different logic)
        
        texts_dir = os.path.join(base_path, f"row_{r}", f"iter{i}", "texts")
        if not os.path.exists(texts_dir):
            continue
            
        total_generations += 1
        
        # Read all texts
        texts = []
        files = sorted(glob.glob(os.path.join(texts_dir, "text_*.txt")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        for fp in files:
            with open(fp, 'r') as f:
                texts.append(f.read().strip())
                
        if len(texts) < 2: continue
        
        # Check Fallback Duplication (Last item == First item)
        # Specifically checking if the LAST generated item matches the FIRST (Elite)
        # usually last item is index -1
        if texts[-1] == texts[0]:
            fallback_duplication_count += 1
            # print(f"Fallback detected: row_{r} iter{i}")
            
        # Check Any Duplication
        if len(texts) != len(set(texts)):
            any_duplication_count += 1
            # Count how many duplicates
            # from collections import Counter
            # counts = Counter(texts)
            # dups = [k for k,v in counts.items() if v > 1]
            # print(f"Duplicates in row_{r} iter{i}: {len(texts) - len(set(texts))} items")

print("-" * 50)
print(f"Total Generations Analyzed: {total_generations}")
print(f"Fallback Duplications (Last == First): {fallback_duplication_count} ({fallback_duplication_count/total_generations*100:.1f}%)")
print(f"Generations with ANY Duplication: {any_duplication_count} ({any_duplication_count/total_generations*100:.1f}%)")
