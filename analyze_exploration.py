import json
import os
import glob
from collections import Counter

def extract_reference_and_analyze(row_idx):
    print(f"\n\n================ Analyzing Row {row_idx} Exploration ================")
    
    # 1. Get Reference from metrics (it usually contains the target or we can infer it)
    # Actually metrics.json in this project structure often just has "score" and "reason"
    # But usually the evaluator logs the error against a reference.
    # Let's try to find the "Reference" from the prompt or task definition if accessible, 
    # OR we can assume the "metrics.json" reason field gives away the reference content.
    # Wait, the previous logs showed reasons like: "The reference focuses on command and control..."
    # So we don't need the exact text, we key concepts.
    
    base_path = f'/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_{row_idx}'
    
    # Define keywords based on our previous knowledge of the "Missing Point"
    if row_idx == 1:
        # Nuclear/Command Focus
        keywords = ["command", "control", "deploy", "authority", "tactical", "field", "commander"]
        missing_concept = "Command and Control / Decentralization"
    elif row_idx == 8:
        # Children/Selfish Focus
        keywords = ["autonomy", "body", "choice", "govern", "decide", "reproduction", "personal"]
        missing_concept = "Bodily Autonomy / Reproductive Rights"
    else:
        keywords = []
        missing_concept = "Unknown"

    print(f"Target Concept: {missing_concept}")
    print(f"Keywords: {keywords}")

    # 2. Analyze Iter 0-2
    hits = []
    themes = []
    
    for iter_i in range(3):
        iter_path = os.path.join(base_path, f'iter{iter_i}')
        texts_dir = os.path.join(iter_path, 'texts')
        metrics_path = os.path.join(iter_path, 'metrics.json')
        
        if not os.path.exists(texts_dir): continue
        
        # Load metrics to check scores of hits
        scores = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                m_list = json.load(f)
                for m in m_list:
                    scores[m.get('file')] = m.get('score', 0.0)

        for txt_file in glob.glob(os.path.join(texts_dir, '*.txt')):
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                
            fname = os.path.basename(txt_file)
            score = scores.get(fname, 0.0)
            
            # Check for keywords
            content_lower = content.lower()
            matched = [k for k in keywords if k in content_lower]
            
            if matched:
                hits.append({
                    "iter": iter_i,
                    "text": content,
                    "matched": matched,
                    "score": score
                })
            
            # Simple Theme Extraction (first noun/verb phrase roughly)
            # Just collecting the text for manual cluster check in output
            themes.append(content)

    # 3. Report Hits
    print(f"\n--- Search Results (Scenario 2 Check) ---")
    if hits:
        print(f"Found {len(hits)} potentially close texts:")
        for h in hits:
            print(f"[Iter {h['iter']} | Score: {h['score']}] Keywords: {h['matched']}")
            print(f"  Text: {h['text']}")
    else:
        print("NO semantic matches found in Iter 0-2 (Exploration Phase). -> Scenario 1 likely.")

    # 4. Analyze Population Bias (Scenario 1 Check)
    print(f"\n--- Population Themes (1st 5 examples from Iter 0) ---")
    for i, t in enumerate(themes[:5]):
        print(f"{i+1}. {t}")

inspect_rows = [1, 8]
for r in inspect_rows:
    extract_reference_and_analyze(r)
