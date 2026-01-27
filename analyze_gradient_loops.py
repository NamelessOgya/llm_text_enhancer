import os
import glob
import json
import re
from collections import defaultdict
import difflib

# Configuration
BASE_DIR_ROOT = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/textgrad/perspectrum_llm"
SIMILARITY_THRESHOLD = 0.8

def load_text_db(base_dir):
    """
    Load all texts and scores for a row.
    db[iter_idx][content] = {score, file, id}
    """
    db = defaultdict(dict)
    for iter_dir in sorted(glob.glob(os.path.join(base_dir, "iter*"))):
        try:
            iter_idx = int(os.path.basename(iter_dir).replace("iter", ""))
        except: continue
        
        metrics_path = os.path.join(iter_dir, "metrics.json")
        texts_dir = os.path.join(iter_dir, "texts")
        if not os.path.exists(metrics_path): continue
        
        with open(metrics_path, 'r') as f: metrics = json.load(f)
        
        for m in metrics:
            fname = m.get("file")
            t_path = os.path.join(texts_dir, fname)
            if os.path.exists(t_path):
                with open(t_path, 'r') as tf: content = tf.read().strip()
                db[iter_idx][content] = {
                    "score": m.get("score"), 
                    "file": fname,
                    "id": f"i{iter_idx}_{fname}"
                }
    return db

def get_gradient_from_logic(base_dir, iter_idx, fname):
    idx_str = fname.replace("text_", "").replace(".txt", "")
    logic_path = os.path.join(base_dir, f"iter{iter_idx}", "logic", f"creation_prompt_{idx_str}.txt")
    if not os.path.exists(logic_path): return None
    
    with open(logic_path, 'r') as f: content = f.read()
    
    # Extract Gradient
    grad_match = re.search(r'Feedback \(Gradient\) for Improvement:\s*(.*?)(?:\n\nDirective:|$)', content, re.DOTALL)
    if grad_match:
        return grad_match.group(1).strip()
    return None

def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def analyze_row(row_idx):
    base_dir = os.path.join(BASE_DIR_ROOT, f"row_{row_idx}")
    if not os.path.exists(base_dir): return [], []
    
    db = load_text_db(base_dir)
    
    # Build simple lineage chains?
    # Since we can't easily rebuild the full perfect graph without the logic parsing again,
    # let's try to track "Lineages" by content matching from Iter N to Iter N+1.
    
    # We want to find: Path of (Text, Gradient, Score) -> (Text, Gradient, Score)...
    
    # Construct edges (Parent -> Child)
    edges = []
    
    for iter_idx in sorted(db.keys()):
        if iter_idx == 0: continue
        
        current_texts = db[iter_idx]
        prev_texts = db.get(iter_idx - 1, {})
        
        logic_dir = os.path.join(base_dir, f"iter{iter_idx}", "logic")
        
        for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
            child_fname = os.path.basename(logic_file).replace("creation_prompt_", "text_")
            
            # Find child node in current_texts to get its content/score
            # Search by filename
            child_node = None
            child_content = None
            for c, info in current_texts.items():
                if info['file'] == child_fname:
                    child_node = info
                    child_content = c
                    break
            if not child_node: continue
            
            with open(logic_file, 'r') as f: l_content = f.read()
            
            # Skip Elite
            if "Elitism:" in l_content: continue
            
            # Find Parent by Content
            match = re.search(r'Current Text:\s*(?:["\'])(.*?)(?:["\'])', l_content, re.DOTALL)
            if not match: match = re.search(r'Current Text:\s*\n(.*?)\n', l_content, re.DOTALL)
            
            if match:
                parent_text_snippet = match.group(1).strip()
                parent_content = None
                
                # Match in prev_texts
                if parent_text_snippet in prev_texts:
                    parent_content = parent_text_snippet
                else:
                    for t in prev_texts:
                        if parent_text_snippet in t or t in parent_text_snippet:
                            parent_content = t
                            break
                            
                if parent_content:
                    parent_node = prev_texts[parent_content]
                    edges.append((parent_node['id'], child_node['id']))
                    
                    # Store metadata for nodes if not present
                    # We can store in a global node_map
                    pass

    # Build Graph
    adj = defaultdict(list)
    nodes_info = {}
    
    # helper to get node info
    def get_info(nid):
        # nid: i{iter}_{fname}
        parts = nid.split('_')
        iter_idx = int(parts[0].replace('i',''))
        fname = f"{parts[1]}_{parts[2]}" # text_X.txt
        idx_str = parts[2].replace('.txt','')
        
        # Get score and gradient
        # We need to access db... passed row_idx
        # Let's just re-fetch
        # Slow but okay
        logic_p = os.path.join(base_dir, f"iter{iter_idx}", "logic", f"creation_prompt_{idx_str}.txt")
        grad = get_gradient_text_robust(logic_p)
        
        # Get score from db
        # Need content lookup? db is content-keyed.
        # This architecture is messy.
        # Better: iterate edges and populate info then.
        return grad

    def get_gradient_text_robust(path):
        if not os.path.exists(path): return ""
        with open(path, 'r') as f: c = f.read()
        if "Elitism:" in c: return "ELITISM"
        m = re.search(r'Feedback \(Gradient\) for Improvement:\s*(.*?)(?:\n\nDirective:|$)', c, re.DOTALL)
        return m.group(1).strip() if m else ""

    def get_score_from_nid(nid, db):
        parts = nid.split('_')
        iter_idx = int(parts[0].replace('i',''))
        fname = f"{parts[1]}_{parts[2]}"
        
        for c, info in db[iter_idx].items():
            if info['file'] == fname:
                return info['score'], c
        return 0, ""

    # Analysis 3: Hub-and-Spoke Repetition
    repetitive_failures = []
    content_loops = []
    
    # Group gradients by Parent Content
    # parent_map: content -> list of (iter, gradient, child_score, parent_score)
    parent_attempt_map = defaultdict(list)
    
    # Iterate all edges to fill attempts
    # We need to rebuild edges with Parent Content info
    # Re-scan DB/Logic
    
    # Actually, we can just scan logic files again
    for iter_idx in sorted(db.keys()):
        if iter_idx == 0: continue
        
        logic_dir = os.path.join(base_dir, f"iter{iter_idx}", "logic")
        current_texts = db[iter_idx]
        prev_texts = db.get(iter_idx - 1, {})
        
        for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
            with open(logic_file, 'r') as f: l_content = f.read()
            
            # Extract Gradient
            grad_match = re.search(r'Feedback \(Gradient\) for Improvement:\s*(.*?)(?:\n\nDirective:|$)', l_content, re.DOTALL)
            if not grad_match: continue # Not an Exploit/Gradient node
            gradient = grad_match.group(1).strip()
            
            # Extract Parent Content
            match = re.search(r'Current Text:\s*(?:["\'])(.*?)(?:["\'])', l_content, re.DOTALL)
            if not match: match = re.search(r'Current Text:\s*\n(.*?)\n', l_content, re.DOTALL)
            if not match: continue
            parent_text = match.group(1).strip()
            
            # Child info
            child_fname = os.path.basename(logic_file).replace("creation_prompt_", "text_")
            child_node = None
            for c, info in current_texts.items():
                if info['file'] == child_fname:
                    child_node = info
                    break
            if not child_node: continue
            
            # Parent Score (approx from prev loop or db)
            # We can try to find parent score in prev_texts
            parent_score = 0
            # Fuzzy match parent_text in prev_texts
            if parent_text in prev_texts:
                parent_score = prev_texts[parent_text]['score']
            else:
                for t, info in prev_texts.items():
                     if parent_text in t or t in parent_text:
                         parent_score = info['score']
                         break
            
            parent_attempt_map[parent_text].append({
                "iter": iter_idx,
                "grad": gradient,
                "p_score": parent_score,
                "c_score": child_node['score']
            })

    print(f"DEBUG: Analyzed {len(parent_attempt_map)} unique parents.")
    attempt_counts = [len(v) for v in parent_attempt_map.values()]
    if attempt_counts:
        print(f"DEBUG: Max attempts on single parent: {max(attempt_counts)}")
        print(f"DEBUG: Avg attempts: {sum(attempt_counts)/len(attempt_counts):.2f}")

    # Check for repetitions in parent_attempt_map
    for p_text, attempts in parent_attempt_map.items():
        if len(attempts) < 2: continue
        
        # Compare every pair
        for i in range(len(attempts)):
            for j in range(i + 1, len(attempts)):
                a1 = attempts[i]
                a2 = attempts[j]
                
                if similar(a1['grad'], a2['grad']) > 0.85:
                    if a1['c_score'] < a1['p_score']: # Repetition of a FAILURE
                         repetitive_failures.append({
                                "row": row_idx,
                                "iter": a2['iter'],
                                "grad": a2['grad'][:50],
                                "past_iter": a1['iter'],
                                "type": "Repeated Failed Gradient on Same Parent"
                         })
                         
    return repetitive_failures, content_loops

def analyze_all():
    print(f"Analyzing all 10 rows...")
    all_reps = []
    all_loops = []
    
    for i in range(10):
        reps, loops = analyze_row(i)
        all_reps.extend(reps)
        all_loops.extend(loops)
        print(f"Row {i}: {len(reps)} repeated failures, {len(loops)} content loops")
        
    print("\n=== SUMMARY ===")
    print(f"Total Repeated Failed Gradients: {len(all_reps)}")
    print(f"Total Content Reversions (Loops): {len(all_loops)}")
    
    if all_reps:
        print("\n[Example Repeated Failures]")
        for x in all_reps[:5]:
             print(f"Row {x['row']} {x['iter']}: Repeated gradient from {x['past_iter']}")
             print(f"  Grad: {x['grad']}...")

    if all_loops:
        print("\n[Example Content Loops]")
        for x in all_loops[:5]:
             print(f"Row {x['row']} {x['iter']}: Reverted to content of {x['reverted_to']}")

if __name__ == "__main__":
    analyze_all()
