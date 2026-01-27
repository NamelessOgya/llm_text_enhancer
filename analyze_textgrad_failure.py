import os
import glob
import json
import re
from collections import defaultdict, Counter

BASE_DIR_ROOT = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm"

def load_text_db(base_dir):
    db = defaultdict(dict) # iter -> content -> {score, file}
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
                db[iter_idx][content] = {"score": m.get("score"), "file": fname}
    return db

def analyze_failures_for_row(base_dir):
    # print(f"Scanning {base_dir}...")
    db = load_text_db(base_dir)
    print(f"Loaded {len(db)} iterations.")
    
    failures = []
    successes = []
    
    for iter_idx in sorted(db.keys()):
        if iter_idx == 0: continue
        
        logic_dir = os.path.join(base_dir, f"iter{iter_idx}", "logic")
        current_iter_texts = db.get(iter_idx, {})
        # Parent could be from ANY previous iteration technically, but usually prev (iter-1)
        # However, TextGrad might pick from current population if population passed is mixed?
        # Assuming Parents come from iter_idx - 1.
        prev_iter_texts = db.get(iter_idx - 1, {})
        
        for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
            with open(logic_file, 'r') as f: content = f.read()
            
            # Identify format: JSON or Text
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "meta_prompt" in data:
                    content = data["meta_prompt"]
            except:
                pass # Treat as plain text

            # Skip Elitism
            if "Elitism:" in content: continue
            
            # Extract Current Text (Parent)
            # Pattern: Current Text:\n"..."
            match = re.search(r'Current Text:\s*(?:["\'])(.*?)(?:["\'])', content, re.DOTALL)
            if not match: match = re.search(r'Current Text:\s*\n(.*?)\n', content, re.DOTALL)
            
            if not match: continue
            parent_text = match.group(1).strip()
            
            # Find in prev db
            # Need fuzzy match or exact?
            # Start with exact
            parent_info = prev_iter_texts.get(parent_text)
            if not parent_info:
                 # Try to search in all keys
                 for t, info in prev_iter_texts.items():
                     if parent_text in t or t in parent_text: # Loose match
                         parent_info = info
                         break
            
            if not parent_info: continue
            
            # Get Child Info
            fname = os.path.basename(logic_file).replace("creation_prompt_", "text_")
            # Need to find content for this fname
            # Reverse lookup in current_iter_texts
            child_info = None
            for t, info in current_iter_texts.items():
                 if info['file'] == fname:
                     child_info = info
                     break
            
            if not child_info: continue
            
            p_score = parent_info['score']
            c_score = child_info['score']
            
            # Extract Gradient
            grad_match = re.search(r'Feedback \(Gradient\) for Improvement:\s*(.*?)(?:\n\nDirective:|$)', content, re.DOTALL)
            gradient = grad_match.group(1).strip() if grad_match else "N/A"
            
            record = {
                "iter": iter_idx,
                "parent_score": p_score,
                "child_score": c_score,
                "diff": c_score - p_score,
                "gradient": gradient
            }
            
            if c_score < p_score:
                failures.append(record)
            elif c_score > p_score:
                successes.append(record)
                
    return failures, successes

def analyze_all_rows():
    all_failures = []
    all_successes = []
    
    print(f"{'Row':<5} | {'Failures':<10} | {'Successes':<10} | {'Rate':<10}")
    print("-" * 45)

    for i in range(10):
        row_dir = os.path.join(BASE_DIR_ROOT, f"row_{i}")
        if not os.path.exists(row_dir): continue
        
        f, s = analyze_failures_for_row(row_dir)
        all_failures.extend(f)
        all_successes.extend(s)
        
        total = len(f) + len(s)
        rate = (len(f) / total * 100) if total else 0
        print(f"{i:<5} | {len(f):<10} | {len(s):<10} | {rate:.1f}%")
        
    print("-" * 45)
    print(f"Total | {len(all_failures):<10} | {len(all_successes):<10} | {(len(all_failures)/(len(all_failures)+len(all_successes))*100):.1f}%")

    print(f"Total | {len(all_failures):<10} | {len(all_successes):<10} | {(len(all_failures)/(len(all_failures)+len(all_successes))*100):.1f}%")

    if all_failures:
        avg_fail_score = sum(f['child_score'] for f in all_failures) / len(all_failures)
        print(f"Avg Child Score on Failures: {avg_fail_score:.3f}")
        
        # Sort by parent score descending (failures on high performers)
        all_failures.sort(key=lambda x: x['parent_score'], reverse=True)
        print("\n--- Top High-Stake Failures ---")
        for f in all_failures[:5]:
             print(f"Row {f.get('iter')}?? Iter {f['iter']} Score {f['parent_score']} -> {f['child_score']}")
             print(f"Gradient: {f['gradient'][:100]}...")

    # Length Analysis
    all_grads = [f['gradient'] for f in all_failures] + [s['gradient'] for s in all_successes]
    if all_grads:
        lengths_char = [len(g) for g in all_grads]
        lengths_word = [len(g.split()) for g in all_grads]
        print("\n--- Gradient Length Statistics ---")
        print(f"Count: {len(all_grads)}")
        print(f"Avg Length (Chars): {sum(lengths_char)/len(lengths_char):.1f}")
        print(f"Max Length (Chars): {max(lengths_char)}")
        print(f"Min Length (Chars): {min(lengths_char)}")
        print(f"Avg Length (Words): {sum(lengths_word)/len(lengths_word):.1f}")

    print("\n--- Common Keyphrases in ALL Failed Gradients ---")
    all_text = " ".join([f['gradient'] for f in all_failures])
    words = [w.lower() for w in re.findall(r'\w+', all_text) if len(w) > 4]
    common = Counter(words).most_common(15)
    for w, c in common:
        print(f"{w}: {c}")

if __name__ == "__main__":
    analyze_all_rows()
