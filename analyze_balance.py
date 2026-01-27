import os
import glob
import json
import sys
from collections import defaultdict, Counter

# Add analysis/plots to path to import visualize_genealogy
sys.path.append(os.path.join(os.getcwd(), 'analysis', 'plots'))
try:
    from visualize_genealogy import load_texts, build_genealogy
except ImportError:
    print("[ERROR] Could not import visualize_genealogy. Please ensure it exists.")
    sys.exit(1)

BASE_DIR = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_0"

def analyze_balance(base_dir):
    print(f"Analyzing {base_dir}...")
    text_db = load_texts(base_dir)
    nodes, edges = build_genealogy(base_dir, text_db)
    
    # Organize nodes by iteration
    iter_nodes = defaultdict(list)
    for nid, info in nodes.items():
        iter_nodes[info['iter']].append((nid, info))
        
    sorted_iters = sorted(iter_nodes.keys())
    
    # 1. Composition per Iteration
    print("\n--- 1. Population Composition per Iteration ---")
    print(f"{'Iter':<5} | {'Elite':<5} {'Exploit':<7} {'Div':<5} {'Explore':<7} | {'Avg Score':<9} | {'Max Score':<9}")
    
    high_score_sources = Counter()
    
    for i in sorted_iters:
        if i == 0: continue
        
        types = [info['type'] for nid, info in iter_nodes[i]]
        scores = [info['score'] for nid, info in iter_nodes[i]]
        
        counts = Counter(types)
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        print(f"{i:<5} | {counts['elitism']:<5} {counts['exploit']:<7} {counts['diversity']:<5} {counts['unknown']:<7} | {avg_score:.3f}     | {max_score:.3f}")
        
        # 2. Source of High Scorers (e.g. >= 0.7)
        for nid, info in iter_nodes[i]:
            if info['score'] >= 0.7:
                # "unknown" in build_genealogy usually implies New/Explore in EED context if not linked
                # But let's check if build_genealogy correctly tagged "explore"
                # My previous build_genealogy snippet had: elif '"type": "eed_explore_persona"' in logic_content: nodes[...]["type"] = "explore"
                # So "unknown" might be strictly nothing found?
                # Actually visualize_genealogy script sets "explore" type.
                high_score_sources[info['type']] += 1

    print("\n--- 2. Source of High Scoring Individuals (Score >= 0.7) ---")
    total_high = sum(high_score_sources.values())
    for t, count in high_score_sources.items():
        pct = (count / total_high) * 100 if total_high else 0
        print(f"{t:<15}: {count} ({pct:.1f}%)")
        
    # 3. Success Rate by Mutation Type (Improvement over parent?)
    # Need to match Parent Score.
    print("\n--- 3. Success Rate (Child Score >= Parent Score) ---")
    type_stats = defaultdict(lambda: {'success': 0, 'fail': 0, 'total': 0, 'score_diff': 0})
    
    for u, v, t in edges:
        if u not in nodes or v not in nodes: continue
        
        p_score = nodes[u]['score']
        c_score = nodes[v]['score']
        
        type_stats[t]['total'] += 1
        type_stats[t]['score_diff'] += (c_score - p_score)
        
        if c_score >= p_score:
            type_stats[t]['success'] += 1
        else:
            type_stats[t]['fail'] += 1
            
    for t in ['elitism', 'exploit', 'diversity', 'explore']: # explore usually has no parent edge in graph? 
        if t not in type_stats: continue
        stats = type_stats[t]
        rate = (stats['success'] / stats['total']) * 100 if stats['total'] else 0
        avg_diff = stats['score_diff'] / stats['total'] if stats['total'] else 0
        print(f"{t:<15}: Success {rate:.1f}% ({stats['success']}/{stats['total']}) | Avg Diff: {avg_diff:+.3f}")

if __name__ == "__main__":
    analyze_balance(BASE_DIR)
