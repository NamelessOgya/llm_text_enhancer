import os
import glob
import json
import re
import sys
from collections import defaultdict
import networkx as nx

# Add analysis/plots to path to import visualize_genealogy
sys.path.append(os.path.join(os.getcwd(), 'analysis', 'plots'))
try:
    from visualize_genealogy import load_texts, build_genealogy
    print("[DEBUG] Using IMPORTED visualize_genealogy")
except ImportError:
    # Fallback if import fails (e.g. if file was modified or path issues)
    # Re-implement minimal logic for robustness
    print("[DEBUG] Using LOCAL implementation")
    
    def load_texts(base_dir):
        text_db = defaultdict(dict)
        for iter_dir in sorted(glob.glob(os.path.join(base_dir, "iter*"))):
            try:
                iter_idx = int(os.path.basename(iter_dir).replace("iter", ""))
            except: continue
            
            metrics_path = os.path.join(iter_dir, "metrics.json")
            texts_dir = os.path.join(iter_dir, "texts")
            
            if not os.path.exists(metrics_path): continue
            
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except: continue
            
            for m in metrics:
                fname = m.get("file")
                t_path = os.path.join(texts_dir, fname)
                if os.path.exists(t_path):
                    with open(t_path, 'r') as tf: content = tf.read().strip()
                    text_db[iter_idx][fname] = {
                        "content": content,
                        "score": m.get("score"),
                        "id": f"i{iter_idx}_{fname.replace('.txt','')}"
                    }
        return text_db

    def build_genealogy(base_dir, text_db):
        edges = []
        nodes = {}
        for iter_dir in sorted(glob.glob(os.path.join(base_dir, "iter*"))):
            try:
                iter_idx = int(os.path.basename(iter_dir).replace("iter", ""))
            except: continue
            
            logic_dir = os.path.join(iter_dir, "logic")
            current_texts = text_db.get(iter_idx, {})
            prev_texts = text_db.get(iter_idx-1, {})
            
            # Map content to ID for prev iter
            prev_content_map = {info['content']: info['id'] for info in prev_texts.values()}
            
            for logic_file in glob.glob(os.path.join(logic_dir, "creation_prompt_*.txt")):
                fname = os.path.basename(logic_file)
                idx_str = fname.replace("creation_prompt_", "").replace(".txt", "")
                text_fname = f"text_{idx_str}.txt"
                
                if text_fname not in current_texts: continue
                child_info = current_texts[text_fname]
                child_id = child_info['id']
                nodes[child_id] = {"iter": iter_idx}
                
                with open(logic_file, 'r') as f: logic = f.read()
                
                if "Elitism:" in logic:
                    if child_info['content'] in prev_content_map:
                        parent_id = prev_content_map[child_info['content']]
                        edges.append((parent_id, child_id, "elitism"))
        return nodes, edges

BASE_PATH = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm"

print(f"{'Row':<5} | {'Max Survival (Gens)':<20} | {'Detail (Chain Lengths)':<30}")
print("-" * 60)

for row_idx in range(10):
    row_dir = os.path.join(BASE_PATH, f"row_{row_idx}")
    if not os.path.exists(row_dir):
        print(f"row_{row_idx} missing")
        continue
        
    text_db = load_texts(row_dir)
    nodes, edges = build_genealogy(row_dir, text_db)
    
    # Build a graph of ONLY Elitism edges
    G = nx.DiGraph()
    for u, v, t in edges:
        if t == "elitism":
            G.add_edge(u, v)
            
    # Find connected components (weakly connected in directed graph? No, distinct paths)
    # Since it's a DAG (time flows forward), we can find longest path in this subgraph
    
    max_len = 0
    chain_lengths = []
    
    if len(G.edges) > 0:
        # Since it is a forest of linear chains (mostly), we can find the extracted components.
        # But branching? Strict elitism usually 1 parent -> 1 child clone.
        # But if duplicates exist, 1 parent -> 2 child clones.
        # We want the DEPTH of the tree.
        
        # Longest path in DAG
        try:
            longest_path = nx.dag_longest_path(G)
            # Length is number of edges = nodes - 1
            max_len = len(longest_path) - 1 
            
            # Get lengths of all disconnected components
            # For each weakly connected component, get its depth
            for component in nx.weakly_connected_components(G):
                subgraph = G.subgraph(component)
                path = nx.dag_longest_path(subgraph)
                chain_lengths.append(len(path) - 1)
        except:
            pass
            
    chain_lengths.sort(reverse=True)
    top_chains = chain_lengths[:3]
    
    # Calculate generations survived
    # If a chain has 1 edge (Iter1 -> Iter2), it survived for 1 transition (2 generations present).
    # User asked "How many generations did it survive?"
    # If it covers Iter1..Iter9, that's 9 generations? or 8 transitions?
    # Usually "survived X generations" implies duration.
    # Let's verify: "row_0 from iter1 to iter9" (prev result) means it survived 8 transitions (9 nodes).
    # I will report "Transitions (Edges)" count. 
    # Max possible for 10 iters (0..9) is 9 transitions.
    
    print(f"row_{row_idx:<3} | {max_len:<20} | {str(top_chains)}")
    
    if row_idx == 0:
        print(f"DEBUG row_0 edges: {len(G.edges)}")
        for u, v in G.edges():
            print(f"  {u} -> {v}")
