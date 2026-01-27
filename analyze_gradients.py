import os
import glob
import re
import sys
import json
import networkx as nx
from collections import defaultdict

# Add analysis/plots/visualize_genealogy.py path
sys.path.append(os.path.join(os.getcwd(), 'analysis', 'plots'))
try:
    from visualize_genealogy import load_texts, build_genealogy
except ImportError:
    print("[ERROR] Could not import visualize_genealogy.")
    sys.exit(1)

BASE_DIR = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_0"

def get_gradient_text(logic_path):
    if not os.path.exists(logic_path): return ""
    with open(logic_path, 'r') as f:
        content = f.read()
    
    # Extract "Gradient" or "Feedback" section
    # Based on previous file views, format is often JSON-like or Prompt text.
    # Pattern: "Gradient:\n\n..." until "Revised Text:" or similar.
    # Or "Feedback (Gradient) for Improvement:"
    
    # Try regex for the section
    match = re.search(r'(?:Gradient|Critique|Feedback).*?:\n(.*?)(?:\n\n(?:Revised|Directive|Altern|Role)|$)', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def analyze_gradients(base_dir):
    print(f"Analyzing Gradients in {base_dir}...")
    text_db = load_texts(base_dir)
    print(f"DEBUG: text_db keys -> {text_db.keys()}")
    nodes, edges = build_genealogy(base_dir, text_db)
    
    G = nx.DiGraph()
    for u, v, t in edges:
        G.add_edge(u, v, type=t)
        
    print(f"DEBUG: Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    if G.number_of_nodes() == 0:
        return
        
    # Find all paths of length >= 2 involving Exploit
    # Filter paths to keep only those containing at least one Exploit edge?
    # Or preferably multiple.
    
    # Let's just traverse from roots to leaves and print lines where Exploit happens frequently.
    
    paths = []
    # Find leaves
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    # Find roots
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    # Get all paths (might be expensive if connected, but tree usually)
    for root in roots:
        # Use DFS to find paths
        # limit depth?
        # Just simple traversal
        stack = [(root, [root])]
        while stack:
            curr, path = stack.pop()
            neighbors = list(G.successors(curr))
            if not neighbors:
                paths.append(path)
            else:
                for n in neighbors:
                    stack.append((n, path + [n]))
                    
    print(f"Found {len(paths)} lineages.")
    
    oscillation_count = 0
    repetition_count = 0
    
    for path in paths:
        # Filter: check if path has at least 2 Exploit/Diversity steps
        # We need logic files for nodes (except root usually).
        # Node ID: i{iter}_text_{idx}
        
        path_details = []
        has_exploit = False
        
        for i, nid in enumerate(path):
            if i == 0: continue # Skip root (no logic usually, or init)
            
            # Get edge type leading to this node
            prev = path[i-1]
            edge_data = G.get_edge_data(prev, nid)
            etype = edge_data['type']
            
            if etype != 'exploit': continue
            has_exploit = True
            
            # Get Logic File
            info = nodes[nid] # {iter: ..., score: ...}
            iter_idx = info['iter']
            # parse nid "iX_text_Y"
            try:
                # "i1_text_0" -> split
                parts = nid.split('_text_')
                idx_str = parts[1]
            except: continue
            
            logic_path = os.path.join(base_dir, f"iter{iter_idx}", "logic", f"creation_prompt_{idx_str}.txt")
            gradient = get_gradient_text(logic_path)
            
            path_details.append({
                "iter": iter_idx,
                "node": nid,
                "score": info['score'],
                "gradient": gradient[:100].replace('\n', ' ') + "..." # Truncate
            })
            
        if len(path_details) >= 2:
            print(f"\nLineage: {' -> '.join(path)}")
            for step in path_details:
                print(f"  Iter {step['iter']} (Score {step['score']}): {step['gradient']}")
                
            # Check similarity
            # Very primitive check
            grads = [s['gradient'] for s in path_details]
            if len(set(grads)) < len(grads):
                print("  [ALERT] Identical gradient repeated!")
                repetition_count += 1
            
            # Oscillation check? (A -> B -> A)
            if len(grads) >= 3:
                if grads[-1] == grads[-3]: 
                     print("  [ALERT] Oscillation detected (A -> B -> A)!")
                     oscillation_count += 1
                     
    print(f"\nSummary: {repetition_count} repetitions, {oscillation_count} oscillations detected in {len(paths)} paths.")

if __name__ == "__main__":
    analyze_gradients(BASE_DIR)
