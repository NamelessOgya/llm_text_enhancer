import os
import json
import glob
import re
from visualize_genealogy import load_texts, build_genealogy

BASE_DIR = "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_0"

print("Loading data...")
text_db = load_texts(BASE_DIR)
nodes, edges = build_genealogy(BASE_DIR, text_db)

print(f"Checking {len(edges)} edges for Elitism drops...")

for u, v, t in edges:
    if t == "elitism":
        # Get scores
        parent_score = nodes[u]['score'] if u in nodes else None
        child_score = nodes[v]['score'] if v in nodes else None
        
        # If node not in nodes dict directly (maybe parent from prev iter), look it up
        # load_texts returns text_db[iter][content] -> info
        # build_genealogy constructs nodes dict dynamically.
        # Let's trust nodes dict has what we need if build_genealogy populated it.
        # Wait, build_genealogy iterates current iter texts.
        # Parent node u might be from prev iter. 
        # `nodes` dictionary behaves: `nodes[child_node_id] = ...`.
        # Does it populate parent nodes?
        # In `build_genealogy`: `nodes[child_node_id] = ...`. 
        # It does NOT explicitly add parent node `u` to `nodes` mapping if `u` was from previous iteration 
        # (unless `u` was added when processing that previous iteration).
        # Since we run over all iterations, `u` should be in `nodes`.
        
        if parent_score is not None and child_score is not None:
            if parent_score > 0.7 and child_score < 0.3:
                print(f"[SUSPICIOUS EDGE] {u} ({parent_score}) -> {v} ({child_score}) [Type: {t}]")
        else:
            print(f"[WARNING] Missing node info: {u} or {v}")
            
print("Done.")
