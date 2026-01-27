from visualize_genealogy import load_texts, build_genealogy, BASE_DIR

print("Building Graph Model...")
text_db = load_texts(BASE_DIR)
nodes, edges = build_genealogy(BASE_DIR, text_db)

print(f"Total Edges: {len(edges)}")
print("Scanning for Gray (Elitism) edges with drops...")

found_issue = False
for u, v, t in edges:
    if t == "elitism":
        u_score = nodes[u]['score']
        v_score = nodes[v]['score']
        
        print(f"[GRAY/ELITISM] {u} ({u_score}) -> {v} ({v_score})")
        
        if u_score > 0.7 and v_score < 0.3:
            print(f"  !!! SUSPICIOUS DROP FOUND !!!")
            found_issue = True

if not found_issue:
    print("No suspicious gray edges found.")
else:
    print("Found suspicious gray edges.")
