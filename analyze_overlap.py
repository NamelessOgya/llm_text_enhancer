import json
import os
from collections import defaultdict

def analyze_overlap(row_idx, row_name):
    path = f'/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/perspectrum_v1.1/run1/eed/perspectrum_llm/row_{row_idx}/eed_knowledge.json'
    
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)
        
    items = data.get('explored_items', [])
    print(f"\n=== Analyzing {row_name} (Row {row_idx}, {len(items)} items) ===")
    
    # Simple Keyword Clustering
    clusters = defaultdict(list)
    
    for item in items:
        p = item['persona']
        arg = item['argument'].lower()
        
        # Row 1 Clusters
        if row_idx == 1:
            if any(k in arg for k in ['resource', 'fund', 'money', 'cost', 'financial', 'tax', 'spending', 'divert']):
                clusters['Resource/Cost Diversion'].append(p)
            elif any(k in arg for k in ['environment', 'contaminat', 'echo', 'climate', 'winter', 'nature', 'biodiversity']):
                clusters['Environment/Climate'].append(p)
            elif any(k in arg for k in ['health', 'medic', 'sick', 'casualt', 'radiation']):
                clusters['Public Health/Safety'].append(p)
            elif any(k in arg for k in ['unstable', 'risk', 'miscalculation', 'accidental', 'security', 'conflict', 'war']):
                clusters['Instability/Risk'].append(p)
            elif any(k in arg for k in ['law', 'legal', 'right']):
                clusters['Legal/Rights'].append(p)
            else:
                clusters['Other'].append(p)

        # Row 8 Clusters (Family Size)
        elif row_idx == 8:
            if any(k in arg for k in ['autonomy', 'choice', 'freedom', 'right', 'decide']):
                clusters['Autonomy/Freedom'].append(p)
            elif any(k in arg for k in ['resource', 'environment', 'sustain', 'climate', 'shortage']):
                clusters['Resource/Environment'].append(p)
            elif any(k in arg for k in ['economic', 'market', 'financial', 'workforce', 'labor', 'consumer']):
                clusters['Economic/Workforce'].append(p)
            elif any(k in arg for k in ['social', 'skill', 'empathy', 'relationship', 'development']):
                clusters['Social/Development'].append(p)
            else:
                clusters['Other'].append(p)

    # Print Clusters
    for category, personas in clusters.items():
        print(f"\n[Cluster: {category}] Count: {len(personas)}")
        print(f"  Personas: {', '.join(personas[:10])}..." if len(personas)>10 else f"  Personas: {', '.join(personas)}")

analyze_overlap(1, "Nuclear Weapons")
analyze_overlap(8, "Family Size")
