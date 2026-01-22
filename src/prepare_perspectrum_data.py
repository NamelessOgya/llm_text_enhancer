import os
import argparse
import pandas as pd
import yaml
import random
import requests
import json

def load_config(config_path="config/data_generation_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_json(url):
    print(f"Downloading {url}...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

def prepare_perspectrum_data(output_path: str, sample_size: int = 100):
    """
    Downloads Perspectrum data from CogComp GitHub repository.
    Merges claims with perspective texts.
    """
    base_url = "https://raw.githubusercontent.com/CogComp/perspectrum/master/data/dataset"
    claims_url = f"{base_url}/perspectrum_with_answers_v1.0.json"
    pool_url = f"{base_url}/perspective_pool_v1.0.json"
    
    # 1. Download Data
    claims_data = download_json(claims_url)
    pool_data = download_json(pool_url)
    
    print(f"Loaded {len(claims_data)} claims and {len(pool_data)} perspectives.")
    
    # 2. Build Lookup Map
    pid_to_text = {item['pId']: item['text'] for item in pool_data}
    
    processed_data = []
    
    # 3. Flatten
    for item in claims_data:
        claim_text = item.get("text", "")
        perspectives = item.get("perspectives", [])
        
        if not claim_text:
            continue
            
        for p in perspectives:
            pids = p.get("pids", [])
            stance = p.get("stance_label_3", "") # SUPPORT, CRITICIZE/UNDERMINE?
            
            # Normalize stance if needed. 
            # In inspection: "SUPPORT", "UNDERMINE".
            # User requirement: "SUPPORT" / "OPPOSE".
            if stance == "UNDERMINE":
                stance = "OPPOSE"
            
            if not stance:
                continue
                
            # Use the first pid's text, or all?
            # Usually these are paraphrases. Let's start with just the first one to avoid near-duplicates in a small sample.
            # Or if we want diversity, we can include all.
            # Let's include ONLY the first one for now to keep it simple, as they are often synonymous.
            if not pids:
                continue
            
            pid = pids[0] 
            p_text = pid_to_text.get(pid)
            
            if not p_text:
                continue
                
            summary = f"{stance}: {p_text}"
            
            processed_data.append({
                "content": claim_text,
                "summary": summary
            })
            
    print(f"Total flattened samples: {len(processed_data)}")
    
    # 4. Shuffle & Save
    random.seed(42)
    random.shuffle(processed_data)
    
    df = pd.DataFrame(processed_data)
    
    if sample_size > 0 and len(df) > sample_size:
        print(f"Sampling {sample_size} items...")
        df = df.head(sample_size)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} items to {output_path}")

def run_preparation(config_path="config/data_generation_config.yaml", dataset_key="perspectrum"):
    config = load_config(config_path)
    if dataset_key not in config.get("datasets", {}):
        print(f"Dataset key '{dataset_key}' not found in configuration.")
        return
        
    ds_config = config["datasets"][dataset_key]
    output_path = ds_config.get("output_path", "data/perspectrum.csv")
    sample_size = ds_config.get("sample_size", 100)
    
    prepare_perspectrum_data(output_path, sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_generation_config.yaml")
    parser.add_argument("--dataset", default="perspectrum")
    args = parser.parse_args()
    
    try:
        run_preparation(args.config, args.dataset)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
