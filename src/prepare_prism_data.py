import os
import json
import argparse
from datasets import load_dataset
import pandas as pd
import random
import yaml

def prepare_prism_data(output_path: str, sample_size: int = 100, min_length_prompt: int = 0, min_length_reason: int = 0, max_score: float = 1.0, min_score: float = 0.0):
    """
    HannahRoseKirk/prism-alignment データセットをロードし、
    本プロジェクトで使用する形式 (JSONリスト) に変換して保存する。
    """
    print(f"Loading dataset: HannahRoseKirk/prism-alignment")
    try:
        # Load conversations
        ds_conv = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
        # Load survey (user demographics)
        ds_survey = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loaded conversations: {len(ds_conv)}, survey: {len(ds_survey)}")
    
    # Create user metadata map
    user_map = {}
    for row in ds_survey:
        if "user_id" in row:
            meta = {k: v for k, v in row.items() if k != "user_id"}
            user_map[row["user_id"]] = meta
            
    processed_data = []
    
    # Iterate over ALL conversations to find valid candidates
    for row in ds_conv:
        user_id = row.get("user_id", "unknown_user")
        user_metadata = user_map.get(user_id, {})
        
        # Global feedback for the conversation
        global_feedback = row.get("open_feedback", "")
        if global_feedback is None:
            global_feedback = ""
        
        # Filter by reason length
        if len(str(global_feedback).strip()) < min_length_reason:
            continue

        history = row.get("conversation_history", [])
        
        # Iterate through history to find Input(User) -> Output(Model) pairs
        last_user_content = ""
        
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            
            if role == "user":
                last_user_content = content
            elif role == "model":
                if not last_user_content:
                    continue 

                # Filter by prompt length
                if len(str(last_user_content).strip()) < min_length_prompt:
                    continue
                
                score = turn.get("score")
                if score is None: 
                    continue # Skip unscored
                    
                norm_score = float(score) / 100.0
                
                # Filter by max score (keep only bad/mediocre responses)
                if norm_score > max_score:
                    continue

                # Filter by min score (avoid broken garbage)
                if norm_score < min_score:
                    continue
                
                item = {
                    "user_id": str(user_id),
                    "user_metadata": user_metadata,
                    "prompt": last_user_content,
                    "model_response": content,
                    "score": norm_score,
                    "original_score": score,
                    "reason": global_feedback # Using conversation-level feedback as proxy
                }
                processed_data.append(item)

    # Shuffle the collected pool to ensure random sampling
    random.seed(42)
    random.shuffle(processed_data)

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Apply sample_size limit
    if sample_size > 0 and len(df) > sample_size:
        df = df.head(sample_size)
    
    # Ensure output directory exists (data/ is gitignored, so we must make sure it exists)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} items to {output_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

def load_config(config_path="config/data_generation_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_preparation(config_path="config/data_generation_config.yaml", dataset_key="prism"):
    config = load_config(config_path)
    if dataset_key not in config.get("datasets", {}):
        print(f"Dataset key '{dataset_key}' not found in configuration.")
        return
        
    ds_config = config["datasets"][dataset_key]
    output_path = ds_config.get("output_path", "data/prism_dataset.csv")
    sample_size = ds_config.get("sample_size", 100)
    min_length_prompt = ds_config.get("min_length_prompt", 0)
    min_length_reason = ds_config.get("min_length_reason", 0)
    max_score = ds_config.get("max_score", 1.0)
    min_score = ds_config.get("min_score", 0.0)
    
    prepare_prism_data(output_path, sample_size, min_length_prompt, min_length_reason, max_score, min_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_generation_config.yaml")
    parser.add_argument("--dataset", default="prism")
    args = parser.parse_args()
    
    run_preparation(args.config, args.dataset)
