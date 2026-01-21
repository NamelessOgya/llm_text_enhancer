from datasets import load_dataset
import pandas as pd
import os
import yaml
import argparse

def load_config(config_path="config/data_generation_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_tldr_dataset(config_path="config/data_generation_config.yaml", dataset_key="tldr"):
    """
    Downloads and prepares the TL;DR dataset based on configuration.
    """
    config = load_config(config_path)
    if dataset_key not in config.get("datasets", {}):
        print(f"Dataset key '{dataset_key}' not found in configuration.")
        return
        
    ds_config = config["datasets"][dataset_key]
    
    source = ds_config.get("source", "CarperAI/openai_summarize_tldr")
    split = ds_config.get("split", "train")
    sample_size = ds_config.get("sample_size", 100)
    output_path = ds_config.get("output_path", "data/tldr.csv")
    
    print(f"Loading {source} (split: {split})...")
    try:
        dataset = load_dataset(source, split=split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    print(f"Original dataset size: {len(dataset)}")
    
    # columns: prompt, label (summary)
    df = pd.DataFrame(dataset)
    print(f"Columns: {df.columns}")
    
    # Map valid columns generic logic
    # usually 'prompt' is content, 'label' is summary for this specific dataset
    if 'prompt' in df.columns:
        df.rename(columns={'prompt': 'content'}, inplace=True)
    if 'label' in df.columns:
        df.rename(columns={'label': 'summary'}, inplace=True)
        
    if 'content' not in df.columns or 'summary' not in df.columns:
         print("Required columns 'content' or 'summary' not found after mapping.")
         return
    
    # Filter out empty content or summaries
    df = df[df['content'].astype(str).str.strip() != ""]
    df = df[df['summary'].astype(str).str.strip() != ""]
    
    # Sample
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        print(f"Sampling first {sample_size} rows based on config...")
        df = df.head(sample_size)
    else:
        print(f"Using full dataset ({len(df)} rows) or sample_size not specified/valid.")
    
    print(f"Saving {len(df)} samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_generation_config.yaml", help="Path to config yaml")
    parser.add_argument("--dataset", default="tldr", help="Dataset key in config")
    args = parser.parse_args()
    
    prepare_tldr_dataset(args.config, args.dataset)
