import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments.csv")
    parser.add_argument("--output", default="run_pipeline.sh")
    args = parser.parse_args()

    # Get absolute path to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(args.config, 'r') as f:
        reader = csv.DictReader(f)
        try:
             # Just iterate, DictReader handles headers automatically
            experiments = list(reader)
        except csv.Error as e:
            print(f"Error reading CSV: {e}")
            return

    with open(args.output, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Generated pipeline script\n\n")
        f.write("set -e\n\n") # Exit on error

        for exp in experiments:
            exp_id = exp['experiment_id']
            max_gen = int(exp['max_generations'])
            pop_size = exp['population_size']
            model = exp['model_name']
            
            # Default to 'llm' if not present (backward compatibility)
            evaluator = exp.get('evaluator_type', 'llm')
            
            task_def = exp['task_definition']
            target = exp['target_preference']
            
            f.write(f"# Experiment: {exp_id}\n")
            f.write(f"echo \"Starting Experiment: {exp_id}\"\n")
            
            for i in range(max_gen):
                iter_dir = os.path.join(project_root, "result", exp_id, f"iter{i}")
                metrics_file = os.path.join(iter_dir, "metrics.json")
                
                # Check idempotency
                f.write(f"if [ ! -f \"{metrics_file}\" ]; then\n")
                f.write(f"    echo \"Running Iteration {i}\"\n")
                
                # Generate
                f.write(f"    ./cmd/generate_next_step.sh \"{exp_id}\" \"{i}\" \"{pop_size}\" \"{model}\" \"{task_def}\"\n")
                
                # Evaluate - PASS EVALUATOR_TYPE
                f.write(f"    ./cmd/evaluate_step.sh \"{exp_id}\" \"{i}\" \"{model}\" \"{evaluator}\" \"{target}\"\n")
                
                f.write("else\n")
                f.write(f"    echo \"Skipping Iteration {i} (already completed)\"\n")
                f.write("fi\n\n")

    os.chmod(args.output, 0o755)
    print(f"Pipeline script generated at {args.output}")

if __name__ == "__main__":
    main()
