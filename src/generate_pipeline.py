import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="実験パイプライン生成スクリプト")
    parser.add_argument("--config", default="config/experiments.csv", help="実験設定CSVファイルへのパス")
    parser.add_argument("--output", default="run_pipeline.sh", help="生成される実行用シェルスクリプトのファイル名")
    args = parser.parse_args()

    # プロジェクトルートの絶対パスを取得 (このスクリプトはsrc/配下にあるため、2階層上がルート)
    # これにより、どのディレクトリから実行してもパス解決が正しく行われるようにする。
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(args.config, 'r') as f:
        reader = csv.DictReader(f)
        try:
             # DictReaderはヘッダー行をキーとして、各行を辞書形式で読み込む
            experiments = list(reader)
        except csv.Error as e:
            print(f"Error reading CSV: {e}")
            return

    # 実行用シェルスクリプトの生成
    # Pythonから直接サブプロセスを起動するのではなく、一度シェルスクリプトとして出力することで、
    # 実行内容の事前確認や手動での再実行、デバッグを容易にする設計としている。
    with open(args.output, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Generated pipeline script\n\n")
        f.write("set -e\n\n") # エラー発生時に即時終了 (安全性確保)
 
        for exp in experiments:
            exp_id = exp['experiment_id']
            max_gen = int(exp['max_generations'])
            pop_size = exp['population_size']
            model = exp['model_name']
            adapter_type = exp['adapter_type']
            
            # 下位互換性維持: 指定がない場合はデフォルトで'llm' (LLMEvaluator) を使用
            evaluator = exp.get('evaluator_type', 'llm')
            
            task_def = exp['task_definition']
            target = exp['target_preference']
            
            f.write(f"# Experiment: {exp_id}\n")
            f.write(f"echo \"Starting Experiment: {exp_id}\"\n")
            
            # 各世代(iteration)ごとの実行コマンドを生成
            for i in range(max_gen):
                iter_dir = os.path.join(project_root, "result", exp_id, f"iter{i}")
                metrics_file = os.path.join(iter_dir, "metrics.json")
                
                # 冪等性(Idempotency)の確保: 
                # 既に評価結果(metrics.json)が存在する場合、その世代の計算は完了しているとみなし、
                # 再計算を行わずにスキップする。これにより、中断した実験の再開が容易になる。
                f.write(f"if [ ! -f \"{metrics_file}\" ]; then\n")
                f.write(f"    echo \"Running Iteration {i}\"\n")
                
                # 生成ステップ (Generation Phase)
                # 次世代のプロンプトとテキストを生成するスクリプトを呼び出す
                f.write(f"    ./cmd/generate_next_step.sh \"{exp_id}\" \"{i}\" \"{pop_size}\" \"{model}\" \"{adapter_type}\" \"{task_def}\"\n")
                
                # 評価ステップ (Evaluation Phase)
                # 生成されたテキストを評価し、スコアを算出するスクリプトを呼び出す
                f.write(f"    ./cmd/evaluate_step.sh \"{exp_id}\" \"{i}\" \"{model}\" \"{adapter_type}\" \"{evaluator}\" \"{target}\"\n")
                
                f.write("else\n")
                f.write(f"    echo \"Skipping Iteration {i} (already completed)\"\n")
                f.write("fi\n\n")

    # 生成したスクリプトに実行権限(755)を付与
    os.chmod(args.output, 0o755)
    print(f"Pipeline script generated at {args.output}")

if __name__ == "__main__":
    main()
