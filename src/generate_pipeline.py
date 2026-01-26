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

        f.write(f"PROJECT_ROOT=\"{project_root}\"\n")
        f.write(f"PYTHONpath=\"$PROJECT_ROOT/src\"\n\n")
 
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
            
            # 下位互換性維持 & 新機能対応
            evolution_method = exp.get('evolution_method', 'ga')
            ensemble_ratios = exp.get('ensemble_ratios', '')
            population_name = exp.get('population_name', 'default')
            
            f.write(f"# Experiment: {exp_id} ({evolution_method}) [Pop: {population_name}]\n")
            f.write(f"echo \"Starting Experiment: {exp_id} with {evolution_method} (Pop: {population_name})\"\n")

            # 初期化チェック (Populationごと)
            # result/[exp_id]/initial_population/[pop_name]
            initial_pop_dir = os.path.join(project_root, "result", exp_id, "initial_population", population_name)
            
            # 初期化コマンドの挿入 (まだ初期化コードが書かれていない場合のみ)
            # シェルスクリプト上でのチェックも入れるが、Python側でも重複して書かないように制御してもよい
            # ここではシンプルに毎回チェックを入れる（シェル側で判定）
            f.write(f"if [ ! -d \"{initial_pop_dir}\" ]; then\n")
            f.write(f"    echo \"Initializing Population: {population_name} for {exp_id}\"\n")
            f.write(f"    export PYTHONPATH=$PYTHONpath\n")
            f.write(f"    python3 \"$PROJECT_ROOT/src/initialize_task.py\" \\\n")
            f.write(f"        --experiment-id \"{exp_id}\" \\\n")
            f.write(f"        --population-name \"{population_name}\" \\\n")
            f.write(f"        --population-size \"{pop_size}\" \\\n")
            f.write(f"        --model-name \"{model}\" \\\n")
            f.write(f"        --adapter-type \"{adapter_type}\" \\\n")
            f.write(f"        --task-definition \"{task_def}\"\n")
            f.write("fi\n\n")
            
            # 各世代(iteration)ごとの実行コマンドを生成
            for i in range(max_gen):
                # フォルダ構成を result/exp_id/pop_name/method/evaluator/row_*/iterN に変更
                
                # check_file path update
                # generate.py uses: result_dir = .../method/evaluator
                # inside generate.py: result_dir/row_0/iterN
                # So full path: result/exp_id/pop_name/method/evaluator/row_0/iterN/metrics.json
                base_iter_dir = os.path.join(project_root, "result", exp_id, population_name, evolution_method, evaluator)
                check_file = os.path.join(base_iter_dir, f"row_0/iter{i}/metrics.json")
                
                # 冪等性(Idempotency)の確保
                f.write(f"if [ ! -f \"{check_file}\" ]; then\n")
                f.write(f"    echo \"Running Iteration {i}\"\n")
                
                # 生成ステップ (Generation Phase)
                # 引数を追加: ..., population_name
                f.write(f"    ./cmd/generate_next_step.sh \"{exp_id}\" \"{i}\" \"{pop_size}\" \"{model}\" \"{adapter_type}\" \"{task_def}\" \"{evolution_method}\" \"{ensemble_ratios}\" \"{evaluator}\" \"{population_name}\"\n")
                
                # 評価ステップ (Evaluation Phase)
                # パス解決のために evolution_method, population_name を渡す
                f.write(f"    ./cmd/evaluate_step.sh \"{exp_id}\" \"{i}\" \"{model}\" \"{adapter_type}\" \"{evaluator}\" \"{target}\" \"{evolution_method}\" \"{population_name}\"\n")
                
                f.write("else\n")
                f.write(f"    echo \"Skipping Iteration {i} (already completed)\"\n")
                f.write("fi\n\n")

    # 生成したスクリプトに実行権限(755)を付与
    os.chmod(args.output, 0o755)
    print(f"Pipeline script generated at {args.output}")

if __name__ == "__main__":
    main()
