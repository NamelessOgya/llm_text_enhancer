# LLM Text Enhancer

遺伝的アルゴリズム (Genetic Algorithm) やその他の手法を用いて、LLMが生成するテキストの品質を反復的に向上させるための実験フレームワークです。

GeneratorはEvaluatorの「隠された嗜好」をスコアフィードバックのみから学習し、ターゲットに適合したテキストを生成するように進化します。

## サポートされている進化戦略 (Evolution Strategies)

本フレームワークは以下の5つのプロンプト最適化手法をサポートしています。

1. **Genetic Algorithm (GA)** [デフォルト]: エリート保存と変異（言い換え、拡張、短縮、トーン変更）を利用。
2. **TextGrad**: スコアに基づく「擬似勾配（修正指示）」をLLMに生成させ、プロンプトを修正。
3. **Trajectory (OPRO)**: 過去のスコア履歴の「改善の軌跡」を分析し、より良いプロンプトを予測。
4. **Demonstration (DSPy)**: 高スコアの成功事例をFew-Shotとしてプロンプトに埋め込む。
5. **Ensemble**: 上記の手法を任意の比率で組み合わせて実行 (例: GA 50% + TextGrad 50%)。

## 必要要件

- Docker (推奨)
- Python 3.9+ (ローカル実行の場合)
- OpenAI API Key

## セットアップ

### 1. 環境変数の設定

`.env` ファイルを作成し、OpenAI APIキーを設定してください。

```bash
cp .env.example .env
# .envを編集して OPENAI_API_KEY を設定
```

### 2. 実験設定

`config/experiments.csv` を編集して実験パラメータを設定します。

```csv
experiment_id,max_generations,population_size,model_name,evaluator_type,task_definition,target_preference,evolution_method,ensemble_ratios
my_exp,5,5,gpt-4o,llm,Generate a short story.,Horror story,ga,
ensemble_exp,5,10,gpt-4o,llm,Title generation,Funny,ensemble,"ga:0.5,textgrad:0.5"
```

- `task_definition`: 生成タスクの一般的な指示（Generatorへの入力）。
- `target_preference`: 評価の正解基準（Evaluatorのみが使用）。
- `evaluator_type`: `llm`, `rule_keyword`, `rule_regex` から選択。
- `evolution_method`: `ga`, `textgrad`, `trajectory`, `demonstration`, `ensemble` から選択。
- `ensemble_ratios`: `ensemble` 選択時のみ有効。`"method:ratio,..."` の形式で指定。

### 3. データセットを用いた実験 (Dataset Experiment)

CSVなどの外部データセットを用いた実験も可能です。

1. **データセットの用意**: `data/` 配下にCSVファイルを配置します (例: `data/dummy_tldr.csv`)。
2. **タスク定義**: `.taml` ファイルの `[ref]` セクションでデータセットとカラムを指定します。

```taml
[content]
以下のテキストを要約してください。
{{ content }}

[ref]
dataset: data/dummy_tldr.csv
column: content
target_column: summary
```

3. **実行**: 通常通りパイプラインを実行すると、各行ごとにディレクトリ (`result/exp_id/ga/row_N/iterM`) が作成され、個別に最適化が行われます。

### 4. データセットの準備 (Data Preparation)

本番用データセット (例: TL;DR) をダウンロード・作成するための専用スクリプトが用意されています。

1. **設定**: `config/data_generation_config.yaml` を編集して、データセットや生成件数を設定します。

```yaml
datasets:
  tldr:
    source: "CarperAI/openai_summarize_tldr"
    split: "train"
    sample_size: 100
    output_path: "data/tldr.csv"
```

2. **実行**: Dockerコンテナ内でデータ生成スクリプトを実行します。

```bash
docker run --rm -v $(pwd):/app llm_text_enhancer python src/prepare_tldr_data.py --dataset tldr
```

## 実行方法 (Docker)

### 1. イメージのビルド

```bash
./build.sh
```

### 2. パイプラインの生成と実行

コンテナ内でパイプラインスクリプトを生成し、実行します。

1. インタラクティブモードでコンテナに入ります:

```bash
docker run --rm -it -v $(pwd):/app llm_text_enhancer /bin/bash
```

2. コンテナ内で以下のコマンドを実行します:

```bash
# パイプライン生成
python3 src/generate_pipeline.py

# データセットの準備 (必要に応じて)
python3 src/prepare_tldr_data.py --dataset tldr

# 実験実行
./run_pipeline.sh

# データ集計
python3 src/aggregate_results.py
```

## 実行方法 (ローカル)

```bash
# 依存関係のインストール
pip install -r requirements.txt

# データセットの準備 (必要に応じて)
python3 src/prepare_tldr_data.py --dataset tldr

# パイプライン生成
python3 src/generate_pipeline.py

# 実験実行
./run_pipeline.sh

# データ集計
python3 src/aggregate_results.py
```

## 結果の集計 (Aggregation)

`src/aggregate_results.py` を実行すると、全実験・全手法・全イテレーションのスコア（最大・最小）が集計され、`result/aggregation_report.csv` に出力されます。

```bash
python3 src/aggregate_results.py
# オプション: --result-dir "path/to/result" --output "my_report.csv"
```

## テストの実行

### Docker内でのテスト（推奨）

```bash
docker run --rm -v $(pwd):/app llm_text_enhancer python3 -m unittest discover tests
```

### ローカルでのテスト

```bash
python3 -m unittest discover tests
```

## ディレクトリ構成

- `src/`: ソースコード
- `cmd/`: 実行用シェルスクリプト
- `config/`: 設定ファイル
- `result/`: 実験結果（テキスト、ログ、メトリクス）
