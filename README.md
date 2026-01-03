# LLM Text Enhancer

遺伝的アルゴリズム (Genetic Algorithm) やその他の手法を用いて、LLMが生成するテキストの品質を反復的に向上させるための実験フレームワークです。

GeneratorはEvaluatorの「隠された嗜好」をスコアフィードバックのみから学習し、ターゲットに適合したテキストを生成するように進化します。

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
experiment_id,max_generations,population_size,model_name,evaluator_type,task_definition,target_preference
my_exp,5,5,gpt-4o,llm,Generate a short story.,Horror story
```

- `task_definition`: 生成タスクの一般的な指示（Generatorへの入力）。
- `target_preference`: 評価の正解基準（Evaluatorのみが使用）。
- `evaluator_type`: `llm`, `rule_keyword`, `rule_regex` から選択。

## 実行方法 (Docker)

### 1. イメージのビルド

```bash
./build.sh
```

### 2. パイプラインの生成と実行

コンテナ内でパイプラインスクリプトを生成し、実行します。

```bash
docker run --rm -v $(pwd):/app llm_text_enhancer /bin/bash -c "python3 src/generate_pipeline.py && ./run_pipeline.sh"
```

## 実行方法 (ローカル)

```bash
# 依存関係のインストール
pip install -r requirements.txt

# パイプライン生成
python3 src/generate_pipeline.py

# 実験実行
./run_pipeline.sh
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
