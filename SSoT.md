## 1. 概要
遺伝的アルゴリズム (Genetic Algorithm) を用いて、テキストの品質を反復的に向上させるための実験用リポジトリを作成する。
参考: [DLN (Deep RFL)](https://arxiv.org/pdf/2211.01910) - ※本プロジェクトはプロンプト最適化ではなく、**テキストそのものの品質向上**（または評価者への適合度向上）を目的とする。

## 2. システム構成
### 2.1. 動作環境
- **Docker**: コンテナ内で動作。ビルドスクリプトを提供 (Python 3.11-slim)。
- **LLM**: 外部API (OpenAI, Google Gemini等) を利用。
  - **Gemini**: `google-genai` SDKを使用 (推奨される最新版)。
  - 将来的な差し替えを考慮し、LLM利用部分は抽象化層(Interface/Adapter)を設ける。

### 2.2. 設定管理と実行パイプライン
- **設定管理**: 実験設定はCSVファイルで管理する。
  - CSVには実験ID、最大世代数、個体数 `k`、モデル名(`gemini-2.0-flash-lite`等)、`adapter_type`、`task_definition`、`target_preference` 等を定義。
  - **Adapter Type**: 使用するLLMアダプターの種類を指定（例: `openai`, `gemini`, `dummy`）。これによりモデル名のみに依存しない明確な切り替えが可能。
  - **Task Definition (ドメイン定義)**: 生成タスクの一般的な指示（例: "実在するポケモンの説明を生成せよ"）。Generatorにはこれのみが与えられる。
  - **Target Preference (隠された嗜好)**: 評価の正解基準（例: "炎タイプのポケモン"）。Generatorには**一切公開されない**。Evaluatorのみが使用する。
  - **Evaluator Type**: 評価手法を指定する。
    - `llm`: LLMを用いた評価。
    - `rule_keyword`: キーワードの一致率による評価。
    - `rule_regex`: 正規表現マッチングによる評価。

- **パイプライン生成**: CSVファイルを読み込み、実験実行用のシェルスクリプトを生成するスクリプト (`src/generate_pipeline.py` 等) を作成する。
  - 生成されたパイプラインスクリプトを実行することで実験を行う。
- **冪等性 (Skip)**: 各実験の最終成果物が既に存在する場合、その実験ステップはスキップされる仕組みとする。

### 2.3. ディレクトリ構成
```
.
├── src/                # ソースコード
│   ├── generate_pipeline.py  # CSVから実行スクリプトを生成
│   ├── llm/
│   │   ├── openai_adapter.py # OpenAI API アダプター
│   │   └── dummy_adapter.py  # テスト用ダミーアダプター
│   ├── evaluation/       # 評価モジュール
│   │   ├── interface.py
│   │   ├── llm_evaluator.py
│   │   └── rule_evaluator.py
├── cmd/                # 実行用シェルスクリプト（.shのみ配置可能）
│   ├── generate_next_step.sh
│   ├── evaluate_step.sh
├── tests/              # テストコード
│   ├── test_pipeline.py
│   ├── test_core_logic.py
│   ├── test_integration_flow.py
│   └── test_gemini_integration.py # Gemini接続テスト（トークン消費あり）
├── config/
│   └── experiments.csv # 実験設定一覧
├── .env.example        # 環境変数テンプレート
├── .gitignore          # Git除外設定
└── result/             # 実験結果出力
    └── [実験設定ID]/   # 実行時の引数で指定
        ├── logs/       # ログ・コスト管理
        │   ├── execution.log
        │   └── token_usage.json
        └── iter[N]/    # 第N世代
            ├── input_prompts/  # テスト生成に使用したプロンプト(Input)
            │   ├── prompt_1.txt
            │   └── ...
            ├── logic/          # ロジック内で使用された中間出力物 (メタプロンプト等)
            │   ├── creation_prompt_1.txt
            │   └── ...
            ├── texts/          # 生成されたテキスト(Output)
            │   ├── text1
            │   └── ...
            └── metrics.json
```

## 3. ロジックとワークフロー
実験は「世代 (Generation)」単位で進行する。
- **終了条件**: ユーザーが設定したサイクル数（世代数）に達したら終了。
- **再開機能**: 実験が中断された場合でも、途中から再開可能とする（既存の世代がある場合はスキップして次から開始）。

### 3.1. テキスト生成 (`cmd/generate_next_step.sh`)
- **入力**: 
  - 履歴、評価 (`metrics.json`)、実験設定ID
- **処理**:
  - **初期化 (n=0)**: `task_definition` に基づき初期候補を生成（`target_preference` は使用しない）。
  - **更新 (n>0)**: 過去の評価に基づき、遺伝的アルゴリズム/LLMで次世代候補 `k` 個を生成。変異(Mutation)の際も `task_definition` のみを使用し、正解情報はリークさせない。
- **出力**:
  - `result/[setting]/iter[n+1]/texts/` (生成テキスト)
  - `result/[setting]/iter[n+1]/input_prompts/` (生成プロンプト)
  - `result/[setting]/iter[n+1]/logic/` (ロジック内で生成された中間出力物)

### 3.2. 評価 (`cmd/evaluate_step.sh`)
- **入力**: 第`n+1`世代テキスト
- **処理**: `target_preference` を正解として使用し、テキストとの適合度をスコアリング。
- **出力**: `result/[setting]/iter[n+1]/metrics.json`
- **Metricsスキーマ**: 各プロンプトサンプルに対応したスコアと**評価理由(reason)**をリスト形式で保持する。
  - `{"file": "...", "score": 8.0, "reason": "..."}`のように出力される。
  - ルールベース評価の場合、reasonは空文字となる。

### 3.3. ルールベース評価器の追加手順
新しいルールベース評価器を追加する場合は、以下の手順に従う。
1. `src/evaluation/rule_evaluator.py` に `Evaluator` を継承したクラスを作成し、`evaluate` メソッドを実装する。
2. 同ファイルの `get_rule_evaluator` ファクトリ関数に、新しい `rule_type` とクラスのマッピングを追加する。
3. `experiments.csv` の `evaluator_type` に新しいタイプを指定して実行する。

## 4. サンプルタスク
**タスク名**: ポケモン推測ゲーム (Pokemon Guessing Game)
- GeneratorがEvaluatorの隠された嗜好（例：炎タイプ好き）を、点数フィードバックのみから学習し、適合するテキストを生成することを目指す。

## 5. 非機能要件
### 5.1. エラーハンドリング
- **リトライ**: 外部API呼び出し等は **最大3回** までリトライを行う。
- **停止**: 3回失敗した場合はエラーとしてプロセスを停止する。

### 5.2. ログ・コスト管理
- **ログ**: 実行ログ（APIレスポンス等含む）を `result/[setting]/logs/` 配下に保存。
- **トークン数**: 消費したトークン数を記録し、同ディレクトリに保存可能な形式で出力する。

### 5.3. 環境変数管理
- **APIキー**: `.env` ファイルに `OPENAI_API_KEY=xxx`, `GEMINI_API_KEY=xxx` の形式で保存することで、スクリプト実行時に自動的に読み込まれる。
- `.env` ファイルはGit管理外とし、テンプレート `.env.example` を提供する。

## 6. テスト・検証機能
### 6.1. Dummy Adapter
- **目的**: APIコストをかけずにパイプラインの動作検証を行うため、`DummyAdapter` (`src/llm/dummy_adapter.py`) を提供する。
- **動作**: 
  - 生成時: ランダムなポケモン名を返す。
  - 生成時: ランダムなポケモン名を返す。
  - 評価時: ランダムなスコア (0-10) を返す。
- **使用方法**: `experiments.csv` の `adapter_type` に `dummy` を指定することで使用可能（`model_name` は無視されるが何らかの指定は必要）。

### 6.2. Gemini Integration Test
- **ファイル**: `tests/test_gemini_integration.py`
- **目的**: 実際のGemini APIへの接続と、Input/Outputの確認を行う。
- **動作**: 
  - `.env` に `GEMINI_API_KEY` が存在する場合のみ実行される。
  - テスト実行時にInput PromptとOutput Responseを標準出力に表示し、実際の挙動を目視確認可能にする。
- **実行**: `python -m unittest tests/test_gemini_integration.py`
