## 1. 概要
遺伝的アルゴリズム (Genetic Algorithm) 等の進化戦略を用いて、テキストの品質を反復的に向上させるための実験用リポジトリ。
参考: [DLN (Deep RFL)](https://arxiv.org/pdf/2211.01910)
従来の「プロンプト最適化 (Prompt Optimization)」から進化し、**「テキスト直接最適化 (Direct Text Optimization)」** を採用しています。生成されたテキストそのものを評価し、直接書き換えることで品質を向上させます。

## 2. アーキテクチャ概要
**特徴**: プロンプトを探索するのではなく、**生成されたテキストそのものを直接修正・進化させる (Direct Text Optimization)** アプローチを採用。

### 構成要素
1.  **Generator (LLM)**: テキスト生成を行うエージェント。
2.  **Evaluator**: 生成されたテキストを評価し、スコア(0-10)と理由を出力する。
3.  **Evolution Strategy**: 評価結果に基づき、テキストを「進化」させるロジック。
4.  **Pipeline**: 上記を連携させ、反復的(Iterative)に品質を向上させるフロー。

### データフロー
1.  **Iter 0**: タスク定義から初期テキスト個体群を生成。
2.  **Evaluation**: テキストを評価。
3.  **Evolution**: 高評価なテキストを親として、変異・改善を加え次世代のテキストを生成 (Evolution = Generation)。
4.  Iter Nまで繰り返し。

### 2.1. 動作環境
- **Docker**: コンテナ内で動作。ビルドスクリプトを提供 (Python 3.11-slim)。
- **LLM**: 外部API (OpenAI, Google Gemini等) を利用。

### 2.2. 設定管理と実行パイプライン
- **設定管理**: `experiments.csv` で管理。
- **ディレクトリ構成**:
  - `src/`: ソースコード
  - `analysis/`: スポット分析・検証用のスクリプトや実験ノートブック
  - `result/`:
    └── [実験設定ID]/   # 実行時の引数で指定
        ├── [進化戦略]/   # 進化戦略 (ga, textgrad, etc.) ごとに分離
        │   ├── logs/       # ログ・コスト管理
        │   │   ├── execution.log
        │   │   └── token_usage.json
        │   └── iter[N]/    # 第N世代
        │       ├── input_prompts/  # テスト生成に使用したメタ情報
        │       │   ├── prompt_1.txt
        │       │   └── ...
        │       ├── logic/          # ロジック内で使用された中間出力物
        │       │   ├── creation_prompt_1.txt
        │       │   └── ...
        │       ├── texts/          # 生成されたテキスト(Output)
        │       │   ├── text1
        │       │   └── ...
        │       ├── metrics.json # 評価結果
        │       └── input_data.json # 生成の元となった入力データ (トレーサビリティ用)
        │   └── row_N/score_summary.json # そのRowの各Iterのスコア集計結果 (Max/Min)

## 3. ロジックとワークフロー
### 3.1. テキスト生成/進化 (`src/generate.py`)
- **Iter 0**: タスク定義に基づき、多様な初期テキストを生成 (Promptなし)。
- **Iter N**: 
  - 過去のテキストと評価スコアを読み込む。
  - 指定された **Evolution Strategy** を実行。
  - ストラテジーが「高評価テキストの変異」「改善指摘に基づくリライト」などを行い、**次世代のテキストを直接出力**する。

### 3.2. 評価 (`src/evaluate.py`)
- テキストを評価し、`metrics.json` を出力。

### 3.2. 評価 (`src/evaluate.py`)
- テキストを評価し、`metrics.json` を出力。
- **Score Summarization**: 各Rowの全Iterationのスコア（Max, Min, Mean）を集計し、`score_summary.json` を出力/更新する。

### 3.3. 結果集計 (`src/aggregate_results.py`)
- プロジェクト全体の結果を集約する。
- 全実験・全手法・全Rowの `score_summary.json` をクロールし、`result/aggregation_report.csv` に出力する。
- 以下の統計量を算出:
  - Avg_Max_Score (平均最大スコア)
  - Global_Max_Score (全体最大スコア)
  - Avg_Min_Score (平均最小スコア)
  - Global_Min_Score (全体最小スコア)

## 4. 進化戦略 (Evolution Strategies)
`--evolution-method` で指定可能。全ての戦略は「テキスト」を入出力とします。

| 戦略名 | キー | 概要 |
| :--- | :--- | :--- |
| **Genetic Algorithm** | `ga` | **(Default)** テキストを直接変異（言い換え、拡張、短縮、トーン変更）させる。エリート保存あり。 |
| **TextGrad** | `textgrad` | テキストに対する改善点（勾配）を言語化し、それに基づいてテキストをリライトする。 |
| **Trajectory** | `trajectory` | テキストの改善履歴（バージョン履歴）を提示し、次の改善版テキストを予測させる。 |
| **Demonstration** | `demonstration` | 過去の高評価テキストをFew-shot事例として提示し、同品質の新作テキストを生成させる。 |
| **Ensemble** | `ensemble` | 複数の戦略を並列実行し、結果を統合する。 |

## 5. コーディング規約
- **コメント**: コード内のコメントは必ず日本語で書くこと。
- **アーキテクチャ**: 新しい進化戦略を追加する際は、`EvolutionStrategy` クラスを継承し、`src/evolution_strategies.py` に実装すること。
