## 1. 概要
遺伝的アルゴリズム (Genetic Algorithm) 等の進化戦略を用いて、テキストの品質を反復的に向上させるための実験用リポジトリ。
参考: [DLN (Deep RFL)](https://arxiv.org/pdf/2211.01910)
従来の「プロンプト最適化 (Prompt Optimization)」から進化し、**「テキスト直接最適化 (Direct Text Optimization)」** を採用しています。生成されたテキストそのものを評価し、直接書き換えることで品質を向上させます。

## 2. アーキテクチャ概要
**特徴**: プロンプトを探索するのではなく、**生成されたテキストそのものを直接修正・進化させる (Direct Text Optimization)** アプローチを採用。

### 構成要素
1.  **Generator (LLM)**: テキスト生成を行うエージェント。TAMLファイルで定義されたプロンプトを使用。
2.  **Evaluator**: 生成されたテキストを評価し、スコア(0-10)と理由を出力する。LLM評価のプロンプトもTAMLで管理。
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
- **設定管理**: `config/experiments.csv` で管理。
- **プロンプト管理 (TAML)**:
    - `config/definitions/prompts/[TaskName]/[Strategy].taml`: タスクごとにプロンプトをオーバーライド可能。
    - `config/definitions/prompts/[Strategy].taml`: デフォルトのプロンプト。
- **ディレクトリ構成**:
  - `src/`: ソースコード
  - `config/`: 実験設定、タスク定義、プロンプト定義(TAML)
  - `result/`:
    └── [実験設定ID]/   # Experiment ID
        └── [集団名]/   # Population Name (Run ID)
            └── [進化戦略]/   # Strategy (ga, textgrad, etc.)
                └── [評価器]/ # Evaluator (llm, rule_*)
                    ├── row_[N]/ # データセットの行ごと (Row)
                    │   ├── logs/
                    │   ├── score_summary.json
                    │   └── iter[N]/
                    │       ├── input_prompts/  # メタ情報
                    │       ├── logic/          # 中間ロジックログ
                    │       ├── texts/          # 生成テキスト
                    │       └── metrics.json    # 評価結果
                    └── input/ # 再現性確保用の入力ファイルコピー

## 3. ロジックとワークフロー
### 3.1. テキスト生成/進化 (`src/generate.py`)
- **Prompts**: 全ての進化戦略は `src/evolution_strategies.py` 内でハードコードされず、**TAMLファイル** から読み込まれます。タスク定義のファイル名に基づき、タスク固有のプロンプトフォルダ `config/definitions/prompts/[TaskName]/` を優先して参照します。

### 3.2. 評価 (`src/evaluate.py`)
- **LLM Judge**: TAML (`judge.taml`) で定義されたプロンプトを使用して評価を実行します。タスク固有の評価基準を適用可能です。
- **Metrics**: 各Iterで `metrics.json` を出力。
- **Score Summarization**: 各Rowの全Iterationのスコア（Max, Min, Mean）を集計し、`score_summary.json` を出力/更新する。

### 3.3. 結果集計 (`src/aggregate_results.py`)
- 全実験・全手法・全Rowの `score_summary.json` をクロールし、`result/aggregation_report.csv` に出力する。
- 統計量 (Avg_Max, Global_Max, Avg_Min, Global_Min) を算出。

## 4. 進化戦略 (Evolution Strategies)
`--evolution-method` で指定可能。

| 戦略名 | キー | 概要 |
| :--- | :--- | :--- |
| **Genetic Algorithm** | `ga` | **(Default)** テキストを直接変異（言い換え、拡張、短縮、トーン変更）させる。 |
| **TextGrad** | `textgrad` | 改善点（勾配）を言語化し、それに基づいてリライトする。 |
| **TextGrad V2** | `textgradv2` | 過去の履歴(Reference)を参照し、成功/失敗パターンを踏まえた勾配を生成する強化版。 |
| **Trajectory** | `trajectory` | 改善の軌跡（低スコア→高スコア）を提示し、さらに上位のバージョンを予測させる。 |
| **Demonstration** | `demonstration` | 過去の高評価テキストをFew-shot事例として提示し、同品質の新作テキストを生成させる。 |
| **Hypothesis & Exploration** | `he` | **(HE)** 成功法則の「仮説(Hypothesis)」を知識ベースとして蓄積・更新しながら探索する。 |
| **Exploit-Explore-Diversity** | `eed` | **(EED)** `Exploit`(既存勾配の深掘り)と`Diversity`(対照的な改善方向の探索)を明示的に分ける。 |
| **GA-TextGrad-Diversity** | `gatd` | **(GATD)** GA(交叉)、TextGrad(勾配)、Persona(多様性)を組み合わせたハイブリッド戦略。 |
| **Ensemble** | `ensemble` | 複数の戦略を並列実行し、結果を統合する。 |

## 5. TAML (Task/Target Augmented Markup Language)
プロンプトやタスク定義を記述するための独自フォーマット。
- `[section_name]`: セクション区切り。
- `src/utils.py` の `load_taml_sections` でパースされ、辞書形式で利用される。
- 変数展開 (`{variable}`) をサポート。

## 6. コーディング規約
- **コメント**: コード内のコメントは必ず日本語で書くこと。
- **アーキテクチャ**: 新しい進化戦略を追加する際は、`EvolutionStrategy` クラスを継承し、`src/evolution_strategies.py` に実装すること。併せて `config/definitions/prompts/` に対応するTAMLを作成すること。
