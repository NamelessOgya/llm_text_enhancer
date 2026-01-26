# eed v2 (Stabilized)

Base: [eed.md](eed.md)

最大スコアの不安定性を解消するための改良版。

## 登場変数 (追加・変更)
`config/logic/eed.yaml` にて変更可能。

- `elite_sample_num`: **追加** (デフォルト: 1)。次世代にそのまま引き継ぐ上位個体の数。
- `grad_temperature`: **変更** (デフォルト: 1.0)。最適化の安定性向上のため、1.2から1.0へ引き下げ。

## ロジック変更点

### 1. Elitism (エリート保存)の導入
- **課題**: 従来のEEDでは、Exploitフェーズで親を種として使用するものの、親そのものは保存しなかったため、最適化が失敗した場合にスコアが低下していた。
- **解決策**: 世代構築の最初に、前世代の上位 `elite_sample_num` 個の個体を、無変更で次世代リストに追加する。
- **構成調整**: エリート枠を確保するため、`exploit_sample_num` で生成される新規サンプル数を調整（減少）する。

### 2. Exploitフェーズのプロンプト厳格化
- **課題**: 従来は「独創的で大胆な指示 (Creative and bold instruction)」を求めており、不必要な分散や破壊的な変更を引き起こしていた。
- **解決策**: `TextGradStrategy` と完全に同一のプロンプトを採用。「厳密な批評と具体的な改善指示」を求める形式に変更。多様性はTemperatureとPrompt内のVarianceではなく、EED全体のDiversity/Exploreフェーズで担保する設計へシフト。

### 3. Temperatureの適正化
- **課題**: `grad_temperature: 1.2` は最適化タスクには高すぎた。
- **解決策**: デフォルトを `1.0` に変更。

### 4. Evaluator Reason (評価理由) の削除
- **課題**: 評価者が生成する `reason` (フィードバック) はデバッグ用であり、進化計算（最適化ループ）に混ぜるべきではない。
- **解決策**: プロンプトに注入されていた `Evaluator Logic/Reasoning` や `Feedback` の項目を削除。純粋にテキストとスコアのみから改善案を生成させる。

### 5. Score Context (スコア情報) の明記
- **課題**: Diversityフェーズなどでスコア情報が欠落しており、改善の基準が不明確。また、スコアのスケール (0-1) が明示されていない場合がある。
- **解決策**: 全ての TextGrad 系プロンプトに `Score: {score}` と `Note: The score is on a scale of 0.0 to 1.0.` を追加する。

## 期待される効果
- 最大スコア (`max_score`) が単調非減少（あるいはそれに近い安定推移）になる。
- TextGradと同等以上の到達点を目指しつつ、Diversity/Exploreフェーズによる局所解脱出能力を維持する。
