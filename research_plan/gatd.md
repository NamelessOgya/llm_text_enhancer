# eed v2 (Stabilized)
`ga`と`textgrad`を組み合わせた遺伝的アルゴリズム。

## 登場変数 (追加・変更)
`config/logic/eed.yaml` にて変更可能。

- `elite_sample_num`:  (デフォルト: 2)。次世代にそのまま引き継ぐ上位個体の数。
- `gd_sample_num`:  (デフォルト: 4)。遺伝的アルゴリズムで生成する数。  
- `grad_sample_num`:  (デフォルト: 2)。`textgrad`で生成する数。  
- `mutation_sample_num`:  (デフォルト: 2)。遺伝的アルゴリズムで生成する数。  



## ロジック

### 集団編成  
それぞれの数は上記変数準拠  
- elite: 前世代の精度最大個体を引き継ぎ  
- text_grad: `textgrad`で生成されたサンプル
- gd: 遺伝的アルゴリズムで生成されたサンプル
- mutation: 遺伝的アルゴリズムで生成されたサンプル

### 更新アルゴリズム  
`選抜`: サンプリングはランク選択で行う。  
ランクが高いほど選ばれる確率が高い。

1. 最も精度の良い`elite_sample_num``個体`をエリート枠としてcopy  
2. 交雑: `gd_sample_num`個体を`選抜`して、`ga`で定義された交雑を行う。 
3. textgrad: `grad_sample_num`個体を`選抜`して、`textgrad`で改善を行う（プロンプトやハイパラはtextgradを同じ）
4. mutation: ペルソナベースのmutationを行う。まず`タスク`を入力として`ペルソナ`を生成し、ペルソナ目線で出力を生成する。
