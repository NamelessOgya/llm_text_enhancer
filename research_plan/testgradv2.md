# textgradv2
## 背景  
現状のtext gradは更新候補のサンプルの入力情報とスコアしか参照できず、限界がある。  
https://arxiv.org/html/2506.00400v3
を参考して、過去の入力/スコア履歴を全て保持して、その中から過去の情報をサンプリングし、
補足情報として加えることで、精度改善を図る。  
  
## 変更点  
textgradの作成の際に、これまで、grad計算対象のテキストとスコアのみを入力としていたが、  
v2では、過去の入力/スコア履歴を全て保持して、その中から過去の情報をサンプリングし、
補足情報として加える。  

### 情報のサンプリング  
確率を
p = Softmax(score * `ref_sampling_weight`)  
としたGumbel-Top-kサンプリングを行い、`ref_sampling_num`数のサンプルを抽出する。  


## 実装方針  
- 変数`ref_sampling_weight`と`ref_sampling_num`を`config/logic/textgradv2.yaml`に追加する。  
- `textgrad`自体は残して、`textgradv2`という新しいロジックとして実装。    
- `eed`ロジックでtextgradを呼び出す際に、`textgradv2`を呼び出すように変更する。  
- 入力とスコアの関係は`result/perspectrum_v1.1/eed/perspectrum_rule/row_0/logic`配下に適切な形式のファイルを作って保存する。  

  
