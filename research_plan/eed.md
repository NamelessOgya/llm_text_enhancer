# eed(exploitation and exploration and diversity)  
適宜必要なファイルなどを追加してください。

## 登場変数  
以下の変数については、`config/logic/eed.yaml`から変更できるようにすること。
`exploit_sample_num`
`diversity_sample_num`
`explore_sample_num`
`grad_temperature`
`sample_pass_rate`


## 各世代のサンプル内容  
1世代はexploit(`exploit_sample_num`個), diversity(`diversity_sample_num`個), explore(`explore_sample_num`個)のサンプルからなる。  

### exploit  
前世代の上位3つを親にして、それぞれから2つのtext gradを作り、それぞれを適用して子を作る。(3×2=6)
text_gradの生成時にはtempreatureを高めにとり(`grad_temperature`)、生成多様性を担保する。
gradは共通gradと個別gradに分ける。  

### diversity  
exploitで利用しなかった7つのサンプルに対してtext gradを1つ作り、それを適用して子を作る。
この中から、以下のロジックで`diversity_sample_num`のサンプルを選ぶ。  

1. novelity(x, exploitサンプルの集合)が最大のものを選択する。
2. novelity(x, exploitサンプルの集合 + 1.で選んだサンプル)が最大のものを選択する。
...
..
.

### explore
前世代のスコア上位半分のサンプルにおいて、全てのサンプルで守られている**規範**をLLMを用いて言語化する。  
**規範**の前提下でテキストを`explore_sample_num`×`sample_pass_rate`個生成する。

その後、生成したサンプルの中から以下のロジックで`explore_sample_num`のサンプルを選ぶ。  
1. novelity(x, exploitサンプルの集合+diversityサンプルの集合)が最大のものを選択する。
2. novelity(x, exploitサンプルの集合+diversityサンプルの集合 + 1.で選んだサンプル)が最大のものを選択する。
...
..
.


novelity(x, K) = 1 - max_{k ¥in K}(similarity(x, k))
similarity(p, q): 3-gram-cosine-similarity
