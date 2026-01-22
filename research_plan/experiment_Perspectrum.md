# perspectrumデータセットでの評価  
  
## 背景・目的  
現状のredditデータセットでの評価は、模範的な要約をどれだけ要約できたかに終始しており、  
特定の嗜好に沿った出力をできているかを評価するには不十分だった。  

prismデータセットに基づく理由生成は、背景ごとに理由が異なることを期待したが、
「返信が途中で切れている」「もう少しcontroversialなほうがいい」などの、モデルの精度起因のノイズが多かった。

そこで、討論サイトから収集されたPerspectrumを用いて、より品質の高いデータセットで「嗜好に合わせた出力ができるか」を評価する。

## 手法  
Perspectrumデータセットを用いて、ユーザーの評価とその理由を忠実に生成できるかを確認する。

```
{
  "claim_id": 101,
  "claim_text": "Is social media beneficial for society?",
  "perspectives": [
    {
      "perspective_id": "p1",
      "text": "Social media facilitates global communication.",
      "stance": "SUPPORT",
      "evidence_ids": [501, 502]
    },
    {
      "perspective_id": "p2",
      "text": "It contributes to the spread of misinformation.",
      "stance": "OPPOSE",
      "evidence_ids": [605]
    }
  ]
}
```  
  
このデータセットのうち、`claim_text`の問いから、いずれかの`perspectives`の`stance`と`text`を生成するタスクを行う。  
生成時は`claim_text`しか確認することができず、評価機の評価をみて、ユーザーの嗜好を判定することが必要になる。
評価はmeteorを用いて行う。  
  
例えばこの例であれば、  
conversation, feedback
"Is social media beneficial for society?", "SUPPORT: because Social media facilitates global communication."

必ず、SUPPORT / OPPOSEを答えた後に、理由を答えるようにプロンプティングしてください。

## 実装時の注意  
redditと同様に、全部のデータを実験するのは無理なので、サンプリングを行ってください。  
`config/data_generation_config.yaml`でサンプリング数を設定できるようにして。  
質問内容が偏らないように、ランダムサンプリングをしてください。

データセット名は`perspectrum`としてください。

## 参照
Perspectrumのgit
https://github.com/CogComp/perspectrum/tree/master/data/dataset

research_plan/experiment_Perspectrum.md
に記したような実験を実現できるような機能を実装してください。

まず日本語で計画書を作成し、私のレビューを受けてください。