# prismデータセットでの評価  
  
## 背景・目的  
現状のredditデータセットでの評価は、模範的な要約をどれだけ要約できたかに終始しており、  
特定の嗜好に沿った出力をできているかを評価するには不十分  

もっと、バックグラウンドによって望ましい出力が大きく変化するようなドメインでの検証が望ましい。  

## 手法  
PRISMデータセットを用いて、ユーザーの評価理由を生成し、ユーザーが作成した評価結果に近づけることができるかに挑戦する。  

```
{
  "user_id": "user_75-US-liberal",
  "user_metadata": {
    "country": "USA",
    "political_leaning": "Liberal",
    "religion": "Atheist",
    "education": "Postgraduate"
  },
  "prompt": "Is abortion morally justifiable?",
  "responses": [
    {
      "model": "Model_A",
      "text": "It is a fundamental right of bodily autonomy...",
      "score_by_user": 5, // このユーザー（リベラル）には高評価
      "reason": "Aligned with my values on personal freedom"
    },
    {
      "model": "Model_B",
      "text": "Many believe life begins at conception...",
      "score_by_user": 2, // このユーザーには低評価
      "reason": "Too much focus on religious perspectives"
    }
  ]
}
```  
  
このデータセットのうち、promptとmodelのtextからanswerを生成するタスクを追加する。  
評価はmeteorを用いて行う。  
  
例えばこの例であれば、  
conversation, feedback
"user: Is abortion morally justifiable?¥nmodel:Many believe life begins at conception...", "Too much focus on religious perspectives"

## 実装時の注意  
redditと同様に、全部のデータを実験するのは無理なので、サンプリングを行ってください。  
