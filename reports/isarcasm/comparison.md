| model                            |   accuracy |   precision_sarcastic |   recall_sarcastic |   f1_sarcastic |   f1_macro |   roc_auc |
|:---------------------------------|-----------:|----------------------:|-------------------:|---------------:|-----------:|----------:|
| bilstm_attention                 |     0.7159 |                0.1797 |              0.275 |         0.2174 |     0.5219 |    0.5162 |
| baseline_tfidf                   |     0.7138 |                0.2149 |              0.375 |         0.2732 |     0.5475 |    0.6023 |
| transformer_twitter-roberta-base |     0.7618 |                0.3125 |              0.55  |         0.3986 |     0.625  |    0.7147 |