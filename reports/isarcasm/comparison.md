| model                            |   accuracy |   precision_sarcastic |   recall_sarcastic |   f1_sarcastic |   f1_macro |   roc_auc |
|:---------------------------------|-----------:|----------------------:|-------------------:|---------------:|-----------:|----------:|
| bilstm_attention                 |     0.6571 |                0.1841 |              0.405 |         0.2531 |     0.5153 |    0.5649 |
| baseline_tfidf                   |     0.7138 |                0.2149 |              0.375 |         0.2732 |     0.5475 |    0.6023 |
| transformer_twitter-roberta-base |     0.7475 |                0.309  |              0.615 |         0.4114 |     0.6253 |    0.7296 |