# Sarcasm Detection in Social Media Text

Binary classification of tweets and Reddit comments as **sarcastic** or **not sarcastic**,
comparing three generations of NLP models on the same data:

1. **TF-IDF + linear classifiers** (Naive Bayes / Logistic Regression / Linear SVM): classical ML
2. **BiLSTM + Attention** (PyTorch): deep learning
3. **Fine-tuned `twitter-roberta-base`** (Hugging Face): transfer learning with transformers

There is also a **Streamlit demo** that shows each model's prediction and explains it.

## Documentation

| File | Contents |
|---|---|
| [docs/TECH_STACK.md](docs/TECH_STACK.md) | Every library/tool and why it was chosen over alternatives |
| [docs/DATASETS.md](docs/DATASETS.md) | Datasets, how they were labelled, how to get them, split decisions |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | Concepts behind every step + FAQ |

## Project structure

```
sarcasm-detection/
├── app.py                    Streamlit demo
├── run_all.py                whole pipeline in one command
├── train_on_colab.ipynb      GPU training on Google Colab
├── requirements.txt
├── src/
│   ├── config.py             paths, random seed
│   ├── data.py               download + clean schema + stratified splits (cached)
│   ├── preprocess.py         social-media text cleaning (shared by all models)
│   ├── eda.py                dataset statistics and figures
│   ├── train_baseline.py     Model 1: TF-IDF + NB / LR / SVM
│   ├── lstm.py               Model 2 architecture, vocabulary, helpers
│   ├── train_lstm.py         Model 2 training
│   ├── train_transformer.py  Model 3: fine-tune RoBERTa
│   ├── metrics.py            metrics, confusion matrix, error analysis
│   ├── compare.py            results table + chart across models
│   └── predict.py            load models and classify new text (CLI + used by app)
├── data/        (created)    raw downloads + processed splits
├── models/      (created)    trained models, one folder per dataset
└── reports/     (created)    metrics, figures, error lists, EDA
```

## Quick start

```bash
# 1. Environment (Python 3.10–3.12 recommended)
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
# PowerShell says "running scripts is disabled"? Either skip activation and use
# .venv\Scripts\python instead of python in every command below, or allow scripts for this window only:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
# NVIDIA GPU? Replace the CPU build of PyTorch with the CUDA one (transformer: ~1 min instead of ~1 h)
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu128

# 2. Everything in one command (data, EDA, baseline, BiLSTM, comparison)
python run_all.py --dataset isarcasm                 # add --transformer on a GPU

# --- or step by step ---
# Data + exploration
python -m src.data --dataset isarcasm
python -m src.eda  --dataset isarcasm

# Models
python -m src.train_baseline    --dataset isarcasm   # ~1 min on CPU
python -m src.train_lstm        --dataset isarcasm   # a few min on CPU
python -m src.train_transformer --dataset isarcasm   # use a GPU (Colab), ~5-10 min on a T4

# Results + demo
python -m src.compare --dataset isarcasm
python -m src.predict --dataset isarcasm "Oh great, another Monday. Just what I needed."
streamlit run app.py
```

Datasets: `isarcasm` (Twitter, default, auto-download), `reddit` (Kaggle download, see
[docs/DATASETS.md](docs/DATASETS.md)), `headlines` (auto-download).

### Training the transformer on Google Colab
Upload the `sarcasm-detection` folder to Google Drive, open `train_on_colab.ipynb` in Colab,
select *Runtime → Change runtime type → T4 GPU* and run all cells. It trains all three models on
every dataset and gives you `results.zip` (models + reports). Unzip it into the local project
and `streamlit run app.py` shows the transformer as well.

## Results (iSarcasmEval, official test set)

| Model | Accuracy | Precision (sarc.) | Recall (sarc.) | F1 (sarc.) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + Linear SVM (best of NB/LR/SVM on validation) | 0.714 | 0.215 | 0.375 | 0.273 | 0.548 | 0.602 |
| BiLSTM + Attention | 0.657 | 0.184 | 0.405 | 0.253 | 0.515 | 0.565 |
| **Fine-tuned twitter-roberta-base** | **0.748** | **0.309** | **0.615** | **0.411** | **0.625** | **0.730** |

The test set has 1,394 tweets, of which 200 are sarcastic.

**News headlines** (28.6k headlines; test set: 2,851 headlines, 48% sarcastic):

| Model | Accuracy | Precision (sarc.) | Recall (sarc.) | F1 (sarc.) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + Linear SVM | 0.875 | 0.859 | 0.881 | 0.870 | 0.875 | 0.944 |
| BiLSTM + Attention | 0.870 | 0.857 | 0.872 | 0.865 | 0.870 | 0.942 |
| **Fine-tuned twitter-roberta-base** | **0.938** | **0.960** | **0.908** | **0.933** | **0.938** | **0.986** |

The same models score 0.87–0.94 macro-F1 on clean, professionally written headlines but only
0.52–0.63 on real tweets. This shows how much harder noisy social media text is.

Training time on an RTX 4050 laptop GPU: transformer on tweets about 1 min, on headlines about 8 min.
On a CPU the tweet model takes about an hour. `python -m src.compare` regenerates this
table in `reports/isarcasm/comparison.md`.

### What the results show
- **Accuracy is misleading here.** A model that always answers "not sarcastic" scores 85.7%
  accuracy and 0 F1 on sarcasm. That is why we report F1 and macro-F1.
- **The BiLSTM does not beat TF-IDF on 3k tweets.** Its training loss keeps falling while
  validation F1 stays flat: it overfits because it has to learn English from scratch on a tiny
  dataset. This motivates transfer learning.
- **What the baseline learned** (`reports/isarcasm/baseline_top_features.png`): the strongest
  sarcasm features are phrases like *"love when"*, *"it when"* ("I love it when…"), *"oh"*,
  *"you so"*, *"thanks @user"*. These are textbook sarcasm markers, and several of them contain stop words.
- **A dataset artifact:** the strongest *non*-sarcastic feature is `http` (a link). In this
  dataset sarcastic tweets were written for the study and rarely contain links. A model can
  learn such shortcuts, which is why error analysis matters.
- **The transformer wins clearly:** sarcastic-class F1 rises from 0.27 to 0.41 (+50%) and macro-F1
  from 0.55 to 0.63. Pre-training on 58M tweets gives it the language knowledge that 3k tweets
  cannot. It finds 61.5% of sarcastic tweets, compared with 37.5% for the baseline.
- **The task is genuinely hard:** the best SemEval-2022 systems reached about 0.6 F1 on the
  sarcastic class of this same test set.

## Trained models are not in the repository

Datasets (`data/`) and trained models (`models/`) are not committed: the transformer alone is
about 500 MB, above GitHub's file limit. After cloning, rebuild them with one command (about 2 minutes on
an NVIDIA GPU, about 1 hour on CPU for the transformer, or use `train_on_colab.ipynb`):

```bash
python run_all.py --dataset isarcasm --transformer
python run_all.py --dataset headlines --transformer
```

The results in `reports/` are committed, so the tables and figures are available without retraining.

## Working on the project together

```bash
git clone https://github.com/aaryanve47-alt/sarcasm-detection.git
cd sarcasm-detection
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

git pull                               # always start from the latest version
git checkout -b my-change              # work on your own branch
# ... edit files ...
git add .
git commit -m "Describe what you changed"
git push -u origin my-change           # then open a Pull Request on GitHub
```

Collaborators added in *Settings → Collaborators* can push directly. Anyone else can **fork** the
repository and send a Pull Request.

## References
- Riloff et al. (2013). *Sarcasm as Contrast between a Positive Sentiment and Negative Situation.* EMNLP.
- Khodak, Saunshi & Vodrahalli (2018). *A Large Self-Annotated Corpus for Sarcasm.* LREC.
- Abu Farha et al. (2022). *SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic.*
- Misra & Arora (2023). *Sarcasm Detection using News Headlines Dataset.* AI Open.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.*
- Vaswani et al. (2017). *Attention Is All You Need.*
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- Barbieri et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.*
