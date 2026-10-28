# Tech Stack: what we use and why

## Overview

| Layer | Tool | Role in the project |
|---|---|---|
| Language | **Python 3.10–3.12** | Everything |
| Data handling | **pandas**, **NumPy** | Loading CSV/JSON, splits, statistics |
| Text cleaning | **re** (regex), **emoji**, **html** | Social-media normalisation |
| Classical ML | **scikit-learn** | TF-IDF, Naive Bayes, Logistic Regression, SVM, metrics |
| Deep learning | **PyTorch** | BiLSTM + attention model |
| Transformers | **Hugging Face `transformers`, `datasets`, `accelerate`** | Pre-trained RoBERTa, tokenizer, fine-tuning loop |
| Visualisation | **matplotlib** | EDA figures, confusion matrices, comparison chart |
| Model saving | **joblib**, `torch.save`, `save_pretrained` | Reusing trained models in the app |
| Demo | **Streamlit** | Interactive web app |
| Compute | Laptop CPU + **Google Colab (free T4 GPU)** | CPU for baseline/LSTM, GPU for transformer |
| Versioning | **Git + GitHub** | Team collaboration, history |

## Why each choice (and what we rejected)

### Python
The standard language for NLP and ML: every library below is Python-first, and most
research code is released in Python.
*Rejected:* R (weaker deep-learning ecosystem), Java (verbose, fewer NLP libraries).

### pandas
Text datasets are tables (text, label, context). pandas reads CSV/JSON in one line and makes
filtering, deduplication and stratified sampling easy.

### Regex + `emoji` for cleaning (instead of NLTK/spaCy pipelines)
Social-media cleaning is mostly *pattern replacement* (URLs, @users, hashtags, elongations),
which regex does exactly. The `emoji` library converts 🙄 into `:face_with_rolling_eyes:` so
word-based models can use it.
We deliberately **do not** apply the "textbook" steps:
- **No stop-word removal:** words like *not*, *so*, *really*, *just*, *oh* are strong sarcasm cues
  ("Oh I *just* love Mondays").
- **No stemming/lemmatisation:** it removes tense and form, and does not help a transformer, which has
  its own sub-word tokenizer.
- **No lowercasing for the transformer:** "GREAT" in capitals signals tone.
- **No punctuation removal:** "!!!" and "?!" are signals.

### scikit-learn
Industry-standard for classical ML: `TfidfVectorizer`, linear models, `Pipeline`, and all
evaluation metrics in one consistent API. The `Pipeline` bundles cleaning, features and
classifier into **one saved object**, so prediction uses exactly the same preprocessing as training.

### PyTorch (for the BiLSTM)
Gives full control to write the model layer by layer (embedding → LSTM → attention), which
makes the architecture easy to explain. It is also the backend of Hugging Face, so one framework
covers both deep models.
*Rejected:* TensorFlow/Keras. It would work, but using PyTorch for both models keeps a single
framework, and PyTorch is the dominant choice in NLP research.

### Hugging Face Transformers
Gives access to thousands of pre-trained models with 3 lines of code, the matching tokenizer,
and a tested `Trainer` (mixed precision, evaluation each epoch, early stopping, best-checkpoint
selection).

### `cardiffnlp/twitter-roberta-base` (the pre-trained model)
- **RoBERTa** = BERT trained longer, on more data, with better settings → stronger than BERT-base.
- This version was **further pre-trained on about 58M tweets** (Barbieri et al., TweetEval, 2020),
  so it already knows Twitter language: slang, hashtags, emojis, @user.
- **Base size (about 125M parameters)** fine-tunes in minutes on a free Colab GPU.
- *Alternatives considered:*
  | Model | Why not default |
  |---|---|
  | `bert-base-uncased` | Trained on Wikipedia/books, lowercases text (loses CAPS signal) |
  | `vinai/bertweet-base` | Also a strong choice; needs special tweet normalisation. Can be swapped in via `--model-name` |
  | `distilroberta-base` | 2× faster, slightly less accurate; good fallback on CPU (`--model-name distilroberta-base`) |
  | `roberta-large` | Better but 3× bigger; often doesn't fit a free GPU |
  | GPT-style LLM prompting | No training needed, but slow, costly per request, hard to evaluate reproducibly |

### Streamlit
Turns a Python script into a web app with no HTML/JS. Ideal for a live demo: text box,
per-model probability bars, explanations, results tables.
*Rejected:* Flask + HTML (more code for the same result), Gradio (also fine; Streamlit gives
easier multi-tab layouts).

### Google Colab
Transformer fine-tuning on a CPU takes hours; on Colab's free T4 GPU it takes minutes. The
baseline and BiLSTM run fine on a laptop.

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

On Colab: upload/clone the project, then `!pip install -r requirements.txt`, and select
*Runtime → Change runtime type → T4 GPU*.
