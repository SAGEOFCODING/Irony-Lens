# Datasets: where they come from and why we chose them

## How sarcasm datasets are built

Labelling sarcasm by hand is slow and annotators disagree, so researchers use one of three
strategies. Knowing these explains the strengths and weaknesses of each dataset:

| Strategy | How | Pro | Con |
|---|---|---|---|
| **Self-labelling by hashtag / marker** | Collect tweets with `#sarcasm` or Reddit comments ending in `/s` | Huge, cheap | Noisy: people forget the marker, use it jokingly; **the marker leaks the label** |
| **Author labelling** | Ask the *writers* whether they meant it sarcastically | Most reliable: it is the *intended* sarcasm | Small, expensive |
| **Source-based** | Sarcastic source vs serious source (The Onion vs HuffPost) | Very clean, no ambiguity | Not real social media; model may learn *source style* |

## 1. iSarcasmEval (Twitter): primary dataset

- **Source:** SemEval-2022 Task 6, Abu Farha et al. 2022. GitHub: `iabufarha/iSarcasmEval`
- **Size:** about 3,468 training tweets (about 25% sarcastic) and an official test set of
  1,400 tweets (about 14% sarcastic).
- **Labels:** written by the tweet authors themselves: each participant gave one sarcastic
  tweet and explained its meaning (the `rephrase` column).
- **Why we chose it as the main benchmark:**
  1. It is **real social media** (Twitter), which matches our topic exactly.
  2. **Author labels** are the most reliable kind of sarcasm label.
  3. It has an **official test set**, so our numbers can be compared with published SemEval
     results (top systems reached about 0.6 F1 on the sarcastic class, which shows how hard the task is).
  4. It is **imbalanced**, like real life, which forces us to use proper metrics.
- **Loaded by:** `python -m src.data --dataset isarcasm` (downloads automatically). Validation
  = 10% of the training data (stratified); the official test set is never used for tuning.

## 2. Reddit SARC (balanced): large-scale dataset

- **Source:** Khodak, Saunshi & Vodrahalli 2018, *A Large Self-Annotated Corpus for Sarcasm*.
  Kaggle version: `danofer/sarcasm`, file `train-balanced-sarcasm.csv`.
- **Size:** about 1 million comments, 50% sarcastic. We sample 100k by default (`--sample-size`).
- **Labels:** Reddit users mark sarcasm with `/s`. The marker is removed from the text.
- **Why we use it:**
  1. **Scale:** enough data for deep models to learn.
  2. **Context:** each comment comes with its **parent comment**, so we can test whether
     context helps (`train_transformer --use-context`).
- **How to get it:** create a free Kaggle account → download the dataset → put
  `train-balanced-sarcasm.csv` in `data/raw/` → `python -m src.data --dataset reddit`.
  (Or: `pip install kaggle`, add your API token, `kaggle datasets download -d danofer/sarcasm`.)

## 3. News Headlines: reference dataset

- **Source:** Misra & Arora, *Sarcasm Detection using News Headlines Dataset*. GitHub:
  `rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection`.
- **Size:** 28,619 headlines, about 48% sarcastic (The Onion) vs non-sarcastic (HuffPost).
- **Why we include it:** professional writers, no typos and clean labels, so it shows how well
  the models do *when the data is easy*. Comparing this with Twitter shows how much harder
  real social media text is.
- **Caveat:** a model can partly learn "Onion writing style" rather than sarcasm itself.

## Other datasets we considered (and why not as primary)

| Dataset | Why not primary |
|---|---|
| SemEval-2018 Task 3 (irony, Twitter) | Labels *irony*, a broader category than sarcasm |
| Riloff et al. 2013 Twitter set | Tweets must be re-downloaded by ID; many are deleted |
| MUStARD (TV-show dialogues) | Multimodal video, not social media text |
| Scraping Twitter/X ourselves | API is paid/limited, and labels would be hashtag-based and noisy |

## Data handling decisions

| Decision | Why |
|---|---|
| **Stratified split (80/10/10)** | Every split keeps the same sarcastic/non-sarcastic ratio |
| **Separate validation set** | All tuning happens on validation, so the test score is an honest estimate |
| **Same split for all models** | Fair comparison, cached in `data/processed/` |
| **Drop duplicate texts** | The same text in train and test would inflate scores (*data leakage*) |
| **Remove #sarcasm / #not** | They give away the answer, and the model would learn the hashtag instead of sarcasm |
| **Fixed random seed (42)** | Reproducible results |
