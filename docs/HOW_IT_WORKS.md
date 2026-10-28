# How It Works: concepts behind every step

This document explains **what** each part of the project does, **how** it works, and **why**
we did it that way. Each section points to the file that implements it.

---

## 1. What is sarcasm, from an NLP point of view?

Sarcasm is saying the **opposite of what you mean**, usually to mock or criticise.
A useful working definition (Riloff et al., 2013): **positive sentiment + negative situation**.

> "I **love** being stuck in traffic for 2 hours" → *love* (positive) + *stuck in traffic* (negative)

Common cues a model can pick up:

| Cue | Example |
|---|---|
| Sentiment contrast | "Great, another exam on my birthday" |
| Hyperbole | "Best. Day. Ever." |
| Interjections | "Oh wow", "yeah right", "sure" |
| Punctuation / caps | "Thanks SO much!!!" |
| Emojis | 🙄 😒 🙃 |
| Elongation | "sooo helpful" |

Cues a text-only model **cannot** see: tone of voice, facial expression, shared knowledge
between speaker and listener. This is why sarcasm detection stays hard.

---

## 2. Preprocessing (`src/preprocess.py`)

Rule: **remove noise, keep signals.**

| Step | Example | Why |
|---|---|---|
| Unescape HTML | `&amp;` → `&` | Data scraped from web pages |
| URLs → `http` | `https://t.co/x1` → `http` | Every URL is unique noise, but *that there is a link* can matter |
| Users → `@user` | `@john_99` → `@user` | Names don't generalise; also privacy |
| Remove label hashtags | `#sarcasm`, `#not` deleted | **Label leakage**: the model would just learn "#sarcasm → sarcastic" |
| Hashtag → word | `#MondayMotivation` → `MondayMotivation` | Keeps the meaning |
| Emoji → text | 🙄 → `:face_with_rolling_eyes:` | Lets TF-IDF/LSTM treat emojis as words (transformer keeps raw emoji) |
| Shorten elongation | `soooooo` → `sooo` | Keeps the emphasis, avoids 50 versions of the same word |

**Same function for training and prediction.** If the app cleaned text differently from
training, the model would see unfamiliar input (*train/serve skew*).

---

## 3. Model 1: TF-IDF + linear classifiers (`src/train_baseline.py`)

### 3.1 Turning text into numbers: TF-IDF
Machine-learning models need numbers. **Bag of words**: each document becomes a vector with
one entry per vocabulary term. TF-IDF weights each term:

```
tfidf(t, d) = tf(t, d) × idf(t)
idf(t)      = log( (1 + N) / (1 + df(t)) ) + 1
```
- **tf**: how often term *t* appears in post *d* (we use `sublinear_tf`: 1 + log(tf), so
  repeating a word 10 times doesn't count 10×).
- **idf**: terms that appear in *every* post ("the", "a") get low weight; rare, distinctive
  terms get high weight.
- Each vector is then L2-normalised, so long and short posts are comparable.

### 3.2 Two kinds of features
- **Word 1–2-grams**: single words and pairs: `love`, `yeah right`, `oh great`. Bigrams capture
  short phrases that single words miss.
- **Character 2–5-grams** (`char_wb`): pieces like `!!!`, `soo`, `ove `, `:face`. They capture
  punctuation, elongation, emojis, and are **robust to typos** ("greaat" still shares n-grams
  with "great").

Result: a sparse matrix of a few hundred thousand columns (most entries are 0).

### 3.3 Classifiers compared
| Model | Idea | Notes |
|---|---|---|
| **Multinomial Naive Bayes** | Uses Bayes' rule, assuming words are independent given the class | Very fast, strong on small data; the independence assumption is wrong but often works |
| **Logistic Regression** | Learns a weight per feature; P(sarcastic) = sigmoid(w·x + b) | Outputs probabilities; weights are directly interpretable |
| **Linear SVM** | Finds the boundary with the largest margin between classes | Often the best on high-dimensional sparse text |

All three are **linear**, which works well because TF-IDF space is very high-dimensional.
`C` / `alpha` control regularisation (how strongly we stop the model memorising training data).
They are tuned on the **validation set**.

`class_weight="balanced"`: errors on the rarer class (sarcastic) cost more, so the model
does not just predict "not sarcastic" for everything.

### 3.4 Explainability
For a linear model, *contribution of a feature = its weight × its TF-IDF value*. The app shows
which words pushed a post towards "sarcastic". `baseline_top_features.png` shows the strongest
features overall.

### 3.5 Limitation
Bag of words **ignores word order and meaning**: "not good, great" and "good, not great" look
almost the same. It cannot understand the *contrast* between sentiment and situation, only
which words appear.

---

## 4. Model 2: BiLSTM + Attention (`src/lstm.py`, `src/train_lstm.py`)

### 4.1 Word embeddings
Each token gets a **dense vector** (200 numbers) instead of a sparse one-hot column. Vectors are
learned so that similar words end up close together (*great*, *awesome*, *fantastic*).
Optionally initialised from **GloVe-Twitter** (pre-trained on 2B tweets).

### 4.2 RNN → LSTM
An RNN reads the post word by word and keeps a **hidden state** (its memory). Plain RNNs forget
long-range information because of **vanishing gradients**. An **LSTM** adds a *cell state* and
three **gates**:
- **forget gate**: what to erase from memory
- **input gate**: what new information to store
- **output gate**: what to expose as the current hidden state

This lets it remember that "love" appeared earlier when it later reads "stuck in traffic".

### 4.3 Bidirectional
Two LSTMs: one reads left→right, one right→left. Their outputs are concatenated, so every
word's representation includes **both** its left and right context.

### 4.4 Attention pooling
Instead of using only the last hidden state, we score every word, apply softmax to get weights
that sum to 1, and take the weighted sum. The model **learns which words matter**.
Bonus: the weights are an explanation: the app highlights the words the model focused on.

### 4.5 Training details
| Technique | Why |
|---|---|
| Padding + `pack_padded_sequence` | Posts have different lengths; packing makes the LSTM skip padding |
| Cross-entropy loss with class weights | Handles imbalance |
| Adam optimiser (lr 1e-3) | Adaptive learning rate per parameter; standard default |
| Dropout 0.4 | Randomly zeroes units during training → prevents overfitting |
| Gradient clipping (1.0) | Prevents exploding gradients, a known RNN problem |
| Early stopping on val macro-F1 | Stops when the model starts overfitting; keeps the best epoch |

### 4.6 Limitation
Learns language **only from our training data**: a few thousand tweets is not enough to learn
English well. This is why it may not beat the baseline on small datasets, which is an
interesting result worth presenting.

---

## 5. Model 3: Fine-tuned Transformer (`src/train_transformer.py`)

### 5.1 Transformer and self-attention
Transformers (Vaswani et al., 2017) drop recurrence. In **self-attention**, every token looks
at **every other token at once** and decides how much each one matters to its own meaning:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```
- Q (query), K (key), V (value) are learned projections of each token.
- Multiple **heads** learn different relations (syntax, sentiment, coreference…).
- Stacking 12 layers builds deep **contextual** representations: "great" gets a different vector in
  "great job!" vs "great, it's broken again".

### 5.2 Pre-training → fine-tuning (transfer learning)
1. **Pre-training** (done for us): RoBERTa learned English by **masked language modelling**,
   predicting hidden words on huge text corpora. `twitter-roberta-base` continued this on
   about 58M tweets.
2. **Fine-tuning** (what we do): add a small classification head on top and train the
   **whole network** for a few epochs on our labelled data with a small learning rate (2e-5).

Why it works: the model already understands language, sentiment and Twitter style, so it needs
only a few thousand examples to learn the sarcasm task. This is the main reason it wins on small
datasets.

### 5.3 Tokenisation: Byte-Pair Encoding (BPE)
RoBERTa splits text into **sub-words**: "unbelievable" → `un`, `believ`, `able`. There are no
unknown words (anything can be built from pieces), and emojis and slang are handled too.
Special tokens: `<s>` (start, its vector is used for classification) and `</s>` (separator).

### 5.4 Context-aware input (Reddit)
With `--use-context`, input = `<s> reply </s></s> parent comment </s>`. Self-attention lets
words in the reply attend to the parent comment, so "Yeah, great idea" can be judged against what
it replies to.

### 5.5 Training settings
| Setting | Value | Why |
|---|---|---|
| Learning rate | 2e-5 | Small: adjust pre-trained knowledge without destroying it (*catastrophic forgetting*) |
| Warm-up | 10% of steps | Large updates at the start can destabilise a pre-trained model |
| Weight decay | 0.01 | Regularisation |
| Epochs | up to 4 + early stopping | Transformers overfit small data quickly |
| Max length | 128 tokens | Covers almost all tweets/comments; attention cost grows with length² |
| Mixed precision (fp16) | on GPU | About 2× faster, half the memory |
| Weighted loss | class weights | Imbalance |
| Best checkpoint | by val macro-F1 | The test set is never used for choices |

---

## 6. Handling class imbalance

iSarcasm has about 25% sarcastic tweets in training and about 14% in test. A model that always says
"not sarcastic" gets about 86% test accuracy and is **useless**. We handle this with:
1. **Stratified splits**: the same ratio in every split.
2. **Class-weighted loss**: weight_c = N / (2 × count_c), so each class contributes equally.
3. **Metrics that expose it**: F1 on the sarcastic class, macro-F1.

---

## 7. Evaluation metrics (`src/metrics.py`)

Confusion matrix (sarcastic = positive class):

|                 | Predicted not | Predicted sarcastic |
|---|---|---|
| **Actually not**       | TN | FP |
| **Actually sarcastic** | FN | TP |

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | (TP+TN) / all | How often is it right? *Misleading when imbalanced* |
| Precision | TP / (TP+FP) | When it says "sarcastic", how often is it right? |
| Recall | TP / (TP+FN) | Of all sarcastic posts, how many did it find? |
| F1 | 2·P·R / (P+R) | Balance of precision and recall (harmonic mean punishes a low value) |
| Macro-F1 | mean of F1 for both classes | Treats both classes equally: **our main metric** |
| ROC-AUC | area under ROC curve | How well scores rank sarcastic above non-sarcastic, at any threshold |

**Why macro-F1:** it's the official SemEval metric family for this task, and it cannot be gamed by
predicting the majority class.

**Precision vs recall trade-off:** the 0.5 threshold can be moved. A moderation tool might
prefer high recall (catch everything); a sentiment-analysis pipeline might prefer high precision.

**Error analysis:** `reports/<dataset>/<model>_errors.csv` lists the mistakes the model was
*most confident* about. Reading them shows *why* models fail (missing context, world knowledge,
label noise), and this is more useful than the score alone.

---

## 8. Train / validation / test: why three splits?
- **Train**: the model learns its weights here.
- **Validation**: we choose hyper-parameters, the best classifier and the best epoch here.
- **Test**: used **once** at the end. Because no decision was based on it, the score is an
  honest estimate of performance on new data.
If we tuned on the test set, we would be overfitting to it, and the reported number would be
optimistic.

---

## 9. Token-Level Saliency Heatmap & Pragmatic KPIs

### 9.1 Saliency & Attention Synthesis
To understand *why* an expression is flagged as sarcastic, IronyLens extracts token-level attribution by synthesizing three signals:
1. **BiLSTM Self-Attention Pooling:** The attention weights $\alpha_t \in [0, 1]$ directly reflect the recurrent network's sequential focus when building the post summary vector.
2. **Linear SVM Feature Impact:** Feature weights $\beta_i \cdot \text{TF-IDF}(i)$ quantify whether a token or n-gram pushed the decision toward the sarcastic or literal class.
3. **Lexical Polarity Categorization:** Tokens are categorized into functional linguistic roles:
   - **Positive Valence (Inverted):** Overt positive words contradicted by context (*great*, *love*, *genius*, *brilliant*).
   - **Situational Friction Targets:** The frustrating topic entity creating pragmatic dissonance (*meeting*, *email*, *delay*, *traffic*).
   - **Modal Intensifiers:** Amplifiers and sarcastic boosters (*definitely*, *totally*, *obviously*, *truly*).
   - **Discourse Markers:** Irony signposts (*oh*, *another*, *sure*, *thanks*).
   - **Syntactic Carriers:** Grammatical stop words (*the*, *a*, *that*, *been*, *could*) which are dampened to background baseline ($0.05 - 0.18$).

### 9.2 Lossless Pragmatic KPIs
- **Valence Inversion Index ($0 - 10$):** Measures the semantic gap between literal positive vocabulary and negative contextual reality.
- **Pragmatic Gap ($\Delta$):** Quantifies semantic incongruity between overt semantics and inferred speaker intent.
- **Hyperbole Index ($0 - 10$):** Detects stylistic intensity using capitalization runs, punctuation clusters (`!`, `?!`), and character elongations (`soooo`).
- **Consensus Engine:** Simultaneously queries Twitter-RoBERTa (P1), BiLSTM Attention (P2), and Linear SVM (P3) to display model alignment.

---

## 10. Frequently asked questions

**Q: Why not just use a sentiment analyser?**
Sarcastic posts usually *look* positive. A sentiment model labels "I love Mondays 🙄" as
positive. Sarcasm detection is a separate task and could be used *before* sentiment analysis
to correct it.

**Q: Why three models instead of only the best one?**
To show *why* the best one is best. The comparison isolates the value of (a) word order and
learned representations (LSTM) and (b) large-scale pre-training (transformer). The baseline also
tells us whether the complex models are worth their cost.

**Q: Why did you not remove stop words?**
Words like *not, so, just, oh, really, sure* are sarcasm signals. Removing them hurts. Our own
top-feature plot confirms it: *"it when"*, *"love when"*, *"you so"* and *"oh"* are among the
strongest sarcasm features, and stop-word removal would have destroyed most of them.

**Q: Why character n-grams?**
Social media is full of typos, elongations ("sooo"), emojis and punctuation runs ("!!!").
Character n-grams capture these and are robust to spelling variation.

**Q: Why Logistic Regression and not deep learning only?**
On small datasets, linear models on TF-IDF are hard to beat, train in seconds, and are fully
interpretable. They're the standard baseline in text classification research.

**Q: How do you know the model isn't just memorising?**
Separate validation set for every decision, early stopping, regularisation (C, dropout,
weight decay), deduplication across splits, and a test set used only once.

**Q: What is the difference between the BiLSTM's attention and the transformer's self-attention?**
BiLSTM attention is a single *pooling* step: one score per word to build a summary vector.
Transformer self-attention is applied in every layer, between **all pairs** of words, with
many heads. It builds the word representations themselves.

**Q: Why RoBERTa and not BERT?**
RoBERTa uses the BERT architecture but was trained longer on 10× more data, with dynamic masking and without
the next-sentence task, and it performs better on most benchmarks. The Twitter version also knows social-media
language. BERT-uncased also lowercases text, which loses the "CAPS" signal.

**Q: Why is the score on Twitter lower than on headlines?**
Headlines are written by professionals for a sarcastic *publication*, so the style is consistent
and labels are clean. Tweets are short, noisy, and the sarcasm is often subtle or needs
context. Even top SemEval-2022 systems reached only about 0.6 F1 on the sarcastic class.

**Q: What does a probability of 0.55 mean?**
The model is uncertain. Scores near 0.5 are borderline cases; the app shows the probability so
users can see confidence, not just the label.

**Q: Could you use ChatGPT/LLMs instead?**
Yes, as a zero-shot comparison (listed as future work). Trade-offs: LLMs need no training data
but are slower, costly per call, change between versions and are harder to evaluate reproducibly. A
fine-tuned 125M-parameter model runs on a laptop in milliseconds.

**Q: What are the ethical concerns?**
Misclassifying sarcasm in moderation could wrongly flag users. Datasets reflect the language of
specific communities (mostly English, US-centric Reddit/Twitter). We anonymise user names
during preprocessing.

**Q: How would you improve it?**
More context (conversation threads, author history), multimodal input (memes), ensembles of the
three models, larger transformers, data augmentation, and LLM-generated explanations.
