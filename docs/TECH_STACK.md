# Tech Stack: Architectural Choices & Justifications

## 1. System Overview

| Layer | Technology | Primary Role in IronyLens |
|---|---|---|
| **Runtime Language** | **Python 3.10 – 3.13** | Core language across pipelines, neural networks, inference servers, and tooling. |
| **Data Engineering** | **pandas**, **NumPy** | High-performance dataframe ingestion, schema standardization, and stratified train/val/test splits. |
| **Text Normalization** | **re** (regex), **emoji**, **html** | Social media artifact cleaning, emoji tokenization, and stylistic cue preservation. |
| **Classical ML** | **scikit-learn** | High-dimensional TF-IDF feature extraction (word & char n-grams), Linear SVM, Logistic Regression, Naive Bayes, hyperparameter optimization. |
| **Deep Learning** | **PyTorch (CUDA 12.x)** | Custom Bidirectional LSTM with Self-Attention pooling layer for token saliency explainability. |
| **Transformer Backbone** | **Hugging Face (`transformers`, `accelerate`)** | Transfer learning via pre-trained `cardiffnlp/twitter-roberta-base` fine-tuned for contextual irony resolution. |
| **Evaluation & Visuals** | **matplotlib**, **seaborn** | High-resolution publication plots: ROC/PR curves, Radar capability spiders, Latency vs F1 frontiers, hyperparameter sweeps, confusion matrices. |
| **Production Inference** | **Flask + Tailwind CSS** | **IronyLens Telemetry Workbench (`server.py`)**: Sub-50ms RTX GPU inference, live token attention heatmap, saliency inspector, dynamic threshold tuning, REST API endpoints. |
| **Research Dashboard** | **Streamlit** | Multi-tab exploratory benchmarking dashboard (`app.py`). |
| **Hardware & Compute** | **Local NVIDIA RTX 4060 GPU / Google Colab T4** | Hardware-accelerated tensor operations, mixed-precision evaluation, and sub-millisecond CPU fallbacks. |
| **Version Control** | **Git & GitHub** | Collaborative code review, experiment tracking, and artifact persistence. |

---

## 2. Model Priority Hierarchy & Production Roles

In production, models are arranged in an empirical priority cascade based on accuracy, interpretability, and latency trade-offs:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INPUT TEXT / POST                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 1: Twitter-RoBERTa Transformer (125M Parameters)             │
│  • Primary Deep Classification Engine                                  │
│  • Contextual self-attention captures cross-clause pragmatic paradoxes │
│  • Macro-F1: 0.6250 (Twitter) / 0.9376 (Headlines) • Latency: ~40ms    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 2: BiLSTM + Self-Attention Pooling (FastText Embeddings)     │
│  • Saliency & Interpretability Engine                                  │
│  • Exposes exact recurrent token weights for attention heatmaps        │
│  • Macro-F1: 0.5219 (Twitter) / 0.8700 (Headlines) • Latency: ~18ms    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 3: TF-IDF + Linear SVM (< 1.4MB Footprint)                   │
│  • High-Throughput CPU Fallback Engine                                 │
│  • Pure CPU execution for failover when GPU capacity is saturated      │
│  • Macro-F1: 0.5475 (Twitter) / 0.8746 (Headlines) • Latency: < 1ms    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Justifications for Key Design Decisions

### Python 3.10–3.13 Ecosystem
Python remains the undisputed foundation for modern NLP and deep learning. All production libraries (PyTorch 2.x, Hugging Face Transformers, scikit-learn) release native C++/CUDA bindings first for Python, ensuring seamless interoperability and zero-overhead hardware acceleration.

### Preprocessing Philosophy: Preserve Signal, Filter Noise
Social-media text contains vital pragmatic cues that conventional textbook NLP pipelines discard:
- **No Stop-Word Stripping:** Words like *"just"*, *"so"*, *"oh"*, and *"never"* are essential irony amplifiers (*"Oh, I just love waiting in line"*).
- **No Punctuation Stripping:** Punctuation markers like `!`, `?!`, and `...` carry distinct sarcastic tone (1.69× higher question marks and 1.68× higher ellipses in sarcastic tweets).
- **Emoji Normalization:** Demojizing converts emojis into semantic descriptors (`:face_with_rolling_eyes:`) for classical models while retaining native Unicode tokens for RoBERTa.
- **Hashtag De-duplication & Leakage Prevention:** Explicit hashtags like `#sarcasm` and `#not` are stripped during preprocessing to prevent trivial label leakage during training.

### Pre-trained Transformer: `cardiffnlp/twitter-roberta-base`
- **Domain Specialization:** RoBERTa pre-trained on 58 million real-world tweets (TweetEval benchmark) captures informal slang, colloquial grammar, and platform-specific framing far better than Wikipedia-trained `bert-base-uncased`.
- **Sub-word BPE Tokenizer:** Byte-Pair Encoding handles novel internet slang, elongations (`sooo`), and code-mixed tokens without degenerating into out-of-vocabulary (`<unk>`) errors.
- **Computational Efficiency:** At 125M parameters, it fine-tunes in minutes on an NVIDIA GPU while serving real-time inferences with an average latency under 45ms.

### Custom BiLSTM with Self-Attention Pooling
While transformers offer state-of-the-art predictive accuracy, extracting token-level interpretability from multi-head attention across 12 layers can be noisy and computationally expensive. Our PyTorch BiLSTM features a dedicated **Self-Attention Pooling Layer**:
$$\alpha_t = \frac{\exp(w^T h_t)}{\sum_{j=1}^T \exp(w^T h_j)}$$
This generates a direct, normalized probability distribution over tokens that powers the live **Token Attention Heatmap** in the web dashboard.

### Sub-Millisecond Fallback: TF-IDF + Linear SVM
Large-scale social platforms require high-throughput failover mechanisms. If GPU inference pools experience spikes or outages, the linear SVM provides an instantaneous fallback:
- Evaluates word (1-2) and character (2-5) n-grams in under 1 millisecond on CPU.
- Requires less than 1.4MB of disk and memory footprint.
- Delivers strong baseline accuracy with interpretable linear coefficients.

---

## 4. Hardware Verification

Benchmarked on **NVIDIA GeForce RTX 4060 GPU (8GB VRAM, CUDA 12.x, PyTorch 2.x)**:
- **Transformer Inference Latency:** 38 – 45 ms per sequence.
- **BiLSTM Inference Latency:** 16 – 20 ms per sequence.
- **Linear SVM Latency:** 0.6 – 0.9 ms on CPU.
- **Throughput:** Capable of processing over 1,200 requests per second under linear fallback and over 22 requests per second under full multi-model GPU consensus.
