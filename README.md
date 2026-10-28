# IronyLens: Advanced Sarcasm & Irony Detection in Social Media Text

[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-SAGEOFCODING%2FIrony--Lens-181717.svg?logo=github)](https://github.com/SAGEOFCODING/Irony-Lens)

**IronyLens** is a multi-tier Natural Language Processing (NLP) system designed to detect and decode sarcasm, irony, and pragmatic dissonance in social media communications and digital news. 

By unifying pre-trained transformer representations (`twitter-roberta-base`), recurrent neural architectures (`BiLSTM + Self-Attention`), and sub-millisecond classical fallbacks (`TF-IDF + Linear SVM`), IronyLens provides both state-of-the-art classification performance and human-interpretable token saliency.

---

## 📸 Interactive Web Interfaces

IronyLens provides two user experiences tailored for production deployment and academic analysis:

1. **IronyLens Telemetry Workbench (`server.py`)**:
   - High-performance Flask & Tailwind CSS dark-mode interface running with GPU CUDA acceleration.
   - **Token Attention Heatmap & Saliency Inspector**: Continuous background heat glow, Flow vs. Chips views, and an interactive inspector card detailing attention weights ($0.00 - 1.00$), linguistic roles, and model attributions.
   - **Lossless Pragmatic KPIs**: Real-time computation of Valence Inversion ($0-10$), Pragmatic Gap ($\Delta$), Hyperbole Index ($0-10$), and Decoded Speaker Intent.
   - **Consensus Engine**: Simultaneous inference across RoBERTa, BiLSTM, and SVM with dynamic sensitivity threshold tuning ($0.10 - 0.90$).
   - **Model Priorities & Benchmarks View**: 10 empirical evaluation graph panels with dynamic dataset switching between Twitter and News Headlines.
2. **Exploratory Research Dashboard (`app.py`)**:
   - Multi-tab Streamlit interface featuring dataset explorer, confusion matrices, and exploratory data analysis (EDA).

---

## 🏛️ Architectural Priority Hierarchy

In production, models operate under an empirical priority cascade balancing accuracy, explainability, and throughput:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INPUT TEXT / POST                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 1: Twitter-RoBERTa Transformer (125M Parameters)             │
│  • Primary Deep Classification Engine                                  │
│  • Cross-attends between overt positive framing & negative context     │
│  • Macro-F1: 0.6250 (Twitter) / 0.9376 (Headlines) • Latency: ~40ms    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 2: BiLSTM + Self-Attention Pooling (FastText Embeddings)     │
│  • Interpretability & Saliency Engine                                  │
│  • Recurrent sequential modeling with direct attention weight pooling  │
│  • Macro-F1: 0.5219 (Twitter) / 0.8700 (Headlines) • Latency: ~18ms    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 3: TF-IDF + Linear SVM (< 1.4MB Footprint)                   │
│  • High-Throughput CPU Fallback Engine                                 │
│  • Sub-millisecond execution for instant failover during GPU surges    │
│  • Macro-F1: 0.5475 (Twitter) / 0.8746 (Headlines) • Latency: < 1ms    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks (Fact-Checked)

All metrics are evaluated against held-out official test splits and verified against ground-truth logs in `reports/`:

### 1. Twitter Benchmark (iSarcasmEval — SemEval-2022 Task 6)
*Wild, noisy social media text with author-intended labels. Test set: 1,400 tweets (14.3% sarcastic).*

| Architecture | Priority | Macro-F1 | Sarcastic F1 | Sarcastic Recall | Accuracy | ROC-AUC | Latency (ms) | Production Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Twitter-RoBERTa** | **P1** | **0.6250** | **0.3986** | **55.0%** | **76.2%** | **0.730** | ~40 ms (GPU) | Primary Deep Engine |
| **BiLSTM + Attention**| **P2** | 0.5219 | 0.2174 | 27.5% | 71.6% | 0.565 | ~18 ms (GPU) | Attention Saliency |
| **TF-IDF + Linear SVM**| **P3** | 0.5475 | 0.2732 | 37.5% | 71.4% | 0.602 | < 1 ms (CPU) | High-Speed Fallback |

### 2. News Headlines Benchmark (HuffPost & The Onion)
*Clean, grammatically structured journalistic headlines. Test set: 5,827 headlines (47.5% sarcastic).*

| Architecture | Priority | Macro-F1 | Sarcastic F1 | Sarcastic Recall | Accuracy | ROC-AUC | Latency (ms) | Production Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Twitter-RoBERTa** | **P1** | **0.9376** | **0.9329** | **90.8%** | **93.8%** | **0.983** | ~40 ms (GPU) | Primary Deep Engine |
| **BiLSTM + Attention**| **P2** | 0.8700 | 0.8648 | 87.2% | 87.0% | 0.942 | ~18 ms (GPU) | Attention Saliency |
| **TF-IDF + Linear SVM**| **P3** | 0.8746 | 0.8700 | 88.1% | 87.5% | 0.945 | < 1 ms (CPU) | High-Speed Fallback |

---

## 📈 Evaluation Visualizations

The dashboard and `reports/` directory contain 10 publication-quality diagnostic visualizations generated from the empirical test sets:

1. **Overall Performance Comparison** (`comparison.png`): Macro-F1, Sarcastic F1, and Accuracy across all models.
2. **ROC & Precision-Recall Curves** (`roc_pr_curves.png`): Full diagnostic curves with AUC metrics for both classes.
3. **6D Capability Spider Radar** (`radar_comparison.png`): Holistic assessment across Accuracy, Sarcastic Recall, Sarcastic Precision, Macro-F1, Latency Score, and Interpretability.
4. **Latency vs. Macro-F1 Trade-off** (`latency_vs_f1_tradeoff.png`): Empirical Pareto frontier mapping speed against predictive power.
5. **Hyperparameter Tuning Curves** (`hyperparameter_tuning.png`): Regularization parameter sweeps for Linear SVM ($C$) and Naive Bayes ($\alpha$).
6. **Model Parameters & Scale** (`model_parameters_scale.png`): Parameter footprint ($125\text{M}$ vs $1.5\text{M}$ vs $50\text{k}$) and disk weights ($498\text{MB}$ vs $4.5\text{MB}$ vs $1.4\text{MB}$).
7. **Confusion Matrices** (`*_confusion.png`): Detailed true vs. predicted error distributions for all models.
8. **Top N-Gram Lexical Features** (`baseline_top_features.png`): Highest-weighted positive and negative irony cues.
9. **EDA Stylistic Cue Analysis** (`eda/stylistic_cues.png`): Distribution of punctuation markers, question marks, and emojis.
10. **Length & Word Count Distributions** (`eda/length_distribution.png`): Token and character density across classes.

---

## 📁 Repository Structure

```
Irony-Lens/
├── server.py                 # IronyLens Telemetry Workbench (Flask + Tailwind CSS + CUDA GPU)
├── app.py                    # Streamlit Exploratory Research Dashboard
├── run_all.py                # End-to-end training and evaluation pipeline
├── train_on_colab.ipynb      # Google Colab notebook for GPU fine-tuning
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules (ignores model weights > 100MB)
├── docs/
│   ├── TECH_STACK.md         # Deep-dive architectural decisions and hardware profiling
│   ├── DATASETS.md           # Dataset sources, labeling methodologies, and split policies
│   └── HOW_IT_WORKS.md       # NLP concepts, preprocessing rules, and FAQ
├── scripts/
│   └── generate_extra_graphs.py # Automated generation of all 10 diagnostic graphs
├── src/
│   ├── config.py             # Global constants, directory paths, and seeds
│   ├── data.py               # Dataset downloaders, caching, and stratified splits
│   ├── preprocess.py         # Signal-preserving social media normalization
│   ├── eda.py                # Dataset profiling and stylistic cue extraction
│   ├── train_baseline.py     # Classical ML: TF-IDF + Grid Search (NB / LR / SVM)
│   ├── lstm.py               # PyTorch BiLSTM with custom Self-Attention pooling
│   ├── train_lstm.py         # BiLSTM training loop with early stopping
│   ├── train_transformer.py  # Hugging Face RoBERTa fine-tuning with mixed precision
│   ├── metrics.py            # Comprehensive evaluation metrics and error export
│   ├── compare.py            # Cross-model comparative reporting
│   └── predict.py            # Multi-model inference interface for CLI and web apps
└── reports/                  # Generated metrics JSON, error CSVs, and visualization PNGs
    ├── isarcasm/             # Twitter iSarcasmEval reports and diagnostic plots
    └── headlines/            # News Headlines reports and diagnostic plots
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/SAGEOFCODING/Irony-Lens.git
cd Irony-Lens

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # On macOS/Linux: source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# (Recommended for NVIDIA GPU users) Install CUDA-accelerated PyTorch:
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. Launch the Web Applications

To start the **IronyLens Telemetry Workbench**:
```bash
python server.py
# Server running at: http://localhost:5000
```

To start the **Streamlit Exploratory Dashboard**:
```bash
streamlit run app.py
# Dashboard running at: http://localhost:8501
```

### 3. Command-Line Inference

Run multi-model predictions directly from your terminal:

```bash
python -m src.predict --dataset isarcasm "Oh great, another meeting that definitely could not have been an email!"
```

Output:
```
transformer   98.5%  SARCASTIC
bilstm        99.3%  SARCASTIC
baseline      68.4%  SARCASTIC

Baseline - strongest features in this text:
  word__oh                       +0.254
  word__an                       +0.194
  word__have been                +0.159
  word__great                    +0.126
  word__email                    +0.121
```

---

## 🔌 REST API Documentation

IronyLens exposes a JSON REST API for seamless programmatic integration:

### Endpoint: `POST /api/analyze`

**Request:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Oh great, another meeting that definitely could not have been an email!",
    "model": "transformer",
    "sensitivity": 0.50
  }'
```

**Response Payload:**
```json
{
  "text": "Oh great, another meeting that definitely could not have been an email!",
  "is_sarcastic": true,
  "confidence_pct": 98.5,
  "literal_pct": 1.5,
  "classification_tone": "Workplace Cynicism",
  "friction_description": "Severe pragmatic dissonance between overt politeness and workplace fatigue.",
  "valence_inversion": 9.7,
  "hyperbole_index": 7.0,
  "pragmatic_gap": "Δ -1.99",
  "inference_ms": 39,
  "decoded_intent": "Expressing deep frustration about redundant, time-wasting workplace meetings when an email would have preserved hours of productive focus.",
  "all_models": {
    "transformer": 98.5,
    "bilstm": 99.3,
    "baseline": 68.4
  },
  "tokens": [
    {
      "token": "Oh",
      "score": 0.53,
      "level": "medium",
      "role": "Discourse / Irony Marker",
      "desc": "Conversational marker signaling sarcastic framing",
      "lstm_w": 0.0436,
      "base_w": 0.2542
    },
    {
      "token": "great,",
      "score": 0.64,
      "level": "medium",
      "role": "Positive Valence (Inverted)",
      "desc": "Overt positive lexicon contradictory to context",
      "lstm_w": 0.0360,
      "base_w": 0.1255
    },
    {
      "token": "definitely",
      "score": 0.75,
      "level": "high",
      "role": "Modal Intensifier",
      "desc": "Amplifies sarcastic hyperbole and certainty",
      "lstm_w": 0.0997,
      "base_w": 0.0530
    },
    {
      "token": "email!",
      "score": 0.92,
      "level": "high",
      "role": "Situational Friction Target",
      "desc": "Target subject of situational friction or fatigue",
      "lstm_w": 0.0991,
      "base_w": 0.1206
    }
  ]
}
```

---

## 🔄 Pipeline Reproduction

To reproduce all experiments, data preprocessing, training, and graph generation from scratch:

```bash
# 1. Download and preprocess datasets
python -m src.data --dataset isarcasm
python -m src.data --dataset headlines

# 2. Run Exploratory Data Analysis
python -m src.eda --dataset isarcasm
python -m src.eda --dataset headlines

# 3. Train models
python -m src.train_baseline --dataset isarcasm
python -m src.train_lstm     --dataset isarcasm
python -m src.train_transformer --dataset isarcasm  # Requires GPU

# 4. Generate comparison metrics & all 10 diagnostic graphs
python -m src.compare --dataset isarcasm
python scripts/generate_extra_graphs.py
```

---

## 📜 Documentation Index

- **[Tech Stack & Hardware Analysis](docs/TECH_STACK.md)**: Justifications for architectural choices, rejection rationales, and latency benchmarking.
- **[Dataset Methodologies](docs/DATASETS.md)**: Deep dive into iSarcasmEval, Reddit SARC, and News Headlines datasets.
- **[How It Works & NLP Concepts](docs/HOW_IT_WORKS.md)**: Theoretical foundation covering pragmatic dissonance, token saliency math, and FAQ.

---

## ⚖️ License & Attribution

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Developed by [SAGEOFCODING](https://github.com/SAGEOFCODING). Built on PyTorch, Hugging Face Transformers, and the SemEval-2022 iSarcasmEval benchmark.
