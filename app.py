"""Streamlit Web Dashboard for Sarcasm Detection.

Provides real-time multi-model sarcasm classification, token-level interpretability,
architectural comparisons, and dataset insights for both technical and non-technical users.
"""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.compare import load_results
from src.config import MODELS_DIR, REPORTS_DIR
from src.predict import SarcasmDetector

# --- Page Configuration ---
st.set_page_config(
    page_title="SarcasmSense | NLP Sarcasm Detection",
    page_icon="🙃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling (Modern Dark/Light Compatible CSS) ---
st.markdown(
    """
    <style>
    /* Card Container */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 50%, #4c1d95 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .badge-sarcastic {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-sincere {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .model-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .model-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    .prob-num {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .info-callout {
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        background: rgba(99, 102, 241, 0.08);
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Preset Examples for Easy Testing ---
PRESET_EXAMPLES = {
    "Select an example or write your own below...": "",
    "✈️ Flight Delay (Classic Irony)": "Oh great, my flight got delayed again. Best day ever!!",
    "🔋 Dead Phone (Praise for Negative Situation)": "I just love it when my phone dies at 5% battery 🙃",
    "📚 Late Submission (Sarcastic Praise)": "Wow, you finished the assignment 5 minutes before the deadline. Truly a genius.",
    "🌧️ Rainy Picnic (Contrast Cue)": "Torrential rain on our beach picnic day, absolutely wonderful weather!",
    "📅 Rescheduled Meeting (Sincere / Non-Sarcastic)": "The team meeting has been rescheduled to Thursday at 3:00 PM in Conference Room B.",
    "🍕 Food Order (Sincere / Factual)": "We are ordering pizza for lunch today, let me know your preferences.",
    "🚗 Traffic Jam (Everyday Sarcasm)": "Sitting in bumper-to-bumper traffic for two hours is totally my favorite hobby.",
}

MODEL_METADATA = {
    "baseline": {
        "title": "Classical Baseline",
        "tag": "TF-IDF + Linear SVM",
        "paradigm": "Generation 1: Statistical n-gram tokens & hyperplane margin classifier",
        "icon": "📊",
        "explain": "Looks for explicit words ('oh', 'great') and character runs ('!!!', 'sooo'). Fast and interpretable, but lacks word order understanding.",
    },
    "bilstm": {
        "title": "Deep Learning",
        "tag": "BiLSTM + Attention",
        "paradigm": "Generation 2: Recurrent neural sequence model with attention pooling",
        "icon": "🧠",
        "explain": "Reads text left-to-right and right-to-left to track context. Attention weights show which tokens the memory state focused on.",
    },
    "transformer": {
        "title": "Transfer Learning",
        "tag": "twitter-roberta-base",
        "paradigm": "Generation 3: Self-attention transformer pre-trained on 58M tweets",
        "icon": "🚀",
        "explain": "State-of-the-art transformer. Pre-trained on social media nuance, slang, and emojis. Handles subtle tone shifts and sentiment contrast.",
    },
}


@st.cache_resource
def get_detector(dataset):
    return SarcasmDetector(dataset)


# --- Sidebar Navigation & Controls ---
trained_datasets = sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()]) if MODELS_DIR.exists() else []

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sarcasm.png", width=64)
    st.markdown("### **SarcasmSense Engine**")
    st.caption("AI-powered sarcasm detection & linguistic analysis across 3 NLP generations.")
    st.divider()

    if not trained_datasets:
        st.error("No trained models found! Please train a model first.")
        st.stop()

    default_idx = trained_datasets.index("isarcasm") if "isarcasm" in trained_datasets else 0
    selected_dataset = st.selectbox(
        "📂 Benchmark Dataset",
        trained_datasets,
        index=default_idx,
        help="Select which dataset's trained models to use for inference and inspection.",
    )

    detector = get_detector(selected_dataset)
    available_models = detector.available

    st.sidebar.markdown("#### **Telemetry Workbench**")
    st.sidebar.link_button("⚡ Open IronyLens Futuristic UI", "http://localhost:5000")
    st.sidebar.divider()
    st.markdown("#### **Active Models in Memory**")
    for m in ["baseline", "bilstm", "transformer"]:
        if m in available_models:
            st.markdown(f"✅ **{MODEL_METADATA[m]['title']}** (`{m}`)")
        else:
            st.markdown(f"⚪ *{MODEL_METADATA[m]['title']}* (not loaded)")

    st.divider()
    st.markdown("#### **System Info**")
    st.caption(f"**Dataset:** `{selected_dataset}`")
    st.caption(f"**Models Available:** {len(available_models)}/3")
    st.caption("**Hardware:** GPU Accelerated (NVIDIA RTX 4060)")

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-container">
        <h1 style="margin:0; font-size:2.4rem; font-weight:800;">🙃 SarcasmSense</h1>
        <p style="margin-top:8px; font-size:1.15rem; opacity:0.9;">
            Multi-Generation NLP Benchmarking & Explainable Sarcasm Detection in Social Media
        </p>
        <div style="margin-top:14px; font-size:0.92rem; opacity:0.85; line-height:1.5;">
            Sarcasm is one of the hardest challenges in computational linguistics: it expresses negative criticism
            using overtly positive words. This platform demonstrates how three generations of AI tackle this challenge.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Main App Navigation Tabs ---
tab_try, tab_arch, tab_compare, tab_eda, tab_faq = st.tabs(
    [
        "🧪 Live Detector & Explainability",
        "🏗️ How NLP Detects Sarcasm",
        "📊 Model Comparison & Leaderboard",
        "📈 Linguistic Signals & Data",
        "❓ Project FAQ & Guide",
    ]
)

# ==============================================================================
# TAB 1: LIVE DETECTOR & EXPLAINABILITY
# ==============================================================================
with tab_try:
    st.markdown("### 💬 Test Any Sentence")
    st.write(
        "Choose an example preset or type your own text below to see how each model interprets the tone and context."
    )

    col_select, col_custom = st.columns([1, 2])
    with col_select:
        preset_choice = st.selectbox("Quick Presets:", list(PRESET_EXAMPLES.keys()))
        preset_text = PRESET_EXAMPLES[preset_choice]

    text_input = st.text_area(
        "Input Text to Analyze:",
        value=preset_text,
        placeholder="e.g. Oh great, another Monday morning meeting. Exactly what I wanted!",
        height=95,
        help="Type any social media post, headline, or message.",
    )

    context_input = ""
    if detector.uses_context:
        context_input = st.text_input(
            "Parent Comment (Optional Context):",
            help="For conversational sarcasm where tone depends on the message being replied to.",
        )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Tone", type="primary", use_container_width=True)
    with col_hint:
        st.caption("Click to run inference through all available models simultaneously.")

    if (analyze_clicked or preset_text) and text_input.strip():
        with st.spinner("Analyzing linguistic cues and sequence attention..."):
            predictions = detector.predict(text_input, context_input)

        st.markdown("---")
        st.markdown("### 🎯 Multi-Model Predictions")

        # Consensus Summary Banner
        probs_list = list(predictions.values())
        avg_prob = sum(probs_list) / len(probs_list) if probs_list else 0.0
        consensus_sarcastic = avg_prob >= 0.50

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 6px solid {'#ef4444' if consensus_sarcastic else '#10b981'};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; color:#94a3b8; font-weight:700;">Overall AI Consensus</span>
                        <div style="font-size:1.8rem; font-weight:800; margin-top:2px;">
                            {'🚨 SARCASTIC TONE DETECTED' if consensus_sarcastic else '🟢 SINCERE / NON-SARCASTIC'}
                        </div>
                        <div style="font-size:0.95rem; color:#64748b; margin-top:4px;">
                            Average confidence across {len(predictions)} model(s): <strong>{avg_prob:.1%}</strong>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="{'badge-sarcastic' if consensus_sarcastic else 'badge-sincere'}" style="font-size:1.1rem; padding:8px 18px;">
                            {avg_prob:.0%} Sarcasm Score
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Individual Model Cards Side by Side
        cols = st.columns(len(predictions))
        for col, (model_key, prob) in zip(cols, predictions.items()):
            meta = MODEL_METADATA.get(model_key, {"title": model_key, "tag": "", "icon": "🤖", "explain": ""})
            is_sarc = prob >= 0.50

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <div class="model-title">{meta['icon']} {meta['title']}</div>
                                <div class="model-sub">{meta['tag']}</div>
                            </div>
                            <span class="{'badge-sarcastic' if is_sarc else 'badge-sincere'}">
                                {'SARCASTIC' if is_sarc else 'NOT SARCASTIC'}
                            </span>
                        </div>
                        <div class="prob-num" style="color: {'#ef4444' if is_sarc else '#10b981'};">
                            {prob:.1%}
                        </div>
                        <div style="font-size:0.82rem; color:#64748b; margin-bottom:8px;">
                            Sarcasm Probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(prob))
                st.caption(meta["explain"])

        # Deep Interpretability Section
        st.markdown("---")
        st.markdown("### 🔬 Explainability: *Why did the models predict this?*")

        col_exp1, col_exp2 = st.columns(2)

        # Baseline Feature Contributions
        with col_exp1:
            st.markdown("#### 📊 Classical Model Feature Signals")
            st.caption("Which specific words or character patterns influenced the linear classifier?")
            if detector.baseline is not None:
                contribs = detector.explain_baseline(text_input, k=8)
                if contribs:
                    feat_df = pd.DataFrame(contribs, columns=["Feature", "Push Weight"])
                    feat_df["Direction"] = feat_df["Push Weight"].apply(
                        lambda w: "Pushes Toward Sarcastic (+)" if w > 0 else "Pushes Toward Sincere (−)"
                    )

                    fig, ax = plt.subplots(figsize=(6, 3.8))
                    colors = ["#ef4444" if w > 0 else "#3b82f6" for w in feat_df["Push Weight"]]
                    ax.barh(feat_df["Feature"], feat_df["Push Weight"], color=colors)
                    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
                    ax.set_xlabel("Contribution Score")
                    ax.set_title("Top Influential Features in this Post", fontsize=11)
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No specific vocabulary weights available for this input.")
            else:
                st.warning("Baseline model not loaded.")

        # BiLSTM Attention Weights
        with col_exp2:
            st.markdown("#### 🧠 BiLSTM Neural Attention Heatmap")
            st.caption("Which words did the recurrent network focus its attention on?")
            if detector.lstm is not None:
                token_weights = detector.explain_lstm(text_input)
                if token_weights:
                    max_w = max(w for _, w in token_weights) or 1.0
                    html_spans = []
                    for tok, w in token_weights:
                        norm_w = w / max_w
                        # Red highlight with opacity based on attention weight
                        bg = f"rgba(239, 68, 68, {max(0.08, norm_w):.2f})"
                        border = "#ef4444" if norm_w > 0.5 else "transparent"
                        html_spans.append(
                            f'<span title="Attention weight: {w:.3f}" style="background:{bg}; border: 1px solid {border}; padding:4px 8px; margin:3px; border-radius:6px; font-weight:{600 if norm_w>0.5 else 400}; display:inline-block;">{html.escape(tok)}</span>'
                        )

                    st.markdown(
                        f"""
                        <div style="background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:18px; line-height:2.4; font-size:1.15rem;">
                            {' '.join(html_spans)}
                        </div>
                        <div style="margin-top:10px; font-size:0.85rem; color:#94a3b8;">
                            ℹ️ <em>Darker red highlight indicates words where the model's bidirectional attention mechanism concentrated the most memory weight. Hover over a token to see its raw score.</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No attention weights extracted.")
            else:
                st.warning("BiLSTM model not loaded.")

# ==============================================================================
# TAB 2: HOW NLP DETECTS SARCASM
# ==============================================================================
with tab_arch:
    st.markdown("### 🏗️ How Natural Language Processing Detects Sarcasm")
    st.write(
        "Sarcasm detection has evolved over three major AI eras. This project implements all three on the exact same data splits to demonstrate the progression:"
    )

    st.markdown(
        """
        <div class="info-callout">
            <strong>The Linguistic Paradox of Sarcasm:</strong><br>
            Standard sentiment analysis relies on word valence (e.g. <em>"love"</em> = positive, <em>"stuck in traffic"</em> = negative).
            Sarcasm breaks this by pairing <strong>positive sentiment words with negative real-world situations</strong>.
            A model must detect the <em>incongruity</em> between the words and the reality.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        st.markdown(
            """
            <div class="metric-card" style="height:100%;">
                <div style="font-size:1.5rem;">📊</div>
                <div class="model-title">Generation 1: Classical ML</div>
                <div class="model-sub">TF-IDF + Linear Classifiers</div>
                <p style="font-size:0.9rem; line-height:1.5;">
                    <strong>How it works:</strong> Converts text into a bag of word 1-2 grams (<em>"oh great"</em>, <em>"love it"</em>)
                    and character 2-5 grams (<em>"!!!"</em>, <em>"sooo"</em>). Uses an SVM to find a decision hyperplane.
                </p>
                <ul style="font-size:0.85rem; padding-left:18px;">
                    <li><strong>Strength:</strong> Extremely fast, highly interpretable, lightweight.</li>
                    <li><strong>Limitation:</strong> Completely ignores word order and syntactic nuance. Cannot distinguish <em>"not good, great"</em> from <em>"good, not great"</em>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_g2:
        st.markdown(
            """
            <div class="metric-card" style="height:100%;">
                <div style="font-size:1.5rem;">🧠</div>
                <div class="model-title">Generation 2: Recurrent Deep Learning</div>
                <div class="model-sub">BiLSTM + Attention Pooling</div>
                <p style="font-size:0.9rem; line-height:1.5;">
                    <strong>How it works:</strong> Maps words to 200-dimensional dense vectors. A Bidirectional LSTM reads the post
                    forward and backward, and an attention layer pools the words that matter most.
                </p>
                <ul style="font-size:0.85rem; padding-left:18px;">
                    <li><strong>Strength:</strong> Captures sequential word order, negation scope, and temporal context.</li>
                    <li><strong>Limitation:</strong> Must learn English from scratch. Overfits easily when trained on only ~3,000 social media posts.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_g3:
        st.markdown(
            """
            <div class="metric-card" style="height:100%;">
                <div style="font-size:1.5rem;">🚀</div>
                <div class="model-title">Generation 3: Transfer Learning</div>
                <div class="model-sub">Twitter-RoBERTa Transformer</div>
                <p style="font-size:0.9rem; line-height:1.5;">
                    <strong>How it works:</strong> Uses 12 layers of multi-head self-attention, pre-trained on 58 million real tweets.
                    Fine-tuned on our sarcasm datasets with class-weighted cross-entropy.
                </p>
                <ul style="font-size:0.85rem; padding-left:18px;">
                    <li><strong>Strength:</strong> Already knows Twitter slang, hashtags, and emojis. Wins by +50% F1 score on social media text.</li>
                    <li><strong>Limitation:</strong> Requires substantial GPU compute (~125M parameters).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🔄 The Sarcasm-Aware Preprocessing Pipeline")
    st.write(
        "Standard NLP removes punctuation, stop words, and numbers. **We deliberately don't do that**, because those are the primary carriers of sarcasm!"
    )

    flow_df = pd.DataFrame(
        [
            {"Action": "Unescape HTML", "Input Example": "&amp; → &", "Reasoning": "Clean web scraping artifacts"},
            {"Action": "Preserve Punctuation", "Input Example": "Best day ever!!!", "Reasoning": "Exclamation and question runs carry tone"},
            {"Action": "Shorten Elongation", "Input Example": "soooooo → sooo", "Reasoning": "Keeps the emphasis signal without blowing up vocab size"},
            {"Action": "Demojize Emojis", "Input Example": "🙄 → :face_with_rolling_eyes:", "Reasoning": "Allows TF-IDF and BiLSTM to treat emojis as word tokens"},
            {"Action": "Strip Label Hashtags", "Input Example": "Delete #sarcasm, #not", "Reasoning": "Prevents label leakage (model learning the hashtag instead of the sarcasm)"},
        ]
    )
    st.dataframe(flow_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: MODEL COMPARISON & LEADERBOARD
# ==============================================================================
with tab_compare:
    st.markdown("### 📊 Benchmark Leaderboard")
    st.write(
        f"Evaluation metrics on the unseen test set for **`{selected_dataset}`**. Metrics are sorted by **Macro-F1**, the official standard SemEval competition metric."
    )

    results_table = load_results(selected_dataset)
    if results_table.empty:
        st.info("No evaluation reports found for this dataset.")
    else:
        st.dataframe(
            results_table.style.highlight_max(axis=0, props="font-weight: bold; background-color: rgba(16, 185, 129, 0.2); color: #10b981;").format("{:.4f}"),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 🎯 Confusion Matrices")
    st.write(
        "A confusion matrix shows True Negatives, False Positives (calling innocent text sarcastic), False Negatives (missing sarcasm), and True Positives."
    )

    conf_imgs = sorted((REPORTS_DIR / selected_dataset).glob("*_confusion.png"))
    if conf_imgs:
        cols_conf = st.columns(len(conf_imgs))
        for col, img_path in zip(cols_conf, conf_imgs):
            with col:
                st.image(str(img_path), caption=img_path.stem.replace("_confusion", "").replace("_", " ").title())

    st.markdown("---")
    st.markdown("#### ⚠️ Real Error Analysis (Most Confident Mistakes)")
    st.write(
        "Inspecting the posts where models had the highest confidence but were WRONG reveals why sarcasm remains challenging for artificial intelligence:"
    )

    error_files = sorted((REPORTS_DIR / selected_dataset).glob("*_errors.csv"))
    if error_files:
        chosen_err_file = st.selectbox(
            "Select model error log to inspect:",
            error_files,
            format_func=lambda p: p.stem.replace("_errors", "").replace("_", " ").title(),
        )
        if chosen_err_file and chosen_err_file.exists():
            err_df = pd.read_csv(chosen_err_file).head(15)
            err_df["true"] = err_df["true"].map({0: "Not Sarcastic", 1: "Sarcastic"})
            err_df["pred"] = err_df["pred"].map({0: "Not Sarcastic", 1: "Sarcastic"})
            st.dataframe(err_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 4: LINGUISTIC SIGNALS & DATA
# ==============================================================================
with tab_eda:
    st.markdown("### 📈 Linguistic Signals & Stylistic Markers")
    st.write(
        f"Statistical analysis of the training data for **`{selected_dataset}`**, showing how humans signal sarcasm in social writing."
    )

    eda_dir = REPORTS_DIR / selected_dataset / "eda"
    if eda_dir.exists():
        eda_imgs = sorted(eda_dir.glob("*.png"))
        if eda_imgs:
            cols_eda = st.columns(len(eda_imgs))
            for col, img in zip(cols_eda, eda_imgs):
                col.image(str(img), use_container_width=True)

        summary_file = eda_dir / "eda_summary.md"
        if summary_file.exists():
            st.markdown("---")
            st.markdown(summary_file.read_text(encoding="utf8"))
    else:
        st.info("Run `python -m src.eda --dataset <name>` to generate EDA charts.")

    st.markdown("---")
    st.markdown("#### 📚 The Datasets Behind This Project")
    st.markdown(
        """
        | Dataset | Source | Characteristics | Why We Include It |
        |---|---|---|---|
        | **iSarcasmEval** | Twitter (SemEval 2022) | ~3.4k train, 1.4k test. Author-labeled intended sarcasm. Imbalanced (~25% sarcastic). | **Primary real-world benchmark.** Proves how hard real noisy social media is. |
        | **News Headlines** | The Onion vs HuffPost | ~28.6k headlines. Professionally written, 48% sarcastic. | **Clean reference benchmark.** Models easily hit 94% F1 here because language is grammatically clean. |
        | **Reddit SARC** | Reddit Comments | ~1M comments with parent comment context. Self-labeled with `/s`. | **Contextual testing.** Enables sentence-pair attention (`reply + context`). |
        """
    )

# ==============================================================================
# TAB 5: FAQ & PROJECT GUIDE
# ==============================================================================
with tab_faq:
    st.markdown("### ❓ Frequently Asked Questions (FAQ)")

    with st.expander("Q: Why is sarcasm detection different from regular sentiment analysis?"):
        st.write(
            "Regular sentiment analysis scores individual words and sums their sentiment polarity. "
            "A sentiment analyzer reading *'I just love waiting two hours at the DMV'* reads 'love' as strong positive emotion! "
            "Sarcasm detection looks for the mismatch between the positive emotion and the negative reality."
        )

    with st.expander("Q: Why is accuracy a misleading metric for this project?"):
        st.write(
            "In real life and in the iSarcasmEval test set, only ~14% of tweets are sarcastic. "
            "A dumb model that always outputs 'NOT SARCASTIC' for every single input achieves 86% accuracy! "
            "Yet that model has zero ability to detect sarcasm. That is why we evaluate using **Macro-F1** and **Sarcastic-class F1**."
        )

    with st.expander("Q: Why did the deep BiLSTM model score lower than the simple TF-IDF baseline on Twitter?"):
        st.write(
            "Neural networks with recurrent memory require large amounts of data to learn grammatical structures and word representations. "
            "With only ~3,000 training tweets, the BiLSTM quickly overfits to specific vocabulary instead of learning general sarcasm. "
            "This directly justifies why **Transfer Learning (Transformers)** is essential: RoBERTa was pre-trained on 58M tweets before fine-tuning."
        )

    with st.expander("Q: Why do models score 94% F1 on Headlines but only 63% F1 on Twitter?"):
        st.write(
            "News headlines (The Onion vs HuffPost) are professionally edited, grammatically complete, and use distinct journalistic satire styles. "
            "Twitter text, in contrast, contains typos, slang, missing context, and relies heavily on unwritten shared cultural knowledge. "
            "Comparing both datasets reveals the gap between laboratory NLP and real-world noisy text."
        )

# --- Footer ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px 0;">
        <strong>SarcasmSense</strong> • Natural Language Processing Research & Benchmarking System<br>
        Built with Python, scikit-learn, PyTorch, Hugging Face Transformers & Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)
