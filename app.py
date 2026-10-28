"""Streamlit demo: type a post, see what each model thinks and why.

Run:  streamlit run app.py
"""
import html

import streamlit as st

from src.compare import load_results
from src.config import MODELS_DIR, REPORTS_DIR
from src.predict import SarcasmDetector

st.set_page_config(page_title="Sarcasm Detector", page_icon="🙃", layout="wide")

EXAMPLES = [
    "Oh great, my flight got delayed again. Best day ever!!",
    "I just love it when my phone dies at 5% battery",
    "The meeting has been moved to 3pm on Thursday.",
    "Wow, you finished the assignment 5 minutes before the deadline. Truly a genius.",
]


@st.cache_resource
def get_detector(dataset):
    return SarcasmDetector(dataset)


trained = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir()) if MODELS_DIR.exists() else []
st.title("🙃 Sarcasm Detection in Social Media Text")
if not trained:
    st.warning("No trained models found. Run `python -m src.train_baseline` first.")
    st.stop()

default = trained.index("isarcasm") if "isarcasm" in trained else 0
dataset = st.sidebar.selectbox("Models trained on", trained, index=default)
detector = get_detector(dataset)
st.sidebar.write("Available models:", ", ".join(detector.available))

tab_try, tab_compare, tab_data = st.tabs(["Try it", "Model comparison", "Dataset insights"])

with tab_try:
    example = st.selectbox("Pick an example or write your own", [""] + EXAMPLES)
    text = st.text_area("Post", value=example, height=100)
    context = st.text_input("Parent comment (optional context)") if detector.uses_context else ""

    if st.button("Analyse", type="primary") and text.strip():
        results = detector.predict(text, context)
        cols = st.columns(len(results))
        for col, (name, p) in zip(cols, results.items()):
            sarcastic = p >= 0.5
            col.metric(name, f"{p:.0%}", "sarcastic" if sarcastic else "not sarcastic",
                       delta_color="normal" if sarcastic else "off")
            col.progress(p)

        if detector.baseline is not None:
            with st.expander("Why? - baseline (TF-IDF features that drove the decision)"):
                rows = detector.explain_baseline(text)
                if rows:
                    st.table({"feature": [r[0] for r in rows], "push towards sarcastic (+) / not (−)": [round(r[1], 3) for r in rows]})
                else:
                    st.write("The selected baseline classifier has no per-feature weights to show.")

        if detector.lstm is not None:
            with st.expander("Why? - BiLSTM attention (darker = more focus)"):
                pairs = detector.explain_lstm(text)
                top = max(w for _, w in pairs) or 1
                spans = "".join(
                    f'<span style="background: rgba(197,48,48,{w / top:.2f}); padding:2px 4px; margin:1px; border-radius:4px">{html.escape(t)}</span>'
                    for t, w in pairs
                )
                st.markdown(spans, unsafe_allow_html=True)

with tab_compare:
    table = load_results(dataset)
    if table.empty:
        st.info("No evaluation reports yet.")
    else:
        st.dataframe(table.style.highlight_max(axis=0, props="font-weight: bold; color: #38a169").format("{:.4f}"), width="stretch")
        images = sorted((REPORTS_DIR / dataset).glob("*_confusion.png"))
        for col, img in zip(st.columns(max(len(images), 1)), images):
            col.image(str(img), caption=img.stem.replace("_confusion", ""))
        features = REPORTS_DIR / dataset / "baseline_top_features.png"
        if features.exists():
            st.image(str(features), caption="Baseline: most informative features", width=520)

with tab_data:
    eda = REPORTS_DIR / dataset / "eda"
    if not eda.exists():
        st.info(f"Run `python -m src.eda --dataset {dataset}` to generate dataset charts.")
    else:
        for col, img in zip(st.columns(3), sorted(eda.glob("*.png"))):
            col.image(str(img))
        summary = eda / "eda_summary.md"
        if summary.exists():
            st.markdown(summary.read_text(encoding="utf8"))
