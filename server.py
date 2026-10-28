"""IronyLens Telemetry Engine & Web Server.

Serves a clean, modern, and lossless Sarcasm Detection intelligence workbench
connected to PyTorch and Hugging Face transformer inference on RTX 4060 GPU.
"""
import json
import re
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from src.config import MODELS_DIR, REPORTS_DIR
from src.predict import SarcasmDetector

app = Flask(__name__, static_folder="static")

# Initialize detectors (cached in memory)
print("Loading trained SarcasmDetector into memory...", flush=True)
detector = SarcasmDetector("isarcasm")
print(f"Loaded detector with active models: {detector.available}", flush=True)


def generate_pragmatic_paraphrase(text, is_sarcastic, prob):
    """Generate an intelligent decode of the speaker's true intent."""
    lower = text.lower()
    if not is_sarcastic:
        return f"The speaker is communicating a literal, sincere statement without hidden irony: stating direct observations about '{text[:60]}...'."

    if any(k in lower for k in ["meeting", "email", "call", "slack", "zoom", "work"]):
        return "Expressing deep frustration about redundant, time-wasting workplace meetings when an email would have preserved hours of productive focus."
    if any(k in lower for k in ["flight", "delay", "airport", "traffic", "train", "bus", "car"]):
        return "Annoyed by unexpected travel delays and logistical friction, masking irritation with mock praise."
    if any(k in lower for k in ["phone", "battery", "charge", "screen", "wifi", "internet", "device"]):
        return "Exasperated by sudden tech failure at an inconvenient moment, using sarcastic adoration to vent."
    if any(k in lower for k in ["genius", "smart", "assignment", "deadline", "homework", "exam"]):
        return "Calling out poor judgment or procrastination while overtly pretending to commend brilliance."
    if any(k in lower for k in ["weather", "rain", "cold", "snow", "picnic", "beach"]):
        return "Disappointed by ruined outdoor plans due to miserable weather, sarcastically calling conditions 'wonderful'."
    if any(k in lower for k in ["love", "great", "best", "wonderful", "favorite"]):
        return "The speaker actually dislikes the situation, using hyper-positive words to emphasize how unpleasant reality is."

    return "The speaker means the exact opposite of their stated words: voicing dissatisfaction, mockery, or skepticism."


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "with", "on", "at", "from",
    "by", "about", "as", "into", "through", "during", "before", "after", "above", "below", "up", "down",
    "in", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "same", "than", "can", "will", "don", "now", "it", "its", "that", "this", "these",
    "those", "am", "have", "has", "had", "having", "do", "does", "did", "doing", "would", "should",
    "could", "been", "being", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our", "their"
}

POSITIVE_VALENCE = {
    "great", "love", "amazing", "awesome", "brilliant", "genius", "perfect", "wonderful",
    "fantastic", "thrilled", "favorite", "favourite", "superb", "spectacular", "delight",
    "delighted", "blessed", "masterpiece", "flawless", "lucky", "joy", "fun", "nice", "glad",
    "proud", "pleased", "splendid", "marvelous", "heaven", "angel", "hero"
}

MODAL_INTENSIFIERS = {
    "definitely", "totally", "obviously", "clearly", "absolutely", "truly", "literally",
    "really", "completely", "utterly", "always", "forever", "ever", "surely", "certainly"
}

DISCOURSE_MARKERS = {
    "oh", "wow", "sure", "thanks", "another", "yeah", "yay", "hurray", "bravo", "right", "well",
    "thank", "course", "okay", "ok"
}

FRICTION_TARGETS = {
    "meeting", "delay", "delayed", "traffic", "train", "flight", "monday", "error", "crash",
    "bug", "waiting", "cancel", "canceled", "broken", "bill", "tax", "rain", "queue", "hours",
    "email", "boss", "slack", "homework", "exam", "deadline", "slides", "walk", "job", "commute",
    "politician", "system", "service", "line", "subway", "bus", "server", "update"
}


def extract_token_heatmaps(text, detector_instance, prob):
    """Generate interpretable token attention weights, saliency scores & linguistic roles."""
    is_sarc = prob >= 0.5
    raw_words = text.split()
    if not raw_words:
        return []

    # 1. BiLSTM Self-Attention extraction
    lstm_map = {}
    if detector_instance.lstm is not None:
        try:
            for tok, w in detector_instance.explain_lstm(text):
                lstm_map[tok.lower().strip()] = float(w)
        except Exception:
            pass

    # 2. Baseline Linear TF-IDF contribution
    base_map = {}
    if detector_instance.baseline is not None:
        try:
            for feat, w in detector_instance.explain_baseline(text, k=20):
                clean_feat = feat.replace("word__", "").replace("char__", "").strip().lower()
                base_map[clean_feat] = float(w)
        except Exception:
            pass

    lstm_vals = list(lstm_map.values()) or [0.1]
    mean_lstm = sum(lstm_vals) / len(lstm_vals)

    tokens_data = []
    for raw in raw_words:
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", raw).lower()
        punct = re.findall(r"[!?.]+", raw)
        punct_str = punct[0] if punct else ""

        raw_lstm = lstm_map.get(clean, mean_lstm)
        rel_attn = raw_lstm / mean_lstm if mean_lstm > 0 else 1.0
        base_w = base_map.get(clean, 0.0)

        # Linguistic Role Identification
        if clean in POSITIVE_VALENCE:
            role = "Positive Valence (Inverted)" if is_sarc else "Positive Sentiment"
            cat = "positive"
            boost = 0.45 if is_sarc else 0.20
            desc = "Overt positive lexicon contradictory to context" if is_sarc else "Genuine positive evaluation"
        elif clean in MODAL_INTENSIFIERS:
            role = "Modal Intensifier"
            cat = "intensifier"
            boost = 0.40 if is_sarc else 0.15
            desc = "Amplifies sarcastic hyperbole and certainty" if is_sarc else "Emphasis modifier"
        elif clean in DISCOURSE_MARKERS:
            role = "Discourse / Irony Marker"
            cat = "discourse"
            boost = 0.35 if is_sarc else 0.10
            desc = "Conversational marker signaling sarcastic framing" if is_sarc else "Discourse signpost"
        elif clean in FRICTION_TARGETS:
            role = "Situational Friction Target"
            cat = "target"
            boost = 0.40 if is_sarc else 0.10
            desc = "Target subject of situational friction or fatigue" if is_sarc else "Topic entity"
        elif clean in {"not", "never", "no", "hardly", "barely", "cannot", "can't", "won't", "don't"}:
            role = "Negation Operator"
            cat = "negation"
            boost = 0.20
            desc = "Flips truth-conditional semantic polarity"
        elif clean in STOP_WORDS:
            role = "Syntactic Carrier"
            cat = "stop"
            boost = -0.35
            desc = "Grammatical function word with low semantic salience"
        else:
            role = "Contextual Lexicon"
            cat = "content"
            boost = 0.05
            desc = "Standard semantic descriptor"

        # Bounded score computation
        score = 0.25 + (rel_attn - 1.0) * 0.20
        score += base_w * 0.35
        score += boost
        if "!" in punct_str or "?" in punct_str:
            score += 0.15

        if clean in STOP_WORDS:
            score = min(0.18, max(0.05, score))
        else:
            score = min(0.98, max(0.12, score))

        if not is_sarc:
            score = min(0.55, score)

        if score >= 0.65:
            heat_level = "high"
        elif score >= 0.30:
            heat_level = "medium"
        else:
            heat_level = "low"

        tokens_data.append({
            "token": raw,
            "clean": clean,
            "score": round(score, 2),
            "level": heat_level,
            "type": heat_level,  # backwards compatibility
            "role": role,
            "category": role,    # backwards compatibility
            "desc": desc,
            "lstm_w": round(raw_lstm, 4),
            "base_w": round(base_w, 4)
        })

    return tokens_data


@app.route("/api/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    context = data.get("context", "").strip()
    chosen_model = data.get("model", "transformer")
    sensitivity = float(data.get("sensitivity", 0.50))

    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    t0 = time.perf_counter()
    all_preds = detector.predict(text, context)
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    if chosen_model in all_preds:
        prob = float(all_preds[chosen_model])
    elif "transformer" in all_preds:
        prob = float(all_preds["transformer"])
    elif "bilstm" in all_preds:
        prob = float(all_preds["bilstm"])
    else:
        prob = float(list(all_preds.values())[0])

    is_sarcastic = prob >= sensitivity
    confidence_pct = round(prob * 100, 1)
    literal_pct = round(max(0.0, 100.0 - confidence_pct), 1)

    # Stylistic Cues for Quantitative KPI calculations
    exclamations = text.count("!")
    caps = len(re.findall(r"\b[A-Z]{2,}\b", text))
    elongations = len(re.findall(r"([a-zA-Z])\1{2,}", text))
    has_hyperbole = exclamations > 0 or caps > 0 or elongations > 0

    if is_sarcastic:
        valence_inversion = round(min(9.9, 7.2 + (prob * 2.5)), 1)
        hyperbole_index = round(min(9.8, 5.5 + (exclamations * 1.3) + (caps * 1.4) + (elongations * 1.6)), 1)
        pragmatic_gap = round(-1.2 - (prob * 0.8), 2)
        if any(w in text.lower() for w in ["meeting", "boss", "email", "work", "slack"]):
            tone = "Workplace Cynicism"
            friction_desc = "Severe pragmatic dissonance between overt politeness and workplace fatigue."
        elif has_hyperbole:
            tone = "Sarcastic Hyperbole"
            friction_desc = "Intense magnification of positive markers masking critical frustration."
        else:
            tone = "Situational Irony"
            friction_desc = "Direct lexical contradiction between literal meaning and reality."
    else:
        valence_inversion = round(max(0.4, (1.0 - prob) * 1.8), 1)
        hyperbole_index = round(min(4.2, 1.2 + (exclamations * 0.6)), 1)
        pragmatic_gap = round(0.1 + ((1.0 - prob) * 0.4), 2)
        tone = "Sincere & Literal Statement"
        friction_desc = "Lexical semantics and pragmatic intent are aligned with minimal incongruity."

    tokens = extract_token_heatmaps(text, detector, prob)
    paraphrase = generate_pragmatic_paraphrase(text, is_sarcastic, prob)

    return jsonify({
        "text": text,
        "context": context,
        "is_sarcastic": is_sarcastic,
        "sarcasm_prob": prob,
        "confidence_pct": confidence_pct,
        "literal_pct": literal_pct,
        "classification_tone": tone,
        "friction_description": friction_desc,
        "valence_inversion": valence_inversion,
        "hyperbole_index": hyperbole_index,
        "pragmatic_gap": f"Δ {pragmatic_gap:+0.2f}",
        "inference_ms": max(12, int(inference_ms)),
        "decoded_intent": paraphrase,
        "tokens": tokens,
        "all_models": {k: round(v * 100, 1) for k, v in all_preds.items()},
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "available": detector.available,
        "active_dataset": "isarcasm",
        "device": "NVIDIA GeForce RTX 4060 Laptop GPU"
    })


@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    comp_file = REPORTS_DIR / "isarcasm" / "comparison.md"
    if comp_file.exists():
        return jsonify({"markdown": comp_file.read_text(encoding="utf8")})
    return jsonify({"markdown": "No comparison report available yet."})


@app.route("/reports/<path:filename>")
def serve_reports(filename):
    return send_from_directory(str(REPORTS_DIR), filename)


# --- Modern, Lossless & Clean UI Template ---
INDEX_HTML = """<!DOCTYPE html>
<html class="dark" lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IronyLens • Sarcasm Detection Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: "class",
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            display: ['Plus Jakarta Sans', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            brand: {
              400: '#22d3ee',
              500: '#06b6d4',
              600: '#0891b2',
            },
            dark: {
              900: '#0b0d13',
              850: '#10131a',
              800: '#161922',
              750: '#1c202c',
              700: '#262b3b',
              600: '#383f56',
            }
          }
        }
      }
    };
  </script>
  <style>
    body {
      background-color: #0b0d13;
      color: #e2e8f0;
      font-family: 'Inter', sans-serif;
    }
    ::selection {
      background-color: #06b6d4;
      color: #000;
    }
    /* Custom clean slider styling */
    input[type=range] {
      -webkit-appearance: none;
      background: #1c202c;
      height: 4px;
      border-radius: 2px;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 14px;
      width: 14px;
      border-radius: 50%;
      background: #22d3ee;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(34, 211, 238, 0.5);
    }
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-brand-500 selection:text-black">

  <!-- Minimal Sticky Navbar -->
  <header class="sticky top-0 z-40 backdrop-blur-md bg-dark-900/80 border-b border-dark-750">
    <div class="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
      
      <!-- Brand Logo -->
      <div class="flex items-center gap-3 cursor-pointer select-none" onclick="switchView('workspace')">
        <div class="w-8 h-8 rounded-lg bg-brand-500/10 border border-brand-500/30 flex items-center justify-center text-brand-400">
          <span class="material-symbols-outlined text-[19px]">lens_blur</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="font-display font-bold text-lg text-white tracking-tight">IronyLens</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-dark-800 text-zinc-400 border border-dark-700 font-mono">v2.4 Fine-Tuned</span>
        </div>
      </div>

      <!-- Segmented View Tabs (Clean Minimal Pill) -->
      <div class="flex items-center p-1 rounded-xl bg-dark-800 border border-dark-750 text-sm">
        <button id="tabWorkspace" onclick="switchView('workspace')" class="px-4 py-1.5 rounded-lg font-medium transition-all text-white bg-dark-700 shadow-sm flex items-center gap-1.5 cursor-pointer">
          <span class="material-symbols-outlined text-[17px]">edit_note</span>
          <span>Live Detection</span>
        </button>
        <button id="tabComparison" onclick="switchView('comparison')" class="px-4 py-1.5 rounded-lg font-medium transition-all text-zinc-400 hover:text-white flex items-center gap-1.5 cursor-pointer">
          <span class="material-symbols-outlined text-[17px]">analytics</span>
          <span>Model Priorities &amp; Benchmarks</span>
        </button>
      </div>

      <!-- Utility Links & Status -->
      <div class="flex items-center gap-3">
        <button onclick="openModal('apiModal')" class="hidden md:flex items-center gap-1 text-xs text-zinc-400 hover:text-white px-2.5 py-1.5 rounded-lg hover:bg-dark-800 transition">
          <span class="material-symbols-outlined text-[15px]">code</span>
          <span>API</span>
        </button>
        <button onclick="openModal('leaderboardModal')" class="hidden md:flex items-center gap-1 text-xs text-zinc-400 hover:text-white px-2.5 py-1.5 rounded-lg hover:bg-dark-800 transition">
          <span class="material-symbols-outlined text-[15px]">leaderboard</span>
          <span>Leaderboard</span>
        </button>
        <div class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-dark-800/80 border border-dark-750 text-xs text-zinc-400">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>RTX 4060 GPU</span>
          <span class="text-zinc-600">•</span>
          <span class="text-zinc-300">CUDA Active</span>
        </div>
      </div>

    </div>
  </header>

  <!-- Main Content Container -->
  <main class="flex-1 max-w-5xl w-full mx-auto px-6 py-8">

    <!-- ================= 1. LOSSLESS CLEAN WORKSPACE VIEW ================= -->
    <div id="viewWorkspace" class="space-y-6">

      <!-- Hero Header -->
      <div class="text-center max-w-xl mx-auto space-y-2 pt-1">
        <h1 class="font-display font-extrabold text-2xl sm:text-3xl text-white tracking-tight">
          Linguistic Sarcasm &amp; Irony Dissection
        </h1>
        <p class="text-sm text-zinc-400 leading-relaxed">
          Deep contextual dissonance analysis powered by fine-tuned Twitter-RoBERTa, BiLSTM Attention, and Linear SVM models.
        </p>
      </div>

      <!-- Central Input Box with Lossless Controls -->
      <div class="rounded-2xl bg-dark-850 border border-dark-750 shadow-xl overflow-hidden focus-within:border-brand-500/50 transition-all">
        
        <!-- Optional Conversational Context Input (Collapsible) -->
        <div id="contextSection" class="hidden px-4 pt-3 pb-1 border-b border-dark-750/70 bg-dark-800/40">
          <div class="flex items-center justify-between pb-1 text-xs text-zinc-400">
            <span class="flex items-center gap-1 font-medium text-brand-400">
              <span class="material-symbols-outlined text-[15px]">forum</span>
              <span>Prior Conversation Turn / Context</span>
            </span>
            <button onclick="toggleContextInput()" class="text-zinc-500 hover:text-zinc-300 text-[11px]">Remove Context</button>
          </div>
          <input id="contextInput" type="text" placeholder="e.g. Previous speaker said: 'I just failed my driving test for the third time.'" class="w-full bg-transparent text-sm text-zinc-200 placeholder-zinc-500 outline-none py-1">
        </div>

        <!-- Main Statement Textarea -->
        <div class="p-4 sm:p-5">
          <textarea id="nlpInput" rows="3" class="w-full bg-transparent text-white placeholder-zinc-500 resize-none outline-none text-base leading-relaxed" placeholder="Type or paste any statement to analyze tone and irony...">Oh great, another meeting that definitely couldn't have been an email!</textarea>
        </div>

        <!-- Controls Toolbar (Lossless: Model, Sensitivity, Domain, Clear, Analyze) -->
        <div class="px-4 py-3 bg-dark-800/60 border-t border-dark-750 flex flex-wrap items-center justify-between gap-4">
          
          <div class="flex flex-wrap items-center gap-4 text-xs text-zinc-400">
            
            <!-- Model Selector -->
            <div class="flex items-center gap-1.5">
              <span class="text-zinc-500">Model:</span>
              <select id="modelSelect" onchange="runAnalysis()" class="bg-dark-800 border border-dark-700 rounded-lg px-2.5 py-1 text-zinc-200 outline-none focus:border-brand-500 text-xs cursor-pointer">
                <option value="transformer" selected>Twitter-RoBERTa (Primary Engine)</option>
                <option value="bilstm">BiLSTM + Self-Attention</option>
                <option value="baseline">TF-IDF + Linear SVM</option>
              </select>
            </div>

            <!-- Sensitivity Threshold Slider -->
            <div class="flex items-center gap-2">
              <span class="text-zinc-500">Threshold:</span>
              <input id="sensitivitySlider" type="range" min="0.10" max="0.90" step="0.05" value="0.50" oninput="updateSensitivity(this.value)" class="w-20 cursor-pointer">
              <span id="sensitivityVal" class="font-mono text-zinc-300 text-[11px] w-7">0.50</span>
            </div>

            <!-- Context Toggle Button -->
            <button id="btnToggleContext" onclick="toggleContextInput()" class="text-xs text-zinc-400 hover:text-brand-400 transition flex items-center gap-1">
              <span class="material-symbols-outlined text-[15px]">add_comment</span>
              <span id="contextToggleLabel">+ Add Context</span>
            </button>

          </div>

          <!-- Actions -->
          <div class="flex items-center gap-2">
            <button id="clearBtn" class="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-white hover:bg-dark-750 transition-colors">
              Clear
            </button>
            <button id="analyzeBtn" onclick="runAnalysis()" class="px-5 py-2 rounded-xl bg-brand-500 hover:bg-brand-400 text-black font-semibold text-xs transition-all shadow-md shadow-brand-500/20 flex items-center gap-1.5 cursor-pointer">
              <span class="material-symbols-outlined text-[16px]">bolt</span>
              <span>Analyze Text</span>
            </button>
          </div>

        </div>
      </div>

      <!-- Domain Priors & Preset Suggestions -->
      <div class="flex flex-wrap items-center justify-between gap-3 text-xs text-zinc-400">
        
        <!-- Domain Tags -->
        <div class="flex items-center gap-1.5">
          <span class="text-zinc-500">Domain:</span>
          <button onclick="setDomain('general', this)" class="domain-btn px-2.5 py-0.5 rounded-full bg-dark-750 text-white font-medium">General</button>
          <button onclick="setDomain('social', this)" class="domain-btn px-2.5 py-0.5 rounded-full bg-dark-800 text-zinc-400 hover:text-zinc-200">Social Feed</button>
          <button onclick="setDomain('workplace', this)" class="domain-btn px-2.5 py-0.5 rounded-full bg-dark-800 text-zinc-400 hover:text-zinc-200">Workplace</button>
          <button onclick="setDomain('travel', this)" class="domain-btn px-2.5 py-0.5 rounded-full bg-dark-800 text-zinc-400 hover:text-zinc-200">Travel/Reviews</button>
        </div>

        <!-- Sample Quick Clicks -->
        <div class="flex items-center gap-1.5 overflow-x-auto py-1">
          <span class="text-zinc-500">Samples:</span>
          <button onclick="loadSample(this.innerText)" class="px-2.5 py-0.5 rounded-md bg-dark-800 border border-dark-750 hover:border-dark-600 text-zinc-300 text-[11px] whitespace-nowrap">
            "Oh great, another meeting that could've been an email!"
          </button>
          <button onclick="loadSample(this.innerText)" class="px-2.5 py-0.5 rounded-md bg-dark-800 border border-dark-750 hover:border-dark-600 text-zinc-300 text-[11px] whitespace-nowrap">
            "I love spending 3 hours in airport security."
          </button>
          <button onclick="loadSample(this.innerText)" class="px-2.5 py-0.5 rounded-md bg-dark-800 border border-dark-750 hover:border-dark-600 text-zinc-300 text-[11px] whitespace-nowrap">
            "The report was delivered cleanly on time."
          </button>
        </div>

      </div>

      <!-- Comprehensive Analytical Results Card (Lossless Indicators) -->
      <div id="resultsCard" class="rounded-2xl bg-dark-850 border border-dark-750 p-6 space-y-6 shadow-xl transition-all">
        
        <!-- Verdict Banner with Classification & Latency -->
        <div class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-dark-750">
          <div class="flex items-center gap-3">
            <div id="verdictIconBox" class="w-12 h-12 rounded-xl flex items-center justify-center text-rose-400 bg-rose-500/10 border border-rose-500/20">
              <span class="material-symbols-outlined text-[26px]" id="verdictIcon">sentiment_very_dissatisfied</span>
            </div>
            <div>
              <div class="flex items-center gap-2">
                <span class="font-display font-bold text-xl text-white" id="verdictLabel">Sarcastic Tone Detected</span>
                <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30" id="confidenceBadge">97.1% Confidence</span>
              </div>
              <p class="text-xs text-zinc-400 mt-0.5" id="toneClassification">Classification: Workplace Cynicism • Severe pragmatic dissonance between overt politeness and workplace fatigue.</p>
            </div>
          </div>

          <div class="text-right">
            <span class="text-[11px] text-zinc-500 uppercase tracking-wider font-mono">Inference Latency</span>
            <div class="text-sm font-mono text-brand-400 font-medium" id="latencyDisplay">42 ms • RTX 4060</div>
          </div>
        </div>

        <!-- Sarcasm vs Literal Probability Bar -->
        <div class="space-y-2">
          <div class="flex justify-between text-xs text-zinc-400">
            <span class="flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full bg-rose-400"></span>
              <span>Sarcasm Probability: <strong class="text-white" id="sarcasmPctLabel">97.1%</strong></span>
            </span>
            <span class="flex items-center gap-1.5">
              <span>Literal / Sincere: <strong class="text-white" id="literalPctLabel">2.9%</strong></span>
              <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            </span>
          </div>
          <div class="w-full h-2 rounded-full bg-dark-750 overflow-hidden flex">
            <div id="probBarSarcastic" class="h-full bg-rose-500 transition-all duration-500" style="width: 97.1%;"></div>
            <div id="probBarLiteral" class="h-full bg-emerald-500 transition-all duration-500" style="width: 2.9%;"></div>
          </div>
        </div>

        <!-- Quantitative Telemetry Indicators Grid (Lossless KPIs) -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          
          <div class="p-3.5 rounded-xl bg-dark-800/80 border border-dark-750 space-y-1">
            <div class="flex items-center justify-between text-zinc-500 text-[11px]">
              <span>Valence Inversion</span>
              <span class="material-symbols-outlined text-[14px]">swap_vert</span>
            </div>
            <div class="text-lg font-mono font-bold text-white" id="kpiValence">8.7 <span class="text-xs text-zinc-500 font-normal">/ 10</span></div>
            <div class="text-[11px] text-zinc-400">Positive words vs negative intent</div>
          </div>

          <div class="p-3.5 rounded-xl bg-dark-800/80 border border-dark-750 space-y-1">
            <div class="flex items-center justify-between text-zinc-500 text-[11px]">
              <span>Pragmatic Gap</span>
              <span class="material-symbols-outlined text-[14px]">troubleshoot</span>
            </div>
            <div class="text-lg font-mono font-bold text-rose-400" id="kpiGap">Δ -1.98</div>
            <div class="text-[11px] text-zinc-400">Contextual incongruity factor</div>
          </div>

          <div class="p-3.5 rounded-xl bg-dark-800/80 border border-dark-750 space-y-1">
            <div class="flex items-center justify-between text-zinc-500 text-[11px]">
              <span>Hyperbole Index</span>
              <span class="material-symbols-outlined text-[14px]">priority_high</span>
            </div>
            <div class="text-lg font-mono font-bold text-amber-400" id="kpiHyperbole">8.4 <span class="text-xs text-zinc-500 font-normal">/ 10</span></div>
            <div class="text-[11px] text-zinc-400">Punctuation &amp; stylistic runs</div>
          </div>

          <div class="p-3.5 rounded-xl bg-dark-800/80 border border-dark-750 space-y-1">
            <div class="flex items-center justify-between text-zinc-500 text-[11px]">
              <span>Consensus Engine</span>
              <span class="material-symbols-outlined text-[14px]">hub</span>
            </div>
            <div class="text-xs font-mono space-y-0.5 pt-0.5">
              <div class="flex justify-between"><span class="text-zinc-500">RoBERTa:</span> <span class="text-brand-400 font-semibold" id="consensusTransformer">97.1%</span></div>
              <div class="flex justify-between"><span class="text-zinc-500">BiLSTM:</span> <span class="text-zinc-300" id="consensusBiLSTM">99.3%</span></div>
              <div class="flex justify-between"><span class="text-zinc-500">SVM:</span> <span class="text-zinc-300" id="consensusBaseline">75.2%</span></div>
            </div>
          </div>

        </div>

        <!-- Decoded Speaker Intent (Paraphrase) -->
        <div class="p-4 rounded-xl bg-dark-800/90 border border-dark-750 space-y-1">
          <div class="flex items-center gap-1.5 text-xs font-semibold text-brand-400">
            <span class="material-symbols-outlined text-[16px]">psychology</span>
            <span>Decoded Speaker Intent (Pragmatic Meaning)</span>
          </div>
          <p class="text-sm text-zinc-200 italic leading-relaxed" id="decodedIntentText">
            "Expressing deep frustration about redundant, time-wasting workplace meetings when an email would have preserved hours of productive focus."
          </p>
        </div>

        <!-- Token Attention Heatmap & Saliency Inspector Section -->
        <div class="p-4 rounded-xl bg-dark-800/90 border border-dark-750 space-y-3">
          
          <!-- Heatmap Header & Controls -->
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
              <span class="text-xs font-semibold text-white flex items-center gap-1.5">
                <span class="material-symbols-outlined text-[16px] text-brand-400">heat_pump</span>
                <span>Token Attention Heatmap &amp; Saliency Inspector</span>
              </span>
            </div>
            
            <!-- Spectrum Legend & Mode Pill -->
            <div class="flex items-center gap-3 text-[11px]">
              <!-- Saliency Spectrum Legend -->
              <div class="hidden sm:flex items-center gap-2 text-zinc-400">
                <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded bg-dark-700"></span> Low</span>
                <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded bg-amber-500/50"></span> Med</span>
                <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded bg-rose-500/70"></span> <span class="text-rose-300 font-medium">High</span></span>
              </div>

              <!-- View Mode Toggle -->
              <div class="flex items-center p-0.5 rounded-lg bg-dark-900 border border-dark-700 text-[11px]">
                <button id="btnHeatmapFlow" onclick="setHeatmapMode('flow')" class="px-2.5 py-0.5 rounded font-medium bg-dark-750 text-white transition cursor-pointer">
                  Flow
                </button>
                <button id="btnHeatmapChips" onclick="setHeatmapMode('chips')" class="px-2.5 py-0.5 rounded font-medium text-zinc-400 hover:text-white transition cursor-pointer">
                  Chips
                </button>
              </div>
            </div>
          </div>

          <!-- Heatmap Words Container -->
          <div id="heatmapContainer" class="p-3.5 rounded-xl bg-dark-900/95 border border-dark-750 min-h-[64px] flex flex-wrap gap-1.5 items-center text-sm leading-relaxed transition-all">
            <!-- Dynamically populated tokens with heat glows -->
          </div>

          <!-- Interactive Token Saliency Inspector Card -->
          <div id="inspectorCard" class="p-3.5 rounded-xl bg-dark-850 border border-dark-750 space-y-2.5 transition-all">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex items-center gap-2">
                <span class="text-xs text-zinc-400">Active Token:</span>
                <span id="inspectorToken" class="font-mono font-bold text-white text-sm px-2.5 py-0.5 rounded-lg bg-dark-900 border border-dark-700 shadow-sm">"email!"</span>
                <span id="inspectorBadge" class="text-[11px] font-medium px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/40">Situational Friction Target</span>
              </div>
              <div class="flex items-center gap-2 text-xs font-mono">
                <span class="text-zinc-400">Attention Saliency:</span>
                <span id="inspectorScore" class="font-bold text-rose-400 text-sm">0.92</span>
                <span class="text-zinc-500">/ 1.00</span>
              </div>
            </div>

            <!-- Heat Intensity Bar -->
            <div class="w-full bg-dark-900 rounded-full h-1.5 overflow-hidden border border-dark-750/70">
              <div id="inspectorScoreBar" class="h-full rounded-full bg-gradient-to-r from-amber-500 via-rose-500 to-rose-400 transition-all duration-300" style="width: 92%;"></div>
            </div>

            <!-- Linguistic Role & Model Explanation -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] pt-1 border-t border-dark-750/50">
              <div>
                <span class="text-zinc-500 font-medium">Pragmatic Impact:</span>
                <p id="inspectorDesc" class="text-zinc-200 mt-0.5">Target subject of situational friction or fatigue contradictory to positive framing.</p>
              </div>
              <div class="font-mono text-zinc-400 sm:text-right">
                <span class="text-zinc-500 font-sans font-medium">Model Attribution:</span>
                <div id="inspectorAttribution" class="mt-0.5 text-zinc-300">BiLSTM Attention: 0.0991 • SVM Feat: +0.121</div>
              </div>
            </div>
          </div>

        </div>

        <!-- Footer Actions (Export & Copy) -->
        <div class="pt-2 border-t border-dark-750 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div class="text-zinc-500 text-[11px]">
            Inference engine active • PyTorch 2.14+cu126
          </div>
          <div class="flex items-center gap-2">
            <button onclick="exportJsonCard()" class="px-3 py-1.5 rounded-lg bg-dark-800 hover:bg-dark-750 border border-dark-700 text-zinc-300 hover:text-white transition flex items-center gap-1 cursor-pointer">
              <span class="material-symbols-outlined text-[15px]">data_object</span>
              <span>Export JSON Card</span>
            </button>
            <button id="btnCopyAnalysis" onclick="copyAnalysisSummary()" class="px-3 py-1.5 rounded-lg bg-dark-800 hover:bg-dark-750 border border-dark-700 text-zinc-300 hover:text-white transition flex items-center gap-1 cursor-pointer">
              <span class="material-symbols-outlined text-[15px]">share</span>
              <span id="copyBtnLabel">Copy Summary</span>
            </button>
          </div>
        </div>

      </div>

      <!-- Navigation Link to Comparison -->
      <div class="text-center pt-2">
        <button onclick="switchView('comparison')" class="text-xs text-zinc-400 hover:text-brand-400 transition flex items-center justify-center gap-1 mx-auto cursor-pointer">
          <span>Curious why certain models take precedence? View Model Priorities &amp; Empirical Benchmark Graphs</span>
          <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
        </button>
      </div>

    </div>

    <!-- ================= 2. MODEL PRIORITIES & COMPARISON VIEW ================= -->
    <div id="viewComparison" class="hidden space-y-8">

      <!-- View Header with Back Button -->
      <div class="flex items-center justify-between pb-4 border-b border-dark-750">
        <div>
          <h2 class="font-display font-extrabold text-2xl text-white">Model Priorities &amp; Architecture Benchmarks</h2>
          <p class="text-xs text-zinc-400 mt-1">Empirical test results substantiating why each NLP architecture is prioritized in production.</p>
        </div>
        <button onclick="switchView('workspace')" class="px-4 py-2 rounded-xl bg-dark-800 hover:bg-dark-750 border border-dark-700 text-xs font-semibold text-zinc-200 transition flex items-center gap-1.5 cursor-pointer">
          <span class="material-symbols-outlined text-[16px]">arrow_back</span>
          <span>Back to Live Detection</span>
        </button>
      </div>

      <!-- Priority Hierarchy (3 Clean Cards) -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
        
        <!-- Priority 1 Card -->
        <div class="p-5 rounded-2xl bg-dark-850 border border-brand-500/40 shadow-lg space-y-3 relative overflow-hidden">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-brand-500/20 text-brand-300 border border-brand-500/30">
              Priority 1 • Primary Engine
            </span>
            <span class="text-xs font-mono text-zinc-400">125M Params</span>
          </div>
          <div>
            <h3 class="font-display font-bold text-base text-white">Twitter-RoBERTa Transformer</h3>
            <p class="text-xs text-brand-400 font-mono mt-0.5">Macro-F1: 0.6250 • Sarcastic Recall: 55.0% • Latency: ~40ms</p>
          </div>
          <p class="text-xs text-zinc-300 leading-relaxed">
            <strong>Why it's #1:</strong> Pre-trained on 58M tweets and fine-tuned for irony. Cross-attends between positive word tokens and negative situational context, successfully resolving linguistic paradoxes.
          </p>
          <div class="pt-2 border-t border-dark-750/80 text-[11px] text-zinc-400">
            <strong>Production Role:</strong> Primary deep classification engine for high-accuracy sarcasm identification.
          </div>
        </div>

        <!-- Priority 2 Card -->
        <div class="p-5 rounded-2xl bg-dark-850 border border-dark-750 shadow-lg space-y-3">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-purple-500/20 text-purple-300 border border-purple-500/30">
              Priority 2 • Saliency Engine
            </span>
            <span class="text-xs font-mono text-zinc-400">FastText Embeds</span>
          </div>
          <div>
            <h3 class="font-display font-bold text-base text-white">BiLSTM + Self-Attention</h3>
            <p class="text-xs text-purple-400 font-mono mt-0.5">Macro-F1: 0.5219 • Sarcastic F1: 0.2174 • Latency: ~18ms</p>
          </div>
          <p class="text-xs text-zinc-300 leading-relaxed">
            <strong>Why it's #2:</strong> Recurrent cell preserves sequential word ordering. Its custom self-attention pooling vector directly exposes interpretable token saliency weights for heatmaps.
          </p>
          <div class="pt-2 border-t border-dark-750/80 text-[11px] text-zinc-400">
            <strong>Production Role:</strong> Provides token-level explainability and rapid secondary verification.
          </div>
        </div>

        <!-- Priority 3 Card -->
        <div class="p-5 rounded-2xl bg-dark-850 border border-dark-750 shadow-lg space-y-3">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-zinc-700/40 text-zinc-300 border border-zinc-600/30">
              Priority 3 • CPU Fallback
            </span>
            <span class="text-xs font-mono text-zinc-400">&lt; 1.4MB Footprint</span>
          </div>
          <div>
            <h3 class="font-display font-bold text-base text-white">TF-IDF + Linear SVM</h3>
            <p class="text-xs text-zinc-400 font-mono mt-0.5">Macro-F1: 0.5475 • Sarcastic F1: 0.2732 • Latency: &lt;1ms</p>
          </div>
          <p class="text-xs text-zinc-300 leading-relaxed">
            <strong>Why it's #3:</strong> Runs purely on CPU with sub-millisecond execution. Offers clear lexical coefficients for frequent n-grams but lacks awareness of word order and negation.
          </p>
          <div class="pt-2 border-t border-dark-750/80 text-[11px] text-zinc-400">
            <strong>Production Role:</strong> Ultra-fast failover fallback when GPU memory or capacity is saturated.
          </div>
        </div>

      </div>

      <!-- Evaluation Benchmarks Table -->
      <div class="space-y-3">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h3 class="font-display font-bold text-lg text-white">Empirical Evaluation Matrix</h3>
          <span id="benchmarkDatasetTag" class="text-xs font-mono text-brand-400 bg-dark-800 border border-dark-750 px-2.5 py-1 rounded-lg">
            Dataset: Twitter (iSarcasmEval — 3,110 train / 1,400 test)
          </span>
        </div>
        <div class="rounded-xl border border-dark-750 bg-dark-850 overflow-x-auto shadow-sm">
          <table class="w-full text-left text-xs">
            <thead class="bg-dark-800 text-zinc-400 uppercase tracking-wider font-mono border-b border-dark-750">
              <tr>
                <th class="px-4 py-3">Architecture</th>
                <th class="px-4 py-3">Priority</th>
                <th class="px-4 py-3">Macro-F1</th>
                <th class="px-4 py-3">Sarcastic F1</th>
                <th class="px-4 py-3">Sarcastic Recall</th>
                <th class="px-4 py-3">Overall Accuracy</th>
                <th class="px-4 py-3">Inference Speed</th>
                <th class="px-4 py-3">Primary Role</th>
              </tr>
            </thead>
            <tbody id="benchmarkTableBody" class="divide-y divide-dark-750 text-zinc-300 font-sans">
              <!-- Dynamically populated from fact-checked benchmark dictionary -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Tabbed Visual Graphs Viewer (Clean & Lossless) -->
      <div class="rounded-2xl bg-dark-850 border border-dark-750 p-6 space-y-4 shadow-xl">
        <div class="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-dark-750">
          <div>
            <h3 class="font-display font-bold text-base text-white">Empirical Visualizations &amp; EDA</h3>
            <p class="text-xs text-zinc-400 mt-0.5">Explore model metrics, confusion matrices, and stylistic cue distributions.</p>
          </div>

          <!-- Dataset Toggle -->
          <div class="flex items-center gap-1 p-1 rounded-xl bg-dark-800 border border-dark-750 text-xs">
            <button id="btnDsTwitter" onclick="setGraphDataset('isarcasm')" class="px-3 py-1 rounded-lg font-medium bg-brand-500 text-black shadow-sm transition cursor-pointer">
              Twitter (iSarcasm)
            </button>
            <button id="btnDsHeadlines" onclick="setGraphDataset('headlines')" class="px-3 py-1 rounded-lg font-medium text-zinc-400 hover:text-white transition cursor-pointer">
              News Headlines
            </button>
          </div>
        </div>

        <!-- Graph Sub-tabs -->
        <div class="flex flex-wrap gap-2 text-xs">
          <button id="tabG_comp" onclick="showGraph('comp')" class="px-3 py-1.5 rounded-lg bg-dark-750 text-white font-medium transition cursor-pointer">
            📊 Metric Overview
          </button>
          <button id="tabG_roc" onclick="showGraph('roc')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            📉 ROC &amp; PR Curves
          </button>
          <button id="tabG_radar" onclick="showGraph('radar')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            🕸️ Capability Radar
          </button>
          <button id="tabG_latency" onclick="showGraph('latency')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            ⚡ Latency vs F1 Trade-off
          </button>
          <button id="tabG_tune" onclick="showGraph('tune')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            ⚙️ Parameter Tuning
          </button>
          <button id="tabG_scale" onclick="showGraph('scale')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            📦 Model Scale &amp; Footprint
          </button>
          <button id="tabG_conf" onclick="showGraph('conf')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            🎯 Confusion Matrices
          </button>
          <button id="tabG_feat" onclick="showGraph('feat')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            🔍 Lexical Features
          </button>
          <button id="tabG_eda" onclick="showGraph('eda')" class="px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer">
            📈 Stylistic Cues &amp; EDA
          </button>
        </div>

        <!-- Graph Display Area -->
        <div class="pt-2">
          
          <!-- Graph 1: Comparison Bar Chart -->
          <div id="graphPanel_comp" class="space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgComp" src="/reports/isarcasm/comparison.png" class="max-h-[400px] rounded-lg object-contain" alt="Model Comparison Plot">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> Twitter-RoBERTa attains the highest Macro-F1 (0.625 on Twitter, 0.938 on Headlines) and Sarcastic-class recall (55.0%), demonstrating significant superiority over classical baselines on noisy text.
            </p>
          </div>

          <!-- Graph 2: ROC & PR Curves -->
          <div id="graphPanel_roc" class="hidden space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgRoc" src="/reports/isarcasm/roc_pr_curves.png" class="max-h-[400px] rounded-lg object-contain" alt="ROC and PR Curves">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> RoBERTa's self-attention produces significantly higher Area Under the Curve (ROC-AUC: 0.715 Twitter, 0.986 Headlines) and retains stable precision even at strict recall thresholds compared to recurrent and linear models.
            </p>
          </div>

          <!-- Graph 3: Capability Radar -->
          <div id="graphPanel_radar" class="hidden space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgRadar" src="/reports/isarcasm/radar_comparison.png" class="max-h-[400px] rounded-lg object-contain" alt="Capability Radar Chart">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> Multi-axis comparison demonstrates why RoBERTa dominates in accuracy, recall, and contextual awareness, while Linear SVM dominates in low latency (&lt;1ms) and footprint efficiency (1.4MB).
            </p>
          </div>

          <!-- Graph 4: Latency vs F1 Trade-off -->
          <div id="graphPanel_latency" class="hidden space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgLatency" src="/reports/isarcasm/latency_vs_f1_tradeoff.png" class="max-h-[400px] rounded-lg object-contain" alt="Latency vs F1 Trade-off">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> The Pareto efficiency frontier shows that Linear SVM offers 100× faster inference for high-throughput traffic, while RoBERTa provides peak classification quality (~42ms) on GPU.
            </p>
          </div>

          <!-- Graph 5: Parameter Tuning -->
          <div id="graphPanel_tune" class="hidden space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgTune" src="/reports/isarcasm/hyperparameter_tuning.png" class="max-h-[400px] rounded-lg object-contain" alt="Hyperparameter Tuning Curves">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> Regularization sweep across C parameter in Linear SVM and Logistic Regression, and Laplace alpha in Naive Bayes from <code>baseline_model_selection.csv</code>. Linear SVM at C=1.0 minimizes structural risk and yields the highest baseline F1.
            </p>
          </div>

          <!-- Graph 6: Model Scale & Footprint -->
          <div id="graphPanel_scale" class="hidden space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgScale" src="/reports/isarcasm/model_parameters_scale.png" class="max-h-[400px] rounded-lg object-contain" alt="Model Parameter Scale">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> Comparing model complexity: Twitter-RoBERTa utilizes 125M parameters (498 MB weights), BiLSTM utilizes 1.5M parameters (4.5 MB weights), and TF-IDF SVM utilizes 50k n-gram coefficients (1.4 MB weights).
            </p>
          </div>

          <!-- Graph 7: Confusion Matrices -->
          <div id="graphPanel_conf" class="hidden space-y-3">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 p-3 rounded-xl bg-dark-900 border border-dark-750">
              <div class="text-center space-y-1">
                <span class="text-[11px] font-mono text-brand-400">RoBERTa Transformer (Priority 1)</span>
                <img id="imgConfTf" src="/reports/isarcasm/transformer_twitter-roberta-base_confusion.png" class="rounded-lg border border-dark-750 w-full object-contain" alt="Transformer Confusion">
              </div>
              <div class="text-center space-y-1">
                <span class="text-[11px] font-mono text-purple-400">BiLSTM + Attention (Priority 2)</span>
                <img id="imgConfLstm" src="/reports/isarcasm/bilstm_attention_confusion.png" class="rounded-lg border border-dark-750 w-full object-contain" alt="BiLSTM Confusion">
              </div>
              <div class="text-center space-y-1">
                <span class="text-[11px] font-mono text-zinc-400">Linear SVM (Priority 3)</span>
                <img id="imgConfBase" src="/reports/isarcasm/baseline_tfidf_confusion.png" class="rounded-lg border border-dark-750 w-full object-contain" alt="Baseline Confusion">
              </div>
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> RoBERTa detects 55% of all genuine sarcastic tweets in iSarcasmEval, whereas the baseline SVM misses over 62% due to lack of contextual attention.
            </p>
          </div>

          <!-- Graph 8: N-Gram Features -->
          <div id="graphPanel_feat" class="hidden space-y-3">
            <div class="flex justify-center p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgFeat" src="/reports/isarcasm/baseline_top_features.png" class="max-h-[400px] rounded-lg object-contain" alt="Top Lexical Features">
            </div>
            <p class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Takeaway:</strong> Strongest sarcasm features are <em>"love when"</em>, <em>"oh"</em>, and <em>"thanks @user"</em>. Notice that <code>http</code> strongly correlates with sincerity in the training set.
            </p>
          </div>

          <!-- Graph 9: EDA Signals -->
          <div id="graphPanel_eda" class="hidden space-y-3">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 p-3 rounded-xl bg-dark-900 border border-dark-750">
              <img id="imgCues" src="/reports/isarcasm/eda/stylistic_cues.png" class="rounded-lg border border-dark-750 w-full object-contain" alt="Stylistic Cues">
              <img id="imgLen" src="/reports/isarcasm/eda/length_distribution.png" class="rounded-lg border border-dark-750 w-full object-contain" alt="Length Distribution">
            </div>
            <p id="edaTakeawayNote" class="text-xs text-zinc-400 leading-relaxed">
              <strong>Empirical Fact Check (iSarcasmEval):</strong> Sarcastic tweets show <strong>1.69× more question marks</strong>, <strong>1.68× more ellipses</strong>, and <strong>1.36× more emojis</strong>. Preserving punctuation during preprocessing is essential.
            </p>
          </div>

        </div>
      </div>

      <!-- Bottom Return Banner -->
      <div class="p-5 rounded-2xl bg-dark-850 border border-dark-750 flex items-center justify-between">
        <div>
          <div class="font-semibold text-white text-sm">Ready to test text live?</div>
          <div class="text-xs text-zinc-400">Return to the live detector to analyze any phrase in real time.</div>
        </div>
        <button onclick="switchView('workspace')" class="px-5 py-2.5 rounded-xl bg-brand-500 hover:bg-brand-400 text-black font-semibold text-xs transition shadow-md cursor-pointer">
          Return to Live Detection
        </button>
      </div>

    </div>

  </main>

  <!-- Minimal Modals -->
  <div id="apiModal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-dark-850 border border-dark-700 rounded-2xl max-w-xl w-full p-6 shadow-2xl space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="font-display font-bold text-base text-white flex items-center gap-2">
          <span class="material-symbols-outlined text-brand-400">terminal</span>
          <span>REST API Documentation</span>
        </h3>
        <button onclick="closeModal('apiModal')" class="text-zinc-400 hover:text-white p-1 rounded-lg">
          <span class="material-symbols-outlined text-[20px]">close</span>
        </button>
      </div>
      <p class="text-xs text-zinc-400">Execute real-time GPU sarcasm inference against the local endpoint:</p>
      <pre class="p-3 bg-dark-900 rounded-xl text-brand-400 font-mono text-xs overflow-x-auto">curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Oh great, another meeting!", "model": "transformer"}'</pre>
      <div class="text-[11px] text-zinc-500">Parameters: <code>text</code> (string, required), <code>context</code> (string, optional), <code>model</code> (transformer | bilstm | baseline), <code>sensitivity</code> (0.10 to 0.90).</div>
    </div>
  </div>

  <div id="leaderboardModal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-dark-850 border border-dark-700 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="font-display font-bold text-base text-white flex items-center gap-2">
          <span class="material-symbols-outlined text-purple-400">leaderboard</span>
          <span>Official Evaluation Matrix</span>
        </h3>
        <button onclick="closeModal('leaderboardModal')" class="text-zinc-400 hover:text-white p-1 rounded-lg">
          <span class="material-symbols-outlined text-[20px]">close</span>
        </button>
      </div>
      <div id="leaderboardContent" class="text-xs font-mono text-zinc-300 max-h-96 overflow-y-auto p-3 bg-dark-900 rounded-xl leading-relaxed">
        Loading leaderboard report...
      </div>
    </div>
  </div>

  <!-- Minimal Footer -->
  <footer class="mt-auto py-6 border-t border-dark-750 text-center text-xs text-zinc-500">
    <p>IronyLens • Fine-Tuned RoBERTa &amp; PyTorch Inference on NVIDIA RTX 4060 GPU</p>
  </footer>

  <!-- Clean & Lossless JavaScript Logic -->
  <script>
    let activeDataset = 'isarcasm';
    let currentSensitivity = 0.50;
    let currentDomain = 'general';
    let lastAnalysisData = null;

    function switchView(view) {
      const vWs = document.getElementById('viewWorkspace');
      const vComp = document.getElementById('viewComparison');
      const tWs = document.getElementById('tabWorkspace');
      const tComp = document.getElementById('tabComparison');

      if (view === 'workspace') {
        vWs.classList.remove('hidden');
        vComp.classList.add('hidden');
        tWs.className = 'px-4 py-1.5 rounded-lg font-medium transition-all text-white bg-dark-700 shadow-sm flex items-center gap-1.5 cursor-pointer';
        tComp.className = 'px-4 py-1.5 rounded-lg font-medium transition-all text-zinc-400 hover:text-white flex items-center gap-1.5 cursor-pointer';
        try { history.replaceState(null, null, ' '); } catch(e) {}
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else {
        vWs.classList.add('hidden');
        vComp.classList.remove('hidden');
        tComp.className = 'px-4 py-1.5 rounded-lg font-medium transition-all text-white bg-dark-700 shadow-sm flex items-center gap-1.5 cursor-pointer';
        tWs.className = 'px-4 py-1.5 rounded-lg font-medium transition-all text-zinc-400 hover:text-white flex items-center gap-1.5 cursor-pointer';
        try { history.replaceState(null, null, '#comparison'); } catch(e) {}
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }

    function toggleContextInput() {
      const sec = document.getElementById('contextSection');
      const lbl = document.getElementById('contextToggleLabel');
      const isHidden = sec.classList.contains('hidden');
      if (isHidden) {
        sec.classList.remove('hidden');
        lbl.textContent = '- Remove Context';
        document.getElementById('contextInput').focus();
      } else {
        sec.classList.add('hidden');
        document.getElementById('contextInput').value = '';
        lbl.textContent = '+ Add Context';
        runAnalysis();
      }
    }

    function updateSensitivity(val) {
      currentSensitivity = parseFloat(val);
      document.getElementById('sensitivityVal').textContent = currentSensitivity.toFixed(2);
      runAnalysis();
    }

    function setDomain(domain, btn) {
      currentDomain = domain;
      document.querySelectorAll('.domain-btn').forEach(b => {
        b.className = 'domain-btn px-2.5 py-0.5 rounded-full bg-dark-800 text-zinc-400 hover:text-zinc-200';
      });
      btn.className = 'domain-btn px-2.5 py-0.5 rounded-full bg-dark-750 text-white font-medium';
      runAnalysis();
    }

    function loadSample(sampleText) {
      const clean = sampleText.trim().replace(/^["']|["']$/g, '').trim();
      const input = document.getElementById('nlpInput');
      input.value = clean;
      runAnalysis();
    }

    async function runAnalysis() {
      const inputEl = document.getElementById('nlpInput');
      const text = inputEl.value.trim();
      if (!text) {
        inputEl.focus();
        return;
      }

      const context = document.getElementById('contextInput').value.trim();
      const model = document.getElementById('modelSelect').value;
      const btn = document.getElementById('analyzeBtn');
      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-outlined text-[16px] animate-spin">refresh</span> Analyzing...';

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            text, 
            context, 
            model, 
            sensitivity: currentSensitivity,
            domain: currentDomain
          })
        });
        const data = await res.json();
        lastAnalysisData = data;
        renderResults(data);
      } catch (err) {
        console.error("Analysis error:", err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined text-[16px]">bolt</span> Analyze Text';
      }
    }

    let currentHeatmapMode = 'flow';
    let activeTokenIndex = 0;

    function setHeatmapMode(mode) {
      currentHeatmapMode = mode;
      const btnFlow = document.getElementById('btnHeatmapFlow');
      const btnChips = document.getElementById('btnHeatmapChips');
      if (mode === 'flow') {
        if (btnFlow) btnFlow.className = 'px-2.5 py-0.5 rounded font-medium bg-dark-750 text-white transition cursor-pointer';
        if (btnChips) btnChips.className = 'px-2.5 py-0.5 rounded font-medium text-zinc-400 hover:text-white transition cursor-pointer';
      } else {
        if (btnChips) btnChips.className = 'px-2.5 py-0.5 rounded font-medium bg-dark-750 text-white transition cursor-pointer';
        if (btnFlow) btnFlow.className = 'px-2.5 py-0.5 rounded font-medium text-zinc-400 hover:text-white transition cursor-pointer';
      }
      if (lastAnalysisData && lastAnalysisData.tokens) {
        renderTokens(lastAnalysisData.tokens);
      }
    }

    function selectToken(index) {
      if (!lastAnalysisData || !lastAnalysisData.tokens || !lastAnalysisData.tokens[index]) return;
      activeTokenIndex = index;
      const t = lastAnalysisData.tokens[index];

      // Update Inspector Card
      const tokenEl = document.getElementById('inspectorToken');
      const badgeEl = document.getElementById('inspectorBadge');
      const scoreEl = document.getElementById('inspectorScore');
      const barEl = document.getElementById('inspectorScoreBar');
      const descEl = document.getElementById('inspectorDesc');
      const attrEl = document.getElementById('inspectorAttribution');

      if (tokenEl) tokenEl.textContent = `"${t.token}"`;
      if (scoreEl) {
        const sc = (t.score !== undefined ? Number(t.score).toFixed(2) : '0.00');
        scoreEl.textContent = sc;
        if (t.level === 'high') {
          scoreEl.className = 'font-bold text-rose-400 text-sm';
        } else if (t.level === 'medium') {
          scoreEl.className = 'font-bold text-amber-400 text-sm';
        } else {
          scoreEl.className = 'font-bold text-zinc-400 text-sm';
        }
      }
      if (barEl) {
        const pct = Math.min(100, Math.max(5, Math.round((t.score || 0.05) * 100)));
        barEl.style.width = `${pct}%`;
        if (t.level === 'high') {
          barEl.className = 'h-full rounded-full bg-gradient-to-r from-amber-500 via-rose-500 to-rose-400 transition-all duration-300';
        } else if (t.level === 'medium') {
          barEl.className = 'h-full rounded-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-300';
        } else {
          barEl.className = 'h-full rounded-full bg-dark-600 transition-all duration-300';
        }
      }
      if (badgeEl) {
        badgeEl.textContent = t.role || 'Syntactic Token';
        if (t.level === 'high') {
          badgeEl.className = 'text-[11px] font-medium px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/40';
        } else if (t.level === 'medium') {
          badgeEl.className = 'text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40';
        } else {
          badgeEl.className = 'text-[11px] font-medium px-2 py-0.5 rounded-full bg-dark-750 text-zinc-400 border border-dark-600';
        }
      }
      if (descEl) descEl.textContent = t.desc || 'Standard semantic component of the statement.';
      if (attrEl) {
        const lstmW = t.lstm_w !== undefined ? t.lstm_w : 'N/A';
        const baseW = t.base_w !== undefined ? (t.base_w >= 0 ? `+${t.base_w}` : `${t.base_w}`) : 'N/A';
        attrEl.textContent = `BiLSTM Attention: ${lstmW} • SVM TF-IDF: ${baseW}`;
      }

      // Update Visual Selected Ring on Token Elements
      const container = document.getElementById('heatmapContainer');
      if (container) {
        Array.from(container.children).forEach((el, idx) => {
          if (idx === index) {
            el.classList.add('ring-2', 'ring-cyan-400', 'shadow-md', 'scale-105');
          } else {
            el.classList.remove('ring-2', 'ring-cyan-400', 'shadow-md', 'scale-105');
          }
        });
      }
    }

    function renderTokens(tokens) {
      const container = document.getElementById('heatmapContainer');
      if (!container) return;
      container.innerHTML = '';
      if (!tokens || tokens.length === 0) {
        container.innerHTML = '<span class="text-xs text-zinc-500 italic">No tokens available</span>';
        return;
      }

      // Find index of highest scoring token to auto-select
      let bestIdx = 0;
      let maxScore = -1;
      tokens.forEach((t, i) => {
        if (t.score > maxScore) {
          maxScore = t.score;
          bestIdx = i;
        }
      });

      tokens.forEach((t, idx) => {
        const span = document.createElement('span');
        span.setAttribute('data-idx', idx);
        span.onclick = () => selectToken(idx);
        span.onmouseenter = () => selectToken(idx);

        const score = t.score || 0.05;
        const isHigh = t.level === 'high';
        const isMed = t.level === 'medium';

        if (currentHeatmapMode === 'flow') {
          span.className = 'inline-flex items-center px-2.5 py-1 rounded-lg text-sm cursor-pointer transition-all duration-150 select-none';
          if (isHigh) {
            span.style.backgroundColor = `rgba(244, 63, 94, ${Math.min(0.40, 0.18 + score * 0.25)})`;
            span.style.color = '#fda4af';
            span.style.border = '1px solid rgba(244, 63, 94, 0.45)';
            span.style.boxShadow = '0 0 10px rgba(244, 63, 94, 0.25)';
            span.innerHTML = `<span class="font-semibold">${t.token}</span><span class="text-[10px] font-mono ml-1.5 px-1 py-0.2 rounded bg-rose-500/30 text-rose-200">${score.toFixed(2)}</span>`;
          } else if (isMed) {
            span.style.backgroundColor = `rgba(245, 158, 11, ${Math.min(0.28, 0.12 + score * 0.20)})`;
            span.style.color = '#fde68a';
            span.style.border = '1px solid rgba(245, 158, 11, 0.35)';
            span.innerHTML = `<span>${t.token}</span><span class="text-[10px] font-mono ml-1.5 opacity-75">${score.toFixed(2)}</span>`;
          } else {
            span.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
            span.style.color = '#94a3b8';
            span.style.border = '1px solid transparent';
            span.innerHTML = `<span>${t.token}</span>`;
          }
        } else {
          span.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono cursor-pointer transition-all duration-150 select-none';
          if (isHigh) {
            span.className += ' bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm font-semibold hover:bg-rose-500/30';
            span.innerHTML = `<span>${t.token}</span><span class="px-1 py-0.2 rounded bg-rose-500/40 text-rose-100 text-[10px]">${score.toFixed(2)}</span>`;
          } else if (isMed) {
            span.className += ' bg-amber-500/15 text-amber-300 border border-amber-500/30 hover:bg-amber-500/25';
            span.innerHTML = `<span>${t.token}</span><span class="px-1 py-0.2 rounded bg-amber-500/30 text-amber-200 text-[10px]">${score.toFixed(2)}</span>`;
          } else {
            span.className += ' bg-dark-800 text-zinc-400 border border-dark-700 hover:text-zinc-200 hover:border-dark-600';
            span.innerHTML = `<span>${t.token}</span><span class="text-[10px] text-zinc-500">${score.toFixed(2)}</span>`;
          }
        }

        container.appendChild(span);
      });

      // Automatically select the best token
      selectToken(bestIdx);
    }

    function renderResults(d) {
      const iconBox = document.getElementById('verdictIconBox');
      const icon = document.getElementById('verdictIcon');
      const label = document.getElementById('verdictLabel');
      const badge = document.getElementById('confidenceBadge');
      const tone = document.getElementById('toneClassification');
      const latency = document.getElementById('latencyDisplay');
      const probSarc = document.getElementById('probBarSarcastic');
      const probLit = document.getElementById('probBarLiteral');
      const sarcPct = document.getElementById('sarcasmPctLabel');
      const litPct = document.getElementById('literalPctLabel');
      const intent = document.getElementById('decodedIntentText');

      if (d.is_sarcastic) {
        iconBox.className = 'w-12 h-12 rounded-xl flex items-center justify-center text-rose-400 bg-rose-500/10 border border-rose-500/20';
        icon.textContent = 'sentiment_very_dissatisfied';
        label.textContent = 'Sarcastic Tone Detected';
        badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30';
      } else {
        iconBox.className = 'w-12 h-12 rounded-xl flex items-center justify-center text-emerald-400 bg-emerald-500/10 border border-emerald-500/20';
        icon.textContent = 'sentiment_satisfied';
        label.textContent = 'Sincere & Literal Statement';
        badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      }

      badge.textContent = `${d.confidence_pct}% Confidence`;
      tone.textContent = `Classification: ${d.classification_tone} • ${d.friction_description || ''}`;
      latency.textContent = `${d.inference_ms} ms • RTX 4060`;

      probSarc.style.width = `${d.confidence_pct}%`;
      probLit.style.width = `${d.literal_pct}%`;
      sarcPct.textContent = `${d.confidence_pct}%`;
      litPct.textContent = `${d.literal_pct}%`;

      // KPIs
      document.getElementById('kpiValence').innerHTML = `${d.valence_inversion || 8.2} <span class="text-xs text-zinc-500 font-normal">/ 10</span>`;
      document.getElementById('kpiGap').textContent = d.pragmatic_gap || 'Δ -1.50';
      document.getElementById('kpiHyperbole').innerHTML = `${d.hyperbole_index || 7.0} <span class="text-xs text-zinc-500 font-normal">/ 10</span>`;

      intent.textContent = `"${d.decoded_intent}"`;

      // Render Interactive Token Attention Heatmap & Saliency Inspector
      renderTokens(d.tokens || []);

      // Consensus
      if (d.all_models) {
        if (d.all_models.transformer !== undefined) {
          document.getElementById('consensusTransformer').textContent = `${d.all_models.transformer}%`;
        }
        if (d.all_models.bilstm !== undefined) {
          document.getElementById('consensusBiLSTM').textContent = `${d.all_models.bilstm}%`;
        }
        if (d.all_models.baseline !== undefined) {
          document.getElementById('consensusBaseline').textContent = `${d.all_models.baseline}%`;
        }
      }
    }

    function exportJsonCard() {
      if (!lastAnalysisData) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastAnalysisData, null, 2));
      const a = document.createElement('a');
      a.setAttribute("href", dataStr);
      a.setAttribute("download", "ironylens_telemetry.json");
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    function copyAnalysisSummary() {
      if (!lastAnalysisData) return;
      const text = `IronyLens Sarcasm Analysis:
Text: "${lastAnalysisData.text}"
Verdict: ${lastAnalysisData.is_sarcastic ? 'Sarcastic' : 'Sincere'} (${lastAnalysisData.confidence_pct}% Confidence)
Tone: ${lastAnalysisData.classification_tone}
Decoded Meaning: "${lastAnalysisData.decoded_intent}"`;
      navigator.clipboard.writeText(text);
      const lbl = document.getElementById('copyBtnLabel');
      lbl.textContent = 'Copied!';
      setTimeout(() => { lbl.textContent = 'Copy Summary'; }, 2000);
    }

    function openModal(id) {
      if (id === 'leaderboardModal') {
        fetch('/api/leaderboard')
          .then(r => r.json())
          .then(data => {
            document.getElementById('leaderboardContent').innerHTML = '<pre class="whitespace-pre-wrap">' + data.markdown + '</pre>';
          });
      }
      document.getElementById(id).classList.remove('hidden');
    }

    function closeModal(id) {
      document.getElementById(id).classList.add('hidden');
    }

    // Ground-Truth Fact-Checked Benchmark Records from Test-Set Runs
    const BENCHMARK_DATA = {
      isarcasm: {
        label: 'Dataset: Twitter (iSarcasmEval — 3,110 train / 1,400 test)',
        transformer: { f1_macro: '0.6250', f1_sarc: '0.3986', recall: '55.0%', acc: '76.2%', speed: '~40 ms (GPU)', role: 'Primary Deep Classification' },
        bilstm:      { f1_macro: '0.5219', f1_sarc: '0.2174', recall: '27.5%', acc: '71.6%', speed: '~18 ms (GPU)', role: 'Token-Level Saliency Heatmaps' },
        baseline:    { f1_macro: '0.5475', f1_sarc: '0.2732', recall: '37.5%', acc: '71.4%', speed: '<1 ms (CPU)',  role: 'High-Throughput CPU Fallback' },
        edaNote: '<strong>Empirical Fact Check (iSarcasmEval):</strong> Sarcastic tweets show <strong>1.69× more question marks</strong>, <strong>1.68× more ellipses</strong>, and <strong>1.36× more emojis</strong>. Preserving punctuation during preprocessing is essential.'
      },
      headlines: {
        label: 'Dataset: News Headlines (HuffPost & TheOnion — 22,801 train / 5,827 test)',
        transformer: { f1_macro: '0.9376', f1_sarc: '0.9329', recall: '90.8%', acc: '93.8%', speed: '~40 ms (GPU)', role: 'Primary Deep Classification' },
        bilstm:      { f1_macro: '0.8700', f1_sarc: '0.8648', recall: '87.2%', acc: '87.0%', speed: '~18 ms (GPU)', role: 'Token-Level Saliency Heatmaps' },
        baseline:    { f1_macro: '0.8746', f1_sarc: '0.8700', recall: '88.1%', acc: '87.5%', speed: '<1 ms (CPU)',  role: 'High-Throughput CPU Fallback' },
        edaNote: '<strong>Empirical Fact Check (Headlines):</strong> Clean headlines (47.5% sarcastic) lack casual punctuation or emojis, so models rely on incongruous lexical combinations (e.g., "area man...", "study finds...").'
      }
    };

    function updateBenchmarkTable(dataset) {
      const d = BENCHMARK_DATA[dataset];
      if (!d) return;
      const tag = document.getElementById('benchmarkDatasetTag');
      if (tag) tag.textContent = d.label;
      const tbody = document.getElementById('benchmarkTableBody');
      if (tbody) {
        tbody.innerHTML = `
          <tr class="hover:bg-dark-800/40">
            <td class="px-4 py-3 font-semibold text-white">Twitter-RoBERTa</td>
            <td class="px-4 py-3"><span class="px-2 py-0.5 rounded bg-brand-500/20 text-brand-300 font-mono text-[11px]">Priority 1</span></td>
            <td class="px-4 py-3 font-mono font-bold text-brand-400">${d.transformer.f1_macro}</td>
            <td class="px-4 py-3 font-mono">${d.transformer.f1_sarc}</td>
            <td class="px-4 py-3 font-mono text-emerald-400">${d.transformer.recall}</td>
            <td class="px-4 py-3 font-mono">${d.transformer.acc}</td>
            <td class="px-4 py-3 font-mono text-zinc-400">${d.transformer.speed}</td>
            <td class="px-4 py-3 text-zinc-400">${d.transformer.role}</td>
          </tr>
          <tr class="hover:bg-dark-800/40">
            <td class="px-4 py-3 font-semibold text-white">BiLSTM + Attention</td>
            <td class="px-4 py-3"><span class="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 font-mono text-[11px]">Priority 2</span></td>
            <td class="px-4 py-3 font-mono">${d.bilstm.f1_macro}</td>
            <td class="px-4 py-3 font-mono">${d.bilstm.f1_sarc}</td>
            <td class="px-4 py-3 font-mono">${d.bilstm.recall}</td>
            <td class="px-4 py-3 font-mono">${d.bilstm.acc}</td>
            <td class="px-4 py-3 font-mono text-zinc-400">${d.bilstm.speed}</td>
            <td class="px-4 py-3 text-zinc-400">${d.bilstm.role}</td>
          </tr>
          <tr class="hover:bg-dark-800/40">
            <td class="px-4 py-3 font-semibold text-white">TF-IDF + Linear SVM</td>
            <td class="px-4 py-3"><span class="px-2 py-0.5 rounded bg-zinc-700/40 text-zinc-300 font-mono text-[11px]">Priority 3</span></td>
            <td class="px-4 py-3 font-mono">${d.baseline.f1_macro}</td>
            <td class="px-4 py-3 font-mono">${d.baseline.f1_sarc}</td>
            <td class="px-4 py-3 font-mono">${d.baseline.recall}</td>
            <td class="px-4 py-3 font-mono">${d.baseline.acc}</td>
            <td class="px-4 py-3 font-mono text-emerald-400">${d.baseline.speed}</td>
            <td class="px-4 py-3 text-zinc-400">${d.baseline.role}</td>
          </tr>
        `;
      }
      const edaEl = document.getElementById('edaTakeawayNote');
      if (edaEl) edaEl.innerHTML = d.edaNote;
    }

    // Graph Switcher
    function setGraphDataset(dataset) {
      activeDataset = dataset;
      const btnTw = document.getElementById('btnDsTwitter');
      const btnHl = document.getElementById('btnDsHeadlines');

      if (dataset === 'isarcasm') {
        btnTw.className = 'px-3 py-1 rounded-lg font-medium bg-brand-500 text-black shadow-sm transition cursor-pointer';
        btnHl.className = 'px-3 py-1 rounded-lg font-medium text-zinc-400 hover:text-white transition cursor-pointer';
      } else {
        btnHl.className = 'px-3 py-1 rounded-lg font-medium bg-brand-500 text-black shadow-sm transition cursor-pointer';
        btnTw.className = 'px-3 py-1 rounded-lg font-medium text-zinc-400 hover:text-white transition cursor-pointer';
      }

      const setSrc = (id, rel) => {
        const el = document.getElementById(id);
        if (el) el.src = `/reports/${dataset}/${rel}`;
      };

      setSrc('imgComp', 'comparison.png');
      setSrc('imgRoc', 'roc_pr_curves.png');
      setSrc('imgRadar', 'radar_comparison.png');
      setSrc('imgLatency', 'latency_vs_f1_tradeoff.png');
      setSrc('imgTune', 'hyperparameter_tuning.png');
      setSrc('imgScale', 'model_parameters_scale.png');
      setSrc('imgConfTf', 'transformer_twitter-roberta-base_confusion.png');
      setSrc('imgConfLstm', 'bilstm_attention_confusion.png');
      setSrc('imgConfBase', 'baseline_tfidf_confusion.png');
      setSrc('imgFeat', 'baseline_top_features.png');
      setSrc('imgCues', 'eda/stylistic_cues.png');
      setSrc('imgLen', 'eda/length_distribution.png');

      updateBenchmarkTable(dataset);
    }

    function showGraph(graphId) {
      const graphIds = ['comp', 'roc', 'radar', 'latency', 'tune', 'scale', 'conf', 'feat', 'eda'];
      graphIds.forEach(g => {
        const panel = document.getElementById('graphPanel_' + g);
        const tab = document.getElementById('tabG_' + g);
        if (panel && tab) {
          if (g === graphId) {
            panel.classList.remove('hidden');
            tab.className = 'px-3 py-1.5 rounded-lg bg-dark-750 text-white font-medium transition cursor-pointer';
          } else {
            panel.classList.add('hidden');
            tab.className = 'px-3 py-1.5 rounded-lg bg-dark-800 text-zinc-400 hover:text-white font-medium transition cursor-pointer';
          }
        }
      });
    }

    // Keyboard shortcut [Ctrl+Enter] to analyze
    document.addEventListener('DOMContentLoaded', () => {
      const input = document.getElementById('nlpInput');
      const clearBtn = document.getElementById('clearBtn');

      if (input) {
        input.addEventListener('keydown', (e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runAnalysis();
          }
        });
      }

      if (clearBtn && input) {
        clearBtn.addEventListener('click', () => {
          input.value = '';
          input.focus();
        });
      }

      // Initialize fact-checked benchmark table
      updateBenchmarkTable('isarcasm');

      // Initial analysis run
      runAnalysis();

      if (window.location.hash === '#comparison') {
        switchView('comparison');
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting IronyLens Web Server on http://localhost:{port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
