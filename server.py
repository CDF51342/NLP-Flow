"""NLP Flow 4 — Flask API with SSE progress streaming + session persistence"""
# ── MODEL STORE: trained model objects keyed by node_id ──────────────────────
_MODEL_STORE: dict = {}    # { node_id: result_dict }
_NODE_MODELS: dict = {}   # { node_id: sklearn model object }

# ── UI language (set by /api/set_lang) ───────────────────────────────────────
_UI_LANG: str = "es"   # default Spanish

_PLOT_LABELS = {
    "es": {
        "pred_actual":   "Predicho vs Real",
        "predicted":     "Predicho",
        "actual":        "Real",
        "residuals":     "Residuos",
        "fitted":        "Ajustados",
        "residuals_vs":  "Residuos vs Ajustados",
        "qq_title":      "Q-Q de residuos (normalidad)",
        "qq_x":          "Cuantiles teóricos",
        "qq_y":          "Cuantiles muestrales",
        "coef_title":    "Coeficientes",
        "variable":      "Variable",
        "coefficient":   "Coeficiente",
        "cm_title":      "Matriz de confusión",
        "pred_lbl":      "Predicho",
        "actual_lbl":    "Real",
        "roc_title":     "Curva ROC",
        "fpr":           "Tasa de falsos positivos",
        "tpr":           "Tasa de verdaderos positivos",
        "chance":        "Azar",
        "boundary_title":"Frontera de decisión",
        "cv_title":      "Validación Cruzada",
        "cv_score":      "Puntuación CV",
        "missing_title": "Valores ausentes (%)",
        "frequency":     "Frecuencia",
        "count":         "Recuento",
        "value":         "Valor",
    },
    "en": {
        "pred_actual":   "Predicted vs Actual",
        "predicted":     "Predicted",
        "actual":        "Actual",
        "residuals":     "Residuals",
        "fitted":        "Fitted",
        "residuals_vs":  "Residuals vs Fitted",
        "qq_title":      "Q-Q of residuals (normality)",
        "qq_x":          "Theoretical quantiles",
        "qq_y":          "Sample quantiles",
        "coef_title":    "Coefficients",
        "variable":      "Variable",
        "coefficient":   "Coefficient",
        "cm_title":      "Confusion matrix",
        "pred_lbl":      "Predicted",
        "actual_lbl":    "Actual",
        "roc_title":     "ROC curve",
        "fpr":           "False positive rate",
        "tpr":           "True positive rate",
        "chance":        "Random",
        "boundary_title":"Decision boundary",
        "cv_title":      "Cross-Validation",
        "cv_score":      "CV score",
        "missing_title": "Missing values (%)",
        "frequency":     "Frequency",
        "count":         "Count",
        "value":         "Value",
    }
}

def PL():
    """Return the current plot-label dict for the active UI language."""
    return _PLOT_LABELS.get(_UI_LANG, _PLOT_LABELS["es"])
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, send_file
import re, io, base64, csv, json, time, threading, zipfile, os, pickle, tempfile, shutil
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              f1_score, precision_score, recall_score,
                              mean_squared_error, mean_absolute_error, r2_score,
                              roc_curve, auc)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
import joblib

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        STOP_EN = set(stopwords.words("english"))
        STOP_ES = set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_EN = set(stopwords.words("english"))
        STOP_ES = set(stopwords.words("spanish"))
    STEMMER = PorterStemmer()
    NLTK_OK = True
except Exception:
    STOP_EN, STOP_ES, STEMMER, NLTK_OK = set(), set(), None, False

PORT = 5053
app  = Flask(__name__, static_folder="static")

@app.route("/api/set_lang", methods=["POST"])
def set_lang():
    global _UI_LANG
    body = request.get_json(force=True, silent=True) or {}
    new_lang = body.get("lang", "es")
    if new_lang in _PLOT_LABELS:
        _UI_LANG = new_lang
    return jsonify({"lang": _UI_LANG})

# ── Global state ─────────────────────────────────────────────────────────────
S = dict(
    texts=[], labels=[], label_names=[], task="classification",
    processed_texts=[], active_steps=[],
    model=None, vectorizer=None, results={}, dataset_name="",
    columns=[], raw_rows=[], csv_source="",  # "external" when loaded from user CSV
    exp_label="",                             # experiment label (one word)
)

# ── Multi-dataset store keyed by data-node id ─────────────────────────────────
# Each entry mirrors the tabular fields of S that are data-specific.
# NLP fields (texts, labels, model…) stay in S (single pipeline for now).
_NODE_DATA: dict = {}   # { node_id_str : { columns, raw_rows, dataset_name, task, csv_source } }

def _node_slot(node_id: str) -> dict:
    """Return (creating if needed) the per-node data slot."""
    if node_id not in _NODE_DATA:
        _NODE_DATA[node_id] = dict(columns=[], raw_rows=[], dataset_name="",
                                   task="classification", csv_source="")
    return _NODE_DATA[node_id]

def _get_data(node_id: str | None) -> dict:
    """Return the correct data slot: per-node if available, else global S."""
    if node_id and node_id in _NODE_DATA and _NODE_DATA[node_id]["raw_rows"]:
        return _NODE_DATA[node_id]
    return S   # fallback — keeps backward compat for single-pipeline apps

def _effective_data(node_id: str | None) -> dict:
    """Like _get_data but applies selected_cols / target_col filters from the slot config.
    Returns a dict with keys: raw_rows, columns, task, target, dataset_name.
    The returned rows only contain the selected columns (if set).
    """
    slot = _get_data(node_id)
    rows = slot.get("raw_rows", [])
    cols = slot.get("columns", [])
    task = slot.get("task", "classification")
    target = slot.get("target", "")
    name = slot.get("dataset_name", "")

    # Apply column filter if configured on the node slot
    cfg = _NODE_DATA.get(node_id or "", {}) if node_id else {}
    selected_cols = cfg.get("selected_cols")  # list of col names or None
    cfg_target    = cfg.get("target_col") or target

    if selected_cols:
        # Keep only selected columns in rows
        sel_set = set(selected_cols)
        rows = [{k: v for k, v in r.items() if k in sel_set} for r in rows]
        cols = [c for c in cols if c in sel_set]

    return {
        "raw_rows": rows,
        "columns": cols,
        "task": task,
        "target": cfg_target or target,
        "dataset_name": name,
    }

# Progress: list of {pct, msg}
_progress = []
_progress_lock = threading.Lock()
_train_done = threading.Event()
# Last progress event — survives drain so late pollers can catch up
_last_progress = {"pct": 0, "msg": "", "seen": True}

def push_progress(pct, msg):
    global _last_progress
    ev = {"pct": pct, "msg": msg}
    with _progress_lock:
        _progress.append(ev)
        _last_progress = {"pct": pct, "msg": msg, "seen": False}

def drain_progress():
    global _last_progress
    with _progress_lock:
        out = list(_progress)
        _progress.clear()
        # If nothing queued but last event was pct=100 and not yet seen, re-emit it once
        if not out and _last_progress["pct"] == 100 and not _last_progress["seen"]:
            out = [{"pct": 100, "msg": _last_progress["msg"]}]
            _last_progress["seen"] = True
    return out

def reset_progress():
    global _last_progress
    with _progress_lock:
        _progress.clear()
        _last_progress = {"pct": 0, "msg": "", "seen": True}

# ── Built-in datasets ────────────────────────────────────────────────────────
DATASETS = {
    "movie_reviews": {
        "name": "🎬 Movie Reviews", "task": "classification",
        "texts": [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me completely engaged throughout.",
            "Terrible film. Boring plot, bad acting, complete waste of time and money.",
            "An outstanding masterpiece with breathtaking visuals and a truly compelling, emotional story.",
            "Worst movie I've ever seen. The dialogue was cringeworthy and utterly predictable.",
            "A delightful experience from start to finish. Highly recommended to everyone!",
            "Dreadful. I fell asleep halfway through. No redeeming qualities whatsoever.",
            "Brilliant performances and a thought-provoking narrative. A must-watch film.",
            "Disappointing and dull. The characters were flat and completely uninteresting.",
            "Incredible storytelling! This film moved me to tears. Pure cinema at its best.",
            "Awful script, poor direction. A huge letdown after the exciting trailers.",
            "Visually stunning and emotionally powerful. One of the best films this year.",
            "Boring and predictable. Nothing new or interesting to offer audiences.",
            "A cinematic triumph! Every scene is crafted with care and precision.",
            "Not worth your time. The pacing is terrible and the ending is a mess.",
            "Funny, heartfelt and thrilling. This movie has it all!",
            "Generic and forgettable. Just another formulaic blockbuster film.",
            "The performances are electric and the soundtrack is absolutely incredible.",
            "A complete disaster. I cannot believe how this got made and released.",
            "Thoughtful, beautiful, and deeply moving. Cinema at its very finest.",
            "Tedious and overlong. Could have been easily cut by at least one hour.",
        ],
        "labels": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
        "label_names": ["Negative","Positive"],
    },
    "spam": {
        "name": "📧 Spam vs Ham", "task": "classification",
        "texts": [
            "Congratulations! You have won a FREE iPhone. Click here to claim now!!!",
            "Hey, are we still meeting for lunch tomorrow at noon?",
            "URGENT: Your account has been compromised. Send your password immediately.",
            "Just wanted to check in — how was your weekend?",
            "WIN $1000 CASH PRIZE! Limited time offer! Act now! Free money waiting!",
            "The meeting notes from yesterday are attached. Let me know if you have questions.",
            "You have been selected for a SPECIAL OFFER. Buy now and save 90% off!!!",
            "Can you review the report before Friday? Thanks in advance.",
            "FREE MEDS online! No prescription needed! Click here to order now!",
            "Looking forward to seeing you at the conference next week.",
            "CLAIM YOUR REWARD NOW! You are our lucky winner today!!!",
            "The project deadline has been moved to next Monday. Please update your tasks.",
            "Make money fast! Work from home! $5000 per week guaranteed income!",
            "I have attached the invoice for last month. Let me know if everything looks correct.",
            "ALERT: Verify your account in 24 hours or it will be permanently deleted!!!",
            "Thanks for the great presentation today. The team loved it.",
            "Lose weight fast! Miracle pill! No diet needed! Buy 2 get 1 free!",
            "Could you send me the updated schedule when you get a chance?",
            "Your credit card has been charged. Call us NOW to get a refund!",
            "Hope you are having a good week. Let us catch up soon!",
        ],
        "labels": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
        "label_names": ["Ham","Spam"],
    },
    "twitter": {
        "name": "😊 Twitter Sentiment", "task": "classification",
        "texts": [
            "I love this new feature! It makes my life so much easier.",
            "This update broke everything. I cannot believe how bad this is.",
            "Just had the best coffee of my life. Morning made!",
            "Traffic is a nightmare today. Going to be late again. So annoying.",
            "Finally finished my project! So proud of what we built together!",
            "My phone died right when I needed it most. Worst timing ever.",
            "Beautiful sunset tonight. Nature never disappoints.",
            "The customer service was absolutely terrible. Never buying from them again.",
            "Just got promoted! All the hard work finally paid off!",
            "Stuck in a 3-hour delay at the airport. This is absolutely ridiculous.",
            "My best friend surprised me with concert tickets! Best day ever!",
            "Lost my wallet. This day could not get any worse.",
            "Cooked an amazing dinner from scratch. Feeling like a real chef tonight!",
            "App keeps crashing. Zero stars. Total garbage software.",
            "Volunteered at the shelter today. Such a rewarding experience.",
            "Internet has been down all day. How is anyone supposed to work like this?",
            "Just adopted a puppy! She is the cutest thing in the world!",
            "Exam was way harder than expected. Do not think I passed. Gutted.",
            "Hiked to the top of the mountain. The view was absolutely breathtaking!",
            "Flight cancelled with no warning or compensation. Absolutely furious.",
        ],
        "labels": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
        "label_names": ["Negative","Positive"],
    },
    "news_topics": {
        "name": "📰 News (Topic Model)", "task": "topic_model",
        "texts": [
            "The stock market reached new highs as investors reacted to positive earnings from tech companies.",
            "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
            "The national football team qualified for the World Cup after a dramatic penalty shootout.",
            "Government officials announced new climate change policies to reduce carbon emissions by 2030.",
            "A breakthrough in cancer research offers new hope for patients with untreatable forms of disease.",
            "The central bank raised interest rates for the third consecutive time to combat rising inflation.",
            "Archaeologists uncovered an ancient Roman villa beneath a modern city center.",
            "The championship game drew record television audiences as two historic rivals faced off.",
            "Renewable energy investments surpassed fossil fuels for the first time according to new data.",
            "Researchers developed a new vaccine showing promise against several variants of the virus.",
            "Trade negotiations concluded with a landmark agreement on tariffs and trade barriers.",
            "The exploration rover sent back stunning images of the Martian surface showing ancient water flow.",
            "Athletes from over 200 countries gathered for the opening ceremony of the sporting event.",
            "Marine biologists documented the largest coral bleaching event ever recorded in the reef.",
            "The technology company unveiled its latest artificial intelligence model for complex reasoning.",
        ],
        "labels": [], "label_names": [],
    },
}

# ── Tabular datasets registry (separate from NLP DATASETS) ───────────────────
TAB_DATASETS = {}   # populated lazily by _build_tab_datasets()

def _build_tab_datasets():
    """Build tabular demo datasets and cache in TAB_DATASETS."""
    import random as _rnd, math as _math

    # ── 1. Patient Health Risk (classification, 100 rows) ────────────────────
    _rnd.seed(42)
    rows_ph = []
    cols_ph = ["age","bmi","blood_pressure","cholesterol","glucose",
               "heart_rate","creatinine","income_k","smoker","risk_label"]
    for _ in range(100):
        age  = _rnd.randint(25, 80)
        bmi  = round(_rnd.gauss(26.5, 5.0), 1)
        bp   = _rnd.randint(60, 180) if _rnd.random() > 0.06 else None
        chol = _rnd.randint(140, 320)
        gluc = round(_rnd.gauss(100, 30), 1)
        hr   = _rnd.randint(55, 105)
        crea = round(_rnd.uniform(0.5, 4.5), 2)
        inc  = round(_rnd.gauss(45, 20), 1)
        smok = _rnd.choice(["yes","no","no","no"])
        score= (age>55) + (bmi>30) + (bp is not None and bp>140) + (chol>260) + (smok=="yes")
        risk = 1 if (score>=2 or _rnd.random()<0.15) else 0
        rows_ph.append({
            "age": str(age),
            "bmi": str(bmi) if _rnd.random()>0.04 else "",
            "blood_pressure": str(bp) if bp and _rnd.random()>0.05 else "",
            "cholesterol": str(chol),
            "glucose": str(gluc) if _rnd.random()>0.04 else "",
            "heart_rate": str(hr),
            "creatinine": str(crea),
            "income_k": str(inc) if _rnd.random()>0.03 else "?",
            "smoker": smok,
            "risk_label": str(risk),
        })
    TAB_DATASETS["patient_health"] = {
        "name": "🏥 Patient Health Risk (clasificación)",
        "task": "classification", "target": "risk_label",
        "columns": cols_ph, "rows": rows_ph,
        "desc": "100 pacientes · variables médicas → riesgo (0/1) · ~5% valores ausentes"
    }

    # ── 2. California Housing (regression, 400 rows realistic) ───────────────
    _rnd.seed(7)
    import math as _math
    rows_ca = []
    cols_ca = ["MedInc","HouseAge","AveRooms","AveBedrms","Population",
               "AveOccup","Latitude","Longitude","MedHouseVal"]
    for _ in range(400):
        medinc  = round(_rnd.uniform(0.5, 15.0), 4)
        age     = _rnd.randint(1, 52)
        rooms   = round(_rnd.uniform(1.5, 10.0), 4)
        bedrms  = round(_rnd.uniform(0.8, 3.5), 4)
        pop     = _rnd.randint(50, 35000)
        occup   = round(_rnd.uniform(1.5, 8.0), 4)
        lat     = round(_rnd.uniform(32.5, 42.0), 4)
        lon     = round(_rnd.uniform(-124.5, -114.0), 4)
        coast_bonus = _math.exp(-0.3 * (lon + 120)**2) * 0.8
        overcrowd   = max(0, occup - 3.5) * (-0.25)
        noise       = _rnd.gauss(0, 0.65)
        base = (0.42*medinc + 0.06*rooms - 0.04*bedrms
                + 0.005*age - 0.000012*pop
                + coast_bonus + overcrowd + noise)
        val = round(max(0.15, min(5.0, base + 0.9)), 4)
        rows_ca.append({
            "MedInc": str(medinc),
            "HouseAge": str(age),
            "AveRooms": str(rooms),
            "AveBedrms": str(bedrms),
            "Population": str(pop),
            "AveOccup": str(occup),
            "Latitude": str(lat),
            "Longitude": str(lon),
            "MedHouseVal": str(val) if _rnd.random() > 0.03 else "",
        })
    TAB_DATASETS["california_housing"] = {
        "name": "🏠 California Housing (regresión)",
        "task": "regression", "target": "MedHouseVal",
        "columns": cols_ca, "rows": rows_ca,
        "desc": "400 distritos · variables demográficas → valor medio vivienda (×$100k)"
    }

    # ── 3. Ruido Puro — dataset completamente aleatorio para probar ───────────
    _rnd2 = __import__("random").Random(99)
    rows_noise = []
    cols_noise = ["X1","X2","X3","X4","X5","X6","Y"]
    for _ in range(300):
        x1 = round(_rnd2.gauss(0, 10), 3)
        x2 = round(_rnd2.uniform(-50, 50), 3)
        x3 = round(_rnd2.gauss(100, 30), 3)
        x4 = round(_rnd2.uniform(0, 1), 4)
        x5 = round(_rnd2.gauss(-5, 20), 3)
        x6 = round(_rnd2.uniform(1, 100), 3)
        # Y barely related to inputs — mostly noise
        y_val = round(
            0.15*x1 - 0.05*x2 + _rnd2.gauss(0, 25),   # signal swamped by noise
            3
        )
        rows_noise.append({
            "X1": str(x1), "X2": str(x2), "X3": str(x3),
            "X4": str(x4), "X5": str(x5), "X6": str(x6),
            "Y":  str(y_val) if _rnd2.random() > 0.05 else "",
        })
    TAB_DATASETS["pure_noise"] = {
        "name": "🎲 Ruido Puro (regresión)",
        "task": "regression", "target": "Y",
        "columns": cols_noise, "rows": rows_noise,
        "desc": "300 filas · 6 features aleatorias → Y casi independiente. R² esperado: 0.01–0.15"
    }

_build_tab_datasets()

# ── Tabular state (separate from NLP state S) ────────────────────────────────
# raw_rows / columns come from S; tabular preprocessing lives here
_TAB = dict(
    sampled_train=[],  # rows after Data Sampler
    sampled_test=[],
    processed_rows=[],  # after normalization / imputation
    proc_params={},     # fit params (min, max, mean, std per column)
    split_ratio=0.7,
    split_mode="random",   # "random" or "stratified"
    target_col="",
)

# ── Synthetic tabular dataset ─────────────────────────────────────────────────
# _make_synthetic_dataset removed — data now served from TAB_DATASETS

# ── Preprocessing ────────────────────────────────────────────────────────────
STEPS = {
    "lowercase":    lambda t: t.lower(),
    "punctuation":  lambda t: re.sub(r'[^a-zA-Z0-9\s]', '', t),
    "numbers":      lambda t: re.sub(r'\d+', '', t),
    "stopwords_en": lambda t: ' '.join(w for w in t.split() if w.lower() not in STOP_EN),
    "stopwords_es": lambda t: ' '.join(w for w in t.split() if w.lower() not in STOP_ES),
    "stemming":     lambda t: ' '.join(STEMMER.stem(w) for w in t.split()) if STEMMER else t,
    "whitespace":   lambda t: re.sub(r'\s+', ' ', t).strip(),
}

def preprocess(text, steps):
    for s in steps:
        if s in STEPS:
            text = STEPS[s](text)
    return text

# ── Plot helpers ─────────────────────────────────────────────────────────────
LIGHT="#ffffff"; BG="#f5f5f5"; INK="#121212"; SEC="#656565"; MINT="#19e68c"; BORDER="#e8e8e8"

# Paleta Claude-style: limpia, bien contrastada, sin colores chillones
PALETTE = ["#5B7FDB","#E8785A","#5CB88A","#C4805F","#8B78C9","#D4875F","#4AA8C4","#C46E7A"]
# Paleta de gradiente para mapas de calor y matrices
HEATMAP_COLORS = ["#EEF2FF","#C7D4F5","#99B2ED","#6B8FE5","#3D6DDD","#1A4EC8"]

def style_ax(ax):
    ax.set_facecolor(LIGHT)
    ax.tick_params(colors=SEC, labelsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)
    ax.spines[["left","bottom"]].set_linewidth(0.8)

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=LIGHT)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data

def _wordfreq_b64():
    """Generate word-frequency comparison plot and return base64 PNG string."""
    texts = S["texts"]; proc = S["processed_texts"]
    if not texts: return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5)); fig.patch.set_facecolor(LIGHT)
    def top_words(corpus, n=12):
        all_w = []
        for t in corpus: all_w.extend(t.lower().split())
        return Counter(all_w).most_common(n)
    for ax_i, (ax, corpus, title, pal) in enumerate([
        (axes[0], texts, "Top words — Raw",          PALETTE[1]),
        (axes[1], proc,  "Top words — Preprocessed", PALETTE[3])
    ]):
        style_ax(ax)
        wf = top_words(corpus)
        if wf:
            words, freqs = zip(*wf)
            bar_colors = [pal if j == 0 else PALETTE[(ax_i*3+j) % len(PALETTE)]
                          for j in range(len(words))]
            ax.barh(list(reversed(words)), list(reversed(freqs)),
                    color=list(reversed(bar_colors)), edgecolor="none", alpha=0.88)
        ax.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.8)
    return fig_b64(fig)


def _dataset_summary():
    texts = S["texts"]
    if not texts: return {"loaded": False}
    wc = [len(t.split()) for t in texts]
    c  = Counter(S["labels"])
    # is_tabular: true only when loaded from TAB_DATASETS or when the CSV has
    # many numeric/mixed columns (not a simple text+label NLP dataset).
    n_cols = len(S["columns"])
    n_numeric = sum(1 for c in S["columns"]
                    if S["raw_rows"] and _col_type([r.get(c,"") for r in S["raw_rows"][:50]])=="numeric")
    is_tab = bool(
        S["raw_rows"] and (
            S["csv_source"] == "demo" or          # always a tab dataset
            (n_cols > 2 and n_numeric >= 2) or    # CSV with multiple numeric cols
            S["task"] == "regression"              # regression is always tabular
        )
    )
    return {
        "loaded": True, "name": S["dataset_name"], "task": S["task"],
        "is_tabular": is_tab,
        "n": len(texts), "label_names": S["label_names"],
        "label_counts": {S["label_names"][k]: v for k,v in c.items()} if S["label_names"] else {},
        "avg_words": round(float(np.mean(wc)), 1),
        "columns": S["columns"],
        "samples": [{"idx":i,"text":t[:200],"label":S["label_names"][S["labels"][i]] if S["labels"] else ""}
                    for i,t in enumerate(texts[:20])],
        "rows": S["raw_rows"][:20],
    }

# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index(): return send_from_directory("static", "index.html")

@app.route("/favicon.ico")
def favicon(): return send_from_directory("static", "favicon.png", mimetype="image/png")

@app.route("/api/status")
def status():
    return jsonify({"nltk": NLTK_OK, "data_loaded": bool(S["texts"]),
                    "trained": S["model"] is not None, "task": S["task"],
                    "n_texts": len(S["texts"])})

@app.route("/api/datasets")
def get_datasets():
    return jsonify([{"id":k,"name":v["name"],"task":v["task"]} for k,v in DATASETS.items()])

@app.route("/api/load_dataset", methods=["POST"])
def load_dataset():
    ds = DATASETS.get(request.json.get("id",""))
    if not ds: return jsonify({"error":"Not found"}), 404
    rows = [{"text": t, "label": ds["label_names"][l] if ds["label_names"] else ""}
            for t, l in zip(ds["texts"], ds["labels"])] if ds["labels"] else [{"text":t} for t in ds["texts"]]
    S.update(texts=ds["texts"], labels=ds["labels"], label_names=ds["label_names"],
             task=ds["task"], dataset_name=ds["name"],
             processed_texts=list(ds["texts"]), results={}, model=None, vectorizer=None,
             columns=list(rows[0].keys()) if rows else [], raw_rows=rows)
    return jsonify(_dataset_summary())

@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    f = request.files.get("file")
    if not f: return jsonify({"error":"No file"}), 400
    text_col  = request.form.get("text_col","text")
    label_col = request.form.get("label_col","")
    task      = request.form.get("task","classification")
    node_id   = request.form.get("node_id","")
    content   = f.read().decode("utf-8", errors="replace")
    rows      = list(csv.DictReader(io.StringIO(content)))
    if not rows: return jsonify({"error":"Empty CSV"}), 400
    cols = list(rows[0].keys())
    if text_col not in cols:
        return jsonify({"error": f"Column '{text_col}' not found. Available: {cols}"}), 400
    texts  = [r[text_col].strip() for r in rows if r.get(text_col,"").strip()]
    raw_rows = [{c: r.get(c,"") for c in cols} for r in rows if r.get(text_col,"").strip()]
    labels, label_names = [], []
    if label_col and label_col in cols and task=="classification":
        raw_lbl = [r[label_col].strip() for r in rows if r.get(text_col,"").strip()]
        uniq    = sorted(set(raw_lbl))
        m       = {v:i for i,v in enumerate(uniq)}
        labels, label_names = [m[l] for l in raw_lbl], uniq
    S.update(texts=texts, labels=labels, label_names=label_names, task=task,
             dataset_name=f.filename, processed_texts=list(texts),
             results={}, model=None, vectorizer=None, columns=cols, raw_rows=raw_rows,
             csv_source="external")
    if node_id:
        slot = _node_slot(node_id)
        slot.update(columns=cols, raw_rows=raw_rows, dataset_name=f.filename,
                    task=task, csv_source="external")
    return jsonify({**_dataset_summary(), "columns": cols, "node_id": node_id})

@app.route("/api/dataset_info")
def dataset_info(): return jsonify(_dataset_summary())

# ── Tabular Data Info (rich — for the Data Info panel) ────────────────────────
def _col_type(values):
    """Return fine-grained type for a list of raw values.

    Returns one of: 'int', 'float', 'categorical', 'list', 'text'.
    Callers that previously compared == 'numeric' should now use _is_numeric_type().
    """
    non_empty = [v for v in values if str(v).strip() not in ("", "?", "NA", "NaN", "nan", "None")]
    if not non_empty:
        return "categorical"

    # Check for list values (Python list objects or JSON-array-like strings)
    list_ok = sum(1 for v in non_empty[:50] if isinstance(v, list) or
                  (isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]")))
    if list_ok / max(len(non_empty[:50]), 1) >= 0.6:
        return "list"

    # Check for numeric
    int_ok = 0; float_ok = 0
    for v in non_empty[:200]:
        try:
            f = float(str(v))
            if str(v).strip().lstrip("-").isdigit():
                int_ok += 1
            else:
                float_ok += 1
        except (ValueError, TypeError):
            pass
    total = len(non_empty[:200])
    numeric_ratio = (int_ok + float_ok) / total
    if numeric_ratio >= 0.85:
        return "int" if int_ok >= float_ok else "float"

    if len(set(str(v) for v in non_empty[:500])) <= max(20, len(non_empty) * 0.05):
        return "categorical"
    return "text"

def _is_numeric_type(t):
    """True for both 'int' and 'float' — replaces == 'numeric' comparisons."""
    return t in ("int", "float")

def _is_missing(v):
    return str(v).strip() in ("", "?", "NA", "NaN", "nan", "None", "null")

# ── List & load tabular datasets ──────────────────────────────────────────────
@app.route("/api/tab_datasets")
def tab_datasets():
    return jsonify([{"id": k, "name": v["name"], "task": v["task"],
                     "target": v["target"], "desc": v["desc"]}
                    for k, v in TAB_DATASETS.items()])

@app.route("/api/load_tab_dataset", methods=["POST"])
def load_tab_dataset():
    body = request.get_json(force=True, silent=True) or {}
    ds_id   = body.get("id", "")
    node_id = str(body.get("node_id", ""))
    ds = TAB_DATASETS.get(ds_id)
    if not ds:
        return jsonify({"error": "Dataset not found"}), 404
    cols, rows = ds["columns"], ds["rows"]
    texts = [" ".join(str(r.get(c,"")) for c in cols) for r in rows]
    # Always update global S for NLP pipeline compatibility
    S.update(
        texts=texts, labels=[], label_names=[], task=ds["task"],
        processed_texts=list(texts), results={}, model=None, vectorizer=None,
        dataset_name=ds["name"], columns=cols, raw_rows=rows, csv_source="demo"
    )
    # Also store in per-node slot when a node_id is provided
    if node_id:
        slot = _node_slot(node_id)
        slot.update(columns=cols, raw_rows=rows, dataset_name=ds["name"],
                    task=ds["task"], csv_source="demo")
    _TAB["target_col"] = ds["target"]
    return jsonify({**_dataset_summary(), "columns": cols,
                    "target": ds["target"], "task": ds["task"], "node_id": node_id})

# ── Node configuration (selected columns, target) ─────────────────────────────
@app.route("/api/set_node_config", methods=["POST"])
def set_node_config():
    """Persist column selection and target column for a data node."""
    body      = request.get_json(force=True, silent=True) or {}
    node_id   = str(body.get("node_id", ""))
    sel_cols  = body.get("selected_cols")   # list[str] or None (= all)
    target    = body.get("target_col", "")
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    slot = _node_slot(node_id)
    if sel_cols is not None:
        slot["selected_cols"] = sel_cols
    if target:
        slot["target_col"]    = target
        slot["target"]        = target        # keep both keys in sync
        _TAB["target_col"]    = target
    return jsonify({"ok": True, "node_id": node_id,
                    "selected_cols": slot.get("selected_cols"),
                    "target_col": slot.get("target_col")})

# ── Data Sampler (random & stratified split) ──────────────────────────────────
@app.route("/api/data_sampler", methods=["POST"])
def data_sampler():
    body   = request.get_json(force=True, silent=True) or {}
    rows   = S["raw_rows"]
    if not rows:
        return jsonify({"error": "No data loaded"}), 400

    ratio  = float(body.get("ratio", 0.7))
    mode   = body.get("mode", "random")        # "random" or "stratified"
    target = body.get("target_col", "")
    seed   = int(body.get("seed", 42))

    import random as _rnd
    rng = _rnd.Random(seed)

    if mode == "stratified" and target and target in S["columns"]:
        groups = {}
        for r in rows:
            g = str(r.get(target, ""))
            groups.setdefault(g, []).append(r)
        train_rows, test_rows = [], []
        for g, grp in groups.items():
            rng.shuffle(grp)
            n_train = max(1, round(len(grp) * ratio))
            train_rows.extend(grp[:n_train])
            test_rows.extend(grp[n_train:])
        rng.shuffle(train_rows); rng.shuffle(test_rows)
    else:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_train = max(1, round(len(rows) * ratio))
        train_rows = [rows[i] for i in idx[:n_train]]
        test_rows  = [rows[i] for i in idx[n_train:]]

    _TAB["sampled_train"] = train_rows
    _TAB["sampled_test"]  = test_rows
    _TAB["split_ratio"]   = ratio
    _TAB["split_mode"]    = mode
    _TAB["target_col"]    = target

    def class_dist(rws, col):
        if not col or not rws: return {}
        c = Counter(str(r.get(col,"")) for r in rws)
        return dict(c)

    return jsonify({
        "n_train": len(train_rows), "n_test": len(test_rows),
        "ratio": ratio, "mode": mode,
        "train_sample": train_rows[:5],
        "test_sample":  test_rows[:5],
        "train_dist": class_dist(train_rows, target),
        "test_dist":  class_dist(test_rows, target),
        "orig_dist":  class_dist(rows, target),
    })

@app.route("/api/data_split", methods=["POST"])
def data_split():
    """Random / Stratified split or Cross-Validation fold analysis."""
    body       = request.get_json(force=True, silent=True) or {}
    node_id    = str(body.get("node_id", ""))
    pre_node_id= str(body.get("preprocess_node_id", ""))
    mode       = body.get("mode", "random")   # "random" | "stratified" | "cv" | "cv_stratified" | "cv_random"
    ratio      = float(body.get("ratio", 0.8))
    n_folds    = int(body.get("n_folds", 5))
    target     = body.get("target_col", "")
    seed       = int(body.get("seed", 42))

    # Resolve data: prefer preprocess slot if populated
    D = _effective_data(pre_node_id) if (pre_node_id and pre_node_id in _NODE_DATA
                                          and _NODE_DATA[pre_node_id].get("raw_rows")) \
        else _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows:
        return jsonify({"error": "No hay datos cargados"}), 400

    target = target or D.get("target", "")

    import random as _rnd
    rng = _rnd.Random(seed)

    def class_dist(rws, col):
        if not col or not rws: return {}
        return dict(Counter(str(r.get(col,"")) for r in rws))

    def pct_dist(dist, total):
        return {k: round(v/total*100,1) for k,v in dist.items()}

    # ── Random / Stratified split ─────────────────────────────────────────────
    if mode in ("random", "stratified"):
        if mode == "stratified" and target and target in D["columns"]:
            groups = {}
            for r in rows:
                groups.setdefault(str(r.get(target,"")), []).append(r)
            train_rows, test_rows = [], []
            for g, grp in groups.items():
                grp2 = grp[:]
                rng.shuffle(grp2)
                n_tr = max(1, round(len(grp2)*ratio))
                train_rows.extend(grp2[:n_tr])
                test_rows.extend(grp2[n_tr:])
            rng.shuffle(train_rows); rng.shuffle(test_rows)
        else:
            idx = list(range(len(rows))); rng.shuffle(idx)
            n_tr = max(1, round(len(rows)*ratio))
            train_rows = [rows[i] for i in idx[:n_tr]]
            test_rows  = [rows[i] for i in idx[n_tr:]]

        _TAB["sampled_train"] = train_rows
        _TAB["sampled_test"]  = test_rows
        _TAB["split_ratio"]   = ratio
        _TAB["split_mode"]    = mode
        _TAB["target_col"]    = target

        orig_dist  = class_dist(rows, target)
        train_dist = class_dist(train_rows, target)
        test_dist  = class_dist(test_rows, target)
        total = len(rows)

        # Class balance plot (bar chart: orig vs train vs test)
        img_b64 = None
        try:
            if target and orig_dist:
                labels = sorted(orig_dist.keys())
                orig_pct  = [orig_dist.get(l,0)/total*100  for l in labels]
                train_pct = [train_dist.get(l,0)/len(train_rows)*100 for l in labels]
                test_pct  = [test_dist.get(l,0)/len(test_rows)*100   for l in labels]
                x = np.arange(len(labels)); w = 0.28
                fig, ax = plt.subplots(figsize=(max(5,len(labels)*1.4), 3.5))
                ax.bar(x-w, orig_pct,  w, label="Original", color=PALETTE[0], alpha=.85)
                ax.bar(x,   train_pct, w, label="Train",    color=PALETTE[1], alpha=.85)
                ax.bar(x+w, test_pct,  w, label="Test",     color=PALETTE[2], alpha=.85)
                ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylabel("% clase", color=INK, fontsize=9)
                ax.set_title("Distribución de clases — Original vs Train vs Test", color=INK, fontsize=10, fontweight="bold")
                ax.legend(fontsize=9); _style_ax(ax)
                plt.tight_layout(pad=1.2)
                img_b64 = fig_b64(fig)
        except Exception:
            pass

        return jsonify({
            "mode": mode, "n_train": len(train_rows), "n_test": len(test_rows),
            "ratio": ratio, "n_total": total,
            "train_pct": round(len(train_rows)/total*100,1),
            "test_pct":  round(len(test_rows)/total*100,1),
            "orig_dist": orig_dist, "train_dist": train_dist, "test_dist": test_dist,
            "orig_pct":  pct_dist(orig_dist, total),
            "train_pct_dist": pct_dist(train_dist, max(len(train_rows),1)),
            "test_pct_dist":  pct_dist(test_dist,  max(len(test_rows),1)),
            "img": img_b64
        })

    # ── Cross-Validation analysis ─────────────────────────────────────────────
    if mode in ("cv", "cv_stratified", "cv_random"):
        force_stratified = (mode == "cv_stratified")
        force_random     = (mode == "cv_random")
    if mode in ("cv", "cv_stratified", "cv_random"):
        # Fall back to last column if no target configured
        if not target and D["columns"]:
            target = D["columns"][-1]
        if not target or target not in D["columns"]:
            return jsonify({"error": "No se encontró columna objetivo. Define el target en el bloque Datos."}), 400

        labels_all = [str(r.get(target,"")) for r in rows]
        unique_cls = sorted(set(labels_all))
        n = len(rows)

        try:
            import numpy as _np4
            idx_arr  = _np4.arange(n)
            y_arr    = _np4.array(labels_all)
            # use_skf: stratified if user requested it AND we have class labels
            use_skf = (not force_random) and len(unique_cls) >= 2

            if use_skf:
                from sklearn.model_selection import StratifiedKFold
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = list(kf.split(idx_arr, y_arr))
            else:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = list(kf.split(idx_arr))

            folds = []
            for fi, (tr_idx, te_idx) in enumerate(splits):
                tr_dist = dict(Counter(y_arr[tr_idx]))
                te_dist = dict(Counter(y_arr[te_idx]))
                folds.append({
                    "fold": fi+1,
                    "n_train": len(tr_idx),
                    "n_test":  len(te_idx),
                    "train_dist": tr_dist,
                    "test_dist":  te_dist,
                    "train_pct_dist": {k: round(v/len(tr_idx)*100,1) for k,v in tr_dist.items()},
                    "test_pct_dist":  {k: round(v/len(te_idx)*100,1)  for k,v in te_dist.items()},
                })

            # Plot: class balance per fold as grouped bar chart
            img_b64 = None
            try:
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
                fold_nums = [f["fold"] for f in folds]
                for ci, cls in enumerate(unique_cls[:6]):  # max 6 classes
                    tr_pcts = [f["train_pct_dist"].get(cls,0) for f in folds]
                    te_pcts = [f["test_pct_dist"].get(cls,0)  for f in folds]
                    axes[0].plot(fold_nums, tr_pcts, marker="o", label=cls, color=PALETTE[ci % len(PALETTE)])
                    axes[1].plot(fold_nums, te_pcts, marker="o", label=cls, color=PALETTE[ci % len(PALETTE)])
                axes[0].set_title("Train — distribución por fold", color=INK, fontsize=10, fontweight="bold")
                axes[1].set_title("Test — distribución por fold",  color=INK, fontsize=10, fontweight="bold")
                for ax in axes:
                    ax.set_xlabel("Fold", color=INK, fontsize=9)
                    ax.set_ylabel("% clase", color=INK, fontsize=9)
                    ax.set_xticks(fold_nums)
                    ax.legend(fontsize=8, loc="upper right")
                    _style_ax(ax)
                plt.tight_layout(pad=1.2)
                img_b64 = fig_b64(fig)
            except Exception:
                pass

            return jsonify({
                "mode": "cv", "n_folds": n_folds, "n_total": n,
                "stratified": use_skf, "classes": unique_cls,
                "folds": folds, "img": img_b64
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": f"Modo desconocido: {mode}. Usa random, stratified, cv, cv_stratified o cv_random."}), 400


# ── Tabular Preprocessing (normalize + impute) ────────────────────────────────
@app.route("/api/preprocess_column", methods=["POST"])
def preprocess_column():
    """Apply one operation to one column and return before/after stats + plot.

    State model
    -----------
    - ``_NODE_DATA[pre_node_id]["raw_rows"]``  → current working state (all cols, some already processed)
    - ``_NODE_DATA[pre_node_id]["col_snapshots"][col]`` → original values for *col* from the upstream
      data node, captured the first time that column is touched.  Re-applying an operation always
      restarts from this snapshot so the user can freely switch strategies.
    """
    body        = request.get_json(force=True, silent=True) or {}
    node_id     = str(body.get("node_id", ""))
    pre_node_id = str(body.get("preprocess_node_id", ""))
    col         = body.get("col", "")
    operation   = body.get("operation", "")

    # ── Resolve upstream (original) data ─────────────────────────────────────
    D_orig = _effective_data(node_id) if node_id else None
    if not D_orig or not D_orig["raw_rows"]:
        return jsonify({"error": "No hay datos cargados en el nodo upstream"}), 400

    # ── Resolve / initialise the preprocess working slot ─────────────────────
    pslot = _node_slot(pre_node_id)
    # If the working slot is empty, seed it from the original data
    if not pslot.get("raw_rows"):
        pslot.update(
            raw_rows     = [dict(r) for r in D_orig["raw_rows"]],
            columns      = list(D_orig["columns"]),
            dataset_name = D_orig["dataset_name"] + " [preprocesado]",
            task         = D_orig["task"],
            target       = D_orig["target"],
            col_snapshots= {},
        )
    if "col_snapshots" not in pslot:
        pslot["col_snapshots"] = {}

    # ── Snapshot the original column values on first touch ───────────────────
    # This lets the user re-apply a different strategy without accumulating transforms.
    if col not in pslot["col_snapshots"]:
        pslot["col_snapshots"][col] = [r.get(col, "") for r in D_orig["raw_rows"]]

    orig_col_vals = pslot["col_snapshots"][col]   # always the untouched originals

    # ── Build a working copy of *all* rows, replacing only this column
    #    with its original snapshot values (so we start fresh for this col).
    rows = [dict(r) for r in pslot["raw_rows"]]
    for i, r in enumerate(rows):
        if i < len(orig_col_vals):
            r[col] = orig_col_vals[i]
    cols = list(pslot.get("columns", D_orig["columns"]))

    if col not in cols:
        return jsonify({"error": f"Column '{col}' not found"}), 400

    vals_before = [r.get(col, "") for r in rows]
    ctype = _col_type(vals_before)
    missing_before = sum(1 for v in vals_before if _is_missing(v))

    # ── Apply operation ───────────────────────────────────────────────────────
    changed = 0
    dropped = 0

    if operation.startswith("impute_"):
        strategy = operation[len("impute_"):]
        non_empty = [v for v in vals_before if not _is_missing(v)]
        fill_val = None
        if strategy == "drop":
            rows = [r for r in rows if not _is_missing(r.get(col, ""))]
            dropped = len(orig_col_vals) - len(rows)

        elif strategy == "knn":
            # KNN imputation: use all numeric columns as features; k=5
            if _is_numeric_type(ctype):
                try:
                    from sklearn.impute import KNNImputer
                    num_cols = [c for c in cols if _is_numeric_type(_col_type([r.get(c,"") for r in rows[:50]]))]
                    if col not in num_cols:
                        num_cols.append(col)
                    import numpy as _np2
                    mat = []
                    for r in rows:
                        mat.append([float(r[c]) if (c in r and _try_float(r[c])) else float("nan") for c in num_cols])
                    arr = _np2.array(mat, dtype=float)
                    col_idx = num_cols.index(col)
                    imputer = KNNImputer(n_neighbors=min(5, sum(1 for row in arr if not _np2.isnan(row[col_idx]))))
                    arr_out = imputer.fit_transform(arr)
                    for i, r in enumerate(rows):
                        if _is_missing(r.get(col, "")):
                            r[col] = str(round(float(arr_out[i, col_idx]), 6)); changed += 1
                except Exception as e:
                    return jsonify({"error": f"KNN imputation error: {e}"}), 500
            else:
                # fallback for categorical: mode
                if non_empty:
                    fill_val = Counter(str(v) for v in non_empty).most_common(1)[0][0]

        elif strategy == "regression":
            # Regression imputation: train linear regression on rows where col is present,
            # using all other numeric columns as features.
            if _is_numeric_type(ctype):
                try:
                    from sklearn.linear_model import LinearRegression as _LR
                    num_cols = [c for c in cols if c != col and _is_numeric_type(_col_type([r.get(c,"") for r in rows[:50]]))]
                    if not num_cols:
                        return jsonify({"error": "No hay suficientes columnas numéricas para imputación por regresión"}), 400
                    import numpy as _np3
                    known, unknown = [], []
                    known_y = []
                    for r in rows:
                        feats = [float(r.get(c, 0)) if _try_float(r.get(c,"")) else 0.0 for c in num_cols]
                        if _try_float(r.get(col, "")):
                            known.append(feats); known_y.append(float(r[col]))
                        else:
                            unknown.append((r, feats))
                    if len(known) < 2:
                        return jsonify({"error": "Pocos datos conocidos para regresión"}), 400
                    lr = _LR()
                    lr.fit(_np3.array(known), _np3.array(known_y))
                    for r, feats in unknown:
                        pred = float(lr.predict([feats])[0])
                        r[col] = str(round(pred, 6)); changed += 1
                except Exception as e:
                    return jsonify({"error": f"Regression imputation error: {e}"}), 500
            else:
                if non_empty:
                    fill_val = Counter(str(v) for v in non_empty).most_common(1)[0][0]

        else:
            if _is_numeric_type(ctype) and non_empty:
                nums = [float(v) for v in non_empty if _try_float(v)]
                if strategy == "mean":   fill_val = str(round(float(np.mean(nums)), 6))
                elif strategy == "median": fill_val = str(round(float(np.median(nums)), 6))
                elif strategy == "mode":
                    c2 = Counter(nums); fill_val = str(c2.most_common(1)[0][0])
                elif strategy == "zero": fill_val = "0"
            else:
                if non_empty:
                    c2 = Counter(str(v) for v in non_empty)
                    fill_val = c2.most_common(1)[0][0]
            if fill_val is not None:
                for r in rows:
                    if _is_missing(r.get(col, "")):
                        r[col] = fill_val; changed += 1

    elif operation.startswith("normalize_"):
        method = operation[len("normalize_"):]
        if method != "none" and _is_numeric_type(ctype):
            nums = [float(r.get(col, 0)) for r in rows if _try_float(r.get(col, ""))]
            if nums:
                if method == "minmax":
                    mn, mx = min(nums), max(nums); rng = mx - mn
                    for r in rows:
                        v = _try_float(r.get(col, ""))
                        if v is not None:
                            r[col] = str(round((v - mn) / rng, 6)) if rng else "0"; changed += 1
                elif method == "zscore":
                    mu, sd = float(np.mean(nums)), float(np.std(nums))
                    for r in rows:
                        v = _try_float(r.get(col, ""))
                        if v is not None:
                            r[col] = str(round((v - mu) / sd, 6)) if sd else "0"; changed += 1

    elif operation == "catfix":
        for r in rows:
            v = r.get(col, "")
            if isinstance(v, str):
                nv = v.strip().lower()
                if nv != v: r[col] = nv; changed += 1

    elif operation == "encode_label":
        uniq = sorted(set(str(r.get(col, "")) for r in rows))
        mapping = {v: str(i) for i, v in enumerate(uniq)}
        for r in rows: r[col] = mapping.get(str(r.get(col, "")), "0"); changed += 1

    elif operation == "encode_onehot":
        uniq = sorted(set(str(r.get(col, "")) for r in rows))
        new_cols = [col + "__" + v for v in uniq]
        for r in rows:
            val = str(r.get(col, ""))
            for u in uniq:
                r[col + "__" + u] = "1" if val == u else "0"
            del r[col]
        cols = [c for c in cols if c != col] + new_cols
        changed = len(rows)

    # ── Persist updated rows in preprocess slot (preserve col_snapshots) ────────
    if pre_node_id:
        snaps = pslot.get("col_snapshots", {})
        pslot.update(raw_rows=rows, columns=cols,
                     dataset_name=D_orig["dataset_name"] + " [preprocesado]",
                     task=D_orig["task"], target=D_orig["target"])
        pslot["col_snapshots"] = snaps

    # ── Stats after ───────────────────────────────────────────────────────────
    vals_after = [r.get(col, "") for r in rows] if col in cols else []
    missing_after = sum(1 for v in vals_after if _is_missing(v))

    def col_stats(vals):
        non_e = [v for v in vals if not _is_missing(v)]
        if not non_e: return {}
        if _is_numeric_type(_col_type(vals)):
            nums = [float(v) for v in non_e if _try_float(v)]
            if nums:
                return {"mean": round(float(np.mean(nums)), 4),
                        "std":  round(float(np.std(nums)),  4),
                        "min":  round(min(nums), 4),
                        "max":  round(max(nums), 4),
                        "n_unique": len(set(nums))}
        return {"n_unique": len(set(str(v) for v in non_e)),
                "top": [{"v": k, "n": n} for k, n in Counter(str(v) for v in non_e).most_common(5)]}

    stats_before = col_stats(vals_before)
    stats_after  = col_stats(vals_after)

    # ── Before/after plot ─────────────────────────────────────────────────────
    img_b64 = None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for ax, vals, title, color in [
            (axes[0], vals_before, "Antes", PALETTE[1]),
            (axes[1], vals_after,  "Después", PALETTE[0])
        ]:
            nums = [float(v) for v in vals if not _is_missing(v) and _try_float(v)]
            if nums:
                ax.hist(nums, bins=min(30, len(set(nums))), color=color, edgecolor="none", alpha=0.85)
                ax.axvline(float(np.mean(nums)), color="#333", lw=1.5, linestyle="--", label="Media")
                ax.set_xlabel(col, color=INK, fontsize=9)
            else:
                cats = Counter(str(v) for v in vals if not _is_missing(v))
                top  = cats.most_common(8)
                ax.barh([x[0] for x in top][::-1], [x[1] for x in top][::-1], color=color)
                ax.set_xlabel("Frecuencia", color=INK, fontsize=9)
            ax.set_title(title, color=INK, fontsize=10, fontweight="bold")
            _style_ax(ax)
        fig.suptitle(col, color=INK, fontsize=11, fontweight="bold")
        plt.tight_layout(pad=1.2)
        img_b64 = fig_b64(fig)
    except Exception:
        pass

    return jsonify({
        "col": col, "operation": operation, "changed": changed, "dropped": dropped,
        "missing_before": missing_before, "missing_after": missing_after,
        "stats_before": stats_before, "stats_after": stats_after,
        "n_rows": len(rows), "img": img_b64,
        "new_cols": [col + "__" + v for v in sorted(set(str(v) for v in orig_col_vals))]
               if operation == "encode_onehot" else []
    })


@app.route("/api/tabular_preprocess", methods=["POST"])
def tabular_preprocess():
    body             = request.get_json(force=True, silent=True) or {}
    normalize        = body.get("normalize", "none")   # "none" | "minmax" | "zscore"
    impute           = body.get("impute",    "mean")    # "drop" | "mean" | "median" | "mode" | "zero"
    cat_fix          = body.get("cat_fix",  True)       # standardize text categories (lowercase+strip)
    use_train        = body.get("use_train", False)     # fit on train, apply to all
    node_id          = str(body.get("node_id", ""))     # upstream data node
    preprocess_node_id = str(body.get("preprocess_node_id", ""))

    # Read from the correct data slot (respects selectedCols / targetCol)
    D = _effective_data(node_id) if node_id else None
    source_rows = _TAB["sampled_train"] if (use_train and _TAB["sampled_train"]) else (
                  D["raw_rows"] if D else S["raw_rows"])
    apply_rows  = D["raw_rows"] if D else S["raw_rows"]
    cols        = D["columns"]  if D else S["columns"]
    target_col  = D["target"]   if D else _TAB.get("target_col", "")
    if not apply_rows:
        return jsonify({"error": "No data loaded"}), 400

    # --- Step 1: standardize categories ---
    if cat_fix:
        apply_rows = [{c: v.strip().lower() if isinstance(v, str) else v
                       for c, v in r.items()} for r in apply_rows]
        source_rows = [{c: v.strip().lower() if isinstance(v, str) else v
                        for c, v in r.items()} for r in source_rows]

    # --- Step 2: imputation ---
    # Build fill values from source (train)
    fill = {}
    for col in cols:
        vals = [r.get(col, "") for r in source_rows]
        ctype = _col_type(vals)
        non_empty = [v for v in vals if not _is_missing(v)]
        if _is_numeric_type(ctype):
            nums = []
            for v in non_empty:
                try: nums.append(float(v))
                except: pass
            if nums:
                if impute == "median":
                    fill[col] = str(round(float(np.median(nums)), 4))
                elif impute == "mode":
                    c2 = Counter(nums); fill[col] = str(c2.most_common(1)[0][0])
                elif impute == "zero":
                    fill[col] = "0"
                else:  # mean (default)
                    fill[col] = str(round(float(np.mean(nums)), 4))
        else:
            if non_empty and impute != "drop":
                c2 = Counter(str(v) for v in non_empty)
                fill[col] = c2.most_common(1)[0][0]

    # Apply imputation / drop
    result_rows = []
    dropped = 0
    for r in apply_rows:
        if impute == "drop":
            if any(_is_missing(r.get(c,"")) for c in cols):
                dropped += 1; continue
        else:
            r = {c: (fill.get(c, r.get(c,"")) if _is_missing(r.get(c,"")) else r.get(c,"")) for c in cols}
        result_rows.append(r)

    # --- Step 3: normalization (fit on source, apply to result) ---
    params = {}
    if normalize != "none":
        for col in cols:
            vals = [r.get(col,"") for r in source_rows]
            if not _is_numeric_type(_col_type(vals)): continue
            nums = []
            for v in vals:
                try:
                    if not _is_missing(v): nums.append(float(v))
                except: pass
            if not nums: continue
            if normalize == "minmax":
                mn, mx = min(nums), max(nums)
                params[col] = {"method":"minmax","min":mn,"max":mx}
            else:  # zscore
                mu, sd = float(np.mean(nums)), float(np.std(nums))
                params[col] = {"method":"zscore","mean":mu,"std":sd}

        for col, p in params.items():
            for r in result_rows:
                try:
                    v = float(r.get(col, 0) or 0)
                    if p["method"] == "minmax":
                        rng_v = p["max"] - p["min"]
                        r[col] = str(round((v - p["min"]) / rng_v, 6)) if rng_v else "0"
                    else:
                        r[col] = str(round((v - p["mean"]) / p["std"], 6)) if p["std"] else "0"
                except: pass

    # ── Persist processed rows in the preprocess node's slot ─────────────────
    # Downstream blocks (Análisis, Plots) will call _effective_data(preprocess_node_id)
    # and receive the processed rows transparently.
    if preprocess_node_id:
        pslot = _node_slot(preprocess_node_id)
        pslot.update(
            raw_rows=result_rows,
            columns=cols,
            dataset_name=(D["dataset_name"] if D else S["dataset_name"]) + " [preprocesado]",
            task=(D["task"] if D else S["task"]),
            target=target_col,
        )
    # Also update global S so legacy endpoints still work
    _TAB["processed_rows"] = result_rows
    _TAB["proc_params"]    = params

    before_after = {}
    orig_rows = D["raw_rows"] if D else S["raw_rows"]
    for col in cols:
        raw_vals = [r.get(col,"") for r in orig_rows]
        if not _is_numeric_type(_col_type(raw_vals)): continue
        raw_nums = [float(v) for v in raw_vals if not _is_missing(v) and _try_float(v)]
        proc_nums= [float(v) for v in [r.get(col,"") for r in result_rows] if _try_float(v)]
        if raw_nums and proc_nums:
            before_after[col] = {
                "before": {"mean": round(float(np.mean(raw_nums)),4),
                            "std": round(float(np.std(raw_nums)),4),
                            "min": round(min(raw_nums),4), "max": round(max(raw_nums),4)},
                "after":  {"mean": round(float(np.mean(proc_nums)),4),
                            "std": round(float(np.std(proc_nums)),4),
                            "min": round(min(proc_nums),4), "max": round(max(proc_nums),4)},
            }
        if len(before_after) >= 6: break

    return jsonify({
        "n_in": len(apply_rows), "n_out": len(result_rows), "dropped": dropped,
        "normalize": normalize, "impute": impute,
        "params": {k: {kk: round(vv,4) if isinstance(vv,float) else vv
                       for kk,vv in v.items()} for k,v in params.items()},
        "before_after": before_after,
        "sample": result_rows[:8],
    })

def _try_float(v):
    """Return the float value if v is convertible, else None."""
    try: return float(v)
    except: return None

# ── Box plot (for outlier detection) ─────────────────────────────────────────
@app.route("/api/plot_boxplot")
def plot_boxplot():
    cols_req = request.args.get("cols", "")
    color_by = request.args.get("color_by", "")
    rows = S["raw_rows"]
    if not rows:
        return jsonify({"error": "No data"}), 400

    # Columns to plot: explicit list or all numeric
    if cols_req:
        selected = [c.strip() for c in cols_req.split(",") if c.strip() in S["columns"]]
    else:
        selected = [c for c in S["columns"] if _is_numeric_type(_col_type([r.get(c,"") for r in rows]))]
    selected = selected[:8]  # max 8

    if not selected:
        return jsonify({"error": "No numeric columns found"}), 400

    fig, axes = plt.subplots(1, len(selected), figsize=(max(5, len(selected)*2.2), 5))
    fig.patch.set_facecolor(LIGHT)
    if len(selected) == 1: axes = [axes]

    for ax, col in zip(axes, selected):
        style_ax(ax)
        vals = [r.get(col,"") for r in rows]
        nums = [float(v) for v in vals if not _is_missing(v) and _try_float(v)]
        if not nums: continue
        bp = ax.boxplot(nums, patch_artist=True, widths=0.5,
                        medianprops=dict(color=MINT, linewidth=2),
                        boxprops=dict(facecolor=PALETTE[0], alpha=0.7),
                        flierprops=dict(marker="o", color=PALETTE[1],
                                        markerfacecolor=PALETTE[1], markersize=5, alpha=0.7),
                        whiskerprops=dict(color=SEC), capprops=dict(color=SEC))
        ax.set_xticklabels([col], color=INK, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Value", color=SEC, fontsize=9)
        q1,med,q3 = np.percentile(nums,[25,50,75])
        iqr = q3 - q1
        n_out = sum(1 for v in nums if v < q1-1.5*iqr or v > q3+1.5*iqr)
        ax.set_title(f"{col}\n({n_out} outliers)", color=INK, fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout(pad=1.4)
    return jsonify({"img": fig_b64(fig)})

# ════════════════════════════════════════════════════════════════════════════
# REGRESSION  (OLS · Ridge · LASSO · metrics · CV)
# ════════════════════════════════════════════════════════════════════════════

def _build_Xy(rows, cols, target_col, normalize="none"):
    """Return (X, y, feature_names) from raw_rows.

    y can be numeric (regression) or string (classification).
    Categorical feature columns are one-hot encoded (up to 10 categories).
    Rows with a missing target are dropped; missing numeric features are
    mean-imputed.
    """
    # Strictly exclude target from features — compare by exact string match
    feature_cols = [c for c in cols if str(c).strip() != str(target_col).strip()]
    if not feature_cols:
        return None, f"No hay columnas de features después de excluir '{target_col}'", []

    # Determine which feature cols are numeric vs categorical
    all_vals = {c: [r.get(c, "") for r in rows] for c in feature_cols}
    num_feat  = [c for c in feature_cols if _is_numeric_type(_col_type(all_vals[c]))]
    cat_feat  = [c for c in feature_cols if _col_type(all_vals[c]) == "categorical"]

    # Build category maps for one-hot encoding (max 10 cats per column)
    cat_maps = {}
    for c in cat_feat:
        cats = sorted(set(str(v) for v in all_vals[c] if not _is_missing(v)))[:10]
        cat_maps[c] = cats

    # Determine target type
    targ_vals = [r.get(target_col, "") for r in rows]
    targ_is_numeric = _is_numeric_type(_col_type([v for v in targ_vals if not _is_missing(v)]))

    X_raw, y_raw = [], []
    for r in rows:
        yv = r.get(target_col, "")
        if _is_missing(yv):
            continue
        # Parse target
        if targ_is_numeric:
            yf = _try_float(yv)
            if yf is None:
                continue
            y_parsed = yf
        else:
            y_parsed = str(yv).strip()

        row_x = []
        # Numeric features
        for c in num_feat:
            v = r.get(c, "")
            fv = _try_float(v)
            row_x.append(float(fv) if fv is not None and not _is_missing(v) else float("nan"))
        # Categorical features → one-hot
        for c in cat_feat:
            v = str(r.get(c, "")).strip()
            for cat in cat_maps[c]:
                row_x.append(1.0 if v == cat else 0.0)

        X_raw.append(row_x)
        y_raw.append(y_parsed)

    if not X_raw:
        return None, "No quedan filas válidas tras filtrar target ausente", []

    X = np.array(X_raw, dtype=float)
    y = np.array(y_raw)          # object dtype for strings, float64 for numeric

    # Feature names including one-hot suffixes
    feat_names = list(num_feat)
    for c in cat_feat:
        for cat in cat_maps[c]:
            feat_names.append(f"{c}={cat}")

    # Mean imputation for NaN in numeric columns only (first len(num_feat) cols)
    if X.ndim == 2 and X.shape[1] > 0:
        for j in range(len(num_feat)):
            col_vals = X[:, j]
            nan_mask = np.isnan(col_vals)
            if nan_mask.any():
                col_mean = float(np.nanmean(col_vals)) if not np.all(nan_mask) else 0.0
                X[nan_mask, j] = col_mean

    if normalize == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif normalize == "zscore":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feat_names

def _reg_metrics(y_true, y_pred):
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(mse ** 0.5)
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))) * 100
    return {"MSE": round(mse,4), "RMSE": round(rmse,4),
            "MAE": round(mae,4),  "MAPE": round(mape,2), "R2": round(r2,4)}

@app.route("/api/regression", methods=["POST"])
def run_regression():
    body       = request.get_json(force=True, silent=True) or {}
    node_id    = str(body.get("node", ""))
    D          = _effective_data(node_id) if node_id else None
    target_col = body.get("target_col") or (D["target"] if D else None) or _TAB.get("target_col","")
    method     = body.get("method", "ols")      # ols | ridge | lasso
    alpha      = float(body.get("alpha", 1.0))
    normalize  = body.get("normalize", "none")  # none | minmax | zscore
    test_size  = float(body.get("test_size", 0.3))
    seed       = int(body.get("seed", 42))

    rows = (D["raw_rows"] if D else None) or S["raw_rows"]
    cols = (D["columns"]  if D else None) or S["columns"]
    if not rows or not target_col:
        return jsonify({"error": "No data or no target column specified"}), 400
    if target_col not in cols:
        return jsonify({"error": f"Column '{target_col}' not found"}), 400

    X, y, feat_names = _build_Xy(rows, cols, target_col, normalize)
    if len(X) < 10:
        return jsonify({"error": "Not enough valid rows (need ≥10)"}), 400

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                           random_state=seed)

    if method == "ridge":
        model = Ridge(alpha=alpha)
    elif method == "lasso":
        model = Lasso(alpha=alpha, max_iter=5000)
    else:
        model = LinearRegression()

    model.fit(Xtr, ytr)
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)

    train_m = _reg_metrics(ytr, ytr_pred)
    test_m  = _reg_metrics(yte, yte_pred)

    coefs = list(zip(feat_names, [round(float(c),6) for c in model.coef_]))
    coefs_sorted = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)

    # ── Plot: predicted vs actual ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(LIGHT)

    ax = axes[0]; style_ax(ax)
    mn_v = min(float(yte.min()), float(yte_pred.min()))
    mx_v = max(float(yte.max()), float(yte_pred.max()))
    ax.scatter(yte, yte_pred, color=PALETTE[0], alpha=0.65, s=28, edgecolors="none", zorder=3)
    ax.plot([mn_v, mx_v], [mn_v, mx_v], color=MINT, linewidth=1.8, linestyle="--", label="Ideal")
    ax.set_xlabel("Actual", color=SEC, fontsize=11)
    ax.set_ylabel("Predicted", color=SEC, fontsize=11)
    ax.set_title(f"Predicted vs Actual  (R²={test_m['R2']})", color=INK, fontsize=12, fontweight="bold")
    ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=9)
    ax.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)

    # Residuals
    ax2 = axes[1]; style_ax(ax2)
    resid = yte - yte_pred
    ax2.axhline(0, color=MINT, linewidth=1.5, linestyle="--")
    ax2.scatter(yte_pred, resid, color=PALETTE[1], alpha=0.65, s=28, edgecolors="none", zorder=3)
    ax2.set_xlabel("Fitted values", color=SEC, fontsize=11)
    ax2.set_ylabel("Residuals", color=SEC, fontsize=11)
    ax2.set_title("Residuals plot", color=INK, fontsize=12, fontweight="bold")
    ax2.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout(pad=1.6)
    plot_img = fig_b64(fig)

    # ── Coefficient bar chart ─────────────────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(7, max(3, len(coefs_sorted)*0.4 + 1)))
    fig2.patch.set_facecolor(LIGHT); style_ax(ax3)
    names_c = [c[0] for c in coefs_sorted]
    vals_c  = [c[1] for c in coefs_sorted]
    colors_c= [PALETTE[0] if v >= 0 else PALETTE[1] for v in vals_c]
    ax3.barh(names_c, vals_c, color=colors_c, edgecolor="none", alpha=0.85)
    ax3.axvline(0, color=SEC, linewidth=0.8)
    ax3.set_xlabel("Coefficient", color=SEC, fontsize=11)
    ax3.set_title(f"Coefficients — {method.upper()} (α={alpha})", color=INK,
                  fontsize=12, fontweight="bold")
    ax3.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    plt.tight_layout(pad=1.4)
    coef_img = fig_b64(fig2)

    return jsonify({
        "method": method, "alpha": alpha, "normalize": normalize,
        "n_train": len(Xtr), "n_test": len(Xte),
        "features": feat_names,
        "coefficients": coefs_sorted,
        "intercept": round(float(model.intercept_), 6),
        "train_metrics": train_m, "test_metrics": test_m,
        "plot_img": plot_img, "coef_img": coef_img,
    })

@app.route("/api/regression_cv", methods=["POST"])
def regression_cv():
    """Legacy alias — delegates to model_cv."""
    return model_cv()

@app.route("/api/model_cv", methods=["POST"])
def model_cv():
    """Grid Search + K-fold CV across hyperparameter × normalization combinations.

    For models with a main hyperparameter (ridge λ, lasso λ, logistic C, knn k):
      - Sweeps all param_vals × all normalizations → 2-D grid
      - Returns one curve per normalization, best combo overall
      - Trains final model with best combo and stores in _MODEL_STORE[node_id]

    For OLS (no hyperparameter):
      - Sweeps normalizations only → single bar chart
      - Trains final model with best normalization
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    body        = request.get_json(force=True, silent=True) or {}
    upstream_id = str(body.get("upstream_id", "")) or None
    node_id     = str(body.get("node_id", "")) or None
    target_col  = str(body.get("target_col", "")) or _TAB.get("target_col", "")
    method      = str(body.get("method", "ridge"))
    param_vals  = body.get("param_vals", None)   # explicit list from UI or None
    k_folds     = int(body.get("k_folds", 5))
    # normalizations to sweep (always all three for grid search)
    norm_sweep  = ["none", "zscore", "minmax"]
    norm_labels = {"none": "Sin norm.", "zscore": "Z-score", "minmax": "Min-Max"}

    # ── Resolve data ──────────────────────────────────────────────────────
    D    = _effective_data(upstream_id) if upstream_id else None
    rows = (D["raw_rows"] if D else None) or S["raw_rows"]
    cols = (D["columns"]  if D else None) or S["columns"]
    if not rows:
        return jsonify({"error": "No hay datos cargados"}), 400
    if not target_col:
        target_col = (D["target"] if D else None) or _TAB.get("target_col","") or (cols[-1] if cols else "")

    # Build raw X, y (no scaler — Pipeline handles per fold)
    result_xy = _build_Xy(rows, cols, target_col, normalize="none")
    if result_xy[0] is None:
        return jsonify({"error": result_xy[1]}), 400
    X, y_raw, feat_names = result_xy

    # ── Detect problem type ───────────────────────────────────────────────
    tgt_vals = [r.get(target_col,"") for r in rows if not _is_missing(r.get(target_col,""))]
    tgt_type = _col_type(tgt_vals)
    _reg_methods = {"ridge", "lasso", "linreg"}
    _cls_methods = {"logistic", "knn"}
    if method in _reg_methods:
        is_cls = False
    elif method in _cls_methods:
        is_cls = True
    else:
        is_cls = not _is_numeric_type(tgt_type) or len(set(str(v).strip() for v in tgt_vals)) <= 10

    # ── Encode y ──────────────────────────────────────────────────────────
    if is_cls:
        def _tstr(v):
            try:
                f = float(v); return str(int(f)) if f == int(f) else str(f)
            except: return str(v).strip()
        classes   = sorted(set(_tstr(v) for v in y_raw))
        label_map = {c: i for i, c in enumerate(classes)}
        y         = np.array([label_map[_tstr(v)] for v in y_raw])
    else:
        y = y_raw.astype(float)

    if len(X) < k_folds * 2:
        return jsonify({"error": f"Pocos datos para {k_folds}-fold CV. Necesitas al menos {k_folds*2} filas."}), 400

    # ── CV config per model ───────────────────────────────────────────────
    MODEL_CV_CFG = {
        "ridge":    {"param": "alpha", "label": "λ (alpha)", "log": True,
                     "default": [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]},
        "lasso":    {"param": "alpha", "label": "λ (alpha)", "log": True,
                     "default": [0.0001,0.001,0.01,0.1,1.0,10.0,100.0]},
        "linreg":   {"param": None,    "label": None,        "log": False,
                     "default": [1]},
        "logistic": {"param": "C",     "label": "C (inv. regularización)", "log": True,
                     "default": [0.001,0.01,0.1,1.0,10.0,100.0,1000.0]},
        "knn":      {"param": "k",     "label": "k (vecinos)", "log": False,
                     "default": [1,3,5,7,9,11,15,21]},
    }
    cfg   = MODEL_CV_CFG.get(method, MODEL_CV_CFG["ridge"])
    pvals = param_vals if param_vals else cfg["default"]
    has_param = cfg["param"] is not None

    # ── Scoring ───────────────────────────────────────────────────────────
    if is_cls:
        cv_split = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        scoring  = "f1_macro"
        metric_label = "F1 (macro)"
        higher_is_better = True
    else:
        cv_split = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scoring  = "neg_mean_squared_error"
        metric_label = "RMSE"
        higher_is_better = False

    def _make_scaler_step(norm):
        if norm == "zscore":  return ("scaler", StandardScaler())
        if norm == "minmax":  return ("scaler", MinMaxScaler())
        return None

    def _make_estimator(pv):
        pv = float(pv)
        if method == "ridge":    return Ridge(alpha=pv)
        if method == "lasso":    return Lasso(alpha=pv, max_iter=10000)
        if method == "logistic": return LogisticRegression(C=pv, max_iter=2000, random_state=42, solver="saga")
        if method == "knn":
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            return KNeighborsClassifier(n_neighbors=int(pv)) if is_cls else KNeighborsRegressor(n_neighbors=int(pv))
        return LinearRegression()

    def _cv_score(est, scaler_step):
        if scaler_step:
            pipe = Pipeline([scaler_step, ("model", est)])
        else:
            pipe = est
        raw = cross_val_score(pipe, X, y, cv=cv_split, scoring=scoring)
        if not is_cls:
            scores = [float((-s)**0.5) for s in raw]
        else:
            scores = [float(s) for s in raw]
        return round(float(np.mean(scores)), 6), round(float(np.std(scores)), 6)

    # ── Grid search: param_vals × normalizations ──────────────────────────
    # results_by_norm: { norm: [ {param_val, score_mean, score_std}, ... ] }
    results_by_norm = {}
    all_combos      = []   # flat list for finding global best

    effective_pvals = pvals if has_param else [None]

    for norm in norm_sweep:
        scaler_step = _make_scaler_step(norm)
        norm_results = []
        for pv in effective_pvals:
            est = _make_estimator(pv if pv is not None else 1.0)
            mean_s, std_s = _cv_score(est, scaler_step)
            entry = {
                "param_val":  pv,
                "norm":       norm,
                "score_mean": mean_s,
                "score_std":  std_s,
            }
            norm_results.append(entry)
            all_combos.append(entry)
        results_by_norm[norm] = norm_results

    # Global best combo
    if higher_is_better:
        best_combo = max(all_combos, key=lambda r: r["score_mean"])
    else:
        best_combo = min(all_combos, key=lambda r: r["score_mean"])

    best_norm  = best_combo["norm"]
    best_pv    = best_combo["param_val"]
    best_score = best_combo["score_mean"]

    # ── Train final model with best combo (full train set) ────────────────
    # Use the same split as model_train would use (from preprocess or 70/30)
    split_info = D.get("split") if D else None
    if split_info and split_info.get("train_idx") is not None:
        tr_idx = split_info["train_idx"]
        te_idx = split_info["test_idx"]
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
    else:
        from sklearn.model_selection import train_test_split as tts
        X_tr, X_te, y_tr, y_te = tts(X, y, test_size=0.3, random_state=42)

    # Apply best normalization to final model
    if best_norm == "zscore":
        from sklearn.preprocessing import StandardScaler as SS
        sc = SS(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
    elif best_norm == "minmax":
        from sklearn.preprocessing import MinMaxScaler as MMS
        sc = MMS(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
    else:
        sc = None; X_tr_s = X_tr; X_te_s = X_te

    final_est = _make_estimator(best_pv if best_pv is not None else 1.0)
    final_est.fit(X_tr_s, y_tr)
    y_tr_pred = final_est.predict(X_tr_s)
    y_te_pred = final_est.predict(X_te_s)

    if not is_cls:
        train_m = _reg_metrics(y_tr, y_tr_pred)
        test_m  = _reg_metrics(y_te, y_te_pred)
    else:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        train_m = {"accuracy": round(float(accuracy_score(y_tr, y_tr_pred)), 4)}
        test_m  = {
            "accuracy":  round(float(accuracy_score(y_te, y_te_pred)), 4),
            "f1_macro":  round(float(f1_score(y_te, y_te_pred, average="macro", zero_division=0)), 4),
            "precision": round(float(precision_score(y_te, y_te_pred, average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_te, y_te_pred, average="macro", zero_division=0)), 4),
        }

    coefs = []
    if hasattr(final_est, "coef_"):
        coefs = list(zip(feat_names, [round(float(c), 6) for c in final_est.coef_]))
        coefs = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
    intercept = round(float(final_est.intercept_), 6) if hasattr(final_est, "intercept_") else 0.0

    # Store in _MODEL_STORE so model_eval can read it
    if node_id:
        _MODEL_STORE[node_id] = {
            "method":        method,
            "normalize":     best_norm,
            "alpha":         best_pv,
            "problem_type":  "classification" if is_cls else "regression",
            "feat_names":    feat_names,
            "coefficients":  coefs,
            "intercept":     intercept,
            "n_train":       int(len(X_tr)),
            "n_test":        int(len(X_te)),
            "train_metrics": train_m,
            "test_metrics":  test_m,
            "X_te":          X_te_s.tolist(),
            "y_te":          y_te.tolist(),
            "y_te_pred":     y_te_pred.tolist(),
            "X_tr":          X_tr_s.tolist(),
            "y_tr":          y_tr.tolist(),
            "y_tr_pred":     y_tr_pred.tolist(),
            "scaler":        sc,
            "model_obj":     final_est,
            "from_cv":       True,
            # CV summary fields — used by model_evaluate to populate CV tab
            "cv_img":           None,   # filled below after plot is generated
            "param_label":      cfg["label"],
            "metric_label":     metric_label,
            "k_folds":          k_folds,
            "best":             best_combo,
            "best_pv_display":  None,   # filled below after formatting
        }

    # ── Plot: one curve per normalization ─────────────────────────────────
    use_log = cfg["log"] and has_param and len(set(float(p) for p in pvals if p)) > 1 and min(float(p) for p in pvals if p) > 0

    if has_param:
        # Line chart: X = param values, one line per normalization
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor(BG); style_ax(ax)

        norm_colors = {"none": PALETTE[0], "zscore": PALETTE[1], "minmax": PALETTE[2]}
        for ni, norm in enumerate(norm_sweep):
            nr = results_by_norm[norm]
            xs  = [r["param_val"] for r in nr]
            ms  = [r["score_mean"] for r in nr]
            sts = [r["score_std"]  for r in nr]
            col = norm_colors[norm]
            plot_fn = ax.semilogx if use_log else ax.plot
            plot_fn(xs, ms, color=col, linewidth=2, marker="o", markersize=5,
                    label=norm_labels[norm], zorder=3+ni)
            ax.fill_between(xs, [m-s for m,s in zip(ms,sts)], [m+s for m,s in zip(ms,sts)],
                            alpha=0.12, color=col, zorder=2)

        # Mark global best
        if best_pv is not None:
            ax.axvline(best_pv, color=MINT, linewidth=1.8, linestyle="--", zorder=6,
                       label=f"Mejor: {norm_labels[best_norm]}, {cfg['label']}={round(best_pv,4)} → {metric_label}={best_score}")

        ax.set_xlabel((cfg["label"] or "Parámetro") + (" (log)" if use_log else ""), color=SEC, fontsize=11)
        ax.set_ylabel(f"{metric_label} ({PL()['cv_score']})", color=SEC, fontsize=11)
        ax.set_title(f"{method.upper()} — Grid Search {k_folds}-fold · {metric_label}", color=INK, fontsize=12, fontweight="bold")
        ax.legend(facecolor=BG, labelcolor=INK, fontsize=9, loc="best")
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
        # Limit X ticks to at most 8, evenly spaced from the actual param values
        all_xs = sorted(set(float(p) for p in pvals if p is not None))
        if len(all_xs) > 8:
            step = max(1, len(all_xs) // 8)
            tick_xs = all_xs[::step]
        else:
            tick_xs = all_xs
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(
            [str(int(v)) if v == int(v) else (f"{v:.2e}" if v < 0.01 else str(round(v, 4))) for v in tick_xs],
            color=SEC, fontsize=9
        )
        plt.tight_layout(pad=1.4)
    else:
        # OLS: bar chart comparing normalizations
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(BG); style_ax(ax)
        bar_names   = [norm_labels[n] for n in norm_sweep]
        bar_scores  = [results_by_norm[n][0]["score_mean"] for n in norm_sweep]
        bar_stds    = [results_by_norm[n][0]["score_std"]  for n in norm_sweep]
        bar_colors  = [MINT if n == best_norm else PALETTE[0] for n in norm_sweep]
        ax.bar(bar_names, bar_scores, color=bar_colors, alpha=0.85, edgecolor="none",
               yerr=bar_stds, capsize=5, error_kw={"ecolor": SEC, "linewidth":1.2})
        ax.set_ylabel(f"{metric_label} ({PL()['cv_score']})", color=SEC, fontsize=11)
        ax.set_title(f"OLS — {'Comparación de normalizaciones' if _UI_LANG=='es' else 'Normalization comparison'} ({k_folds}-fold)", color=INK, fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
        plt.tight_layout(pad=1.4)

    cv_img = fig_b64(fig)

    # Format best param for display
    if best_pv is not None:
        if cfg.get("isInt"):
            best_pv_display = str(int(round(best_pv)))
        elif best_pv < 0.01:
            best_pv_display = f"{best_pv:.2e}"
        else:
            best_pv_display = str(round(best_pv, 4))
    else:
        best_pv_display = None

    # Backfill cv_img + best_pv_display into _MODEL_STORE now that they're available
    if node_id and node_id in _MODEL_STORE:
        _MODEL_STORE[node_id]["cv_img"] = cv_img
        _MODEL_STORE[node_id]["best_pv_display"] = best_pv_display

    return jsonify({
        "results_by_norm": results_by_norm,
        "best":            best_combo,
        "best_pv_display": best_pv_display,
        "method":          method,
        "param_label":     cfg["label"],
        "metric_label":    metric_label,
        "k_folds":         k_folds,
        "problem_type":    "classification" if is_cls else "regression",
        "cv_img":          cv_img,
        "has_param":       has_param,
        "train_metrics":   train_m,
        "test_metrics":    test_m,
        "n_train":         int(len(X_tr)),
        "n_test":          int(len(X_te)),
        "norm_labels":     norm_labels,
    })

# ══════════════════════════════════════════════════════════════════════════════
# LAB 4 — TABULAR CLASSIFICATION
# Endpoints: /api/tab_classify  /api/tab_classify_cv  /api/tab_classify_scatter
# /api/tab_classify_boundary  /api/tab_classify_imbalance
# ══════════════════════════════════════════════════════════════════════════════

def _cls_metrics(y_true, y_pred, all_labels=None):
    """Return dict of accuracy, precision, recall, f1 (macro) + confusion matrix.

    all_labels: full list of known integer class labels — ensures cm shape is
    always n_classes × n_classes even when a fold contains only one class.
    """
    labels_arg = sorted(all_labels) if all_labels is not None else None
    acc  = round(float(accuracy_score(y_true, y_pred)), 4)
    prec = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_arg)), 4)
    rec  = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_arg)), 4)
    f1   = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_arg)), 4)
    cm   = confusion_matrix(y_true, y_pred, labels=labels_arg).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}

def _build_classifier(name, hp):
    """Instantiate a sklearn classifier from name + hp dict."""
    c = hp.get("C", 1.0)
    k = hp.get("k", 5)
    n = hp.get("n_estimators", 100)
    d = hp.get("max_depth", None) if hp.get("max_depth", 0) != 0 else None
    if name == "logistic":
        return LogisticRegression(C=float(c), max_iter=2000, random_state=42, solver="saga")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=int(k))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=int(n), max_depth=d, random_state=42, n_jobs=-1)
    if name == "svm":
        return LinearSVC(C=float(c), max_iter=3000, random_state=42)
    return LogisticRegression(max_iter=2000, random_state=42)

def _smote_oversample(X, y):
    """Simple manual SMOTE-lite: duplicate minority class with small jitter."""
    import random as rnd
    classes, counts = np.unique(y, return_counts=True)
    majority_cls = classes[np.argmax(counts)]
    minority_cls = classes[np.argmin(counts)]
    X_min = X[y == minority_cls]
    n_to_add = int(counts.max()) - int(counts.min())
    new_rows = []
    for _ in range(n_to_add):
        i, j = rnd.sample(range(len(X_min)), 2)
        alpha = rnd.random()
        new_rows.append(X_min[i] * alpha + X_min[j] * (1 - alpha))
    X_new = np.vstack([X, np.array(new_rows)])
    y_new = np.concatenate([y, np.full(n_to_add, minority_cls)])
    return X_new, y_new

@app.route("/api/tab_classify", methods=["POST"])
def tab_classify():
    """Train + evaluate one or more classifiers on tabular data."""
    body       = request.get_json(force=True, silent=True) or {}
    node_id    = str(body.get("node", ""))
    D          = _effective_data(node_id) if node_id else None
    target_col = body.get("target_col") or (D["target"] if D else None) or ""
    models_req = body.get("models", ["logistic"])   # list of model names
    normalize  = body.get("normalize", "zscore")
    test_size  = float(body.get("test_size", 0.3))
    hp         = body.get("hp", {})
    imbalance  = body.get("imbalance_strategy", "none")  # none | oversample | undersample | weights

    rows = (D["raw_rows"] if D else None) or S["raw_rows"]
    if not rows:
        return jsonify({"error": "No data loaded"}), 400
    if not target_col:
        return jsonify({"error": "Selecciona la variable objetivo"}), 400

    all_cols = (D["columns"] if D else None) or S["columns"]
    cols = [c for c in all_cols if c != target_col]
    X_all, y_all, feat_names = _build_Xy(rows, cols, target_col, normalize=normalize)
    if X_all is None:
        return jsonify({"error": y_all}), 400

    # Encode target to int labels
    # If target came back as float (0.0 / 1.0) normalise to int strings ("0","1")
    def _tstr(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        except (ValueError, TypeError):
            return str(v).strip()
    classes = sorted(list(set(_tstr(v) for v in y_all)))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_map[_tstr(v)] for v in y_all])

    # Stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_int, test_size=test_size, random_state=42, stratify=y_int)

    # Class distribution
    tr_dist = {classes[i]: int(np.sum(y_tr==i)) for i in range(len(classes))}
    te_dist = {classes[i]: int(np.sum(y_te==i)) for i in range(len(classes))}
    imbalance_ratio = round(float(max(tr_dist.values())) / max(1, float(min(tr_dist.values()))), 2)

    # Imbalance strategy on training set
    X_tr_use, y_tr_use = X_tr.copy(), y_tr.copy()
    if imbalance == "oversample":
        X_tr_use, y_tr_use = _smote_oversample(X_tr_use, y_tr_use)
    elif imbalance == "undersample":
        classes_u, counts_u = np.unique(y_tr_use, return_counts=True)
        min_count = int(counts_u.min())
        idx_keep = []
        for cls_u in classes_u:
            idx_cls = np.where(y_tr_use == cls_u)[0]
            idx_keep.extend(np.random.choice(idx_cls, min_count, replace=False).tolist())
        X_tr_use = X_tr_use[idx_keep]; y_tr_use = y_tr_use[idx_keep]

    results = []
    coef_imgs = {}
    for model_name in models_req:
        clf_hp = hp.get(model_name, {})
        # class_weight for logistic / svm
        use_weights = (imbalance == "weights")
        if use_weights and model_name in ("logistic",):
            clf_hp["class_weight"] = "balanced"
        clf = _build_classifier(model_name, clf_hp)
        try:
            clf.fit(X_tr_use, y_tr_use)
        except Exception as e:
            results.append({"model": model_name, "error": str(e)})
            continue
        y_pred_tr = clf.predict(X_tr_use)
        y_pred_te = clf.predict(X_te)
        all_lbl   = list(range(len(classes)))
        tr_m = _cls_metrics(y_tr_use, y_pred_tr, all_labels=all_lbl)
        te_m = _cls_metrics(y_te,     y_pred_te,  all_labels=all_lbl)

        # Coefficient plot (logistic / svm)
        coef_img = None
        if hasattr(clf, "coef_"):
            coef = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
            top_n = min(12, len(feat_names))
            idx_sorted = np.argsort(np.abs(coef))[::-1][:top_n]
            names_top = [feat_names[i] for i in idx_sorted]
            vals_top  = [float(coef[i]) for i in idx_sorted]
            fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.35)))
            fig.patch.set_facecolor(LIGHT); style_ax(ax)
            colors = [PALETTE[0] if v >= 0 else "#ef4444" for v in vals_top]
            ax.barh(range(len(names_top)), vals_top[::-1], color=colors[::-1],
                    edgecolor="none", zorder=3)
            ax.set_yticks(range(len(names_top)))
            ax.set_yticklabels(names_top[::-1], fontsize=9, color=INK)
            ax.axvline(0, color=INK, linewidth=0.8)
            ax.set_title("Coeficientes — " + model_name, color=INK,
                         fontsize=11, fontweight="bold")
            ax.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
            plt.tight_layout(pad=1.2)
            coef_img = fig_b64(fig)

        # Feature importance (random forest)
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            top_n = min(12, len(feat_names))
            idx_sorted = np.argsort(imp)[::-1][:top_n]
            names_top = [feat_names[i] for i in idx_sorted]
            vals_top  = [float(imp[i]) for i in idx_sorted]
            fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.35)))
            fig.patch.set_facecolor(LIGHT); style_ax(ax)
            ax.barh(range(len(names_top)), vals_top[::-1],
                    color=PALETTE[:len(names_top)][::-1], edgecolor="none", zorder=3)
            ax.set_yticks(range(len(names_top)))
            ax.set_yticklabels(names_top[::-1], fontsize=9, color=INK)
            ax.set_title("Importancia de variables — " + model_name, color=INK,
                         fontsize=11, fontweight="bold")
            ax.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
            plt.tight_layout(pad=1.2)
            coef_img = fig_b64(fig)

        results.append({
            "model": model_name,
            "train": tr_m, "test": te_m,
            "n_train": len(y_tr_use), "n_test": len(y_te),
            "coef_img": coef_img,
            "classes": classes,
        })

    # Confusion matrix plot for first model
    cm_img = None
    if results and "cm" in results[0].get("test", {}):
        cm = np.array(results[0]["test"]["cm"])
        fig, ax = plt.subplots(figsize=(max(4, len(classes)), max(3.5, len(classes)*0.8)))
        fig.patch.set_facecolor(LIGHT); style_ax(ax)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right", color=INK)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=INK)
        ax.set_xlabel(PL()["pred_lbl"], color=SEC, fontsize=11)
        ax.set_ylabel(PL()["actual_lbl"], color=SEC, fontsize=11)
        ax.set_title(PL()["cm_title"] + " — " + (results[0]["model"] if results else ""), color=INK, fontsize=12, fontweight="bold")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else INK, fontsize=13, fontweight="bold")
        plt.tight_layout(pad=1.4)
        cm_img = fig_b64(fig)

    return jsonify({
        "results": results, "cm_img": cm_img,
        "train_dist": tr_dist, "test_dist": te_dist,
        "imbalance_ratio": imbalance_ratio,
        "classes": classes,
        "imbalance_strategy": imbalance,
    })

@app.route("/api/tab_classify_cv", methods=["POST"])
def tab_classify_cv():
    """K-fold cross-validation for one classifier."""
    body       = request.get_json(force=True, silent=True) or {}
    node_id    = str(body.get("node", ""))
    target_col = body.get("target_col", "")
    model_name = body.get("model", "logistic")
    normalize  = body.get("normalize", "zscore")
    k_folds    = int(body.get("k_folds", 5))
    hp         = body.get("hp", {})

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    D    = _effective_data(node_id) if node_id else None
    rows = (D["raw_rows"] if D else None) or S["raw_rows"]
    all_cols = (D["columns"] if D else None) or S["columns"]
    if not rows: return jsonify({"error": "No data"}), 400

    cols = [c for c in all_cols if c != target_col]
    X, y_raw, feat_names = _build_Xy(rows, cols, target_col, normalize=normalize)
    if X is None: return jsonify({"error": y_raw}), 400

    def _tstr(v):
        try:
            f = float(v); return str(int(f)) if f == int(f) else str(f)
        except (ValueError, TypeError): return str(v).strip()
    classes = sorted(list(set(_tstr(v) for v in y_raw)))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[_tstr(v)] for v in y_raw])

    clf = _build_classifier(model_name, hp)
    all_lbl = list(range(len(classes)))

    # Reduce k if fewer samples than requested folds
    min_class_count = int(np.min([np.sum(y == c) for c in all_lbl]))
    k_folds = min(k_folds, min_class_count)
    if k_folds < 2:
        return jsonify({"error": f"Muy pocas muestras por clase ({min_class_count}) para hacer CV. Reduce K o usa más datos."}), 400

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_rows = []
    scores_acc, scores_f1 = [], []
    for i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        # Skip fold if train split has only one class
        unique_tr = np.unique(y[tr_idx])
        if len(unique_tr) < 2:
            fold_rows.append({"fold": i+1, "accuracy": None, "f1": None,
                               "precision": None, "recall": None, "skipped": True})
            continue
        c2 = _build_classifier(model_name, hp)
        try:
            c2.fit(X[tr_idx], y[tr_idx])
            yp = c2.predict(X[te_idx])
            m  = _cls_metrics(y[te_idx], yp, all_labels=all_lbl)
            fold_rows.append({"fold": i+1, "accuracy": m["accuracy"],
                               "f1": m["f1"], "precision": m["precision"],
                               "recall": m["recall"]})
            scores_acc.append(m["accuracy"])
            scores_f1.append(m["f1"])
        except Exception as e:
            fold_rows.append({"fold": i+1, "accuracy": None, "f1": None,
                               "precision": None, "recall": None,
                               "error": str(e)})

    if not scores_acc:
        return jsonify({"error": "Todos los folds fallaron — probablemente el dataset tiene muy pocas muestras de alguna clase."}), 400

    scores_acc = np.array(scores_acc)
    scores_f1  = np.array(scores_f1)

    # Plot only valid folds
    valid_folds = [r for r in fold_rows if r.get("accuracy") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(LIGHT)
    for ax, metric_key, label in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1 (macro)"]):
        style_ax(ax)
        vals = [r[metric_key] for r in valid_folds]
        xs   = [r["fold"] for r in valid_folds]
        ax.bar(xs, vals, color=PALETTE[:len(xs)], edgecolor="none", zorder=3)
        ax.axhline(float(np.mean(vals)), color=INK, linewidth=1.4,
                   linestyle="--", label=f"Media {np.mean(vals):.3f}")
        ax.set_xticks(xs); ax.set_xticklabels([f"Fold {x}" for x in xs], fontsize=9, color=INK)
        ax.set_ylim(0, 1.05)
        ax.set_title(label, color=INK, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
        ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=9)
    plt.tight_layout(pad=1.4)
    cv_img = fig_b64(fig)

    return jsonify({
        "fold_rows": fold_rows,
        "mean_acc": round(float(np.mean(scores_acc)), 4),
        "std_acc":  round(float(np.std(scores_acc)), 4),
        "mean_f1":  round(float(np.mean(scores_f1)), 4),
        "std_f1":   round(float(np.std(scores_f1)), 4),
        "cv_img": cv_img, "model": model_name, "k_folds": k_folds,
    })

@app.route("/api/tab_classify_scatter", methods=["POST"])
def tab_classify_scatter():
    """2D scatter of two features coloured by target class."""
    body       = request.get_json(force=True, silent=True) or {}
    target_col = body.get("target_col", "")
    feat_x     = body.get("feat_x", "")
    feat_y     = body.get("feat_y", "")

    rows = S["raw_rows"]
    if not rows: return jsonify({"error": "No data"}), 400
    cols = S["columns"]
    num_cols = [c for c in cols if _col_type([r.get(c,"") for r in rows])=="numeric" and c != target_col]
    if not feat_x: feat_x = num_cols[0] if num_cols else ""
    if not feat_y: feat_y = num_cols[1] if len(num_cols) > 1 else feat_x

    def get_vals(col):
        return [r.get(col,"") for r in rows]

    targ = [str(r.get(target_col,"")) for r in rows]
    classes = sorted(list(set(targ)))
    xs = [_try_float(v) for v in get_vals(feat_x)]
    ys = [_try_float(v) for v in get_vals(feat_y)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(LIGHT); style_ax(ax)
    for i, cls in enumerate(classes):
        idx = [j for j, t in enumerate(targ) if t == cls]
        xs_c = [xs[j] for j in idx if xs[j] is not None]
        ys_c = [ys[j] for j in idx if ys[j] is not None]
        ax.scatter(xs_c, ys_c, color=PALETTE[i % len(PALETTE)],
                   alpha=0.65, edgecolors="white", linewidth=0.4,
                   label=cls, s=40, zorder=3)
    ax.set_xlabel(feat_x, color=SEC, fontsize=11)
    ax.set_ylabel(feat_y, color=SEC, fontsize=11)
    ax.set_title(f"Distribución: {feat_x} vs {feat_y}", color=INK, fontsize=12, fontweight="bold")
    ax.legend(title=target_col, facecolor=LIGHT, labelcolor=INK, fontsize=9)
    ax.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    plt.tight_layout(pad=1.4)
    return jsonify({"img": fig_b64(fig), "feat_x": feat_x, "feat_y": feat_y})

@app.route("/api/tab_classify_imbalance", methods=["POST"])
def tab_classify_imbalance():
    """Simulate class imbalance: filter rows and return class distribution plot."""
    body       = request.get_json(force=True, silent=True) or {}
    target_col = body.get("target_col", "")
    filter_col = body.get("filter_col", "")
    filter_op  = body.get("filter_op", "gt")
    filter_val = body.get("filter_val", 0.0)
    node_id    = body.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows: return jsonify({"error": "No data"}), 400

    fv = float(filter_val)
    if filter_col:
        if filter_op == "gt":
            filtered = [r for r in rows if _try_float(r.get(filter_col,"")) is not None and float(_try_float(r.get(filter_col,0))) > fv]
        elif filter_op == "lt":
            filtered = [r for r in rows if _try_float(r.get(filter_col,"")) is not None and float(_try_float(r.get(filter_col,0))) < fv]
        else:
            filtered = [r for r in rows if str(r.get(filter_col,"")) == str(filter_val)]
    else:
        filtered = rows

    targ = [str(r.get(target_col,"")) for r in filtered]
    classes = sorted(list(set(targ)))
    orig_targ = [str(r.get(target_col,"")) for r in rows]
    orig_dist = Counter(orig_targ)
    filt_dist = Counter(targ)

    # Distribution comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor(LIGHT)
    for ax, dist, title in zip(axes, [orig_dist, filt_dist], ["Original", "Filtrado (simulado)"]):
        style_ax(ax)
        lbls = sorted(dist.keys()); vals = [dist[l] for l in lbls]
        bars = ax.bar(range(len(lbls)), vals, color=PALETTE[:len(lbls)], edgecolor="none", zorder=3)
        ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, color=INK, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, str(v), ha="center", color=INK, fontsize=10)
    plt.suptitle("Distribución de clases", color=INK, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.4)

    ratio = round(max(filt_dist.values()) / max(1, min(filt_dist.values())), 2) if len(filt_dist) > 1 else 1.0
    return jsonify({
        "img": fig_b64(fig),
        "original_dist": dict(orig_dist),
        "filtered_dist": dict(filt_dist),
        "n_filtered": len(filtered),
        "ratio": ratio,
        "classes": classes,
    })

# ── Tabular info ──────────────────────────────────────────────────────────────

@app.route("/api/tabular_info")
def tabular_info():
    node_id = request.args.get("node", "") or request.args.get("node_id", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows:
        return jsonify({"error": "No data loaded" + (f" for node {node_id}" if node_id else "")}), 400
    cols = D["columns"]
    total = len(rows)
    col_info = []
    for col in cols:
        vals = [r.get(col, "") for r in rows]
        missing = sum(1 for v in vals if _is_missing(v))
        non_empty = [v for v in vals if not _is_missing(v)]
        ctype = _col_type(vals)
        info = {"name": col, "type": ctype, "missing": missing,
                "missing_pct": round(missing / max(total, 1) * 100, 1)}
        if _is_numeric_type(ctype):
            nums = []
            for v in non_empty:
                try: nums.append(float(v))
                except: pass
            if nums:
                info.update({
                    "min": round(min(nums), 4), "max": round(max(nums), 4),
                    "mean": round(float(np.mean(nums)), 4),
                    "std":  round(float(np.std(nums)),  4),
                    "n_unique": len(set(nums))
                })
        else:
            counts = Counter(str(v) for v in non_empty)
            info.update({"n_unique": len(counts),
                         "top_values": [{"v": k, "n": v} for k, v in counts.most_common(5)]})
        col_info.append(info)
    n_numeric = sum(1 for c in col_info if _is_numeric_type(c["type"]))
    n_cat     = sum(1 for c in col_info if c["type"] == "categorical")
    n_text    = sum(1 for c in col_info if c["type"] == "text")
    total_missing = sum(c["missing"] for c in col_info)
    return jsonify({
        "rows": total, "cols": len(cols),
        "n_numeric": n_numeric, "n_categorical": n_cat, "n_text": n_text,
        "total_missing": total_missing,
        "columns": col_info,
        "name": D["dataset_name"],
        "target": D.get("target", ""),   # expose so frontend knows the target col
    })

# ── Distributions plot ────────────────────────────────────────────────────────
@app.route("/api/plot_distribution")
def plot_distribution():
    col     = request.args.get("col", "")
    node_id = request.args.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows or col not in D["columns"]:
        return jsonify({"error": "No data or column not found"}), 400

    vals      = [r.get(col, "") for r in rows]
    ctype     = _col_type(vals)
    non_empty = [v for v in vals if not _is_missing(v)]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(LIGHT)
    style_ax(ax)

    if _is_numeric_type(ctype):
        nums = []
        for v in non_empty:
            try: nums.append(float(v))
            except: pass
        if not nums:
            return jsonify({"error": "No numeric values"}), 400
        n_bins = min(30, max(8, len(set(nums))))
        ax.hist(nums, bins=n_bins, color=PALETTE[0], edgecolor="white",
                linewidth=0.5, alpha=0.88)
        mean_v = float(np.mean(nums)); med_v = float(np.median(nums))
        _mean_lbl = "Media" if _UI_LANG == "es" else "Mean"
        _med_lbl  = "Mediana" if _UI_LANG == "es" else "Median"
        _dist_lbl = "Distribución de" if _UI_LANG == "es" else "Distribution of"
        ax.axvline(mean_v, color=PALETTE[1], linewidth=1.8, linestyle="--",
                   label=f"{_mean_lbl} {mean_v:.2f}", alpha=0.9)
        ax.axvline(med_v, color=PALETTE[2], linewidth=1.8, linestyle=":",
                   label=f"{_med_lbl} {med_v:.2f}", alpha=0.9)
        ax.set_xlabel(col, color=SEC, fontsize=11)
        ax.set_ylabel(PL()["frequency"], color=SEC, fontsize=11)
        ax.set_title(f"{_dist_lbl} {col}", color=INK, fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
        ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=10,
                  framealpha=0.85, edgecolor=BORDER)
    else:
        counts = Counter(str(v) for v in non_empty)
        top = counts.most_common(20)
        if not top:
            return jsonify({"error": "No values"}), 400
        labels_b, freqs = zip(*top)
        # Horizontal bar chart — mucho más legible para categóricas
        y_pos = range(len(labels_b))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels_b))]
        bars = ax.barh(list(reversed(list(y_pos))), list(reversed(list(freqs))),
                       color=list(reversed(colors)), edgecolor="none", height=0.65)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(list(reversed(list(labels_b))), color=INK, fontsize=10)
        ax.set_xlabel(PL()["frequency"], color=SEC, fontsize=11)
        _dist_lbl2 = "Distribución de" if _UI_LANG == "es" else "Distribution of"
        ax.set_title(f"{_dist_lbl2} {col}", color=INK, fontsize=13, fontweight="bold")
        ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
        for b in bars:
            w = b.get_width()
            ax.text(w + max(freqs)*0.01, b.get_y() + b.get_height()/2,
                    str(int(w)), va="center", color=INK, fontsize=9, fontweight="600")

    plt.tight_layout(pad=1.6)
    return jsonify({"img": fig_b64(fig), "type": ctype})

# ── Feature statistics (summary table for all columns) ───────────────────────
@app.route("/api/feature_stats")
def feature_stats():
    node_id = request.args.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows:
        return jsonify({"error": "No data loaded"}), 400
    cols  = D["columns"]
    total = len(rows)
    stats = []
    for col in cols:
        vals = [r.get(col, "") for r in rows]
        missing = sum(1 for v in vals if _is_missing(v))
        non_empty = [v for v in vals if not _is_missing(v)]
        ctype = _col_type(vals)
        row = {"name": col, "type": ctype, "missing": missing,
               "missing_pct": round(missing / max(total, 1) * 100, 1),
               "n_unique": 0}
        if _is_numeric_type(ctype):
            nums = []
            for v in non_empty:
                try: nums.append(float(v))
                except: pass
            if nums:
                row.update({
                    "n_unique": len(set(nums)),
                    "mean":   round(float(np.mean(nums)), 3),
                    "std":    round(float(np.std(nums)),  3),
                    "min":    round(min(nums), 3),
                    "max":    round(max(nums), 3),
                    "median": round(float(np.median(nums)), 3),
                })
        else:
            c = Counter(str(v) for v in non_empty)
            row.update({"n_unique": len(c), "top": c.most_common(1)[0][0] if c else ""})
        stats.append(row)
    return jsonify({"stats": stats, "rows": total, "cols": len(cols)})

# ── Scatter plot ──────────────────────────────────────────────────────────────
@app.route("/api/plot_scatter")
def plot_scatter():
    x_col    = request.args.get("x", "")
    y_col    = request.args.get("y", "")
    color_by = request.args.get("color_by", "")
    node_id  = request.args.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    cols = D["columns"]
    if not rows:
        return jsonify({"error": "No data loaded"}), 400
    if x_col not in cols or y_col not in cols:
        return jsonify({"error": "Column not found"}), 400

    # Build xy pairs
    xs, ys, groups = [], [], []
    for r in rows:
        xv = r.get(x_col, ""); yv = r.get(y_col, "")
        if _is_missing(xv) or _is_missing(yv):
            continue
        try:
            xs.append(float(xv)); ys.append(float(yv))
        except:
            continue
        groups.append(str(r.get(color_by, "all")) if color_by and color_by in cols else "all")

    if not xs:
        return jsonify({"error": "No numeric data for selected columns"}), 400

    uniq_groups = sorted(set(groups))
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(LIGHT)
    style_ax(ax)

    for i, grp in enumerate(uniq_groups):
        idx = [j for j, g in enumerate(groups) if g == grp]
        gx  = [xs[j] for j in idx]
        gy  = [ys[j] for j in idx]
        ax.scatter(gx, gy, color=PALETTE[i % len(PALETTE)], alpha=0.65,
                   s=28, edgecolors="none",
                   label=grp if color_by else None, zorder=3)

    ax.set_xlabel(x_col, color=SEC, fontsize=11)
    ax.set_ylabel(y_col, color=SEC, fontsize=11)
    title = f"{x_col} vs {y_col}"
    if color_by: title += f"  (colour: {color_by})"
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold")
    ax.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
    if color_by and len(uniq_groups) > 1:
        ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=10,
                  title=color_by, markerscale=1.4)
    plt.tight_layout(pad=1.6)
    return jsonify({"img": fig_b64(fig), "n_points": len(xs)})

# ── Missing-value heatmap ─────────────────────────────────────────────────────
@app.route("/api/plot_missing")
def plot_missing():
    node_id = request.args.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows:
        return jsonify({"error": "No data"}), 400
    cols = D["columns"]
    total = len(rows)
    if not cols:
        return jsonify({"error": "No columns"}), 400

    # Barras horizontales compactas: % de nulos por columna
    miss_data = []
    for col in cols:
        vals = [r.get(col, "") for r in rows]
        pct = sum(1 for v in vals if _is_missing(v)) / max(total, 1) * 100
        miss_data.append((col, round(pct, 1)))

    # Solo columnas con algún nulo, ordenadas desc
    miss_data = sorted([m for m in miss_data if m[1] > 0], key=lambda x: -x[1])
    if not miss_data:
        # Sin nulos — devuelve imagen de "todo bien"
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
        ax.text(0.5, 0.5, "✓  Sin valores ausentes", ha="center", va="center",
                fontsize=14, color=MINT, fontweight="bold", transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout(pad=1)
        return jsonify({"img": fig_b64(fig)})

    n = len(miss_data)
    fig_h = max(2.5, n * 0.42 + 1.0)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    fig.patch.set_facecolor(LIGHT)
    style_ax(ax)

    names, pcts = zip(*miss_data)
    colors = ["#EF4444" if p > 10 else "#F59E0B" if p > 5 else "#FCD34D" for p in pcts]
    bars = ax.barh(range(n), pcts, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, color=INK, fontsize=10)
    ax.set_xlabel(PL()["missing_title"], color=SEC, fontsize=11)
    _miss_title = "Valores ausentes por columna" if _UI_LANG == "es" else "Missing values per column"
    ax.set_title(_miss_title, color=INK, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.set_xlim(0, 100)   # siempre sobre 100%
    for b, p in zip(bars, pcts):
        ax.text(min(p + 1.5, 96), b.get_y() + b.get_height()/2,
                f"{p}%", va="center", color=INK, fontsize=9, fontweight="700")
    plt.tight_layout(pad=1.4)
    return jsonify({"img": fig_b64(fig)})

# ── Table endpoint (paginated) ────────────────────────────────────────────────
@app.route("/api/table")
def table():
    q        = request.args.get("q","").lower()
    page     = int(request.args.get("page",1))
    sort_col = request.args.get("sort","")
    sort_dir = int(request.args.get("dir","1"))
    node_id  = request.args.get("node","")
    per      = 20
    D    = _effective_data(node_id)
    rows = list(D["raw_rows"])
    if q:
        rows = [r for r in rows if any(q in str(v).lower() for v in r.values())]
    if sort_col and sort_col in D["columns"]:
        def sort_key(r):
            v = r.get(sort_col, "")
            try: return (0, float(v))
            except: return (1, str(v).lower())
        rows = sorted(rows, key=sort_key, reverse=(sort_dir < 0))
    total  = len(rows)
    start  = (page-1)*per
    return jsonify({"columns": D["columns"], "rows": rows[start:start+per],
                    "total": total, "page": page, "pages": max(1,(total+per-1)//per)})

# ── Explore ───────────────────────────────────────────────────────────────────
@app.route("/api/plot_explore")
def plot_explore():
    """NLP explore: word-length histogram + class distribution."""
    texts, labels, lnames = S["texts"], S["labels"], S["label_names"]
    if not texts: return jsonify({"error":"No data"}), 400
    ncols = 2 if labels else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11 if ncols==2 else 6, 4))
    fig.patch.set_facecolor(LIGHT)
    if ncols==1: axes=[axes]
    ax=axes[0]; style_ax(ax)
    lengths=[len(t.split()) for t in texts]
    n_bins=min(20,len(set(lengths)))
    ax.hist(lengths, bins=n_bins, color=PALETTE[0], edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Words per document",color=SEC,fontsize=11)
    ax.set_ylabel("Frequency",color=SEC,fontsize=11)
    ax.set_title("Document length distribution",color=INK,fontsize=13,fontweight="bold")
    ax.yaxis.grid(True,color=BORDER,linestyle="--",linewidth=0.6,zorder=0)
    # stats annotation
    ax.axvline(float(np.mean(lengths)), color=PALETTE[2], linewidth=1.6,
               linestyle="--", label=f"Mean {np.mean(lengths):.0f}")
    ax.axvline(float(np.median(lengths)), color=PALETTE[4], linewidth=1.6,
               linestyle=":", label=f"Median {np.median(lengths):.0f}")
    ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=9)
    if labels:
        ax2=axes[1]; style_ax(ax2)
        c=Counter(labels)
        cols_b=PALETTE[:len(c)]
        bars=ax2.bar([lnames[k] for k in sorted(c)],[c[k] for k in sorted(c)],
                     color=cols_b,width=.5,edgecolor="none",zorder=3)
        ax2.yaxis.grid(True,color=BORDER,linestyle="--",linewidth=0.6,zorder=0)
        ax2.set_ylabel("Count",color=SEC,fontsize=11)
        ax2.set_title("Class distribution",color=INK,fontsize=13,fontweight="bold")
        for b in bars:
            ax2.text(b.get_x()+b.get_width()/2,b.get_height()+.1,
                     str(int(b.get_height())),ha="center",color=INK,fontsize=12,fontweight="bold")
    plt.tight_layout(pad=1.8)
    return jsonify({"img": fig_b64(fig), "type": "nlp"})

@app.route("/api/plot_explore_tabular")
def plot_explore_tabular():
    """Tabular explore: numeric distributions grid + categorical bars + missing bar."""
    node_id = request.args.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    cols = D["columns"]
    if not rows: return jsonify({"error":"No data"}), 400

    num_cols = [c for c in cols if _col_type([r.get(c,"") for r in rows])=="numeric"]
    cat_cols = [c for c in cols if _col_type([r.get(c,"") for r in rows])=="categorical"]

    # ── Summary stats table (like df.describe()) ──────────────────────────
    summary = []
    for col in num_cols:
        vals = [r.get(col,"") for r in rows]
        nums = [float(v) for v in vals if not _is_missing(v) and _try_float(v)]
        if not nums: continue
        summary.append({
            "col": col,
            "count": len(nums),
            "missing": len(vals)-len(nums),
            "mean": round(float(np.mean(nums)),3),
            "std": round(float(np.std(nums)),3),
            "min": round(min(nums),3),
            "p25": round(float(np.percentile(nums,25)),3),
            "p50": round(float(np.median(nums)),3),
            "p75": round(float(np.percentile(nums,75)),3),
            "max": round(max(nums),3),
        })

    # ── Figure 1: numeric histograms — one per row to avoid crowding ─────
    n_num = len(num_cols[:8])   # max 8
    imgs = []
    if n_num:
        # 2 cols, N rows — guarantees enough vertical space per subplot
        ncols_g = min(2, n_num)
        nrows_g = (n_num + ncols_g - 1) // ncols_g
        cell_h  = 3.2        # height per row
        fig1, axes1 = plt.subplots(nrows_g, ncols_g,
                                   figsize=(ncols_g * 4.2, nrows_g * cell_h))
        fig1.patch.set_facecolor(LIGHT)
        # Normalise axes to always be a 2-D list
        if n_num == 1:
            axes1 = [[axes1]]
        elif nrows_g == 1:
            axes1 = [list(axes1)]
        else:
            axes1 = [list(row) for row in axes1]
        flat = [ax for row in axes1 for ax in row]
        for i, col in enumerate(num_cols[:8]):
            ax = flat[i]; style_ax(ax)
            vals = [r.get(col,"") for r in rows]
            nums = [float(v) for v in vals if not _is_missing(v) and _try_float(v)]
            if nums:
                n_bins = min(20, max(6, len(set(nums))))
                ax.hist(nums, bins=n_bins,
                        color=PALETTE[i % len(PALETTE)], edgecolor="white",
                        linewidth=0.4, alpha=0.88)
                ax.axvline(float(np.mean(nums)), color=INK,
                           linewidth=1.2, linestyle="--", alpha=0.5)
            ax.set_title(col, color=INK, fontsize=10, fontweight="bold")
            ax.set_ylabel("Frecuencia", color=SEC, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
        for ax in flat[n_num:]:
            ax.set_visible(False)
        fig1.suptitle("Distribuciones numéricas", color=INK,
                      fontsize=12, fontweight="bold")
        plt.tight_layout(pad=1.4)
        imgs.append(("numeric", fig_b64(fig1)))

    # ── Figure 2: categorical bar charts — horizontal, 1 col layout ───────
    n_cat = len(cat_cols[:6])
    if n_cat:
        fig2, axes2 = plt.subplots(n_cat, 1,
                                   figsize=(7, n_cat * 3.0))
        fig2.patch.set_facecolor(LIGHT)
        if n_cat == 1: axes2 = [axes2]
        for i, col in enumerate(cat_cols[:6]):
            ax = axes2[i]; style_ax(ax)
            vals = [r.get(col,"") for r in rows]
            non_empty = [v for v in vals if not _is_missing(v)]
            counts = Counter(str(v) for v in non_empty)
            top = counts.most_common(10)
            if top:
                lbls, freqs = zip(*top)
                colors_c = [PALETTE[j % len(PALETTE)] for j in range(len(lbls))]
                # Horizontal bars for legibility
                y_pos = range(len(lbls))
                bars = ax.barh(list(y_pos), list(freqs),
                               color=colors_c, edgecolor="none", height=0.6)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(list(lbls), color=INK, fontsize=8)
                ax.xaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
                ax.set_xlabel("Frecuencia", color=SEC, fontsize=8)
                for b in bars:
                    w = b.get_width()
                    ax.text(w + max(freqs)*0.01, b.get_y()+b.get_height()/2,
                            str(int(w)), va="center", color=INK, fontsize=8)
            ax.set_title(col, color=INK, fontsize=10, fontweight="bold")
        fig2.suptitle("Variables categóricas", color=INK,
                      fontsize=12, fontweight="bold")
        plt.tight_layout(pad=1.4)
        imgs.append(("categorical", fig_b64(fig2)))

    # ── Figure 3: missing values bar ─────────────────────────────────────
    total = len(rows)
    miss_pcts = []
    for col in cols:
        vals = [r.get(col,"") for r in rows]
        pct = sum(1 for v in vals if _is_missing(v)) / max(total,1) * 100
        if pct > 0:
            miss_pcts.append((col, round(pct,1)))
    if miss_pcts:
        n_m = len(miss_pcts)
        fig3, ax3 = plt.subplots(figsize=(7, max(2.5, n_m * 0.42 + 1.0)))
        fig3.patch.set_facecolor(LIGHT); style_ax(ax3)
        mc, mp = zip(*miss_pcts)
        colors_m = ["#EF4444" if p > 10 else "#F59E0B" if p > 5 else "#FCD34D" for p in mp]
        bars3 = ax3.barh(range(n_m), mp, color=colors_m, edgecolor="none", height=0.55)
        ax3.set_yticks(range(n_m))
        ax3.set_yticklabels(mc, color=INK, fontsize=9)
        ax3.set_xlabel("% valores ausentes", color=SEC, fontsize=10)
        ax3.set_xlim(0, max(mp) * 1.2)
        ax3.set_title("Valores ausentes por columna", color=INK,
                      fontsize=12, fontweight="bold")
        ax3.xaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
        for b, p in zip(bars3, mp):
            ax3.text(p + max(mp)*0.02, b.get_y()+b.get_height()/2,
                     f"{p}%", va="center", color=INK, fontsize=9, fontweight="700")
        plt.tight_layout(pad=1.4)
        imgs.append(("missing", fig_b64(fig3)))

    return jsonify({"type": "tabular", "imgs": imgs, "summary": summary,
                    "n_numeric": len(num_cols), "n_categorical": len(cat_cols)})

# ── Preprocessing preview ─────────────────────────────────────────────────────
_pre_thread = None

@app.route("/api/preprocess", methods=["POST"])
def run_preprocess():
    global _pre_thread
    steps = request.json.get("steps", [])
    S["active_steps"] = steps
    texts = S["texts"]
    if not texts:
        return jsonify({"error": "Load data first"}), 400

    def pre_worker():
        reset_progress()
        total = len(texts)
        proc = []
        push_progress(2, f"Preprocessing {total} documents…")
        chunk = max(1, total // 20)          # emit ~20 progress updates
        for i, t in enumerate(texts):
            proc.append(preprocess(t, steps))
            if (i + 1) % chunk == 0 or (i + 1) == total:
                pct = int((i + 1) / total * 95) + 2
                push_progress(pct, f"Processed {i+1}/{total} docs…")
        S["processed_texts"] = proc
        push_progress(100, f"Done — {total} docs preprocessed ✓")

    if _pre_thread and _pre_thread.is_alive():
        return jsonify({"error": "Already running"}), 429
    _pre_thread = threading.Thread(target=pre_worker, daemon=True)
    _pre_thread.start()
    return jsonify({"ok": True, "started": True})

@app.route("/api/preview_text", methods=["POST"])
def preview_text():
    body=request.json; idx=int(body.get("idx",0)); steps=body.get("steps",[])
    if idx>=len(S["texts"]): return jsonify({"error":"Index out of range"}),400
    orig=S["texts"][idx]
    intermediates=[]; cur=orig
    for s in steps:
        if s in STEPS:
            cur=STEPS[s](cur)
            intermediates.append({"step":s,"text":cur[:300]})
    tokens=cur.split()
    return jsonify({"original":orig[:200],"processed":cur[:200],
                    "intermediates":intermediates,"tokens":tokens[:80],
                    "orig_words":len(orig.split()),"proc_words":len(tokens),
                    "reduction":round((1-len(tokens)/max(len(orig.split()),1))*100,1)})

@app.route("/api/plot_wordfreq")
def plot_wordfreq():
    if not S["texts"]: return jsonify({"error": "No data"}), 400
    img = _wordfreq_b64()
    return jsonify({"img": img} if img else {"error": "Could not generate"})

# ── Word cloud ────────────────────────────────────────────────────────────────
@app.route("/api/plot_wordcloud", methods=["POST"])
def plot_wordcloud():
    corpus_key = request.json.get("corpus","processed")
    topic_words = request.json.get("topic_words", None)

    if topic_words:
        freq = {w["word"]: float(w["weight"]) for w in topic_words}
    else:
        corpus = S["processed_texts"] if corpus_key=="processed" else S["texts"]
        if not corpus: return jsonify({"error":"No data"}),400
        all_w=[]
        for t in corpus: all_w.extend(t.split())
        top = Counter(all_w).most_common(80)
        freq = {w:float(c) for w,c in top}

    if not freq: return jsonify({"error":"No words"}),400

    # Vibrant palette for word cloud
    wc_palette = PALETTE + ["#FF6B9D","#00B4D8","#06D6A0","#FFB703"]
    import random as _rnd
    _rnd.seed(42)
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return _rnd.choice(wc_palette)

    wc = WordCloud(
        width=900, height=420,
        background_color="#f5f5f5",
        max_words=80,
        prefer_horizontal=0.85,
        color_func=color_func,
        margin=6,
        relative_scaling=0.55,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor(LIGHT)
    ax.set_facecolor(LIGHT)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return jsonify({"img":fig_b64(fig)})

# ── SSE progress stream ───────────────────────────────────────────────────────
@app.route("/api/progress_stream")
def progress_stream():
    def generate():
        while True:
            events = drain_progress()
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
            time.sleep(0.12)
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Poll endpoint (alternative to SSE for environments that buffer) ────────────
@app.route("/api/progress_poll")
def progress_poll():
    events = drain_progress()
    pre_running   = _pre_thread   is not None and _pre_thread.is_alive()
    train_running = _train_thread is not None and _train_thread.is_alive()
    topic_running = _topic_thread is not None and _topic_thread.is_alive()
    job_running   = pre_running or train_running or topic_running
    return jsonify({"events": events, "job_running": job_running})

# ── Training (background thread + SSE) ───────────────────────────────────────
_train_thread = None

@app.route("/api/train", methods=["POST"])
def train():
    global _train_thread
    body       = request.json
    model_name = body.get("model","naive_bayes")
    vec_name   = body.get("vectorizer","tfidf")
    split      = float(body.get("split",0.3))
    compare    = body.get("compare",True)
    nb_alpha   = float(body.get("nb_alpha",1.0))
    lr_c       = float(body.get("lr_c",1.0))
    lr_penalty = body.get("lr_penalty","l2")
    svm_c      = float(body.get("svm_c",1.0))
    knn_k      = int(body.get("knn_k",5))
    knn_metric = body.get("knn_metric","euclidean")
    rf_trees   = int(body.get("rf_trees",100))
    rf_depth   = body.get("rf_depth",None)
    rf_depth   = int(rf_depth) if rf_depth else None
    gb_trees   = int(body.get("gb_trees",100))
    gb_lr      = float(body.get("gb_lr",0.1))
    gb_depth   = int(body.get("gb_depth",3))
    dt_depth   = body.get("dt_depth",None)
    dt_depth   = int(dt_depth) if dt_depth else None
    dt_crit    = body.get("dt_criterion","gini")
    max_feat   = body.get("max_features",None)
    max_feat   = int(max_feat) if max_feat else None
    exp_label  = (body.get("exp_label") or "").strip().split()[0] if body.get("exp_label") else ""
    S["exp_label"] = exp_label
    ngram      = body.get("ngram","1")

    texts, proc = S["texts"], S["processed_texts"]
    labels, lnames = S["labels"], S["label_names"]
    if not texts or not labels:
        return jsonify({"error":"Load a labelled dataset first"}), 400

    ngram_map = {"1":(1,1),"2":(2,2),"12":(1,2)}
    ngram_range = ngram_map.get(ngram,(1,1))

    def mk_vec():
        kw = dict(min_df=1, ngram_range=ngram_range)
        if max_feat: kw["max_features"]=max_feat
        return TfidfVectorizer(**kw) if vec_name=="tfidf" else CountVectorizer(**kw)

    def mk_clf():
        if model_name=="naive_bayes":   return MultinomialNB(alpha=nb_alpha)
        if model_name=="logistic":      return LogisticRegression(C=lr_c,penalty=lr_penalty,max_iter=2000,random_state=42,solver="saga")
        if model_name=="svm":           return LinearSVC(C=svm_c,max_iter=3000,random_state=42)
        if model_name=="knn":           return KNeighborsClassifier(n_neighbors=knn_k,metric=knn_metric)
        if model_name=="random_forest": return RandomForestClassifier(n_estimators=rf_trees,max_depth=rf_depth,random_state=42,n_jobs=-1)
        if model_name=="grad_boost":    return GradientBoostingClassifier(n_estimators=gb_trees,learning_rate=gb_lr,max_depth=gb_depth,random_state=42)
        if model_name=="decision_tree": return DecisionTreeClassifier(max_depth=dt_depth,criterion=dt_crit,random_state=42)
        return LinearSVC(C=svm_c,max_iter=3000,random_state=42)

    def train_worker():
        try:
            reset_progress()
            push_progress(2,"Starting…")
            results = {}
            pairs = [("raw",texts),("processed",proc)] if compare else [("processed",proc)]
            n_pairs = len(pairs)
            for pi,(mode,corpus) in enumerate(pairs):
                base = pi * 45
                push_progress(base+5, f"Vectorizing ({mode}) — {len(corpus)} docs…")
                time.sleep(0.05)
                vec=mk_vec(); clf=mk_clf()
                X=vec.fit_transform(corpus)
                push_progress(base+18, f"Splitting data ({mode})…")
                time.sleep(0.05)
                Xtr,Xte,ytr,yte=train_test_split(X,labels,test_size=split,random_state=42,
                                                  stratify=labels if len(set(labels))>1 else None)
                push_progress(base+28, f"Training {model_name} ({mode})…")
                time.sleep(0.05)
                clf.fit(Xtr,ytr)
                push_progress(base+40, f"Evaluating ({mode})…")
                time.sleep(0.05)
                ypred=clf.predict(Xte).tolist()
                acc=round(accuracy_score(yte,ypred)*100,1)
                f1_macro=round(f1_score(yte,ypred,average="macro")*100,1)
                f1_weighted=round(f1_score(yte,ypred,average="weighted")*100,1)
                cm=confusion_matrix(yte,ypred).tolist()
                rep=classification_report(yte,ypred,target_names=lnames,output_dict=True)
                report_txt=classification_report(yte,ypred,target_names=lnames)
                results[mode]={"acc":acc,"f1_macro":f1_macro,"f1_weighted":f1_weighted,
                               "cm":cm,"y_test":list(yte),"y_pred":ypred,
                               "report":rep,"report_txt":report_txt}
                if mode=="processed":
                    S["model"]=clf; S["vectorizer"]=vec
            results["exp_label"] = S.get("exp_label", "")
            S["results"]=results; S["label_names"]=lnames
            push_progress(100,"Done ✓")
        except Exception as e:
            push_progress(100, f"Error: {str(e)}")

    if _train_thread and _train_thread.is_alive():
        return jsonify({"error":"Training already running"}), 429

    _train_thread = threading.Thread(target=train_worker, daemon=True)
    _train_thread.start()
    return jsonify({"ok":True,"started":True})

@app.route("/api/train_status")
def train_status():
    running = _train_thread is not None and _train_thread.is_alive()
    return jsonify({"running": running, "ready": bool(S["results"])})

@app.route("/api/train_results")
def train_results():
    if not S["results"]: return jsonify({"ready":False})
    res = S["results"]
    # Topic model results have a flat structure (not nested by mode)
    if res.get("task") == "topic_model" or "topics" in res:
        return jsonify({
            "ready": True,
            "task": "topic_model",
            "results": {
                "topics":           res.get("topics", []),
                "perplexity":       res.get("perplexity"),
                "coherence_cv":     res.get("coherence_cv",    res.get("coherence", [])),
                "coherence_cnpmi":  res.get("coherence_cnpmi", []),
                "algorithm":        res.get("algorithm", "lda"),
                "doc_topics":       res.get("doc_topics", []),
                "label_result":     res.get("label_result"),
            }
        })
    return jsonify({"ready":True,"results":{k:{
        "acc":v["acc"],
        "f1_macro":v.get("f1_macro",""),
        "f1_weighted":v.get("f1_weighted",""),
        "report_txt":v.get("report_txt",""),
        "report":v.get("report",{})
    } for k,v in res.items()}})

# ── Result plots ──────────────────────────────────────────────────────────────
@app.route("/api/plot_results")
def plot_results():
    res=S["results"]; lnames=S["label_names"]
    if not res: return jsonify({"error":"Train first"}),400
    # Detect topic model by result structure, not just S["task"]
    if S["task"]=="topic_model" or res.get("task")=="topic_model" or "topics" in res:
        return _plot_topics(res)

    has_raw=("raw" in res)
    ncols=3 if has_raw else 2
    fig,axes=plt.subplots(1,ncols,figsize=(5.5*ncols,5)); fig.patch.set_facecolor(LIGHT)
    if ncols==1: axes=[axes]
    i=0
    if has_raw:
        ax=axes[i]; i+=1; style_ax(ax)
        modes=["Raw","Preprocessed"]; accs=[res["raw"]["acc"],res["processed"]["acc"]]
        colors=[PALETTE[0],PALETTE[1]]; best=int(np.argmax(accs)); colors[best]=PALETTE[2]
        bars=ax.bar(modes,accs,color=colors,width=.4,edgecolor="none",zorder=3)
        ax.set_ylim(0,115); ax.set_ylabel("Accuracy (%)",color=SEC,fontsize=11)
        ax.set_title("Preprocessing impact",color=INK,fontsize=13,fontweight="bold")
        ax.yaxis.grid(True,color=BORDER,linestyle="--",linewidth=0.6,zorder=0)
        for b,v in zip(bars,accs):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{v}%",
                    ha="center",color=INK,fontsize=13,fontweight="bold")
    # Confusion matrix with colour
    ax=axes[i]; i+=1
    ax.set_facecolor(LIGHT); ax.tick_params(colors=INK)
    cm=np.array(res["processed"]["cm"])
    im=ax.imshow(cm,cmap="RdYlGn",vmin=0,vmax=cm.max())
    ax.set_xticks(range(len(lnames))); ax.set_yticks(range(len(lnames)))
    ax.set_xticklabels(lnames,color=INK,fontsize=9,rotation=20,ha="right")
    ax.set_yticklabels(lnames,color=INK,fontsize=9)
    ax.set_xlabel("Predicted",color=SEC,fontsize=11); ax.set_ylabel("Actual",color=SEC,fontsize=11)
    ax.set_title("Confusion matrix",color=INK,fontsize=13,fontweight="bold")
    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
            ax.text(jj,ii,str(cm[ii,jj]),ha="center",va="center",fontsize=16,fontweight="bold",
                    color=INK)
    # Actual vs predicted
    ax=axes[i]; style_ax(ax)
    yt=res["processed"]["y_test"]; yp=res["processed"]["y_pred"]
    cats=sorted(set(yt)); x=np.arange(len(cats)); w=.3
    ax.bar(x-w/2,[(np.array(yt)==c).sum() for c in cats],w,label="Actual",
           color=PALETTE[3],edgecolor="none",zorder=3)
    ax.bar(x+w/2,[(np.array(yp)==c).sum() for c in cats],w,label="Predicted",
           color=PALETTE[4],edgecolor="none",zorder=3)
    ax.yaxis.grid(True,color=BORDER,linestyle="--",linewidth=0.6,zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([lnames[c] for c in cats],color=INK,fontsize=10)
    ax.set_ylabel("Samples",color=SEC,fontsize=11)
    ax.set_title("Actual vs predicted (test set)",color=INK,fontsize=13,fontweight="bold")
    ax.legend(facecolor=LIGHT,labelcolor=INK,fontsize=10,framealpha=0.8)
    plt.tight_layout(pad=2)
    return jsonify({"img":fig_b64(fig),"acc":res["processed"]["acc"],
                    "f1_macro":res["processed"].get("f1_macro",""),
                    "f1_weighted":res["processed"].get("f1_weighted",""),
                    "report":res["processed"]["report"],
                    "report_txt":res["processed"].get("report_txt","")})

def _plot_topics(res):
    topics=res.get("topics",[])
    if not topics: return jsonify({"error":"No topics"}),400
    topic_labels = res.get("topic_labels", {})
    n=len(topics)
    cols=min(n,4); rows=max(1,(n+cols-1)//cols)
    fig,axes=plt.subplots(rows,cols,figsize=(4.5*cols,4.5*rows))
    fig.patch.set_facecolor(LIGHT)
    axes_flat=np.array(axes).flatten() if n>1 else [axes]
    for idx,(ax,tp) in enumerate(zip(axes_flat,topics)):
        style_ax(ax)
        words=tp["words"][:10]; weights=tp["weights"][:10]
        color=PALETTE[idx % len(PALETTE)]
        ax.barh(list(reversed(words)),list(reversed(weights)),
                color=color,edgecolor="none",alpha=0.85)
        lbl = topic_labels.get(str(tp["id"]), "")
        title = f"Topic {tp['id']+1}" + (f" — {lbl}" if lbl else "")
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold")
        ax.tick_params(axis="y",labelsize=10)
    # hide unused subplots
    for ax in axes_flat[n:]: ax.set_visible(False)
    plt.tight_layout(pad=2)
    return jsonify({"img":fig_b64(fig),"topics":topics})

# ── Topic model ───────────────────────────────────────────────────────────────
_topic_thread = None

def _coherence_cv(topic_words_list, tokenized_docs, top_n=10):
    """
    C_V coherence approximation (no gensim needed):
    For each topic, compute pairwise normalised PMI of top-N words,
    averaged across all word pairs.
    """
    vocab = {}
    for doc in tokenized_docs:
        for w in set(doc): vocab[w] = vocab.get(w, 0) + 1
    N = len(tokenized_docs)
    scores = []
    for words in topic_words_list:
        wds = words[:top_n]
        pair_scores = []
        for i in range(len(wds)):
            for j in range(i+1, len(wds)):
                wi, wj = wds[i], wds[j]
                fi  = vocab.get(wi, 0)
                fj  = vocab.get(wj, 0)
                fij = sum(1 for doc in tokenized_docs if wi in doc and wj in doc)
                if fi == 0 or fj == 0 or fij == 0:
                    pair_scores.append(0.0)
                    continue
                pmi = np.log((fij * N) / (fi * fj) + 1e-10)
                norm = -np.log(fij / N + 1e-10)
                pair_scores.append(float(pmi / (norm + 1e-10)))
        scores.append(round(float(np.mean(pair_scores)) if pair_scores else 0.0, 4))
    return scores


def _coherence_cnpmi(topic_words_list, tokenized_docs, top_n=10):
    """
    C_NPMI coherence approximation:
    NPMI = PMI / -log(p(wi,wj)) — normalised to [-1, 1].
    """
    vocab = {}
    for doc in tokenized_docs:
        for w in set(doc): vocab[w] = vocab.get(w, 0) + 1
    N = len(tokenized_docs)
    scores = []
    for words in topic_words_list:
        wds = words[:top_n]
        pair_scores = []
        for i in range(len(wds)):
            for j in range(i+1, len(wds)):
                wi, wj = wds[i], wds[j]
                fi  = vocab.get(wi, 0)
                fj  = vocab.get(wj, 0)
                fij = sum(1 for doc in tokenized_docs if wi in doc and wj in doc)
                if fi == 0 or fj == 0 or fij == 0:
                    pair_scores.append(-1.0)
                    continue
                p_ij = fij / N
                p_i  = fi  / N
                p_j  = fj  / N
                pmi  = np.log(p_ij / (p_i * p_j) + 1e-10)
                npmi = pmi / (-np.log(p_ij + 1e-10))
                pair_scores.append(float(npmi))
        scores.append(round(float(np.mean(pair_scores)) if pair_scores else -1.0, 4))
    return scores


@app.route("/api/topic_model", methods=["POST"])
def topic_model():
    global _topic_thread
    body       = request.json
    algorithm  = body.get("algorithm", "lda")   # lda | nmf | lsa
    n_topics   = int(body.get("n_topics",  5))
    max_vocab  = int(body.get("max_vocab", 500))
    top_n_words= int(body.get("top_words", 15))
    # Labeling params (optional — if label_classes provided, run weak labeling too)
    label_classes   = body.get("label_classes", [])    # [{name, keywords}]
    label_strategy  = body.get("label_strategy", "most")
    label_ci        = body.get("label_ci", True)

    proc = S["processed_texts"] or S["texts"]
    if not proc:
        return jsonify({"error": "No texts loaded — connect a Data block first"}), 400

    def topic_worker():
        try:
            reset_progress()
            push_progress(5, f"Vectorizando corpus ({algorithm.upper()})…")
            time.sleep(0.05)

            min_df = 2 if len(proc) > 50 else 1
            tokenized = [t.lower().split() for t in proc]

            # Choose vectorizer and model
            if algorithm == "lda":
                vec = CountVectorizer(max_features=max_vocab, min_df=min_df)
                X   = vec.fit_transform(proc)
                push_progress(20, "Ajustando LDA…")
                n_iter = min(40, max(10, len(proc) // 8))
                model  = LatentDirichletAllocation(
                    n_components=n_topics, random_state=42,
                    max_iter=n_iter, learning_method="batch", evaluate_every=5)
                model.fit(X)
                components = model.components_
                doc_topic_matrix = model.transform(X)
                perplexity = round(float(model.perplexity(X)), 1)

            elif algorithm == "nmf":
                vec = TfidfVectorizer(max_features=max_vocab, min_df=min_df)
                X   = vec.fit_transform(proc)
                push_progress(20, "Ajustando NMF…")
                model = NMF(n_components=n_topics, random_state=42, max_iter=400,
                            init="nndsvda", l1_ratio=0.5)
                W = model.fit_transform(X)
                components = model.components_
                # Normalise rows for soft assignment
                row_sums = W.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                doc_topic_matrix = W / row_sums
                perplexity = None  # NMF has no perplexity

            else:  # lsa / svd
                vec = TfidfVectorizer(max_features=max_vocab, min_df=min_df)
                X   = vec.fit_transform(proc)
                push_progress(20, "Ajustando LSA (SVD)…")
                model = TruncatedSVD(n_components=n_topics, random_state=42)
                W = model.fit_transform(X)
                components = model.components_
                # For LSA, take abs value for topic word importance
                components = np.abs(components)
                doc_topic_matrix = np.abs(W)
                row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                doc_topic_matrix = doc_topic_matrix / row_sums
                perplexity = None

            push_progress(60, "Extrayendo tópicos…")
            feat   = vec.get_feature_names_out()
            topics = [
                {
                    "id": i,
                    "words":   [feat[j] for j in c.argsort()[-top_n_words:][::-1]],
                    "weights": [round(float(c[j]), 4) for j in c.argsort()[-top_n_words:][::-1]]
                }
                for i, c in enumerate(components)
            ]
            doc_topics = doc_topic_matrix.argmax(axis=1).tolist()

            push_progress(75, "Calculando coherencia C_V…")
            time.sleep(0.05)
            topic_word_lists = [tp["words"][:10] for tp in topics]
            cv_scores   = _coherence_cv(topic_word_lists, tokenized, top_n=10)

            push_progress(88, "Calculando coherencia C_NPMI…")
            time.sleep(0.05)
            cnpmi_scores = _coherence_cnpmi(topic_word_lists, tokenized, top_n=10)

            # ── Weak labeling (if requested) ──────────────────────────────
            label_result = None
            if label_classes:
                push_progress(93, "Etiquetando corpus…")
                class_kws = []
                for cls in label_classes:
                    kws = [k.lower() if label_ci else k for k in cls.get("keywords", []) if k.strip()]
                    class_kws.append({"name": cls["name"], "kws": kws})
                labels_out = []
                counts = {c["name"]: 0 for c in class_kws}
                for text in proc:
                    t = text.lower() if label_ci else text
                    matched = []
                    for cls in class_kws:
                        hits = sum(1 for kw in cls["kws"] if kw in t)
                        if hits > 0:
                            matched.append({"name": cls["name"], "hits": hits})
                    if not matched:
                        labels_out.append(None)
                    else:
                        assigned = max(matched, key=lambda x: x["hits"])["name"] if label_strategy == "most" else matched[0]["name"]
                        labels_out.append(assigned)
                        counts[assigned] = counts.get(assigned, 0) + 1
                labeled = sum(1 for l in labels_out if l is not None)
                label_result = {
                    "labels": labels_out, "counts": counts,
                    "labeled": labeled, "total": len(proc),
                    "pct": round(labeled / max(len(proc), 1) * 100, 1)
                }
                _pending_labels["labels"]  = labels_out
                _pending_labels["classes"] = [c["name"] for c in class_kws]

            S["results"] = {
                "topics":      topics,
                "doc_topics":  doc_topics,
                "perplexity":  perplexity,
                "coherence_cv":    cv_scores,
                "coherence_cnpmi": cnpmi_scores,
                "algorithm":   algorithm,
                "label_result": label_result,
                "task": "topic_model"
            }

            # ── Auto-label topics via HF if toggle is active ──────────────
            if HF_LABELING_ACTIVE:
                push_progress(95, "Etiquetando tópicos con IA (HF)…")
                topic_list_str = "\n".join(
                    f"Topic {tp['id']+1}: {', '.join(tp['words'][:10])}"
                    for tp in topics
                )
                prompt_lbl = (
                    "You are an expert in text analysis. Given the following topics discovered by a topic model, "
                    "propose a short descriptive label (2-4 words in Spanish) for each topic that captures its main theme.\n\n"
                    f"Topics:\n{topic_list_str}\n\n"
                    "Reply ONLY with a valid JSON object (no explanation):\n"
                    '{"labels": [{"id": 0, "label": "..."}, {"id": 1, "label": "..."}, ...]}'
                )
                raw_lbl = _call_llm(prompt_lbl)
                auto_labels = {}
                if raw_lbl:
                    jm = re.search(r'\{[\s\S]*\}', raw_lbl)
                    if jm:
                        try:
                            parsed_lbl = json.loads(jm.group())
                            for item in parsed_lbl.get("labels", []):
                                auto_labels[str(item["id"])] = item["label"]
                        except Exception:
                            pass
                # fallback for any missing
                for tp in topics:
                    if str(tp["id"]) not in auto_labels:
                        auto_labels[str(tp["id"])] = _rule_label(tp["words"])
                S["results"]["topic_labels"] = auto_labels

            push_progress(100, "Listo ✓")
        except Exception as e:
            push_progress(100, f"Error: {str(e)}")

    if _topic_thread and _topic_thread.is_alive():
        return jsonify({"error": "Ya en ejecución"}), 429

    _topic_thread = threading.Thread(target=topic_worker, daemon=True)
    _topic_thread.start()
    return jsonify({"ok": True, "started": True})

# ── Weak labeling ────────────────────────────────────────────────────────────
# Store pending labeled results before applying to dataset
_pending_labels = {}

@app.route("/api/label", methods=["POST"])
def label():
    body      = request.json
    classes   = body.get("classes", [])      # [{name, keywords}, ...]
    strategy  = body.get("strategy", "first")
    ci        = body.get("case_insensitive", True)

    texts = S["processed_texts"] or S["texts"]
    if not texts:
        return jsonify({"error": "No texts loaded — connect a Data block first"}), 400
    if not classes:
        return jsonify({"error": "Define at least one class"}), 400

    # Normalize keywords
    class_kws = []
    for cls in classes:
        kws = [k.lower() if ci else k for k in cls.get("keywords", []) if k.strip()]
        class_kws.append({"name": cls["name"], "kws": kws})

    labels_out = []
    counts = {c["name"]: 0 for c in class_kws}

    for text in texts:
        t = text.lower() if ci else text
        matched = []
        for cls in class_kws:
            hits = sum(1 for kw in cls["kws"] if kw in t)
            if hits > 0:
                matched.append({"name": cls["name"], "hits": hits})
        if not matched:
            labels_out.append(None)
            continue
        if strategy == "first":
            assigned = matched[0]["name"]
        elif strategy == "most":
            assigned = max(matched, key=lambda x: x["hits"])["name"]
        else:  # any
            assigned = matched[0]["name"]
        labels_out.append(assigned)
        counts[assigned] = counts.get(assigned, 0) + 1

    labeled   = sum(1 for l in labels_out if l is not None)
    unlabeled = len(labels_out) - labeled
    pct       = round(labeled / max(len(labels_out), 1) * 100, 1)

    # Build sample (first 30 rows)
    sample = [
        {"idx": i, "text": texts[i][:200], "label": (labels_out[i] or "—")}
        for i in range(min(30, len(texts)))
    ]

    # Store pending so /api/apply_labels can commit
    _pending_labels["labels"]   = labels_out
    _pending_labels["classes"]  = [c["name"] for c in class_kws]

    # Distribution chart
    chart_img = None
    try:
        names  = [c["name"] for c in class_kws] + ["Sin etiquetar"]
        values = [counts.get(c["name"], 0) for c in class_kws] + [unlabeled]
        colors = PALETTE[:len(names)]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(LIGHT)
        style_ax(ax)
        bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="none", zorder=3)
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
        ax.set_title("Distribución de etiquetas", color=INK, fontsize=13, fontweight="bold")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, str(v),
                    ha="center", color=INK, fontsize=12, fontweight="bold")
        plt.tight_layout(pad=1.5)
        chart_img = fig_b64(fig)
    except Exception:
        pass

    return jsonify({
        "labeled": labeled, "unlabeled": unlabeled, "pct": pct,
        "counts": counts, "sample": sample, "chart_img": chart_img,
        "total": len(texts)
    })


@app.route("/api/apply_labels", methods=["POST"])
def apply_labels():
    if not _pending_labels:
        return jsonify({"error": "Run /api/label first"}), 400
    labels_out = _pending_labels["labels"]
    cls_names  = _pending_labels["classes"]

    texts = S["processed_texts"] or S["texts"]
    # Filter to labeled only
    filtered_texts  = [t for t, l in zip(texts, labels_out) if l is not None]
    filtered_labels = [l for l in labels_out if l is not None]

    uniq_names = list(dict.fromkeys(filtered_labels))  # preserve order
    name_to_idx = {n: i for i, n in enumerate(uniq_names)}
    int_labels  = [name_to_idx[l] for l in filtered_labels]

    # Also update raw_rows if available
    raw = S.get("raw_rows", [])
    filtered_raw = [r for r, l in zip(raw, labels_out) if l is not None] if len(raw) == len(labels_out) else []

    S.update(
        texts=filtered_texts,
        processed_texts=filtered_texts,
        labels=int_labels,
        label_names=uniq_names,
        task="classification",
        raw_rows=filtered_raw,
        columns=S.get("columns", []),
        model=None, vectorizer=None, results={}
    )
    _pending_labels.clear()
    return jsonify({"ok": True, "n": len(filtered_texts), "classes": len(uniq_names)})


# ── Classify ──────────────────────────────────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
def classify():
    text=request.json.get("text","").strip()
    if not text: return jsonify({"error":"Empty text"}),400
    if not S["model"]: return jsonify({"error":"Train a model first"}),400
    proc=preprocess(text,S["active_steps"])
    X=S["vectorizer"].transform([proc]); pred=int(S["model"].predict(X)[0])
    lnames=S["label_names"]; label=lnames[pred] if pred<len(lnames) else str(pred)
    proba=""
    if hasattr(S["model"],"predict_proba"):
        p=S["model"].predict_proba(X)[0]; proba=f"{round(float(max(p))*100,1)}%"
    return jsonify({"label":label,"proba":proba,"processed":proc[:150],
                    "acc":S["results"].get("processed",{}).get("acc","")})


# ════════════════════════════════════════════════════════════════════════════
# SESSION SAVE / LOAD  (canvas + model + data)
# ════════════════════════════════════════════════════════════════════════════

def _safe_results_for_json(res):
    """Strip numpy types so results are JSON-serialisable."""
    if not res:
        return {}
    out = {}
    for mode, v in res.items():
        if isinstance(v, dict):
            out[mode] = {
                k: (val.tolist() if hasattr(val, "tolist") else val)
                for k, val in v.items()
            }
        else:
            out[mode] = v
    return out


@app.route("/api/save_session", methods=["POST"])
def save_session():
    """
    Body: { canvas: <canvas JSON string>, filename: <optional name> }
    Returns a ZIP file containing:
      - canvas.json       — node/edge layout
      - data.json         — dataset metadata + texts + labels
      - preprocessed.csv  — processed texts (if available)
      - model.pkl         — trained sklearn model + vectorizer (if available)
      - results.json      — training metrics + report (if available)
      - plots/            — all result plots as PNG
    """
    body      = request.json or {}
    canvas    = body.get("canvas", "{}")
    filename  = body.get("filename", "nlpflow_session")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # 1. Canvas layout
        zf.writestr("canvas.json", canvas)

        # 2. Dataset (texts + labels + metadata)
        data_payload = {
            "dataset_name": S["dataset_name"],
            "task":         S["task"],
            "texts":        S["texts"],
            "labels":       S["labels"],
            "label_names":  S["label_names"],
            "active_steps": S["active_steps"],
            "columns":      S["columns"],
        }
        zf.writestr("data.json", json.dumps(data_payload, ensure_ascii=False, indent=2))

        # 3. Preprocessed CSV
        if S["processed_texts"]:
            csv_buf = io.StringIO()
            writer  = csv.writer(csv_buf)
            has_labels = bool(S["labels"])
            header = ["idx", "original_text", "processed_text"]
            if has_labels:
                header.append("label")
            writer.writerow(header)
            for i, (orig, proc) in enumerate(zip(S["texts"], S["processed_texts"])):
                row = [i, orig, proc]
                if has_labels and i < len(S["labels"]):
                    lname = S["label_names"][S["labels"][i]] if S["label_names"] else S["labels"][i]
                    row.append(lname)
                writer.writerow(row)
            zf.writestr("preprocessed.csv", csv_buf.getvalue())

        # 4. Model + vectorizer (pickle via joblib in-memory)
        if S["model"] is not None and S["vectorizer"] is not None:
            model_buf = io.BytesIO()
            joblib.dump({"model": S["model"], "vectorizer": S["vectorizer"],
                         "label_names": S["label_names"], "active_steps": S["active_steps"]},
                        model_buf)
            zf.writestr("model.pkl", model_buf.getvalue())

        # 5. Results JSON (metrics + classification report)
        if S["results"]:
            res_clean = _safe_results_for_json(S["results"])
            zf.writestr("results.json", json.dumps(res_clean, ensure_ascii=False, indent=2))

            # Also save report_txt as plain text
            rep_txt = ""
            if "processed" in S["results"]:
                rep_txt = S["results"]["processed"].get("report_txt", "")
            elif "topics" in S["results"]:
                topics = S["results"].get("topics", [])
                lines  = ["NLP Flow — Topic Model Report", "=" * 40]
                for tp in topics:
                    lines.append(f"\nTopic {tp['id']+1}: {', '.join(tp['words'][:10])}")
                rep_txt = "\n".join(lines)
            if rep_txt:
                zf.writestr("report.txt", rep_txt)

        # 6. Plots as PNG
        plots_generated = []

        # 6a. Results / topic plot
        try:
            res = S["results"]
            lnames = S["label_names"]
            if res:
                if S["task"] == "topic_model" or "topics" in res:
                    topics = res.get("topics", [])
                    if topics:
                        n = len(topics)
                        cols = min(n, 4); rows = max(1, (n + cols - 1) // cols)
                        import numpy as _np
                        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
                        fig.patch.set_facecolor(LIGHT)
                        axes_flat = _np.array(axes).flatten() if n > 1 else [axes]
                        for idx, (ax, tp) in enumerate(zip(axes_flat, topics)):
                            style_ax(ax)
                            ax.barh(list(reversed(tp["words"][:10])),
                                    list(reversed(tp["weights"][:10])),
                                    color=PALETTE[idx % len(PALETTE)], edgecolor="none", alpha=0.85)
                            ax.set_title(f"Topic {tp['id']+1}", color=INK, fontsize=12, fontweight="bold")
                        for ax in axes_flat[n:]: ax.set_visible(False)
                        plt.tight_layout(pad=2)
                        png_buf = io.BytesIO()
                        fig.savefig(png_buf, format="png", dpi=130, bbox_inches="tight", facecolor=LIGHT)
                        plt.close(fig)
                        zf.writestr("plots/topic_weights.png", png_buf.getvalue())
                        plots_generated.append("topic_weights.png")
                else:
                    # Classification plots
                    import numpy as _np
                    has_raw = "raw" in res
                    ncols = 3 if has_raw else 2
                    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
                    fig.patch.set_facecolor(LIGHT)
                    if ncols == 1: axes = [axes]
                    i = 0
                    if has_raw:
                        ax = axes[i]; i += 1; style_ax(ax)
                        modes = ["Raw", "Preprocessed"]
                        accs  = [res["raw"]["acc"], res["processed"]["acc"]]
                        colors = [PALETTE[0], PALETTE[1]]
                        colors[int(_np.argmax(accs))] = PALETTE[2]
                        bars = ax.bar(modes, accs, color=colors, width=.4, edgecolor="none", zorder=3)
                        ax.set_ylim(0, 115); ax.set_ylabel("Accuracy (%)", color=SEC)
                        ax.set_title("Preprocessing impact", color=INK, fontweight="bold")
                        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
                        for b, v in zip(bars, accs):
                            ax.text(b.get_x() + b.get_width()/2, b.get_height()+1,
                                    f"{v}%", ha="center", color=INK, fontsize=12, fontweight="bold")
                    ax = axes[i]; i += 1
                    ax.set_facecolor(LIGHT); ax.tick_params(colors=INK)
                    cm = _np.array(res["processed"]["cm"])
                    ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=cm.max())
                    ax.set_xticks(range(len(lnames))); ax.set_yticks(range(len(lnames)))
                    ax.set_xticklabels(lnames, color=INK, fontsize=9, rotation=20, ha="right")
                    ax.set_yticklabels(lnames, color=INK, fontsize=9)
                    ax.set_title("Confusion matrix", color=INK, fontweight="bold")
                    for ii in range(cm.shape[0]):
                        for jj in range(cm.shape[1]):
                            ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                                    fontsize=14, fontweight="bold", color=INK)
                    ax = axes[i]; style_ax(ax)
                    yt = res["processed"]["y_test"]; yp = res["processed"]["y_pred"]
                    cats = sorted(set(yt)); x = _np.arange(len(cats)); w = .3
                    ax.bar(x-w/2, [(_np.array(yt)==c).sum() for c in cats], w,
                           label="Actual",    color=PALETTE[3], edgecolor="none", zorder=3)
                    ax.bar(x+w/2, [(_np.array(yp)==c).sum() for c in cats], w,
                           label="Predicted", color=PALETTE[4], edgecolor="none", zorder=3)
                    ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
                    ax.set_xticks(x); ax.set_xticklabels([lnames[c] for c in cats], color=INK)
                    ax.set_title("Actual vs predicted", color=INK, fontweight="bold")
                    ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=10)
                    plt.tight_layout(pad=2)
                    png_buf = io.BytesIO()
                    fig.savefig(png_buf, format="png", dpi=130, bbox_inches="tight", facecolor=LIGHT)
                    plt.close(fig)
                    zf.writestr("plots/results.png", png_buf.getvalue())
                    plots_generated.append("results.png")
        except Exception as e:
            zf.writestr("plots/error.txt", str(e))

        # 6b. Word frequency plot (direct call, no HTTP overhead)
        try:
            if S["texts"] and S["processed_texts"]:
                img_b64 = _wordfreq_b64()
                if img_b64:
                    zf.writestr("plots/word_frequency.png", base64.b64decode(img_b64))
                    plots_generated.append("word_frequency.png")
        except Exception:
            pass

        # 7. README
        readme = f"""NLP Flow Session
================
Dataset  : {S['dataset_name'] or '—'}
Task     : {S['task']}
Texts    : {len(S['texts'])}
Steps    : {', '.join(S['active_steps']) or 'none'}
Trained  : {'yes' if S['model'] else 'no'}

Files
-----
canvas.json       — Canvas layout (nodes + edges)
data.json         — Dataset texts, labels and metadata
preprocessed.csv  — Preprocessed corpus (one row per text)
model.pkl         — Trained model + vectorizer (joblib)
results.json      — Training metrics and classification report
report.txt        — Plain-text classification report
plots/            — Result plots as PNG images
"""
        zf.writestr("README.txt", readme)

    buf.seek(0)
    safe_name = re.sub(r"[^\w\-]", "_", filename) or "nlpflow_session"
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_name}.zip"
    )


@app.route("/api/load_session", methods=["POST"])
def load_session():
    """
    Receives a ZIP file upload.  Restores data, model and results.
    Returns { canvas, info } so the frontend can rebuild the canvas.
    """
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    canvas_json = "{}"
    info        = {}

    try:
        buf = io.BytesIO(f.read())
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()

            # Canvas
            if "canvas.json" in names:
                canvas_json = zf.read("canvas.json").decode("utf-8")

            # Dataset
            if "data.json" in names:
                data = json.loads(zf.read("data.json").decode("utf-8"))
                S.update(
                    texts        = data.get("texts", []),
                    labels       = data.get("labels", []),
                    label_names  = data.get("label_names", []),
                    task         = data.get("task", "classification"),
                    dataset_name = data.get("dataset_name", "Loaded session"),
                    active_steps = data.get("active_steps", []),
                    columns      = data.get("columns", []),
                    raw_rows     = [],
                    processed_texts = data.get("texts", []),  # will be overwritten below
                    model        = None,
                    vectorizer   = None,
                    results      = {},
                    csv_source   = data.get("csv_source", ""),
                )
                info["texts"]  = len(S["texts"])
                info["task"]   = S["task"]

            # Preprocessed CSV — restore processed_texts
            if "preprocessed.csv" in names:
                csv_data = zf.read("preprocessed.csv").decode("utf-8", errors="replace")
                rows     = list(csv.DictReader(io.StringIO(csv_data)))
                if rows and "processed_text" in rows[0]:
                    S["processed_texts"] = [r["processed_text"] for r in rows]
                    info["preprocessed"] = len(S["processed_texts"])

            # Model
            if "model.pkl" in names:
                model_bytes = zf.read("model.pkl")
                loaded      = joblib.load(io.BytesIO(model_bytes))
                S["model"]       = loaded.get("model")
                S["vectorizer"]  = loaded.get("vectorizer")
                if loaded.get("label_names"):
                    S["label_names"] = loaded["label_names"]
                if loaded.get("active_steps"):
                    S["active_steps"] = loaded["active_steps"]
                info["model_loaded"] = True

            # Results
            if "results.json" in names:
                S["results"] = json.loads(zf.read("results.json").decode("utf-8"))
                info["results_loaded"] = True
                res = S["results"]
                # Extract training metrics for node badge restoration
                proc_res = res.get("processed", res.get("raw", {}))
                if proc_res.get("acc"):
                    info["acc"]       = proc_res["acc"]
                    info["f1_macro"]  = proc_res.get("f1_macro")
                info["exp_label"] = res.get("exp_label", "")
                # Topic model info
                if "topics" in res:
                    info["n_topics"]  = len(res["topics"])
                    info["algorithm"] = res.get("algorithm", "lda")

    except Exception as e:
        return jsonify({"error": f"Failed to load session: {str(e)}"}), 500

    # Flag si el dataset original era un CSV externo (no podemos restaurar el raw)
    info["needs_csv_reload"] = (S.get("csv_source") == "external" and bool(S["texts"]))
    info["dataset_name"]     = S.get("dataset_name", "")

    return jsonify({"canvas": canvas_json, "info": info})


# ── Individual export endpoints ───────────────────────────────────────────────

@app.route("/api/export_csv")
def export_csv():
    """Export the preprocessed corpus as a standalone CSV."""
    texts = S["texts"]; proc = S["processed_texts"]
    if not texts:
        return jsonify({"error": "No data loaded"}), 400

    csv_buf = io.StringIO()
    writer  = csv.writer(csv_buf)
    has_labels = bool(S["labels"])
    header = ["idx", "original_text", "processed_text"]
    if has_labels:
        header.append("label")
    writer.writerow(header)
    for i, (orig, pr) in enumerate(zip(texts, proc if proc else texts)):
        row = [i, orig, pr]
        if has_labels and i < len(S["labels"]):
            lname = S["label_names"][S["labels"][i]] if S["label_names"] else S["labels"][i]
            row.append(lname)
        writer.writerow(row)

    csv_bytes = csv_buf.getvalue().encode("utf-8")
    name = re.sub(r"[^\w\-]", "_", S["dataset_name"] or "corpus") + "_preprocessed.csv"
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name=name
    )


@app.route("/api/export_report")
def export_report():
    """Export the classification report as plain text."""
    if not S["results"]:
        return jsonify({"error": "No results — train a model first"}), 400

    lines = [
        f"NLP Flow — Classification Report",
        f"Dataset : {S['dataset_name']}",
        f"Task    : {S['task']}",
        f"Steps   : {', '.join(S['active_steps']) or 'none'}",
        "=" * 52,
    ]

    if "topics" in S["results"]:
        lines.append("\nTopic Model Results")
        lines.append(f"Perplexity : {S['results'].get('perplexity', '—')}")
        for tp in S["results"].get("topics", []):
            lines.append(f"\nTopic {tp['id']+1}: {', '.join(tp['words'][:15])}")
    else:
        for mode in ("processed", "raw"):
            if mode not in S["results"]:
                continue
            r = S["results"][mode]
            lines.append(f"\n[{mode.upper()}]")
            lines.append(f"Accuracy    : {r.get('acc', '—')}%")
            lines.append(f"F1 Macro    : {r.get('f1_macro', '—')}%")
            lines.append(f"F1 Weighted : {r.get('f1_weighted', '—')}%")
            if r.get("report_txt"):
                lines.append("\n" + r["report_txt"])

    text = "\n".join(lines)
    name = re.sub(r"[^\w\-]", "_", S["dataset_name"] or "results") + "_report.txt"
    return send_file(
        io.BytesIO(text.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=name
    )


@app.route("/api/export_tab_csv")
def export_tab_csv():
    """
    Export a tabular dataset as CSV.
    ?node_id=<id>            → full preprocessed dataset for that node
    ?node_id=<id>&split=train → train split (from model result stored for that node)
    ?node_id=<id>&split=test  → test split
    """
    node_id = request.args.get("node_id", "")
    split   = request.args.get("split", "")   # "train", "test", or ""

    base_name = re.sub(r"[^\w\-]", "_", _TAB.get("dataset_name") or node_id or "dataset")

    # ── Split mode: reconstruct from raw_rows using stored ratio ─────────────
    if split in ("train", "test"):
        import math, random as _rnd

        # Get the raw preprocessed rows for this node
        D = _effective_data(node_id) if node_id and node_id != "global" else None
        rows = (D["raw_rows"] if D else None) or _TAB.get("raw_rows", [])
        if not rows:
            return jsonify({"error": "No hay datos preprocesados disponibles para exportar la partición"}), 400

        # Ratio is passed as query param from the client (e.g. "0.7")
        ratio_str = request.args.get("ratio", "0.7")
        try:
            ratio = float(ratio_str)
        except ValueError:
            ratio = 0.7
        ratio = max(0.1, min(0.95, ratio))

        n_total = len(rows)
        n_train = math.floor(n_total * ratio)
        # Reproducible shuffle using a fixed seed so train/test are consistent
        rng = _rnd.Random(42)
        indices = list(range(n_total))
        rng.shuffle(indices)
        train_idx = set(indices[:n_train])

        selected = [rows[i] for i in indices if (split == "train") == (i in train_idx)]
        cols = list(rows[0].keys()) if rows else []
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        w.writerows(selected)
        fname = base_name + f"_{split}.csv"
        return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                         mimetype="text/csv", as_attachment=True, download_name=fname)

    # ── Full dataset mode ──────────────────────────────────────────────────
    D = _effective_data(node_id) if node_id and node_id != "global" else None
    rows = (D["raw_rows"] if D else None) or _TAB.get("raw_rows", [])
    if not rows:
        return jsonify({"error": "No hay datos tabulares cargados"}), 400
    cols = list(rows[0].keys()) if rows else []
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=base_name + ".csv"
    )


@app.route("/api/export_model")
def export_model():
    """Export the trained model + vectorizer as a .pkl file (NLP flow legacy)."""
    if S["model"] is None:
        return jsonify({"error": "No trained model"}), 400

    model_buf = io.BytesIO()
    joblib.dump({
        "model":        S["model"],
        "vectorizer":   S["vectorizer"],
        "label_names":  S["label_names"],
        "active_steps": S["active_steps"],
    }, model_buf)
    model_buf.seek(0)
    name = re.sub(r"[^\w\-]", "_", S["dataset_name"] or "model") + "_model.pkl"
    return send_file(
        model_buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=name
    )


@app.route("/api/save_tab_model", methods=["POST"])
def save_tab_model():
    """
    Save the full tabular model result as a .pkl.
    Body: { node_id: str, filename: str }
    The .pkl contains: sklearn model, preprocessing info, X_test/y_test,
    metrics, confusion matrix, feature names, target col, classes.
    """
    body    = request.get_json(force=True, silent=True) or {}
    node_id = str(body.get("node_id", ""))
    fname   = re.sub(r"[^\w\-]", "_", body.get("filename", "modelo")) + ".pkl"

    result = _MODEL_STORE.get(node_id) or _mtrain_result
    if not result:
        return jsonify({"error": "No hay modelo entrenado. Entrena primero."}), 400

    # Rebuild sklearn model object from _MODEL_STORE metadata
    # We store the full result dict; also persist the raw sklearn model if available
    # Normalise keys: CV uses "method"/"feat_names"/"X_te"/"y_te"/"y_te_pred"
    # Train-direct uses "model_type"/"features"/"_Xte"/"_yte"/"_yte_pred"
    _model_type  = result.get("model_type") or result.get("method")
    _features    = result.get("features")   or result.get("feat_names", [])
    _target_col  = result.get("target_col") or _TAB.get("target_col", "")
    _X_test      = result.get("_Xte")       or result.get("X_te")
    _y_test      = result.get("_yte")       or result.get("y_te")
    _y_pred      = result.get("_yte_pred")  or result.get("y_te_pred")
    _X_train     = result.get("_Xtr")       or result.get("X_tr")
    _y_train     = result.get("_ytr")       or result.get("y_tr")

    payload = {
        # ── Identity ──────────────────────────────────────────────────────
        "nlpflow_version": "4",
        "model_type":      _model_type,
        "problem_type":    result.get("problem_type"),
        "target_col":      _target_col,
        "features":        _features,
        "classes":         result.get("classes", []),
        "normalize":       result.get("normalize"),
        "alpha":           result.get("alpha"),
        "fit_intercept":   result.get("fit_intercept"),
        "fixed_intercept": result.get("fixed_intercept"),
        "intercept":       result.get("intercept"),
        # ── Metrics ───────────────────────────────────────────────────────
        "train_metrics":   result.get("train_metrics"),
        "test_metrics":    result.get("test_metrics"),
        "confusion_matrix":result.get("confusion_matrix"),
        "coefficients":    result.get("coefficients"),
        # ── Test set (for re-evaluation without new data) ─────────────────
        "X_test":          _X_test,
        "y_test":          _y_test,
        "y_pred":          _y_pred,
        "X_train":         _X_train,
        "y_train":         _y_train,
        # ── Dataset info ──────────────────────────────────────────────────
        "n_train":         result.get("n_train"),
        "n_test":          result.get("n_test"),
        # ── CV summary (if model was trained via grid search) ─────────────
        "from_cv":         result.get("from_cv", False),
        "cv_img":          result.get("cv_img"),
        "param_label":     result.get("param_label"),
        "metric_label":    result.get("metric_label"),
        "k_folds":         result.get("k_folds"),
        "best":            result.get("best"),
        "best_pv_display": result.get("best_pv_display") or (
            # Fallback: model trained without CV — build display from alpha
            (lambda a: (f"{a:.2e}" if a < 0.01 else str(round(a, 4))) if a is not None else None)(
                result.get("alpha")
            )
        ),
    }

    # Attach the live sklearn model object — try str and int keys
    def _get_model(nid):
        m = _NODE_MODELS.get(nid)
        if m is None and str(nid).isdigit():
            m = _NODE_MODELS.get(int(nid))
        if m is None:
            m = _NODE_MODELS.get(str(nid))
        return m
    payload["sklearn_model"] = _get_model(node_id)
    print(f"[save_tab_model] target_col={payload.get('target_col')!r}, sklearn_model found: {payload['sklearn_model'] is not None}")

    model_buf = io.BytesIO()
    joblib.dump(payload, model_buf)
    model_buf.seek(0)
    return send_file(
        model_buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=fname
    )


# Storage for loaded tabular models per node
_LOADED_TAB_MODELS: dict = {}


@app.route("/api/load_tab_model", methods=["POST"])
def load_tab_model():
    """
    Receive a .pkl file and store it for the given node.
    Returns metadata for the UI.
    """
    node_id = str(request.form.get("node_id", "load_default"))
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file received"}), 400
    try:
        payload = joblib.load(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({"error": f"No se pudo leer el .pkl: {e}"}), 400

    if not isinstance(payload, dict) or "model_type" not in payload:
        return jsonify({"error": "El fichero no parece un modelo de NLP Flow."}), 400

    _LOADED_TAB_MODELS[node_id] = payload

    # Also inject into _MODEL_STORE so model_evaluate works transparently
    # Map .pkl keys → _MODEL_STORE schema (same as train-direct result)
    _MODEL_STORE[node_id] = {
        "model_type":      payload.get("model_type"),
        "problem_type":    payload.get("problem_type"),
        "target_col":      payload.get("target_col"),
        "features":        payload.get("features", []),
        "classes":         payload.get("classes", []),
        "normalize":       payload.get("normalize"),
        "alpha":           payload.get("alpha"),
        "intercept":       payload.get("intercept"),
        "coefficients":    payload.get("coefficients"),
        "train_metrics":   payload.get("train_metrics"),
        "test_metrics":    payload.get("test_metrics"),
        "confusion_matrix":payload.get("confusion_matrix"),
        "coef_img":        None,   # will be regenerated by model_evaluate
        "n_train":         payload.get("n_train"),
        "n_test":          payload.get("n_test"),
        # CV summary (present if model was trained via grid search and saved with pkl)
        "from_cv":          payload.get("from_cv", False),
        "cv_img":           payload.get("cv_img"),
        "param_label":      payload.get("param_label"),
        "metric_label":     payload.get("metric_label"),
        "k_folds":          payload.get("k_folds"),
        "best":             payload.get("best"),
        "best_pv_display":  payload.get("best_pv_display") or (
            (lambda a: (f"{a:.2e}" if a < 0.01 else str(round(a, 4))) if a is not None else None)(
                payload.get("alpha")
            )
        ),
        # test arrays for plots/tests (use both key schemas for compatibility)
        "_Xte":      payload.get("X_test"),
        "_yte":      payload.get("y_test"),
        "_yte_pred": payload.get("y_pred"),
        "_Xtr":      payload.get("X_train"),
        "_ytr":      payload.get("y_train"),
    }
    if payload.get("sklearn_model") is not None:
        _NODE_MODELS[node_id] = payload["sklearn_model"]

    # Normalise metric keys to lowercase for consistent frontend display
    def _norm_metrics(m):
        if not m: return m
        return {k.lower(): v for k, v in m.items()}

    target_col_out = payload.get("target_col") or ""
    print(f"[load_tab_model] target_col from pkl: {target_col_out!r}")

    return jsonify({
        "model_type":   payload.get("model_type"),
        "problem_type": payload.get("problem_type"),
        "target_col":   target_col_out,
        "features":     payload.get("features", []),
        "classes":      payload.get("classes", []),
        "n_train":      payload.get("n_train"),
        "n_test":       payload.get("n_test"),
        "train_metrics": _norm_metrics(payload.get("train_metrics")),
        "test_metrics":  _norm_metrics(payload.get("test_metrics")),
    })


@app.route("/api/export_eval_metrics", methods=["GET"])
def export_eval_metrics():
    """
    Export train + test metrics from the last model_evaluate call.
    ?node_id=<id>&format=json|csv
    """
    node_id = str(request.args.get("node_id", ""))
    fmt     = request.args.get("format", "json")

    print(f"\n[export_eval_metrics] node_id={repr(node_id)}, format={fmt}")
    print(f"[export_eval_metrics] claves en _MODEL_STORE: {list(_MODEL_STORE.keys())}")

    result = _MODEL_STORE.get(node_id)
    if result is None and node_id.isdigit():
        result = _MODEL_STORE.get(int(node_id))
    if result is None and len(_MODEL_STORE) == 1:
        result = next(iter(_MODEL_STORE.values()))
        print(f"[export_eval_metrics] usando único resultado disponible")
    if result is None:
        result = _mtrain_result
    if not result:
        print(f"[export_eval_metrics] ERROR: sin resultado")
        return jsonify({"error": "No hay resultados de evaluación"}), 400

    train_m = result.get("train_metrics") or {}
    test_m  = result.get("test_metrics")  or {}
    data = {
        "model_type":   result.get("model_type") or result.get("method"),
        "problem_type": result.get("problem_type"),
        "target_col":   result.get("target_col"),
        "n_train":      result.get("n_train"),
        "n_test":       result.get("n_test"),
        "train_metrics": train_m,
        "test_metrics":  test_m,
    }

    if fmt == "csv":
        import csv as _csv
        buf = io.StringIO()
        w   = _csv.writer(buf)
        w.writerow(["split", "metric", "value"])
        for k, v in train_m.items():
            w.writerow(["train", k, v])
        for k, v in test_m.items():
            w.writerow(["test", k, v])
        content = buf.getvalue().encode("utf-8")
        return send_file(io.BytesIO(content), mimetype="text/csv",
                         as_attachment=True, download_name="eval_metrics.csv")
    else:
        import json as _json
        content = _json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        return send_file(io.BytesIO(content), mimetype="application/json",
                         as_attachment=True, download_name="eval_metrics.json")


@app.route("/api/export_split_zip", methods=["POST"])
def export_split_zip():
    """
    Build a ZIP with train.csv + test.csv using the preprocessed dataset
    and the split ratio/seed stored for a given node.
    Body: { node_id, ratio, mode }
    """
    import zipfile as _zf, math as _math, random as _rnd
    body    = request.get_json(force=True, silent=True) or {}
    node_id = str(body.get("node_id", ""))
    try:
        ratio = float(body.get("ratio", 0.7))
    except (ValueError, TypeError):
        ratio = 0.7
    ratio = max(0.1, min(0.95, ratio))

    print(f"\n[export_split_zip] ── Inicio ─────────────────────────────")
    print(f"[export_split_zip]   node_id={node_id!r}, ratio={ratio}")

    D = _effective_data(node_id) if node_id and node_id != "global" else None
    rows = (D["raw_rows"] if D else None) or _TAB.get("raw_rows", [])
    if not rows:
        return jsonify({"error": "No hay datos preprocesados disponibles"}), 400

    cols    = list(rows[0].keys())
    n_total = len(rows)
    n_train = _math.floor(n_total * ratio)

    rng     = _rnd.Random(42)
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_rows = [rows[i] for i in indices[:n_train]]
    test_rows  = [rows[i] for i in indices[n_train:]]

    print(f"[export_split_zip]   total={n_total}, train={len(train_rows)}, test={len(test_rows)}")

    def _rows_to_csv(rws):
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        w.writerows(rws)
        return buf.getvalue().encode("utf-8")

    train_csv = _rows_to_csv(train_rows)
    test_csv  = _rows_to_csv(test_rows)

    zip_buf = io.BytesIO()
    with _zf.ZipFile(zip_buf, "w", _zf.ZIP_DEFLATED) as zf:
        zf.writestr("train.csv", train_csv)
        print(f"[export_split_zip]   ✓ train.csv ({len(train_csv)} bytes)")
        zf.writestr("test.csv",  test_csv)
        print(f"[export_split_zip]   ✓ test.csv  ({len(test_csv)} bytes)")

    size = zip_buf.tell()
    zip_buf.seek(0)
    print(f"[export_split_zip]   ZIP total: {size} bytes")
    print(f"[export_split_zip] ── Fin ──────────────────────────────────\n")

    return send_file(zip_buf, mimetype="application/zip",
                     as_attachment=True, download_name="dataset_split.zip")


@app.route("/api/export_plots_zip", methods=["POST"])
def export_plots_zip():
    """
    Receive a list of { label, img } (base64 PNG) from the Plots block
    and return a ZIP file containing all of them as PNGs.
    Also supports native dialog via pick_export_path.
    """
    import zipfile as _zf, base64 as _b64, re as _re
    body  = request.get_json(force=True, silent=True) or {}
    plots = body.get("plots", [])   # [{ label: str, img: "data:image/png;base64,..." }]

    print(f"\n[export_plots_zip] ── Inicio ─────────────────────────────")
    print(f"[export_plots_zip]   plots recibidos: {len(plots)}")

    if not plots:
        return jsonify({"error": "No hay plots para exportar"}), 400

    zip_buf = io.BytesIO()
    with _zf.ZipFile(zip_buf, "w", _zf.ZIP_DEFLATED) as zf:
        for i, p in enumerate(plots):
            raw_b64 = p.get("img", "")
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            label = p.get("label") or f"plot_{i+1}"
            # Sanitise filename
            fname = _re.sub(r"[^\w\-\. ]", "_", label).strip() + ".png"
            try:
                raw = _b64.b64decode(raw_b64)
                zf.writestr(fname, raw)
                print(f"[export_plots_zip]   ✓ {fname} ({len(raw)} bytes)")
            except Exception as e:
                print(f"[export_plots_zip]   ✗ {fname}: {e}")

    size = zip_buf.tell()
    zip_buf.seek(0)
    print(f"[export_plots_zip]   ZIP total: {size} bytes")
    print(f"[export_plots_zip] ── Fin ──────────────────────────────────\n")

    return send_file(zip_buf, mimetype="application/zip",
                     as_attachment=True, download_name="plots.zip")


@app.route("/api/export_eval_zip", methods=["POST"])
def export_eval_zip():
    """
    Build a ZIP with metrics (JSON + CSV) and all available plots
    for the given model node_id. All steps are logged to terminal.
    """
    import zipfile, json as _json, csv as _csv, base64 as _b64

    body    = request.get_json(force=True, silent=True) or {}
    node_id = str(body.get("node_id", ""))

    print(f"\n[export_eval_zip] ── Inicio ──────────────────────────────")
    print(f"[export_eval_zip]   node_id recibido: {repr(node_id)}")
    print(f"[export_eval_zip]   claves en _MODEL_STORE: {list(_MODEL_STORE.keys())}")

    # Buscar resultado: primero por node_id exacto, luego como int, luego el último disponible
    result = _MODEL_STORE.get(node_id)
    if result is None and node_id.isdigit():
        result = _MODEL_STORE.get(int(node_id))
    if result is None:
        # Fallback: usar el único resultado si solo hay uno, o el más reciente
        if len(_MODEL_STORE) == 1:
            result = next(iter(_MODEL_STORE.values()))
            print(f"[export_eval_zip]   node_id no encontrado, usando único resultado en _MODEL_STORE")
        elif _mtrain_result:
            result = _mtrain_result
            print(f"[export_eval_zip]   usando _mtrain_result como fallback")

    if not result:
        print(f"[export_eval_zip]   ERROR: no hay resultados de evaluación")
        return jsonify({"error": "Sin resultados de evaluación. Pulsa Evaluar primero."}), 400

    print(f"[export_eval_zip]   resultado encontrado: model_type={result.get('model_type') or result.get('method')}, "
          f"problem_type={result.get('problem_type')}")

    train_m = result.get("train_metrics") or {}
    test_m  = result.get("test_metrics")  or {}
    print(f"[export_eval_zip]   métricas train: {list(train_m.keys())}")
    print(f"[export_eval_zip]   métricas test:  {list(test_m.keys())}")

    metrics_data = {
        "model_type":    result.get("model_type") or result.get("method"),
        "problem_type":  result.get("problem_type"),
        "target_col":    result.get("target_col"),
        "n_train":       result.get("n_train"),
        "n_test":        result.get("n_test"),
        "train_metrics": train_m,
        "test_metrics":  test_m,
    }

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── eval_metrics.json ──────────────────────────────────────────────
        json_str = _json.dumps(metrics_data, indent=2, ensure_ascii=False)
        zf.writestr("eval_metrics.json", json_str)
        print(f"[export_eval_zip]   ✓ eval_metrics.json  ({len(json_str)} bytes)")

        # ── eval_metrics.csv ───────────────────────────────────────────────
        csv_buf = io.StringIO()
        w = _csv.writer(csv_buf)
        w.writerow(["split", "metric", "value"])
        for k, v in train_m.items(): w.writerow(["train", k, v])
        for k, v in test_m.items():  w.writerow(["test",  k, v])
        csv_str = csv_buf.getvalue()
        zf.writestr("eval_metrics.csv", csv_str)
        print(f"[export_eval_zip]   ✓ eval_metrics.csv   ({len(csv_str)} bytes)")

        # ── Plots (base64 → PNG, todos en raíz del ZIP, sin subdirectorios) ─
        PLOT_KEYS = [
            ("pred_img",  "plot_pred_vs_real.png"),
            ("resid_img", "plot_residuos.png"),
            ("qq_img",    "plot_qq.png"),
            ("coef_img",  "plot_coeficientes.png"),
            ("cm_img",    "plot_confusion_matrix.png"),
        ]
        for key, fname in PLOT_KEYS:
            b64 = result.get(key)
            if b64:
                try:
                    raw = _b64.b64decode(b64)
                    zf.writestr(fname, raw)
                    print(f"[export_eval_zip]   ✓ {fname}  ({len(raw)} bytes)")
                except Exception as e:
                    print(f"[export_eval_zip]   ✗ {fname}  error al decodificar: {e}")
            else:
                print(f"[export_eval_zip]   – {fname}  no disponible (no cacheado)")

    zip_size = zip_buf.tell()
    zip_buf.seek(0)
    print(f"[export_eval_zip]   ZIP total: {zip_size} bytes")
    print(f"[export_eval_zip] ── Fin ────────────────────────────────────\n")

    return send_file(zip_buf, mimetype="application/zip",
                     as_attachment=True, download_name="evaluacion.zip")


@app.route("/api/eval_tab_model", methods=["POST"])
def eval_tab_model():
    """
    Evaluate a loaded tabular model.
    Mode 'saved'  → use stored X_test / y_test from the .pkl
    Mode 'new'    → receive a CSV, apply same column schema, run predict
    """
    import sklearn.metrics as skm

    # Support both JSON and multipart (new CSV mode)
    if request.content_type and "multipart" in request.content_type:
        node_id = str(request.form.get("node_id", "load_default"))
        mode    = request.form.get("mode", "new")
        csv_file = request.files.get("file")
    else:
        body    = request.get_json(force=True, silent=True) or {}
        node_id = str(body.get("node_id", "load_default"))
        mode    = body.get("mode", "saved")
        csv_file = None

    print(f"\n[eval_tab_model] content_type={request.content_type!r}")
    print(f"[eval_tab_model] node_id={node_id!r}, mode={mode!r}, csv_file={csv_file is not None}")
    print(f"[eval_tab_model] _LOADED_TAB_MODELS keys: {list(_LOADED_TAB_MODELS.keys())}")
    print(f"[eval_tab_model] _MODEL_STORE keys:       {list(_MODEL_STORE.keys())}")
    print(f"[eval_tab_model] _NODE_MODELS keys:       {list(_NODE_MODELS.keys())}")

    payload = _LOADED_TAB_MODELS.get(node_id)

    # Fallback: rebuild payload from _MODEL_STORE (train result) + _NODE_MODELS (sklearn obj)
    if not payload:
        ms = _MODEL_STORE.get(node_id) or _MODEL_STORE.get(int(node_id) if node_id.isdigit() else None)
        mdl = _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
        if ms:
            print(f"[eval_tab_model] payload not in _LOADED_TAB_MODELS — rebuilding from _MODEL_STORE")
            payload = dict(ms)
            if mdl:
                payload["sklearn_model"] = mdl
        else:
            print(f"[eval_tab_model] ERROR: no payload found for node_id={node_id!r}")

    if not payload:
        return jsonify({"error": "No hay modelo cargado para este nodo. Carga el .pkl primero."}), 400

    print(f"[eval_tab_model] payload keys: {[k for k in payload.keys() if k != 'sklearn_model']}")
    print(f"[eval_tab_model] model_type={payload.get('model_type')!r}, problem_type={payload.get('problem_type')!r}")
    print(f"[eval_tab_model] features ({len(payload.get('features',[]))}): {payload.get('features',[][:5])}")

    problem_type = payload.get("problem_type", "regression")
    classes      = payload.get("classes", [])

    def _compute_metrics(y_true, y_pred, problem_type, classes):
        if problem_type == "regression":
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            import math
            r2   = round(float(r2_score(y_true, y_pred)), 4)
            rmse = round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 4)
            mae  = round(float(mean_absolute_error(y_true, y_pred)), 4)
            return {"r2": r2, "rmse": rmse, "mae": mae}
        else:
            avg = "binary" if len(classes) == 2 else "macro"
            return {
                "accuracy":  round(float(skm.accuracy_score(y_true, y_pred)), 4),
                "f1":        round(float(skm.f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
                "precision": round(float(skm.precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
                "recall":    round(float(skm.recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            }

    def _cm_img(y_true, y_pred, classes):
        import numpy as np, matplotlib.pyplot as plt
        cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(max(4, len(classes)), max(3.5, len(classes) * 0.8)))
        fig.patch.set_facecolor(LIGHT); style_ax(ax)
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right", color=INK)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=INK)
        ax.set_xlabel(PL()["pred_lbl"], color=SEC, fontsize=11)
        ax.set_ylabel(PL()["actual_lbl"], color=SEC, fontsize=11)
        ax.set_title(PL()["cm_title"], color=INK, fontsize=12, fontweight="bold")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=13, fontweight="bold")
        plt.tight_layout(pad=1.4)
        return fig_b64(fig)

    def _roc_img(y_true, y_score, classes):
        """Generate ROC curve. y_score: (n, n_classes) proba or (n,) for binary."""
        import numpy as np, matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(LIGHT); style_ax(ax)
        n_cls = len(classes)
        if n_cls == 2:
            # Binary: use proba of positive class
            score = y_score[:, 1] if y_score.ndim == 2 else y_score
            fpr, tpr, _ = roc_curve(y_true, score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[0], linewidth=2,
                    label=f"AUC = {roc_auc:.3f}")
        else:
            # Multiclass: one-vs-rest
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            for i, cls_name in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=1.8,
                        label=f"{cls_name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color=SEC, linewidth=1, linestyle="--", label=PL()["chance"])
        ax.set_xlabel(PL()["fpr"], color=SEC, fontsize=11)
        ax.set_ylabel(PL()["tpr"], color=SEC, fontsize=11)
        ax.set_title(PL()["roc_title"], color=INK, fontsize=12, fontweight="bold")
        ax.legend(facecolor=BG, labelcolor=INK, fontsize=9, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5)
        plt.tight_layout(pad=1.4)
        return fig_b64(fig)

    # ── Mode: saved test set ───────────────────────────────────────────────
    print(f"[eval_tab_model] routing to mode={mode!r}")
    if mode == "saved":
        X_test = payload.get("X_test")
        y_test = payload.get("y_test")
        y_pred = payload.get("y_pred")
        if X_test is None or y_test is None:
            return jsonify({"error": "El .pkl no contiene test set guardado."}), 400

        import numpy as np
        y_test = np.array(y_test)
        y_pred = np.array(y_pred) if y_pred is not None else None

        # Try to re-predict with live model if available
        mdl = payload.get("sklearn_model")
        if mdl is not None:
            try: y_pred = mdl.predict(np.array(X_test))
            except Exception: pass

        if y_pred is None:
            return jsonify({"error": "No hay predicciones ni modelo sklearn en el .pkl."}), 400

        metrics = _compute_metrics(y_test, y_pred, problem_type, classes)
        cm_img  = None
        roc_img = None
        report  = None
        if problem_type != "regression" and classes:
            cm_img = _cm_img(y_test, y_pred, classes)
            report = skm.classification_report(y_test, y_pred,
                         target_names=classes, zero_division=0)
            # ROC: need predict_proba
            mdl2 = payload.get("sklearn_model")
            if mdl2 is not None and hasattr(mdl2, "predict_proba"):
                try:
                    y_score = mdl2.predict_proba(np.array(payload.get("X_test", [])))
                    roc_img = _roc_img(y_test, y_score, classes)
                except Exception as e:
                    print(f"[eval_tab_model] ROC error: {e}")
        return jsonify({
            "source":       "saved",
            "problem_type": problem_type,
            "metrics":      metrics,
            "cm_img":       cm_img,
            "roc_img":      roc_img,
            "report_txt":   report,
        })

    # ── Mode: new CSV ─────────────────────────────────────────────────────
    if mode == "new":
        if not csv_file:
            return jsonify({"error": "No se recibió ningún CSV."}), 400
        mdl = payload.get("sklearn_model")

        # Fallback: reconstruct from _NODE_MODELS in case the .pkl was saved without sklearn_model
        if mdl is None:
            mdl = _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
            if mdl:
                print(f"[eval_tab_model/new] sklearn_model recovered from _NODE_MODELS")

        # Last resort: rebuild a linear model from saved coefficients
        if mdl is None:
            coefs = payload.get("coefficients")
            intercept = payload.get("intercept")
            model_type = payload.get("model_type", "")
            if coefs is not None and model_type == "linreg":
                import numpy as np
                # coefficients may be [(name, val), ...] or [val, ...]
                if coefs and isinstance(coefs[0], (list, tuple)):
                    coef_vals = [float(c[1]) for c in coefs]
                else:
                    coef_vals = [float(c) for c in coefs]
                coef_arr = np.array(coef_vals, dtype=np.float64)
                intercept_val = float(intercept) if intercept is not None else 0.0
                class _LinearPredictor:
                    def __init__(self, coef, intercept):
                        self.coef_ = coef
                        self.intercept_ = intercept
                    def predict(self, X):
                        import numpy as _np
                        X = _np.array(X, dtype=_np.float64)
                        return X.dot(self.coef_) + self.intercept_
                mdl = _LinearPredictor(coef_arr, intercept_val)
                print(f"[eval_tab_model/new] sklearn_model reconstructed from coefficients (coef shape={coef_arr.shape})")

        if mdl is None:
            return jsonify({"error": "El .pkl no contiene el modelo sklearn.\nGuarda el modelo de nuevo (el .pkl actual es de una versión anterior)."}), 400

        import csv as csvmod, numpy as np
        txt = csv_file.read().decode("utf-8", errors="replace")
        reader = csvmod.DictReader(io.StringIO(txt))
        rows   = [r for r in reader]
        if not rows:
            return jsonify({"error": "CSV vacío."}), 400

        target_col = payload.get("target_col") or ""
        features   = payload.get("features", [])
        normalize  = payload.get("normalize", "none")

        csv_cols = list(rows[0].keys())

        # If target_col missing from payload (old .pkl), infer from CSV columns not in features
        if not target_col:
            extra_cols = [c for c in csv_cols if c not in features]
            if len(extra_cols) == 1:
                target_col = extra_cols[0]
                print(f"[eval_tab_model/new] target_col inferred from CSV: {target_col!r}")
            elif len(extra_cols) > 1:
                print(f"[eval_tab_model/new] multiple extra cols, cannot infer target: {extra_cols}")

        # Validate that required feature columns exist in the uploaded CSV
        missing    = [f for f in features if f not in csv_cols]
        has_target = bool(target_col and target_col in csv_cols)
        print(f"[eval_tab_model/new] CSV cols ({len(csv_cols)}): {csv_cols}")
        print(f"[eval_tab_model/new] Required features ({len(features)}): {features}")
        print(f"[eval_tab_model/new] target_col={target_col!r}, present={has_target}")
        print(f"[eval_tab_model/new] Missing cols: {missing}")
        if missing:
            return jsonify({
                "error": f"El CSV no contiene las columnas necesarias para este modelo.\n"
                         f"Faltan ({len(missing)}): {', '.join(missing[:10])}" +
                         (f" … y {len(missing)-10} más" if len(missing) > 10 else "")
            }), 400

        # Build X using only the feature columns
        feat_cols = features if features else [k for k in csv_cols if k != target_col]
        X_rows = []
        valid_rows = []
        skipped = 0
        for r in rows:
            row_x = []
            row_ok = True
            for c in feat_cols:
                v = r.get(c, "")
                fv = _try_float(v)
                if fv is None or _is_missing(v):
                    row_ok = False
                    break
                row_x.append(float(fv))
            if row_ok:
                X_rows.append(row_x)
                valid_rows.append(r)
            else:
                skipped += 1

        if skipped > 0:
            print(f"[eval_tab_model/new] {skipped} filas eliminadas por valores no numéricos")

        if not X_rows:
            return jsonify({"error": "El dataset no contiene filas válidas para este modelo.\n"
                                     "Todas las filas tienen valores no numéricos o vacíos en las columnas requeridas.\n"
                                     "Usa un dataset con datos numéricos válidos."}), 400

        rows = valid_rows   # use only valid rows for y_true calculation below
        X_new = np.array(X_rows, dtype=float)

        # Normalise if model was trained with normalisation
        if normalize and normalize != "none" and X_new.shape[0] > 0:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            sc = StandardScaler() if normalize == "standard" else MinMaxScaler()
            X_new = sc.fit_transform(X_new)

        # Align feature count to trained model
        n_feat = len(features)
        if X_new.shape[1] < n_feat:
            X_new = np.hstack([X_new, np.zeros((X_new.shape[0], n_feat - X_new.shape[1]))])
        elif X_new.shape[1] > n_feat:
            X_new = X_new[:, :n_feat]

        print(f"[eval_tab_model/new] X_new shape: {X_new.shape}")

        try:
            y_pred_new = mdl.predict(X_new)
            print(f"[eval_tab_model/new] predict OK, y_pred_new shape={y_pred_new.shape}")
        except Exception as e:
            import traceback
            print(f"[eval_tab_model/new] predict ERROR: {e}")
            print(traceback.format_exc())
            return jsonify({"error": f"Error al predecir: {e}"}), 400
        roc_img = None
        if has_target and problem_type != "regression":
            def _tstr(v):
                try: f = float(v); return str(int(f)) if f == int(f) else str(f)
                except: return str(v).strip()
            cls_list = payload.get("classes", [])
            lmap = {c: i for i, c in enumerate(cls_list)}
            y_true_enc = np.array([lmap.get(_tstr(r.get(target_col, "")), 0) for r in rows])
            metrics = _compute_metrics(y_true_enc, y_pred_new, problem_type, classes)
            cm_img  = _cm_img(y_true_enc, y_pred_new, classes) if classes else None
            report  = skm.classification_report(y_true_enc, y_pred_new,
                          target_names=classes, zero_division=0) if classes else None
            if classes and hasattr(mdl, "predict_proba"):
                try:
                    y_score = mdl.predict_proba(X_new)
                    roc_img = _roc_img(y_true_enc, y_score, classes)
                except Exception as e:
                    print(f"[eval_tab_model/new] ROC error: {e}")
        elif has_target and problem_type == "regression":
            y_true_f = np.array([float(r.get(target_col, 0)) for r in rows])
            metrics  = _compute_metrics(y_true_f, y_pred_new.astype(float), problem_type, classes)
            cm_img   = None
            report   = None
        else:
            metrics = {}
            cm_img  = None
            report  = "Sin columna objetivo en el CSV — no se pueden calcular métricas."

        return jsonify({
            "source":       "new",
            "problem_type": problem_type,
            "target_col":   target_col,
            "metrics":      metrics,
            "cm_img":       cm_img,
            "roc_img":      roc_img,
            "report_txt":   report,
            "n_rows":       len(rows),
            "skipped_rows": skipped,
        })

    return jsonify({"error": "Modo desconocido."}), 400


@app.route("/api/decision_boundary", methods=["POST"])
def decision_boundary():
    """
    Generate a 2D decision boundary plot for a classification model.
    Body: { node_id, feat_x, feat_y }
    feat_x / feat_y: feature names to use as axes.
    """
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    body    = request.get_json(force=True, silent=True) or {}
    node_id = str(body.get("node_id", ""))
    feat_x  = body.get("feat_x", "")
    feat_y  = body.get("feat_y", "")

    # Resolve payload
    payload = _LOADED_TAB_MODELS.get(node_id)
    if not payload:
        ms  = _MODEL_STORE.get(node_id) or _MODEL_STORE.get(int(node_id) if node_id.isdigit() else None)
        mdl = _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
        if ms:
            payload = dict(ms)
            if mdl: payload["sklearn_model"] = mdl
    if not payload:
        return jsonify({"error": "No hay modelo cargado."}), 400

    problem_type = payload.get("problem_type", "regression")
    if problem_type == "regression":
        return jsonify({"error": "La frontera de decisión solo está disponible para clasificación."}), 400

    mdl      = payload.get("sklearn_model")
    if mdl is None:
        mdl = _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
    if mdl is None or not hasattr(mdl, "predict"):
        return jsonify({"error": "Modelo no disponible. Reentrena el modelo."}), 400

    features = payload.get("features", [])
    classes  = payload.get("classes", [])
    X_test   = np.array(payload.get("X_test", []))
    y_test   = np.array(payload.get("y_test", []))

    if feat_x not in features or feat_y not in features:
        return jsonify({"error": f"Features '{feat_x}' o '{feat_y}' no encontradas en el modelo."}), 400
    if len(X_test) == 0:
        return jsonify({"error": "No hay datos de test guardados en el modelo."}), 400

    ix = features.index(feat_x)
    iy = features.index(feat_y)

    # Build 2D grid over the range of the two selected features
    x_min, x_max = X_test[:, ix].min() - 0.5, X_test[:, ix].max() + 0.5
    y_min, y_max = X_test[:, iy].min() - 0.5, X_test[:, iy].max() + 0.5
    h = (x_max - x_min) / 120.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Fill all other features with their mean from X_test
    n_feat = len(features)
    means  = X_test.mean(axis=0)
    grid_full = np.tile(means, (xx.ravel().shape[0], 1))
    grid_full[:, ix] = xx.ravel()
    grid_full[:, iy] = yy.ravel()

    try:
        Z = mdl.predict(grid_full)
    except Exception as e:
        return jsonify({"error": f"Error al predecir la malla: {e}"}), 400

    Z = Z.reshape(xx.shape)

    # Plot
    n_cls   = len(classes) if classes else int(Z.max()) + 1
    cmap_bg = ListedColormap([f"#{c}" for c in ["b3d4f5","f5b3b3","b3f5c8","f5e6b3","d4b3f5"][:n_cls]])
    cmap_pt = ListedColormap([PALETTE[i % len(PALETTE)] for i in range(n_cls)])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(LIGHT); style_ax(ax)

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors=[SEC], linewidths=0.8, linestyles="--", alpha=0.6)

    for i, cls_name in enumerate(classes or [str(c) for c in range(n_cls)]):
        mask = y_test == i
        ax.scatter(X_test[mask, ix], X_test[mask, iy],
                   color=PALETTE[i % len(PALETTE)], label=cls_name,
                   s=30, edgecolors="white", linewidth=0.4, alpha=0.85, zorder=3)

    ax.set_xlabel(feat_x, color=SEC, fontsize=11)
    ax.set_ylabel(feat_y, color=SEC, fontsize=11)
    ax.set_title(f"{PL()['boundary_title']} — {feat_x} vs {feat_y}", color=INK, fontsize=12, fontweight="bold")
    ax.legend(facecolor=BG, labelcolor=INK, fontsize=9, loc="best")
    plt.tight_layout(pad=1.4)

    return jsonify({"img": fig_b64(fig), "feat_x": feat_x, "feat_y": feat_y})


@app.route("/api/session_status")
def session_status():
    """Quick summary of what's available to save."""
    return jsonify({
        "has_data":       bool(S["texts"]),
        "has_processed":  bool(S["processed_texts"] and S["processed_texts"] != S["texts"]),
        "has_model":      S["model"] is not None,
        "has_results":    bool(S["results"]),
        "dataset_name":   S["dataset_name"],
        "task":           S["task"],
        "n_texts":        len(S["texts"]),
        "active_steps":   S["active_steps"],
    })


@app.route("/api/save_status")
def save_status():
    """
    Returns what is currently available to save across all sources:
    tabular datasets, trained models, plots, NLP corpus.
    """
    # ── Tabular models ────────────────────────────────────────────────────
    tab_models = []
    for nid, res in _MODEL_STORE.items():
        mtype  = res.get("model_type") or res.get("method") or "modelo"
        target = res.get("target_col") or _TAB.get("target_col", "")
        prob   = res.get("problem_type", "")
        ntrain = res.get("n_train", "?")
        ntest  = res.get("n_test",  "?")
        tab_models.append({
            "node_id":      nid,
            "model_type":   mtype,
            "target_col":   target,
            "problem_type": prob,
            "n_train":      ntrain,
            "n_test":       ntest,
        })

    # ── Tabular datasets ──────────────────────────────────────────────────
    tab_datasets = []
    for nid, d in _TAB_DATA.items() if hasattr(globals(), "_TAB_DATA") else []:
        tab_datasets.append({"node_id": nid, "name": d.get("name", nid), "rows": len(d.get("raw_rows", []))})
    # Also check _TAB global
    if _TAB.get("raw_rows"):
        tab_datasets.append({"node_id": "global", "name": _TAB.get("dataset_name", "dataset"), "rows": len(_TAB["raw_rows"])})

    # ── NLP corpus ────────────────────────────────────────────────────────
    nlp = {}
    if S["texts"]:
        nlp["has_raw"]       = True
        nlp["has_processed"] = bool(S["processed_texts"] and S["processed_texts"] != S["texts"])
        nlp["n_texts"]       = len(S["texts"])
        nlp["dataset_name"]  = S["dataset_name"] or "corpus"

    # ── Plots (stored in plotHistory per node via client — server doesn't hold them) ──
    # plots are base64 in browser memory; we signal the client to include them
    return jsonify({
        "tab_models":   tab_models,
        "tab_datasets": tab_datasets,
        "nlp":          nlp,
    })


# ════════════════════════════════════════════════════════════════════════════
# LLM TOPIC LABELING  — Hugging Face Inference API (free, no key needed)
# ════════════════════════════════════════════════════════════════════════════

# ── Hugging Face InferenceClient (lazy init) ─────────────────────────────────
try:
    from huggingface_hub import InferenceClient as _HFClient
    _HF_AVAILABLE = True
except ImportError:
    _HFClient     = None
    _HF_AVAILABLE = False

_HF_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
_hf_client = None   # created on first use

def _get_hf_client():
    global _hf_client
    if _hf_client is None and _HF_AVAILABLE:
        _hf_client = _HFClient(_HF_MODEL)
    return _hf_client

# Toggle: True = auto-label topics via HF when topic model runs
HF_LABELING_ACTIVE = False

@app.route("/api/labeling_toggle", methods=["POST"])
def labeling_toggle():
    global HF_LABELING_ACTIVE
    body = request.get_json(force=True, silent=True) or {}
    HF_LABELING_ACTIVE = bool(body.get("active", not HF_LABELING_ACTIVE))
    return jsonify({"active": HF_LABELING_ACTIVE})

@app.route("/api/labeling_status", methods=["GET"])
def labeling_status():
    return jsonify({"active": HF_LABELING_ACTIVE})

def _call_llm(prompt: str) -> str:
    """
    Call Hugging Face Inference API via huggingface_hub InferenceClient.
    Uses the free serverless endpoint (no API key required for public models).
    Returns the generated text or empty string on failure.
    """
    client = _get_hf_client()
    if client is None:
        return ""
    try:
        result = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
        )
        # result is a string when return_full_text=False
        return result.strip() if isinstance(result, str) else ""
    except Exception:
        return ""

def cleanup_hf_cache() -> None:
    """
    Remove Hugging Face cache directories created during this session.
    Called when the app window closes.
    """
    import pathlib
    hf_home = os.environ.get("HF_HOME") or os.path.join(pathlib.Path.home(), ".cache", "huggingface")
    hub_cache = os.path.join(hf_home, "hub")
    # Only wipe the hub model cache, not credentials or tokens
    if os.path.isdir(hub_cache):
        try:
            shutil.rmtree(hub_cache)
        except Exception:
            pass


def _rule_label(words: list[str]) -> str:
    """Fast deterministic fallback: title-case first 3 words."""
    return " / ".join(w.capitalize() for w in words[:3])


@app.route("/api/llm_label_topics", methods=["POST"])
def llm_label_topics():
    res = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return jsonify({"error": "Ejecuta primero el Topic Model"}), 400

    topic_list = "\n".join(
        f"Tópico {tp['id']+1}: {', '.join(tp['words'][:10])}"
        for tp in topics
    )
    prompt = f"""Eres un experto en análisis de texto. Se te dan los tópicos descubiertos por un modelo LDA/NMF/LSA.
Para cada tópico, propón una etiqueta descriptiva corta (2-4 palabras en español) que capture el tema principal.

Tópicos:
{topic_list}

Responde ÚNICAMENTE con un JSON válido con esta estructura exacta (sin explicación extra):
{{
  "labels": [
    {{"id": 0, "label": "Etiqueta del tópico 1"}},
    {{"id": 1, "label": "Etiqueta del tópico 2"}}
  ],
  "reasoning": "Una frase breve explicando el criterio de etiquetado"
}}"""

    raw = _call_llm(prompt)

    # Parse JSON from LLM response
    labels_out = []
    reasoning  = ""
    llm_ok     = False

    if raw:
        # Extract first JSON block from response
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                parsed    = json.loads(json_match.group())
                llm_labels = parsed.get("labels", [])
                reasoning  = parsed.get("reasoning", "")
                if llm_labels:
                    for tp in topics:
                        match = next((l for l in llm_labels if l.get("id") == tp["id"]), None)
                        lbl   = match["label"] if match else _rule_label(tp["words"])
                        labels_out.append({"id": tp["id"], "label": lbl, "words": tp["words"]})
                    llm_ok = True
            except Exception:
                pass

    if not llm_ok:
        # Fallback: rule-based labels (first 3 words, capitalised)
        for tp in topics:
            labels_out.append({
                "id":    tp["id"],
                "label": _rule_label(tp["words"]),
                "words": tp["words"]
            })
        reasoning = "Etiquetado automático por reglas (Hugging Face no disponible o sin respuesta)."

    # Store labels in results for later reference
    S["results"]["topic_labels"] = {str(item["id"]): item["label"] for item in labels_out}

    return jsonify({"labels": labels_out, "reasoning": reasoning, "llm_used": llm_ok})


# ════════════════════════════════════════════════════════════════════════════
# NATIVE SAVE DIALOG  — uses pywebview file dialog
# ════════════════════════════════════════════════════════════════════════════

def _build_session_zip(canvas_json: str) -> bytes:
    """Build the session ZIP in memory and return raw bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("canvas.json", canvas_json)

        data_payload = {
            "dataset_name": S["dataset_name"], "task": S["task"],
            "texts": S["texts"], "labels": S["labels"],
            "label_names": S["label_names"], "active_steps": S["active_steps"],
            "columns": S["columns"], "csv_source": S.get("csv_source", ""),
        }
        zf.writestr("data.json", json.dumps(data_payload, ensure_ascii=False, indent=2))

        if S["processed_texts"]:
            csv_buf = io.StringIO()
            writer  = csv.writer(csv_buf)
            has_labels = bool(S["labels"])
            writer.writerow(["idx", "original_text", "processed_text"] + (["label"] if has_labels else []))
            for i, (orig, proc) in enumerate(zip(S["texts"], S["processed_texts"])):
                row = [i, orig, proc]
                if has_labels and i < len(S["labels"]):
                    row.append(S["label_names"][S["labels"][i]] if S["label_names"] else S["labels"][i])
                writer.writerow(row)
            zf.writestr("preprocessed.csv", csv_buf.getvalue())

        if S["model"] is not None and S["vectorizer"] is not None:
            model_buf = io.BytesIO()
            joblib.dump({"model": S["model"], "vectorizer": S["vectorizer"],
                         "label_names": S["label_names"], "active_steps": S["active_steps"]}, model_buf)
            zf.writestr("model.pkl", model_buf.getvalue())

        if S["results"]:
            zf.writestr("results.json", json.dumps(_safe_results_for_json(S["results"]),
                                                    ensure_ascii=False, indent=2))
            rep_txt = ""
            if "processed" in S["results"]:
                rep_txt = S["results"]["processed"].get("report_txt", "")
            elif "topics" in S["results"]:
                lines = ["NLP Flow — Topic Model Report", "="*40]
                for tp in S["results"].get("topics", []):
                    lbl = S["results"].get("topic_labels", {}).get(str(tp["id"]), "")
                    lines.append(f"\nTópico {tp['id']+1}" + (f" [{lbl}]" if lbl else "") +
                                 f": {', '.join(tp['words'][:10])}")
                rep_txt = "\n".join(lines)
            if rep_txt:
                zf.writestr("report.txt", rep_txt)

        # Word freq plot
        try:
            img = _wordfreq_b64()
            if img:
                zf.writestr("plots/word_frequency.png", base64.b64decode(img))
        except Exception:
            pass

        # Results plot
        try:
            if S["results"]:
                plot_resp = app.test_client().get("/api/plot_results")
                pd = json.loads(plot_resp.data)
                if pd.get("img"):
                    zf.writestr("plots/results.png", base64.b64decode(pd["img"]))
        except Exception:
            pass

        readme = f"""NLP Flow Session
================
Dataset : {S['dataset_name'] or '—'}
Task    : {S['task']}
Texts   : {len(S['texts'])}
Steps   : {', '.join(S['active_steps']) or 'none'}
Trained : {'yes' if S['model'] else 'no'}
"""
        zf.writestr("README.txt", readme)
    buf.seek(0)
    return buf.read()


def _native_save_dialog(default_name: str, file_types_wv: tuple, filetypes_tk: list) -> str | None:
    """Open a native Save dialog. Returns chosen path or None if cancelled.
    Tries pywebview first; falls back to tkinter (always available on macOS/Win/Linux).
    Raises RuntimeError only if both methods are unavailable.
    """
    # ── 1. Try pywebview ──────────────────────────────────────────────────────
    try:
        import webview
        windows = webview.windows
        if windows:
            win = windows[0]
            result = win.create_file_dialog(
                webview.SAVE_DIALOG,
                directory    = os.path.expanduser("~"),
                save_filename= default_name,
                file_types   = file_types_wv
            )
            if result is None:
                return None          # user cancelled
            path = result[0] if isinstance(result, (list, tuple)) else result
            return path or None
    except Exception:
        pass

    # ── 2. Fallback: tkinter ──────────────────────────────────────────────────
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            initialfile   = default_name,
            defaultextension = os.path.splitext(default_name)[1] or "",
            filetypes     = filetypes_tk,
            parent        = root,
        )
        root.destroy()
        return path or None
    except Exception as tk_err:
        raise RuntimeError(f"No native dialog available: {tk_err}")


@app.route("/api/pick_save_path", methods=["POST"])
def pick_save_path():
    """Open a native Save-File dialog, write the session ZIP to the chosen path."""
    body        = request.json or {}
    canvas_json = body.get("canvas", "{}")
    try:
        path = _native_save_dialog(
            "nlpflow_session.zip",
            ("ZIP Archive (*.zip)", "All files (*.*)"),
            [("ZIP Archive", "*.zip"), ("All files", "*.*")]
        )
        if not path:
            return jsonify({"cancelled": True})
        if not path.lower().endswith(".zip"):
            path += ".zip"
        zip_bytes = _build_session_zip(canvas_json)
        with open(path, "wb") as f:
            f.write(zip_bytes)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e), "fallback": True})


@app.route("/api/pick_export_img", methods=["POST"])
def pick_export_img():
    """Save a single base64-encoded PNG/JPG image via native Save dialog."""
    body         = request.json or {}
    img_b64      = body.get("img_b64", "")
    default_name = body.get("default_name", "imagen.png")

    if not img_b64:
        return jsonify({"error": "No image data"}), 400

    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        return jsonify({"error": "Invalid base64: " + str(e)}), 400

    ext = os.path.splitext(default_name)[1] or ".png"
    try:
        path = _native_save_dialog(
            default_name,
            ("PNG Image (*.png)", "JPEG Image (*.jpg)", "All files (*.*)"),
            [("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*.*")]
        )
        if not path:
            return jsonify({"cancelled": True})
        if not path.lower().endswith(ext):
            path += ext
        with open(path, "wb") as f:
            f.write(img_bytes)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e), "fallback": True})


@app.route("/api/pick_export_wc_all", methods=["POST"])
def pick_export_wc_all():
    """Generate all topic word clouds and save them as a ZIP via native dialog."""
    res    = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return jsonify({"error": "Ejecuta primero el Topic Model"}), 400

    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for tp in topics:
                words  = tp.get("words", [])
                scores = tp.get("scores", [])
                freq   = {w: float(s) for w, s in zip(words, scores)} if scores else {w: 1.0 for w in words}
                wc = WordCloud(width=800, height=400, background_color="white",
                               max_words=40, colormap="viridis").generate_from_frequencies(freq)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                lbl = res.get("topic_labels", {}).get(str(tp["id"]), "")
                ax.set_title("Tópico " + str(tp["id"]+1) + (" — " + lbl if lbl else ""),
                             fontsize=13, pad=10)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                safe_lbl = lbl.replace(" ", "_").replace("/", "-") if lbl else ""
                fname    = "topico_{:02d}".format(tp["id"]+1) + ("_" + safe_lbl if safe_lbl else "") + ".png"
                zf.writestr(fname, buf.read())
        zip_buf.seek(0)
        zip_bytes = zip_buf.read()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        path = _native_save_dialog(
            "wordclouds_topicos.zip",
            ("ZIP Archive (*.zip)", "All files (*.*)"),
            [("ZIP Archive", "*.zip"), ("All files", "*.*")]
        )
        if not path:
            return jsonify({"cancelled": True})
        if not path.lower().endswith(".zip"):
            path += ".zip"
        with open(path, "wb") as f:
            f.write(zip_bytes)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e), "fallback": True})


@app.route("/api/pick_export_path", methods=["POST"])
def pick_export_path():
    """
    Open a native Save-File dialog, generate the requested export content
    (csv / model / report) and write it to the chosen path.
    """
    body         = request.json or {}
    endpoint     = body.get("endpoint", "")
    default_name = body.get("default_name", "export")
    ext          = os.path.splitext(default_name)[1] or ".bin"

    # Build the content bytes locally (no HTTP round-trip)
    try:
        if endpoint == "/api/export_csv":
            texts = S["texts"]; proc = S["processed_texts"] or texts
            if not texts: return jsonify({"error": "No data loaded"}), 400
            csv_buf = io.StringIO()
            writer  = csv.writer(csv_buf)
            has_labels = bool(S["labels"])
            writer.writerow(["idx","original_text","processed_text"] + (["label"] if has_labels else []))
            for i,(orig,pr) in enumerate(zip(texts, proc)):
                row = [i, orig, pr]
                if has_labels and i < len(S["labels"]):
                    row.append(S["label_names"][S["labels"][i]] if S["label_names"] else S["labels"][i])
                writer.writerow(row)
            content_bytes = csv_buf.getvalue().encode("utf-8")
            file_types = ("CSV File (*.csv)", "All files (*.*)")

        elif endpoint == "/api/export_model":
            if S["model"] is None: return jsonify({"error": "No trained model"}), 400
            model_buf = io.BytesIO()
            joblib.dump({"model": S["model"], "vectorizer": S["vectorizer"],
                         "label_names": S["label_names"], "active_steps": S["active_steps"]}, model_buf)
            content_bytes = model_buf.getvalue()
            file_types = ("Pickle File (*.pkl)", "All files (*.*)")

        elif endpoint == "/api/export_report":
            if not S["results"]: return jsonify({"error": "No results"}), 400
            lines = [f"NLP Flow — Report\nDataset: {S['dataset_name']}\n" + "="*52]
            if "topics" in S["results"]:
                for tp in S["results"].get("topics", []):
                    lbl = S["results"].get("topic_labels", {}).get(str(tp["id"]), "")
                    lines.append(f"\nTópico {tp['id']+1}" + (f" [{lbl}]" if lbl else "") +
                                 f": {', '.join(tp['words'][:10])}")
            else:
                for mode in ("processed","raw"):
                    if mode not in S["results"]: continue
                    r = S["results"][mode]
                    lines.append(f"\n[{mode.upper()}]\nAccuracy: {r.get('acc','—')}%\nF1 Macro: {r.get('f1_macro','—')}%")
                    if r.get("report_txt"): lines.append("\n" + r["report_txt"])
            content_bytes = "\n".join(lines).encode("utf-8")
            file_types = ("Text File (*.txt)", "All files (*.*)")

        # ── Tabular model .pkl ────────────────────────────────────────────
        elif endpoint == "/api/save_tab_model":
            extra   = body.get("extra", {})
            node_id = str(extra.get("node_id", ""))
            result  = _MODEL_STORE.get(node_id) or _mtrain_result
            if not result:
                return jsonify({"error": "No hay modelo entrenado"}), 400
            _model_type = result.get("model_type") or result.get("method") or "modelo"
            _features   = result.get("features")   or result.get("feat_names", [])
            _target_col = result.get("target_col") or _TAB.get("target_col", "")
            _X_test  = result.get("_Xte") or result.get("X_te")
            _y_test  = result.get("_yte") or result.get("y_te")
            _y_pred  = result.get("_yte_pred") or result.get("y_te_pred")
            _X_train = result.get("_Xtr") or result.get("X_tr")
            _y_train = result.get("_ytr") or result.get("y_tr")
            pkl_payload = {
                "nlpflow_version": "4",
                "model_type":   _model_type,
                "problem_type": result.get("problem_type"),
                "target_col":   _target_col,
                "features":     _features,
                "classes":      result.get("classes", []),
                "normalize":    result.get("normalize"),
                "alpha":        result.get("alpha"),
                "intercept":    result.get("intercept"),
                "train_metrics":result.get("train_metrics"),
                "test_metrics": result.get("test_metrics"),
                "confusion_matrix": result.get("confusion_matrix"),
                "coefficients": result.get("coefficients"),
                "X_test": _X_test, "y_test": _y_test, "y_pred": _y_pred,
                "X_train": _X_train, "y_train": _y_train,
                "n_train": result.get("n_train"), "n_test": result.get("n_test"),
                "sklearn_model": _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if str(node_id).isdigit() else node_id),
                # CV fields — present when model was trained via grid search
                "from_cv":        result.get("from_cv", False),
                "cv_img":         result.get("cv_img"),
                "param_label":    result.get("param_label"),
                "metric_label":   result.get("metric_label"),
                "k_folds":        result.get("k_folds"),
                "best":           result.get("best"),
                "best_pv_display":result.get("best_pv_display"),
            }
            model_buf = io.BytesIO()
            joblib.dump(pkl_payload, model_buf)
            content_bytes = model_buf.getvalue()
            file_types = ("Pickle File (*.pkl)", "All files (*.*)")

        # ── Tabular dataset .csv ──────────────────────────────────────────
        elif endpoint == "/api/export_tab_csv":
            extra   = body.get("extra", {})
            node_id = str(extra.get("node_id", ""))
            D = _effective_data(node_id) if node_id and node_id != "global" else None
            rows = (D["raw_rows"] if D else None) or _TAB.get("raw_rows", [])
            if not rows:
                return jsonify({"error": "No hay datos tabulares"}), 400
            cols = list(rows[0].keys()) if rows else []
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
            content_bytes = buf.getvalue().encode("utf-8")
            file_types = ("CSV File (*.csv)", "All files (*.*)")

        elif endpoint == "/api/export_split_zip":
            import zipfile as _zfs, math as _ms, random as _rs
            extra   = body.get("extra", {})
            node_id = str(extra.get("node_id", ""))
            try:
                ratio = float(extra.get("ratio", 0.7))
            except (ValueError, TypeError):
                ratio = 0.7
            ratio = max(0.1, min(0.95, ratio))
            D    = _effective_data(node_id) if node_id and node_id != "global" else None
            rows = (D["raw_rows"] if D else None) or _TAB.get("raw_rows", [])
            if not rows:
                return jsonify({"error": "No hay datos preprocesados"}), 400
            cols    = list(rows[0].keys())
            n_train = _ms.floor(len(rows) * ratio)
            rng     = _rs.Random(42)
            idxs    = list(range(len(rows))); rng.shuffle(idxs)
            tr_rows = [rows[i] for i in idxs[:n_train]]
            te_rows = [rows[i] for i in idxs[n_train:]]
            def _to_csv(rws):
                buf2 = io.StringIO()
                w2   = csv.DictWriter(buf2, fieldnames=cols)
                w2.writeheader(); w2.writerows(rws)
                return buf2.getvalue().encode("utf-8")
            zip_io3 = io.BytesIO()
            print(f"\n[pick_export_path/split_zip] {len(tr_rows)} train + {len(te_rows)} test")
            with _zfs.ZipFile(zip_io3, "w", _zfs.ZIP_DEFLATED) as zf3:
                tr_csv = _to_csv(tr_rows); zf3.writestr("train.csv", tr_csv)
                te_csv = _to_csv(te_rows); zf3.writestr("test.csv",  te_csv)
                print(f"  ✓ train.csv ({len(tr_csv)} bytes)")
                print(f"  ✓ test.csv  ({len(te_csv)} bytes)")
            content_bytes = zip_io3.getvalue()
            print(f"  ZIP total: {len(content_bytes)} bytes")
            file_types = ("ZIP File (*.zip)", "All files (*.*)")

        elif endpoint == "/api/export_plots_zip":
            import zipfile as _zfp, base64 as _b64p, re as _rep
            extra  = body.get("extra", {})
            plots  = extra.get("plots", [])
            if not plots:
                return jsonify({"error": "No hay plots"}), 400
            zip_io2 = io.BytesIO()
            print(f"\n[pick_export_path/plots_zip] Construyendo ZIP con {len(plots)} plots")
            with _zfp.ZipFile(zip_io2, "w", _zfp.ZIP_DEFLATED) as zf2:
                for i, p in enumerate(plots):
                    raw_b64 = p.get("img", "")
                    if "," in raw_b64: raw_b64 = raw_b64.split(",", 1)[1]
                    label = p.get("label") or f"plot_{i+1}"
                    fname = _rep.sub(r"[^\w\-\. ]", "_", label).strip() + ".png"
                    try:
                        raw = _b64p.b64decode(raw_b64)
                        zf2.writestr(fname, raw)
                        print(f"  ✓ {fname} ({len(raw)} bytes)")
                    except Exception as e:
                        print(f"  ✗ {fname}: {e}")
            content_bytes = zip_io2.getvalue()
            print(f"  ZIP total: {len(content_bytes)} bytes")
            file_types = ("ZIP File (*.zip)", "All files (*.*)")

        elif endpoint == "/api/export_eval_zip":
            import zipfile as _zf, json as _json, csv as _csv2, base64 as _b64
            extra   = body.get("extra", {})
            node_id = str(extra.get("node_id", ""))
            result  = _MODEL_STORE.get(node_id)
            if result is None and node_id.isdigit():
                result = _MODEL_STORE.get(int(node_id))
            if result is None and len(_MODEL_STORE) == 1:
                result = next(iter(_MODEL_STORE.values()))
            if result is None:
                result = _mtrain_result
            if not result:
                return jsonify({"error": "Sin resultados. Pulsa Evaluar primero."}), 400

            train_m = result.get("train_metrics") or {}
            test_m  = result.get("test_metrics")  or {}
            metrics_data = {
                "model_type":   result.get("model_type") or result.get("method"),
                "problem_type": result.get("problem_type"),
                "target_col":   result.get("target_col"),
                "n_train":      result.get("n_train"),
                "n_test":       result.get("n_test"),
                "train_metrics": train_m,
                "test_metrics":  test_m,
            }
            zip_io = io.BytesIO()
            print(f"\n[pick_export_path/eval_zip] Construyendo ZIP para node_id={node_id}")
            with _zf.ZipFile(zip_io, "w", _zf.ZIP_DEFLATED) as zf:
                json_str = _json.dumps(metrics_data, indent=2, ensure_ascii=False)
                zf.writestr("eval_metrics.json", json_str)
                print(f"  ✓ eval_metrics.json ({len(json_str)} bytes)")

                csv_io = io.StringIO()
                cw = _csv2.writer(csv_io)
                cw.writerow(["split", "metric", "value"])
                for k, v in train_m.items(): cw.writerow(["train", k, v])
                for k, v in test_m.items():  cw.writerow(["test",  k, v])
                csv_str = csv_io.getvalue()
                zf.writestr("eval_metrics.csv", csv_str)
                print(f"  ✓ eval_metrics.csv  ({len(csv_str)} bytes)")

                PLOT_KEYS = [
                    ("pred_img",  "plot_pred_vs_real.png"),
                    ("resid_img", "plot_residuos.png"),
                    ("qq_img",    "plot_qq.png"),
                    ("coef_img",  "plot_coeficientes.png"),
                    ("cm_img",    "plot_confusion_matrix.png"),
                ]
                for key, fname in PLOT_KEYS:
                    b64 = result.get(key)
                    if b64:
                        raw = _b64.b64decode(b64)
                        zf.writestr(fname, raw)
                        print(f"  ✓ {fname} ({len(raw)} bytes)")
                    else:
                        print(f"  – {fname} no disponible")

            content_bytes = zip_io.getvalue()
            print(f"  ZIP total: {len(content_bytes)} bytes")
            file_types = ("ZIP File (*.zip)", "All files (*.*)")

        elif endpoint == "/api/export_eval_metrics":
            import json as _json2, csv as _csv3
            extra   = body.get("extra", {})
            node_id = str(extra.get("node_id", ""))
            fmt     = extra.get("format", "json")
            result  = _MODEL_STORE.get(node_id)
            if result is None and node_id.isdigit():
                result = _MODEL_STORE.get(int(node_id))
            if result is None and len(_MODEL_STORE) == 1:
                result = next(iter(_MODEL_STORE.values()))
            if result is None:
                result = _mtrain_result
            if not result:
                return jsonify({"error": "Sin resultados de evaluación"}), 400

            train_m = result.get("train_metrics") or {}
            test_m  = result.get("test_metrics")  or {}
            if fmt == "csv":
                csv_io2 = io.StringIO()
                cw2 = _csv3.writer(csv_io2)
                cw2.writerow(["split", "metric", "value"])
                for k, v in train_m.items(): cw2.writerow(["train", k, v])
                for k, v in test_m.items():  cw2.writerow(["test",  k, v])
                content_bytes = csv_io2.getvalue().encode("utf-8")
                file_types = ("CSV File (*.csv)", "All files (*.*)")
            else:
                data = {"model_type": result.get("model_type") or result.get("method"),
                        "problem_type": result.get("problem_type"),
                        "train_metrics": train_m, "test_metrics": test_m}
                content_bytes = _json2.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
                file_types = ("JSON File (*.json)", "All files (*.*)")

        else:
            return jsonify({"error": "Unknown endpoint"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Open native dialog (pywebview → tkinter fallback)
    tk_types = [(ft.split("(")[0].strip(), "*." + ext.lstrip(".")) for ft in [file_types[0]]]
    tk_types.append(("All files", "*.*"))
    try:
        path = _native_save_dialog(default_name, file_types, tk_types)
        if not path:
            return jsonify({"cancelled": True})
        if not path.lower().endswith(ext):
            path += ext
        with open(path, "wb") as f:
            f.write(content_bytes)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e), "fallback": True})

# ── Custom plot builder ──────────────────────────────────────────────────────
@app.route("/api/plot_custom", methods=["POST"])
def plot_custom():
    """Flexible plot builder for the 📊 Plots block."""
    body       = request.get_json(force=True, silent=True) or {}
    plot_type  = body.get("type", "histogram")
    x_col      = body.get("x", "")
    y_col      = body.get("y", "")
    color_col  = body.get("color", "")
    node_id    = body.get("node", "")
    task       = body.get("task", "classification")   # "classification" | "regression"
    target_col = body.get("target_col", "")

    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    cols = D["columns"]
    task = task or D.get("task", "classification")
    is_cls = (task == "classification")

    if not rows:
        return jsonify({"error": "No hay datos. Conecta este bloque a un bloque Datos y carga un dataset primero."}), 400

    # Guard: regression-only types blocked for regression
    if not is_cls and plot_type in ("bar", "pie"):
        return jsonify({"error": f"El tipo '{plot_type}' no es aplicable a problemas de regresión (no hay variables categóricas de clase)."}), 400

    def get_nums(col):
        return [float(r.get(col,"")) for r in rows
                if not _is_missing(r.get(col,"")) and _try_float(r.get(col,""))]
    def get_cats(col):
        return [str(r.get(col,"")) for r in rows if not _is_missing(r.get(col,""))]

    num_cols_all = [c for c in cols if _is_numeric_type(_col_type([r.get(c,"") for r in rows[:50]]))]
    cat_cols_all = [c for c in cols if _col_type([r.get(c,"") for r in rows[:50]]) == "categorical"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(LIGHT)
    style_ax(ax)

    # ── HISTOGRAMA — todas las variables numéricas en un grid ─────────────────
    if plot_type == "histogram":
        sel_cols = num_cols_all[:12]   # cap at 12 subplots
        if not sel_cols:
            return jsonify({"error": "No hay columnas numéricas"}), 400
        plt.close(fig)
        n = len(sel_cols)
        ncols_g = min(3, n)
        nrows_g = (n + ncols_g - 1) // ncols_g
        fig, axes = plt.subplots(nrows_g, ncols_g,
                                 figsize=(ncols_g * 4.2, nrows_g * 3.2))
        fig.patch.set_facecolor(LIGHT)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for i, c in enumerate(sel_cols):
            a = axes_flat[i]
            style_ax(a)
            nums = get_nums(c)
            if not nums:
                a.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                       transform=a.transAxes, color=SEC)
                a.set_title(c, color=INK, fontsize=10, fontweight="bold"); continue
            n_bins = min(25, max(8, len(set(nums))))
            a.hist(nums, bins=n_bins, color=PALETTE[i % len(PALETTE)],
                   edgecolor="white", linewidth=0.4, alpha=0.88)
            mean_v   = float(np.mean(nums))
            median_v = float(np.median(nums))
            # Mode: most frequent value
            from collections import Counter as _Cnt
            mode_raw = _Cnt(round(v, 4) for v in nums).most_common(1)
            mode_v   = mode_raw[0][0] if mode_raw else mean_v
            a.axvline(mean_v,   color="#E8785A", linewidth=1.6, linestyle="--",
                      label=f"Media {mean_v:.2f}", alpha=0.9)
            a.axvline(median_v, color="#5CB88A", linewidth=1.6, linestyle=":",
                      label=f"Mediana {median_v:.2f}", alpha=0.9)
            a.axvline(mode_v,   color="#8B78C9", linewidth=1.6, linestyle="-.",
                      label=f"Moda {mode_v:.2f}", alpha=0.9)
            a.set_title(c, color=INK, fontsize=10, fontweight="bold")
            a.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
            a.legend(facecolor=LIGHT, labelcolor=INK, fontsize=7,
                     edgecolor=BORDER, loc="upper right")
        # Hide unused axes
        for j in range(len(sel_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Distribución de variables numéricas",
                     color=INK, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout(pad=1.2, h_pad=1.8, w_pad=1.4)
        return jsonify({"img": fig_b64(fig)})

    # ── BARRAS — frecuencia por clase target (clasificación) ─────────────────
    elif plot_type == "bar":
        # Use target col if available, else fall back to x_col
        col_b = target_col if (target_col and target_col in cols) else x_col
        if col_b not in cols:
            return jsonify({"error": "No se encontró la columna objetivo. Elige una columna categórica."}), 400
        cats = get_cats(col_b)
        if not cats: return jsonify({"error": "Sin valores"}), 400
        counts_b = Counter(cats)
        top = counts_b.most_common(20)
        labels_b, freqs = zip(*top)
        total_b = sum(freqs)
        bars = ax.barh(range(len(labels_b)), freqs,
                       color=[PALETTE[i % len(PALETTE)] for i in range(len(labels_b))],
                       edgecolor="none", height=0.65)
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b, color=INK, fontsize=10)
        ax.set_xlabel("Número de muestras", color=SEC, fontsize=11)
        is_target = (col_b == target_col)
        ax.set_title(
            f"Distribución de clases — {col_b}" if is_target
            else f"Frecuencia de {col_b}",
            color=INK, fontsize=13, fontweight="bold")
        if is_target:
            ax.set_title(f"Distribución de la clase a predecir: {col_b}",
                         color=INK, fontsize=13, fontweight="bold")
        ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
        for b, f in zip(bars, freqs):
            pct = round(f / total_b * 100, 1)
            ax.text(f + max(freqs)*0.01, b.get_y() + b.get_height()/2,
                    f"{int(f)} ({pct}%)", va="center", color=INK, fontsize=9, fontweight="600")

    # ── BOXPLOT — normalizado por z-score si rangos muy dispares ─────────────
    elif plot_type == "boxplot":
        sel_b = num_cols_all[:8]
        if not sel_b: return jsonify({"error": "No hay columnas numéricas"}), 400
        data_boxes, labels_box = [], []
        for c in sel_b:
            n = get_nums(c)
            if n: data_boxes.append(n); labels_box.append(c)
        if not data_boxes: return jsonify({"error": "Sin datos"}), 400

        # Detect if ranges are too disparate to show in one plot
        # Metric: ratio of max IQR to min IQR (more robust than mean-based)
        iqrs = []
        for d in data_boxes:
            q75, q25 = np.percentile(d, [75, 25])
            iqrs.append(float(q75 - q25))
        valid_iqrs = [v for v in iqrs if v > 0]
        iqr_ratio = (max(valid_iqrs) / min(valid_iqrs)) if len(valid_iqrs) >= 2 else 1.0
        # Also check raw range ratio
        ranges = [max(d) - min(d) for d in data_boxes if d]
        range_ratio = (max(ranges) / max(1e-9, min(r for r in ranges if r > 0))) if ranges else 1.0
        # Only split into subplots when scales differ by more than 50×
        use_subplots = (iqr_ratio > 50 or range_ratio > 100)

        n_b = len(labels_box)
        plt.close(fig)

        if not use_subplots:
            # All in one boxplot — most readable when scales are similar
            fig, ax = plt.subplots(figsize=(max(6, n_b * 1.4), 5))
            fig.patch.set_facecolor(LIGHT); style_ax(ax)
            bp = ax.boxplot(data_boxes, patch_artist=True,
                            medianprops=dict(color="#5CB88A", linewidth=2.2),
                            boxprops=dict(linewidth=0),
                            whiskerprops=dict(color=SEC, linewidth=1.2),
                            capprops=dict(color=SEC, linewidth=1.2),
                            flierprops=dict(marker="o", color=SEC, markersize=4,
                                            alpha=0.4, linestyle="none"))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            ax.set_xticks(range(1, n_b + 1))
            ax.set_xticklabels(labels_box, rotation=20, ha="right", color=INK, fontsize=10)
            ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
            ax.set_title("Boxplot de variables numéricas", color=INK, fontsize=13, fontweight="bold")
        else:
            # Subplots: each variable with its own y-axis scale
            ncols_box = min(4, n_b)
            nrows_box = (n_b + ncols_box - 1) // ncols_box
            fig, axes_box = plt.subplots(nrows_box, ncols_box,
                                         figsize=(ncols_box * 3.2, nrows_box * 3.5))
            fig.patch.set_facecolor(LIGHT)
            ax_list = np.array(axes_box).flatten() if n_b > 1 else [axes_box]
            for i, (d, lbl) in enumerate(zip(data_boxes, labels_box)):
                a = ax_list[i]; style_ax(a)
                bp = a.boxplot([d], patch_artist=True,
                               medianprops=dict(color="#5CB88A", linewidth=2.2),
                               boxprops=dict(linewidth=0),
                               whiskerprops=dict(color=SEC, linewidth=1.2),
                               capprops=dict(color=SEC, linewidth=1.2),
                               flierprops=dict(marker="o", color=SEC, markersize=4,
                                               alpha=0.4, linestyle="none"))
                bp["boxes"][0].set_facecolor(PALETTE[i % len(PALETTE)])
                bp["boxes"][0].set_alpha(0.75)
                a.set_title(lbl, color=INK, fontsize=10, fontweight="bold")
                a.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
                a.set_xticks([])
            for j in range(n_b, len(ax_list)):
                ax_list[j].set_visible(False)
            plt.suptitle("Boxplot por variable (escalas independientes)",
                         color=INK, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout(pad=1.4)
        return jsonify({"img": fig_b64(fig)})

    # ── DISPERSIÓN ───────────────────────────────────────────────────────────
    elif plot_type == "scatter":
        if x_col not in cols or y_col not in cols:
            return jsonify({"error": "Selecciona columnas X e Y numéricas"}), 400
        # For classification: color by target automatically; for regression: no color
        eff_color = target_col if (is_cls and target_col and target_col in cols) else (
                    color_col if (is_cls and color_col and color_col in cols) else "")
        pairs = [(float(r.get(x_col,"")), float(r.get(y_col,"")),
                  str(r.get(eff_color,"clase")) if eff_color else "puntos")
                 for r in rows
                 if not _is_missing(r.get(x_col,"")) and not _is_missing(r.get(y_col,""))
                 and _try_float(r.get(x_col,"")) and _try_float(r.get(y_col,""))]
        if not pairs: return jsonify({"error": "Sin datos numéricos para esos ejes"}), 400
        uniq_g = sorted(set(p[2] for p in pairs))
        for i, g in enumerate(uniq_g):
            gp = [(p[0], p[1]) for p in pairs if p[2] == g]
            gx, gy = zip(*gp)
            ax.scatter(gx, gy, color=PALETTE[i % len(PALETTE)], alpha=0.68,
                       s=28, edgecolors="none",
                       label=g if eff_color and len(uniq_g) > 1 else None,
                       zorder=3)
        ax.set_xlabel(x_col, color=SEC, fontsize=11)
        ax.set_ylabel(y_col, color=SEC, fontsize=11)
        color_label = f" (color: {eff_color})" if eff_color else ""
        ax.set_title(f"Dispersión: {x_col} vs {y_col}{color_label}",
                     color=INK, fontsize=13, fontweight="bold")
        ax.grid(True, color=BORDER, linewidth=0.5, zorder=0)
        if eff_color and len(uniq_g) > 1:
            ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=10,
                      title=eff_color, markerscale=1.4, edgecolor=BORDER)

    # ── CORRELACIÓN — rojo=+1, azul=-1 ───────────────────────────────────────
    elif plot_type == "correlation":
        num_cols_sel = num_cols_all
        if len(num_cols_sel) < 2:
            return jsonify({"error": "Se necesitan al menos 2 columnas numéricas"}), 400
        num_cols_sel = num_cols_sel[:10]
        matrix_data = []
        for c in num_cols_sel:
            matrix_data.append(get_nums(c))
        min_len = min(len(d) for d in matrix_data)
        matrix_data = [d[:min_len] for d in matrix_data]
        corr = np.corrcoef(matrix_data)
        plt.close(fig)
        n = len(num_cols_sel)
        fig, ax = plt.subplots(figsize=(max(5, n*0.9+1), max(4, n*0.9)))
        fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
        from matplotlib.colors import LinearSegmentedColormap
        # Rojo=+1 (correlación positiva fuerte), azul=-1 (negativa fuerte)
        cmap = LinearSegmentedColormap.from_list("corr", ["#5B7FDB", "#ffffff", "#E8785A"])
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(num_cols_sel, rotation=40, ha="right", color=INK, fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(num_cols_sel, color=INK, fontsize=9)
        for i in range(n):
            for j in range(n):
                v = corr[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.55 else INK,
                        fontsize=8, fontweight="600")
        ax.set_title("Matriz de correlación",
                     color=INK, fontsize=13, fontweight="bold")
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cb.ax.tick_params(labelsize=8, colors=SEC)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors=SEC, labelsize=9)
        plt.tight_layout(pad=1.4)
        return jsonify({"img": fig_b64(fig)})

    # ── TARTA (solo clasificación) ────────────────────────────────────────────
    elif plot_type == "pie":
        col_p = target_col if (target_col and target_col in cols) else x_col
        if col_p not in cols:
            return jsonify({"error": "No se encontró la columna objetivo."}), 400
        cats = get_cats(col_p)
        if not cats: return jsonify({"error": "Sin valores"}), 400
        counts_p = Counter(cats)
        top_p = counts_p.most_common(10)
        if len(counts_p) > 10:
            others = sum(v for k, v in counts_p.items()
                         if k not in dict(top_p))
            top_p.append(("Otros", others))
        labels_p, freqs_p = zip(*top_p)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
        wedges, texts, autotexts = ax.pie(
            freqs_p, labels=labels_p, autopct="%1.1f%%",
            colors=PALETTE[:len(labels_p)], startangle=90,
            pctdistance=0.80, labeldistance=1.10,
            wedgeprops=dict(linewidth=0.5, edgecolor="white"))
        for t in texts:    t.set_fontsize(10); t.set_color(INK)
        for t in autotexts: t.set_fontsize(9); t.set_color("white"); t.set_fontweight("bold")
        ax.set_title(f"Distribución de la clase a predecir: {col_p}",
                     color=INK, fontsize=13, fontweight="bold")
        plt.tight_layout(pad=1.2)
        return jsonify({"img": fig_b64(fig)})

    # ── LÍNEAS — una línea por variable, normaliza si escalas muy dispares ────
    elif plot_type == "line":
        sel_l = num_cols_all[:6]
        if not sel_l:
            return jsonify({"error": "Sin columnas numéricas para gráfico de líneas"}), 400

        all_vals = [get_nums(c) for c in sel_l]
        all_vals = [(c, v) for c, v in zip(sel_l, all_vals) if v]
        if not all_vals:
            return jsonify({"error": "Sin datos"}), 400

        # Detect if scales are too disparate — only split then
        per_ranges = [max(vs) - min(vs) for _, vs in all_vals]
        valid_ranges = [r for r in per_ranges if r > 0]
        range_ratio_l = (max(valid_ranges) / min(valid_ranges)) if len(valid_ranges) >= 2 else 1.0
        use_subplots_l = range_ratio_l > 50 and len(all_vals) > 1

        plt.close(fig)
        n_l = len(all_vals)

        if not use_subplots_l:
            # All series in one plot — most readable
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor(LIGHT); style_ax(ax)
            for i, (c, vs) in enumerate(all_vals):
                ax.plot(range(len(vs)), vs, color=PALETTE[i % len(PALETTE)],
                        linewidth=1.6, alpha=0.85, label=c)
            ax.set_xlabel("Índice", color=SEC, fontsize=11)
            ax.set_ylabel("Valor", color=SEC, fontsize=11)
            ax.set_title("Evolución de variables numéricas",
                         color=INK, fontsize=13, fontweight="bold")
            ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
            ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=10, edgecolor=BORDER)
        else:
            # Subplots: each with its own y-axis scale
            ncols_l = min(3, n_l)
            nrows_l = (n_l + ncols_l - 1) // ncols_l
            fig, axes_l = plt.subplots(nrows_l, ncols_l,
                                       figsize=(ncols_l * 4.0, nrows_l * 2.8))
            fig.patch.set_facecolor(LIGHT)
            ax_list_l = np.array(axes_l).flatten() if n_l > 1 else [axes_l]
            for i, (c, vs) in enumerate(all_vals):
                a = ax_list_l[i]; style_ax(a)
                a.plot(range(len(vs)), vs, color=PALETTE[i % len(PALETTE)],
                       linewidth=1.6, alpha=0.85)
                a.set_title(c, color=INK, fontsize=10, fontweight="bold")
                a.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
                a.set_xlabel("Índice", color=SEC, fontsize=9)
            for j in range(n_l, len(ax_list_l)):
                ax_list_l[j].set_visible(False)
            plt.suptitle("Evolución por variable (escalas independientes)",
                         color=INK, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout(pad=1.4, h_pad=1.6, w_pad=1.2)
        return jsonify({"img": fig_b64(fig)})

    else:
        return jsonify({"error": f"Tipo de gráfico desconocido: {plot_type}"}), 400

    plt.tight_layout(pad=1.6)
    return jsonify({"img": fig_b64(fig)})

# ── Imbalance analysis (nuevo: comprensible) ─────────────────────────────────
@app.route("/api/imbalance_analyze", methods=["POST"])
def imbalance_analyze():
    """Analyze class imbalance for a target column. Returns stats + chart."""
    body       = request.get_json(force=True, silent=True) or {}
    target_col = body.get("target_col", "")
    node_id    = body.get("node", "")
    D    = _effective_data(node_id)
    rows = D["raw_rows"]
    if not rows: return jsonify({"error": "No hay datos. Conecta a un bloque Datos."}), 400
    if target_col not in D["columns"]:
        return jsonify({"error": f"Columna '{target_col}' no encontrada"}), 400

    targ = [str(r.get(target_col, "")) for r in rows if not _is_missing(r.get(target_col, ""))]
    if not targ: return jsonify({"error": "La columna no tiene valores"}), 400

    dist = Counter(targ)
    total = sum(dist.values())
    classes = sorted(dist.keys())
    counts  = [dist[c] for c in classes]
    pcts    = [round(dist[c] / total * 100, 1) for c in classes]
    n_cls   = len(classes)
    ratio   = round(max(counts) / max(1, min(counts)), 2) if n_cls > 1 else 1.0

    # Ideal distribution % per class
    ideal = round(100 / n_cls, 1) if n_cls > 0 else 100

    # Status classification
    if ratio < 1.5:
        status = "balanced"; status_label = "Balanceado ✓"; status_color = MINT
    elif ratio < 3:
        status = "mild"; status_label = "Leve desbalanceo ⚠"; status_color = "#F59E0B"
    elif ratio < 10:
        status = "moderate"; status_label = "Desbalanceo moderado ⚠"; status_color = "#EF4444"
    else:
        status = "severe"; status_label = "Desbalanceo severo ✗"; status_color = "#991B1B"

    # ── Figure: horizontal bars with ideal reference line ─────────────────
    fig, ax = plt.subplots(figsize=(7, max(3, n_cls * 0.6 + 1.5)))
    fig.patch.set_facecolor(LIGHT)
    style_ax(ax)

    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(n_cls)]
    bars = ax.barh(classes, pcts, color=bar_colors, edgecolor="none", height=0.55)

    # Ideal line
    ax.axvline(ideal, color=SEC, linewidth=1.4, linestyle="--",
               label=f"Ideal ({ideal}%)", alpha=0.7)

    for b, p in zip(bars, pcts):
        ax.text(p + 0.5, b.get_y() + b.get_height()/2,
                f"{p}%", va="center", color=INK, fontsize=10, fontweight="700")

    ax.set_xlabel("% de muestras", color=SEC, fontsize=11)
    ax.set_title(f"Distribución de clases — {target_col}", color=INK,
                 fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.set_xlim(0, max(pcts) * 1.2)
    ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=10, edgecolor=BORDER)

    plt.tight_layout(pad=1.4)

    return jsonify({
        "img": fig_b64(fig),
        "classes": classes,
        "counts": counts,
        "pcts": pcts,
        "total": total,
        "ratio": ratio,
        "ideal_pct": ideal,
        "status": status,
        "status_label": status_label,
        "n_classes": n_cls,
    })

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: MODELOS — Linear Regression block + Model Evaluation block
# Endpoints:
#   POST /api/model_train       — train a model, stream SSE progress
#   GET  /api/model_train_poll  — polled progress (for non-SSE clients)
#   POST /api/model_evaluate    — full evaluation of last trained model
#   GET  /api/model_result      — quick summary of last trained model
# ══════════════════════════════════════════════════════════════════════════════

# ── helpers ───────────────────────────────────────────────────────────────────

def _vif_numpy(X, feat_names):
    """Compute VIF for each column of X using numpy OLS (no statsmodels needed).
    Returns list of {feature, vif} dicts. VIF = 1/(1-R²_j) where R²_j is the
    R² from regressing column j on all other columns.
    """
    results = []
    n, p = X.shape
    for j in range(p):
        y_j = X[:, j]
        X_other = np.delete(X, j, axis=1)
        # Add intercept column
        Xb = np.column_stack([np.ones(n), X_other])
        try:
            # OLS via normal equations: beta = (XtX)^-1 Xt y
            beta, _, _, _ = np.linalg.lstsq(Xb, y_j, rcond=None)
            y_hat = Xb @ beta
            ss_res = float(np.sum((y_j - y_hat) ** 2))
            ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else 999.0
        except Exception:
            vif = float("nan")
        results.append({"feature": feat_names[j], "vif": round(float(vif), 3)})
    return results


def _durbin_watson(residuals):
    """Durbin-Watson statistic: d = sum((e_t - e_{t-1})^2) / sum(e_t^2).
    Values near 2 → no autocorrelation; <1 or >3 → concern.
    """
    e = np.array(residuals)
    diff = np.diff(e)
    dw = float(np.sum(diff ** 2) / np.sum(e ** 2)) if np.sum(e ** 2) > 1e-12 else 2.0
    return round(dw, 4)


def _breusch_pagan_numpy(residuals, X_fitted):
    """Simplified Breusch-Pagan test using OLS of squared residuals on fitted values.
    Returns (bp_stat, interpretation_string). Not a formal p-value but indicative.
    """
    e2 = residuals ** 2
    Xb = np.column_stack([np.ones(len(X_fitted)), X_fitted])
    try:
        beta, _, _, _ = np.linalg.lstsq(Xb, e2, rcond=None)
        e2_hat = Xb @ beta
        ss_res = float(np.sum((e2 - e2_hat) ** 2))
        ss_tot = float(np.sum((e2 - np.mean(e2)) ** 2))
        r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        bp = float(len(residuals) * r2_aux)
    except Exception:
        bp = float("nan")
    return round(bp, 4)


def _qq_plot_b64(residuals):
    """Generate a Q-Q plot of residuals vs theoretical normal quantiles."""
    n = len(residuals)
    sorted_r = np.sort(residuals)
    # Theoretical quantiles using Blom formula
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    # Approx normal quantile via rational approximation (Abramowitz & Stegun)
    def _norm_ppf(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        sign = np.where(p < 0.5, -1.0, 1.0)
        p2 = np.where(p < 0.5, p, 1.0 - p)
        t = np.sqrt(-2.0 * np.log(p2))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        num = c0 + c1 * t + c2 * t ** 2
        den = 1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3
        return sign * (t - num / den)
    theoretical = _norm_ppf(probs)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(LIGHT); style_ax(ax)
    ax.scatter(theoretical, sorted_r, color=PALETTE[0], alpha=0.7, s=24, edgecolors="none", zorder=3)
    # Reference line
    mn, mx = float(theoretical.min()), float(theoretical.max())
    std_r = float(np.std(sorted_r)) if np.std(sorted_r) > 1e-12 else 1.0
    mean_r = float(np.mean(sorted_r))
    ax.plot([mn, mx], [mean_r + mn * std_r, mean_r + mx * std_r],
            color=MINT, linewidth=1.8, linestyle="--", label=PL()["qq_x"])
    ax.set_xlabel(PL()["qq_x"], color=SEC, fontsize=10)
    ax.set_ylabel(PL()["qq_y"], color=SEC, fontsize=10)
    ax.set_title(PL()["qq_title"], color=INK, fontsize=12, fontweight="bold")
    ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=9)
    ax.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    plt.tight_layout(pad=1.4)
    return fig_b64(fig)


def _residuals_plot_b64(y_pred, residuals):
    """Residuals vs fitted values plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(LIGHT); style_ax(ax)
    ax.axhline(0, color=MINT, linewidth=1.6, linestyle="--")
    ax.scatter(y_pred, residuals, color=PALETTE[1], alpha=0.65, s=26, edgecolors="none", zorder=3)
    ax.set_xlabel(PL()["fitted"], color=SEC, fontsize=10)
    ax.set_ylabel(PL()["residuals"], color=SEC, fontsize=10)
    ax.set_title(PL()["residuals_vs"], color=INK, fontsize=12, fontweight="bold")
    ax.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    plt.tight_layout(pad=1.4)
    return fig_b64(fig)


def _pred_vs_actual_b64(y_true, y_pred, r2):
    """Predicted vs actual scatter."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(LIGHT); style_ax(ax)
    mn_v = min(float(y_true.min()), float(y_pred.min()))
    mx_v = max(float(y_true.max()), float(y_pred.max()))
    ax.scatter(y_true, y_pred, color=PALETTE[0], alpha=0.65, s=26, edgecolors="none", zorder=3)
    ax.plot([mn_v, mx_v], [mn_v, mx_v], color=MINT, linewidth=1.8, linestyle="--", label="Ideal")
    ax.set_xlabel(PL()["actual"], color=SEC, fontsize=10)
    ax.set_ylabel(PL()["predicted"], color=SEC, fontsize=10)
    ax.set_title(f"{PL()['pred_actual']}  (R²={round(r2,4)})", color=INK, fontsize=12, fontweight="bold")
    ax.legend(facecolor=LIGHT, labelcolor=INK, fontsize=9)
    ax.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    plt.tight_layout(pad=1.4)
    return fig_b64(fig)


def _coef_bar_b64(coefs_sorted, title="Coeficientes"):
    """Horizontal bar chart of coefficients."""
    names_c = [c[0] for c in coefs_sorted]
    vals_c  = [c[1] for c in coefs_sorted]
    colors_c = [PALETTE[0] if v >= 0 else PALETTE[1] for v in vals_c]
    fig2, ax3 = plt.subplots(figsize=(7, max(3, len(coefs_sorted) * 0.38 + 1.2)))
    fig2.patch.set_facecolor(LIGHT); style_ax(ax3)
    ax3.barh(names_c, vals_c, color=colors_c, edgecolor="none", alpha=0.85)
    ax3.axvline(0, color=SEC, linewidth=0.8)
    ax3.set_xlabel(PL()["coefficient"], color=SEC, fontsize=10)
    ax3.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax3.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
    plt.tight_layout(pad=1.4)
    return fig_b64(fig2)


# ── /api/model_train ──────────────────────────────────────────────────────────

_mtrain_progress: list = []   # [{pct, msg}]
_mtrain_lock = threading.Lock()
_mtrain_result: dict = {}     # last train result stored here

@app.route("/api/model_train", methods=["POST"])
def model_train():
    """Train a model with polled progress. Stores result in _MODEL_STORE[node_id]."""
    body        = request.get_json(force=True, silent=True) or {}
    node_id       = str(body.get("node_id", "linreg_default"))
    model_type    = str(body.get("model_type", "linreg"))   # linreg | ridge | lasso
    alpha         = float(body.get("alpha", 1.0))
    fit_intercept    = bool(body.get("fit_intercept", True))
    fixed_intercept  = body.get("fixed_intercept", None)     # float or None
    if fixed_intercept is not None:
        try: fixed_intercept = float(fixed_intercept)
        except: fixed_intercept = None
    normalize     = str(body.get("normalize", "none"))       # none | minmax | zscore
    test_size     = float(body.get("test_size", 0.3))
    seed          = int(body.get("seed", 42))
    upstream_id   = str(body.get("upstream_id", "")) or None
    target_col    = str(body.get("target_col", ""))

    # Resolve data — do this BEFORE the thread so we capture a snapshot
    D = _effective_data(upstream_id) if upstream_id else None
    snap_rows = list((D["raw_rows"] if D else None) or S["raw_rows"])
    snap_cols = list((D["columns"]  if D else None) or S["columns"])

    # Resolve target: explicit arg > slot target > _TAB > last column
    if not target_col:
        target_col = str(
            (D["target"] if D else None)
            or _TAB.get("target_col", "")
            or (snap_cols[-1] if snap_cols else "")
        )

    # SAFETY: if target_col is still not in snap_cols, use the last column
    if target_col and snap_cols and target_col not in snap_cols:
        target_col = snap_cols[-1]

    # Detect problem type — method takes priority over data heuristics
    _tgt_vals = [r.get(target_col, "") for r in snap_rows if not _is_missing(r.get(target_col, ""))]
    _tgt_type = _col_type(_tgt_vals)
    _reg_methods = {"ridge", "lasso", "linreg"}
    if model_type in _reg_methods:
        # Explicit regression model → always regression
        _is_classification = False
    else:
        # Auto-detect: categorical dtype or very few unique numeric values (≤10)
        _is_classification = not _is_numeric_type(_tgt_type)
        if _is_numeric_type(_tgt_type):
            _unique_tgt = set(str(v).strip() for v in _tgt_vals)
            if len(_unique_tgt) <= 10:
                _is_classification = True

    # Snapshot all params for the thread (avoid closure over mutable request state)
    _nid   = node_id
    _mtype = model_type
    _alpha = alpha
    _fi    = fit_intercept
    _fxint = fixed_intercept   # float or None
    _norm  = normalize
    _tsz   = test_size
    _seed  = seed
    _tgt   = target_col
    _is_cls = _is_classification

    def _run():
        global _mtrain_result
        def _push(pct, msg):
            with _mtrain_lock:
                _mtrain_progress.append({"pct": pct, "msg": msg})

        try:
            prob_label = "clasificación" if _is_cls else "regresión"
            _push(5, f"Preparando datos… {len(snap_rows)} filas · problema detectado: {prob_label}")
            time.sleep(0.08)

            if not snap_rows:
                _push(100, "__error__:No hay datos cargados. Carga un dataset en el bloque Datos primero.")
                return
            if not _tgt:
                _push(100, "__error__:No se ha definido variable objetivo. Defínela en el bloque Datos.")
                return
            if _tgt not in snap_cols:
                _push(100, f"__error__:Columna objetivo '{_tgt}' no encontrada en las columnas disponibles.")
                return

            _push(20, f"Construyendo matriz de features… ({len(snap_rows)} filas, {len(snap_cols)} cols, target='{_tgt}')")
            time.sleep(0.05)
            result_xy = _build_Xy(snap_rows, snap_cols, _tgt, _norm)
            if result_xy[0] is None:
                _push(100, f"__error__:{result_xy[1]}")
                return
            X, y, feat_names = result_xy
            if len(X) < 10:
                _push(100, "__error__:Necesitas al menos 10 filas válidas para entrenar.")
                return

            # ── CLASIFICACIÓN ────────────────────────────────────────────────
            if _is_cls:
                # Encode labels
                def _tstr(v):
                    try:
                        f = float(v); return str(int(f)) if f == int(f) else str(f)
                    except: return str(v).strip()
                classes = sorted(set(_tstr(v) for v in y))
                label_map = {c: i for i, c in enumerate(classes)}
                y_enc = np.array([label_map[_tstr(v)] for v in y])

                _push(38, f"Dividiendo… {X.shape[0]} filas × {X.shape[1]} features · {len(classes)} clases · test={_tsz}")
                time.sleep(0.05)
                Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=_tsz, random_state=_seed, stratify=y_enc if len(classes)<=20 else None)

                _push(52, f"Entrenando modelo de clasificación ({_mtype})…")
                time.sleep(0.05)
                # Map linreg/ridge/lasso → classification equivalents
                if _mtype in ("ridge", "lasso", "linreg"):
                    mdl = LogisticRegression(C=1.0/max(_alpha,1e-6), max_iter=2000, random_state=42, solver="saga")
                else:
                    mdl = LogisticRegression(max_iter=2000, random_state=42)
                mdl.fit(Xtr, ytr)

                _push(75, "Calculando métricas de clasificación…")
                from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                             recall_score, confusion_matrix, classification_report)
                ytr_pred = mdl.predict(Xtr)
                yte_pred = mdl.predict(Xte)
                avg = "binary" if len(classes)==2 else "macro"
                def _cls_metrics(yt, yp):
                    return {
                        "accuracy":  round(float(accuracy_score(yt, yp)), 4),
                        "f1":        round(float(f1_score(yt, yp, average=avg, zero_division=0)), 4),
                        "precision": round(float(precision_score(yt, yp, average=avg, zero_division=0)), 4),
                        "recall":    round(float(recall_score(yt, yp, average=avg, zero_division=0)), 4),
                    }
                train_m = _cls_metrics(ytr, ytr_pred)
                test_m  = _cls_metrics(yte, yte_pred)
                cm = confusion_matrix(yte, yte_pred, labels=list(range(len(classes)))).tolist()

                # Coefs (only for logistic, 2-class → single coef vector)
                if hasattr(mdl, "coef_"):
                    coef_arr = mdl.coef_[0] if mdl.coef_.shape[0]==1 else np.mean(np.abs(mdl.coef_), axis=0)
                    coefs_sorted = sorted(zip(feat_names, [round(float(c),6) for c in coef_arr]),
                                          key=lambda x: abs(x[1]), reverse=True)
                else:
                    coefs_sorted = []

                _push(88, "Generando gráfica de coeficientes…")
                coef_img = _coef_bar_b64(coefs_sorted, title=f"Importancia de features — {_mtype.upper()}")

                res = {
                    "node_id":      _nid,
                    "model_type":   _mtype,
                    "problem_type": "classification",
                    "alpha":        _alpha,
                    "normalize":    _norm,
                    "n_train":      int(len(Xtr)),
                    "n_test":       int(len(Xte)),
                    "target_col":   _tgt,
                    "features":     feat_names,
                    "classes":      classes,
                    "coefficients": coefs_sorted,
                    "train_metrics": train_m,
                    "test_metrics":  test_m,
                    "confusion_matrix": cm,
                    "coef_img":     coef_img,
                    "_Xte":  Xte.tolist(), "_yte":  yte.tolist(), "_yte_pred": yte_pred.tolist(),
                    "_Xtr":  Xtr.tolist(), "_ytr":  ytr.tolist(),
                }

            # ── REGRESIÓN ────────────────────────────────────────────────────
            else:
                try:
                    y_float = y.astype(float)
                except Exception:
                    _push(100, "__error__:La variable objetivo no es numérica. Revisa el dataset.")
                    return

                _push(38, f"Dividiendo… {X.shape[0]} filas × {X.shape[1]} features · test={_tsz}")
                time.sleep(0.05)
                Xtr, Xte, ytr, yte = train_test_split(X, y_float, test_size=_tsz, random_state=_seed)

                _push(52, "Entrenando modelo de regresión…")
                time.sleep(0.05)
                ytr_fit = ytr - _fxint if _fxint is not None else ytr
                use_fi  = _fi if _fxint is None else False
                if _mtype == "ridge":
                    mdl = Ridge(alpha=_alpha, fit_intercept=use_fi)
                elif _mtype == "lasso":
                    mdl = Lasso(alpha=_alpha, max_iter=10000, fit_intercept=use_fi)
                else:
                    mdl = LinearRegression(fit_intercept=use_fi)
                mdl.fit(Xtr, ytr_fit)

                _push(75, "Calculando métricas…")
                ytr_pred = mdl.predict(Xtr) + (_fxint or 0)
                yte_pred = mdl.predict(Xte) + (_fxint or 0)
                train_m  = _reg_metrics(ytr, ytr_pred)
                test_m   = _reg_metrics(yte, yte_pred)

                coefs_sorted = sorted(zip(feat_names, [round(float(c),6) for c in mdl.coef_]),
                                      key=lambda x: abs(x[1]), reverse=True)

                _push(88, "Generando gráfica de coeficientes…")
                title_str = f"{_mtype.upper()} · λ={_alpha}" if _mtype != "linreg" else "OLS"
                coef_img  = _coef_bar_b64(coefs_sorted, title=f"Coeficientes — {title_str}")

                res = {
                    "node_id":      _nid,
                    "model_type":   _mtype,
                    "problem_type": "regression",
                    "alpha":        _alpha,
                    "normalize":    _norm,
                    "n_train":      int(len(Xtr)),
                    "n_test":       int(len(Xte)),
                    "target_col":   _tgt,
                    "features":     feat_names,
                    "coefficients": list(coefs_sorted),
                    "fit_intercept":   _fi,
                    "fixed_intercept": _fxint,
                    "intercept": (
                        round(_fxint, 6) if _fxint is not None
                        else (round(float(mdl.intercept_), 6) if use_fi else 0.0)
                    ),
                    "train_metrics": train_m,
                    "test_metrics":  test_m,
                    "coef_img":      coef_img,
                    "_Xte":  Xte.tolist(), "_yte":  yte.tolist(), "_yte_pred": yte_pred.tolist(),
                    "_Xtr":  Xtr.tolist(), "_ytr":  ytr.tolist(),
                }

            _MODEL_STORE[_nid] = res
            _NODE_MODELS[_nid] = mdl   # persist sklearn object for save_tab_model
            _mtrain_result     = res
            _push(100, "__done__")

        except Exception as exc:
            import traceback
            _push(100, f"__error__:{exc} | {traceback.format_exc().splitlines()[-1]}")

    with _mtrain_lock:
        _mtrain_progress.clear()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/model_train_poll")
def model_train_poll():
    """Return pending progress events since last call."""
    with _mtrain_lock:
        events = list(_mtrain_progress)
        _mtrain_progress.clear()
    return jsonify({"events": events})


@app.route("/api/model_result")
def model_result():
    """Quick summary of last trained model for the model block."""
    # Keys that hold sklearn objects or heavy arrays — never JSON-serialize these
    _SKIP_KEYS = {"scaler", "model_obj", "sklearn_model"}

    node_id = request.args.get("node_id", "")
    result = _MODEL_STORE.get(node_id) or _mtrain_result
    if not result:
        return jsonify({"error": "No hay modelo entrenado aún"}), 404

    # Filter: drop private arrays (start with _) and known sklearn objects
    def _is_safe(k, v):
        if k.startswith("_") or k in _SKIP_KEYS:
            return False
        try:
            json.dumps(v)
            return True
        except (TypeError, ValueError):
            return False

    safe = {k: v for k, v in result.items() if _is_safe(k, v)}

    # Normalise field names: CV uses "method"/"feat_names"/"X_te"/"y_te"/"y_te_pred"
    # Train-direct uses "model_type"/"features"/"_Xte"/"_yte"/"_yte_pred"
    # Expose both so downstream blocks work regardless of origin.
    if "model_type" not in safe and "method" in safe:
        safe["model_type"] = safe["method"]
    if "features" not in safe and "feat_names" in safe:
        safe["features"] = safe["feat_names"]
    if "target_col" not in safe:
        safe["target_col"] = result.get("target_col") or _TAB.get("target_col", "")
    if "n_train" not in safe:
        safe["n_train"] = result.get("n_train")
    if "n_test" not in safe:
        safe["n_test"] = result.get("n_test")

    # Move sklearn objects from _MODEL_STORE into _NODE_MODELS for save_tab_model
    for sk_key in _SKIP_KEYS:
        obj = result.get(sk_key)
        if obj is not None and node_id:
            _NODE_MODELS[node_id] = obj
            break

    return jsonify(safe)


# ── /api/model_evaluate ───────────────────────────────────────────────────────

@app.route("/api/model_evaluate", methods=["POST"])
def model_evaluate():
    """Full evaluation of the model stored for a given node_id.
    For regression: metrics, coef chart, predicted vs actual, residuals,
    Q-Q plot, VIF, Durbin-Watson, Breusch-Pagan.
    """
    body    = request.get_json(force=True, silent=True) or {}
    node_id = str(body.get("node_id", ""))

    result = _MODEL_STORE.get(node_id)
    if not result:
        # Try last trained
        result = _mtrain_result
    if not result:
        return jsonify({"error": "No hay modelo entrenado. Entrena primero desde el bloque Modelo."}), 404

    problem_type = result.get("problem_type", "regression")

    # Normalise keys: CV uses X_te/y_te/y_te_pred/feat_names, train-direct uses _Xte/_yte/_yte_pred/features
    _Xte       = result.get("_Xte")       or result.get("X_te")
    _yte       = result.get("_yte")       or result.get("y_te")
    _yte_pred  = result.get("_yte_pred")  or result.get("y_te_pred")
    _feat_names = result.get("features")  or result.get("feat_names", [])

    if problem_type == "regression":
        if _Xte is None or _yte is None or _yte_pred is None:
            return jsonify({"error": "No hay datos de test disponibles. Reentrena el modelo."}), 400
        Xte      = np.array(_Xte)
        yte      = np.array(_yte)
        yte_pred = np.array(_yte_pred)
        residuals = yte - yte_pred
        feat_names = _feat_names

        # ── Plots ──────────────────────────────────────────────────────────
        _tm        = result.get("test_metrics") or {}
        pred_img   = _pred_vs_actual_b64(yte, yte_pred, _tm.get("R2") or _tm.get("r2", 0))
        resid_img  = _residuals_plot_b64(yte_pred, residuals)
        qq_img     = _qq_plot_b64(residuals)
        coef_img   = result.get("coef_img")

        # ── Statistical tests ──────────────────────────────────────────────
        dw = _durbin_watson(residuals)
        if dw < 1.5:
            dw_interp = "⚠️ Posible autocorrelación positiva"
        elif dw > 2.5:
            dw_interp = "⚠️ Posible autocorrelación negativa"
        else:
            dw_interp = "✅ Sin autocorrelación aparente"

        bp = _breusch_pagan_numpy(residuals, yte_pred)
        n  = len(residuals)
        # Rough threshold: BP > chi2(1, 0.05) ≈ 3.84
        bp_interp = "⚠️ Posible heterocedasticidad" if bp > 3.84 else "✅ Homocedasticidad razonable"

        # ── VIF (only if enough features and rows) ─────────────────────────
        vif_data = []
        if Xte.shape[1] >= 2 and Xte.shape[0] > Xte.shape[1] + 2:
            try:
                vif_data = _vif_numpy(Xte, feat_names)
            except Exception:
                vif_data = []

        _model_type = result.get("model_type") or result.get("method", "—")
        # CV summary — include if model was trained via grid search
        _cv_summary = None
        if result.get("from_cv") or result.get("cv_img"):
            _cv_best = result.get("best") or {"param_val": result.get("alpha"), "score_mean": "—"}
            _cv_pv_display = result.get("best_pv_display")
            # Reconstruct display string if missing (old .pkl)
            if not _cv_pv_display:
                _pv = _cv_best.get("param_val") if isinstance(_cv_best, dict) else None
                if _pv is None:
                    _pv = result.get("alpha")
                if _pv is not None:
                    try:
                        _pv = float(_pv)
                        if _pv == int(_pv) and _pv >= 1:
                            _cv_pv_display = str(int(_pv))
                        elif _pv < 0.01:
                            _cv_pv_display = f"{_pv:.2e}"
                        else:
                            _cv_pv_display = str(round(_pv, 4))
                    except (TypeError, ValueError):
                        _cv_pv_display = str(_pv)
            _cv_summary = {
                "cv_img":         result.get("cv_img"),
                "best":           _cv_best,
                "best_pv_display":_cv_pv_display,
                "param_label":    result.get("param_label") or "λ",
                "metric_label":   result.get("metric_label") or "RMSE",
                "k_folds":        result.get("k_folds") or "—",
                "method":         _model_type,
                "problem_type":   result.get("problem_type", "regression"),
            }
        # Cache plots in _MODEL_STORE BEFORE returning so export_eval_zip finds them
        _MODEL_STORE.setdefault(node_id, {}).update({
            "pred_img": pred_img, "resid_img": resid_img,
            "qq_img": qq_img, "coef_img": coef_img,
        })
        cached = [k for k in ("pred_img","resid_img","qq_img","coef_img") if _MODEL_STORE[node_id].get(k)]
        print(f"[model_evaluate] plots cacheados en _MODEL_STORE[{node_id!r}]: {cached}")
        return jsonify({
            "problem_type":  "regression",
            "model_type":    _model_type,
            "normalize":     result.get("normalize"),
            "n_train":       result.get("n_train"),
            "n_test":        result.get("n_test"),
            "target_col":    result.get("target_col") or _TAB.get("target_col", ""),
            "features":      feat_names,
            "coefficients":  result.get("coefficients"),
            "intercept":     result.get("intercept"),
            "train_metrics": result.get("train_metrics"),
            "test_metrics":  result.get("test_metrics"),
            # plots
            "pred_img":  pred_img,
            "resid_img": resid_img,
            "qq_img":    qq_img,
            "coef_img":  coef_img,
            # tests
            "durbin_watson": dw,
            "dw_interp": dw_interp,
            "breusch_pagan": bp,
            "bp_interp": bp_interp,
            "vif": vif_data,
            "cv_summary": _cv_summary,
        })

    # ── CLASIFICACIÓN ────────────────────────────────────────────────────────
    if problem_type == "classification":
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, confusion_matrix, ConfusionMatrixDisplay)
        if _yte is None or _yte_pred is None:
            return jsonify({"error": "No hay datos de test disponibles. Reentrena el modelo."}), 400
        yte      = np.array(_yte)
        yte_pred = np.array(_yte_pred)
        classes  = result.get("classes", [])
        avg = "binary" if len(classes) == 2 else "macro"

        cm = confusion_matrix(yte, yte_pred, labels=list(range(len(classes)))).tolist()

        fig, ax = plt.subplots(figsize=(max(4, len(classes)*1.2), max(3.5, len(classes)*1.0)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        cm_arr = np.array(cm)
        ax.imshow(cm_arr, cmap="Greens")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels([PL()["pred_lbl"]+": "+c for c in classes], color=SEC, fontsize=9)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels([PL()["actual_lbl"]+": "+c for c in classes], color=SEC, fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm_arr[i,j]), ha="center", va="center",
                        color=INK if cm_arr[i,j] < cm_arr.max()*0.6 else "white", fontsize=11, fontweight="bold")
        ax.set_title(PL()["cm_title"] + " (test)", color=INK, fontsize=12, fontweight="bold")
        plt.tight_layout()
        cm_img = fig_b64(fig)

        _model_type = result.get("model_type") or result.get("method", "—")
        _cv_summary_cls = None
        if result.get("from_cv") or result.get("cv_img"):
            _cv_best_cls = result.get("best") or {"param_val": result.get("alpha"), "score_mean": "—"}
            _cv_pv_display_cls = result.get("best_pv_display")
            if not _cv_pv_display_cls:
                _pv_cls = _cv_best_cls.get("param_val") if isinstance(_cv_best_cls, dict) else None
                if _pv_cls is None:
                    _pv_cls = result.get("alpha")
                if _pv_cls is not None:
                    try:
                        _pv_cls = float(_pv_cls)
                        if _pv_cls == int(_pv_cls) and _pv_cls >= 1:
                            _cv_pv_display_cls = str(int(_pv_cls))
                        elif _pv_cls < 0.01:
                            _cv_pv_display_cls = f"{_pv_cls:.2e}"
                        else:
                            _cv_pv_display_cls = str(round(_pv_cls, 4))
                    except (TypeError, ValueError):
                        _cv_pv_display_cls = str(_pv_cls)
            _cv_summary_cls = {
                "cv_img":         result.get("cv_img"),
                "best":           _cv_best_cls,
                "best_pv_display":_cv_pv_display_cls,
                "param_label":    result.get("param_label") or "C",
                "metric_label":   result.get("metric_label") or "F1",
                "k_folds":        result.get("k_folds") or "—",
                "method":         _model_type,
                "problem_type":   "classification",
            }
        # ROC curve
        roc_img_cls = None
        live_mdl = _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
        if live_mdl is not None and hasattr(live_mdl, "predict_proba") and _Xte is not None:
            try:
                y_score_cls = live_mdl.predict_proba(np.array(_Xte))
                n_cls = len(classes)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                fig_roc.patch.set_facecolor(LIGHT); style_ax(ax_roc)
                if n_cls == 2:
                    fpr, tpr, _ = roc_curve(yte, y_score_cls[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax_roc.plot(fpr, tpr, color=PALETTE[0], linewidth=2, label=f"AUC = {roc_auc:.3f}")
                else:
                    y_bin = label_binarize(yte, classes=list(range(n_cls)))
                    for i, cls_name in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score_cls[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax_roc.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=1.8,
                                    label=f"{cls_name} (AUC={roc_auc:.3f})")
                ax_roc.plot([0,1],[0,1], color=SEC, linewidth=1, linestyle="--", label=PL()["chance"])
                ax_roc.set_xlabel(PL()["fpr"], color=SEC, fontsize=11)
                ax_roc.set_ylabel(PL()["tpr"], color=SEC, fontsize=11)
                ax_roc.set_title(PL()["roc_title"], color=INK, fontsize=12, fontweight="bold")
                ax_roc.legend(facecolor=BG, labelcolor=INK, fontsize=9, loc="lower right")
                ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.02])
                ax_roc.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5)
                plt.tight_layout(pad=1.4)
                roc_img_cls = fig_b64(fig_roc)
            except Exception as e:
                print(f"[model_evaluate] ROC error: {e}")

        # Cache plots in _MODEL_STORE BEFORE returning so export_eval_zip finds them
        _MODEL_STORE.setdefault(node_id, {}).update({
            "cm_img":   cm_img,
            "roc_img":  roc_img_cls,
            "coef_img": result.get("coef_img"),
        })
        cached_cls = [k for k in ("cm_img","roc_img","coef_img") if _MODEL_STORE[node_id].get(k)]
        print(f"[model_evaluate] plots cacheados en _MODEL_STORE[{node_id!r}]: {cached_cls}")
        return jsonify({
            "problem_type":    "classification",
            "model_type":      _model_type,
            "normalize":       result.get("normalize"),
            "n_train":         result.get("n_train"),
            "n_test":          result.get("n_test"),
            "target_col":      result.get("target_col") or _TAB.get("target_col", ""),
            "features":        _feat_names,
            "classes":         classes,
            "coefficients":    result.get("coefficients"),
            "train_metrics":   result.get("train_metrics"),
            "test_metrics":    result.get("test_metrics"),
            "confusion_matrix": cm,
            "cm_img":          cm_img,
            "roc_img":         roc_img_cls,
            "coef_img":        result.get("coef_img"),
            "cv_summary":      _cv_summary_cls,
        })

    return jsonify({"error": f"Tipo de problema '{problem_type}' no reconocido"}), 400
