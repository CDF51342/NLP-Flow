"""NLP Flow 4 — Flask API with SSE progress streaming + session persistence"""
import threading, queue as _queue_mod  # needed by module-level locks below (full imports follow later)

# ── Native dialog queue (must run on the main thread / Cocoa thread) ──────────
# Flask threads post requests here; main.py drains them via a pywebview hook.
_DIALOG_REQ: "_queue_mod.Queue" = _queue_mod.Queue()   # (req_id, kwargs) → main thread
_DIALOG_RES: "_queue_mod.Queue" = _queue_mod.Queue()   # (req_id, path|None) ← main thread
_DIALOG_LOCK = threading.Lock()   # serialise concurrent callers
_DIALOG_CTR  = 0
# ── MODEL STORE: trained model objects keyed by node_id ──────────────────────
_MODEL_STORE: dict = {}    # { node_id: result_dict }
_NODE_MODELS: dict = {}    # { node_id: sklearn model object }

# ── CV background thread state ────────────────────────────────────────────────
_cv_thread   = None
_cv_progress: list = []       # list of {"pct", "msg"} events
_cv_result:   dict = {}       # final result keyed by node_id
_cv_lock   = threading.Lock()
_cv_cancel = threading.Event()  # set() to request cancellation

def _cv_push(pct, msg):
    with _cv_lock:
        _cv_progress.append({"pct": pct, "msg": msg})

def _cv_drain():
    with _cv_lock:
        out = list(_cv_progress)
        _cv_progress.clear()
    return out

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
import re, io, base64, csv, json, time, threading, zipfile, os, pickle, tempfile, shutil, sys

# Load .env if present (HF_TOKEN, HF_LLM_MODEL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually
from collections import Counter

import pandas as pd

# Raise CSV field-size limit as fallback for any legacy code path still using csv.DictReader
csv.field_size_limit(min(2147483647, sys.maxsize))

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
    from nltk.stem import PorterStemmer, SnowballStemmer
    try:
        STOP_EN = set(stopwords.words("english"))
        STOP_ES = set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_EN = set(stopwords.words("english"))
        STOP_ES = set(stopwords.words("spanish"))
    STEMMER    = PorterStemmer()
    STEMMER_EN = SnowballStemmer("english")
    STEMMER_ES = SnowballStemmer("spanish")
    NLTK_OK = True
except Exception:
    STOP_EN, STOP_ES, STEMMER, NLTK_OK = set(), set(), None, False
    STEMMER_EN = STEMMER_ES = None

# ── Lemmatizer (spaCy if available, else nltk WordNetLemmatizer) ───────────
# Supports EN and ES. Language is auto-detected at call time.
LEMMATIZER = None
_LEMMATIZE_FN = None
_nlp_spacy_en = None
_nlp_spacy_es = None

def _spacy_load_model(name):
    import spacy as _spacy
    try:
        return _spacy.load(name, disable=["parser", "ner"])
    except OSError:
        from spacy.cli import download as _dl
        _dl(name)
        return _spacy.load(name, disable=["parser", "ner"])

try:
    import spacy as _spacy
    _nlp_spacy_en = _spacy_load_model("en_core_web_sm")
    print("[LEMMA] spaCy en_core_web_sm OK", flush=True)
    try:
        _nlp_spacy_es = _spacy_load_model("es_core_news_sm")
        print("[LEMMA] spaCy es_core_news_sm OK", flush=True)
    except Exception as _ese:
        print(f"[LEMMA] es_core_news_sm no disponible: {_ese}", flush=True)

    def _lemmatize_spacy(text):
        # Detect language by Spanish stopword overlap
        words = text.split()
        es_hits = sum(1 for w in words if w.lower() in STOP_ES)
        if _nlp_spacy_es and es_hits >= max(1, len(words) * 0.05):
            return " ".join(tok.lemma_ for tok in _nlp_spacy_es(text))
        return " ".join(tok.lemma_ for tok in _nlp_spacy_en(text))

    _LEMMATIZE_FN = _lemmatize_spacy
except Exception:
    try:
        from nltk.stem import WordNetLemmatizer as _WNL
        import nltk as _nltk2
        try:
            from nltk.corpus import wordnet as _wn; _wn.synsets("test")
        except LookupError:
            _nltk2.download("wordnet", quiet=True)
            _nltk2.download("omw-1.4", quiet=True)
        _wn_lemmatizer = _WNL()
        def _lemmatize_nltk(text):
            return " ".join(_wn_lemmatizer.lemmatize(w) for w in text.split())
        _LEMMATIZE_FN = _lemmatize_nltk
        print("[LEMMA] usando nltk WordNetLemmatizer", flush=True)
    except Exception as _le:
        print(f"[LEMMA] no disponible: {_le}", flush=True)

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

# ── Image dataset store keyed by data-node id ────────────────────────────────
# Each entry: { dataset: "mnist"|"dogs_muffins", class_names: [...], n_train: int,
#               n_test: int, n_classes: int, img_size: (H,W), loaded: bool,
#               train_data: [(pil_img, label), ...], test_data: [...] }
_IMAGE_NODE_DATA: dict = {}

def _image_slot(node_id: str) -> dict:
    if node_id not in _IMAGE_NODE_DATA:
        _IMAGE_NODE_DATA[node_id] = {"loaded": False, "dataset": None,
                                      "class_names": [], "n_train": 0,
                                      "n_test": 0, "n_classes": 0,
                                      "img_size": (0, 0),
                                      "train_data": [], "test_data": []}
    return _IMAGE_NODE_DATA[node_id]

# ── Pandas DataFrame store (new layer — parallel to _NODE_DATA) ───────────────
# Keyed by node_id_str. New endpoints read from here; legacy endpoints keep
# using raw_rows for now. Both are kept in sync on every upload/load.
_DF_STORE: dict = {}    # { node_id_str: pd.DataFrame }
_DF_GLOBAL: list = [None]  # single-element list so it's mutable from closures; [0] = global DataFrame

def _store_df(node_id: str | None, df: "pd.DataFrame") -> None:
    """Persist a DataFrame for a node (and always update the global slot)."""
    _DF_GLOBAL[0] = df
    if node_id:
        _DF_STORE[node_id] = df

def _get_df(node_id: str | None) -> "pd.DataFrame | None":
    """Return the DataFrame for a node, falling back to the global one."""
    if node_id and node_id in _DF_STORE:
        return _DF_STORE[node_id]
    return _DF_GLOBAL[0]

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

# ── Built-in datasets removed ────────────────────────────────────────────────
DATASETS = {}  # kept for compatibility; no built-in datasets
# ── Tabular datasets registry ─────────────────────────────────────────────────
TAB_DATASETS = {}  # no built-in datasets; users upload their own CSV

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
def _remove_punctuation(t):
    """Remove punctuation but preserve accented letters, ñ, ü and other Unicode word chars."""
    import unicodedata
    # Keep: Unicode letters (including á é í ó ú ñ ü), digits, whitespace
    return re.sub(r'[^\w\s]', '', t, flags=re.UNICODE).strip()

def _detect_corpus_lang(texts: list, sample: int = 200, threshold: float = 0.08) -> str:
    """Detect whether a corpus is predominantly Spanish or English.

    Samples up to `sample` documents, counts Spanish stopword hits per token,
    and returns 'es' if the hit-rate exceeds `threshold`, else 'en'.
    """
    sample_texts = texts[:sample]
    all_words = []
    for t in sample_texts:
        all_words.extend(t.lower().split())
    if not all_words:
        return "es"   # default
    es_hits = sum(1 for w in all_words if w in STOP_ES)
    rate = es_hits / len(all_words)
    detected = "es" if rate >= threshold else "en"
    print(f"[LANG] es_hit_rate={rate:.3f} → detected='{detected}'", flush=True)
    return detected


def _smart_stem(t):
    """Stem with language-aware Snowball stemmer.
    Detects Spanish by checking overlap with Spanish stopwords; falls back to English."""
    if not STEMMER_EN:
        return t
    words = t.split()
    es_hits = sum(1 for w in words if w.lower() in STOP_ES)
    stemmer = STEMMER_ES if (STEMMER_ES and es_hits >= max(1, len(words) * 0.05)) else STEMMER_EN
    return ' '.join(stemmer.stem(w) for w in words)

STEPS = {
    "lowercase":      lambda t: t.lower(),
    "punctuation":    _remove_punctuation,
    "numbers":        lambda t: re.sub(r'\d+', '', t),
    "stopwords_en":   lambda t: ' '.join(w for w in t.split() if w.lower() not in STOP_EN),
    "stopwords_es":   lambda t: ' '.join(w for w in t.split() if w.lower() not in STOP_ES),
    "stemming":       _smart_stem,
    "lemmatization":  lambda t: _LEMMATIZE_FN(t) if _LEMMATIZE_FN else t,
    "whitespace":     lambda t: re.sub(r'\s+', ' ', t).strip(),
}

# Mutually exclusive steps: if both arrive (shouldn't happen after UI fix),
# keep only the first one in the list.
_EXCLUSIVE_PAIRS = [{"stemming", "lemmatization"}]

def _sanitize_steps(steps):
    """Remove the second member of any exclusive pair if both are present."""
    steps = list(steps)
    for pair in _EXCLUSIVE_PAIRS:
        present = [s for s in steps if s in pair]
        if len(present) > 1:
            # keep the first one that appears, drop the rest
            for drop in present[1:]:
                steps.remove(drop)
    return steps

def preprocess(text, steps):
    steps = _sanitize_steps(steps)
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
        "target": (S.get("nlp_label_col", "") or "") if S.get("task") != "topic_model" else "",
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
def favicon(): return send_from_directory("static", "icon-nlpflow.png", mimetype="image/png")

@app.route("/api/status")
def status():
    return jsonify({"nltk": NLTK_OK, "data_loaded": bool(S["texts"]),
                    "trained": S["model"] is not None, "task": S["task"],
                    "n_texts": len(S["texts"])})

@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    f = request.files.get("file")
    if not f: return jsonify({"error":"No file"}), 400
    mode       = request.form.get("mode","nlp")       # "nlp" | "tab"
    text_col   = request.form.get("text_col","text")
    label_col  = request.form.get("label_col","")
    target_col = request.form.get("target_col","")
    task       = request.form.get("task","classification")
    node_id    = request.form.get("node_id","")

    # ── Read with pandas ─────────────────────────────────────────────────────
    try:
        content = f.read()
        df = pd.read_csv(
            io.BytesIO(content),
            dtype=str,
            keep_default_na=False,
        )
    except Exception as e:
        return jsonify({"error": f"Error leyendo CSV: {e}"}), 400

    if df.empty: return jsonify({"error": "Empty CSV"}), 400
    df = df.fillna("")
    cols = list(df.columns)

    # ── Store DataFrame (shared layer) ───────────────────────────────────────
    _store_df(node_id if node_id else None, df)
    raw_rows = df.to_dict(orient="records")

    if mode == "tab":
        # ── Modo tabular ─────────────────────────────────────────────────────
        # Convertir columnas numéricas de str a float donde sea posible
        df_num = df.copy()
        for col in cols:
            try:
                df_num[col] = pd.to_numeric(df_num[col])
            except (ValueError, TypeError):
                pass
        _store_df(node_id if node_id else None, df_num)
        raw_rows = df_num.to_dict(orient="records")

        tgt = target_col if target_col in cols else (cols[-1] if cols else "")
        S.update(
            texts=[], labels=[], label_names=[], task=task,
            dataset_name=f.filename, processed_texts=[],
            results={}, model=None, vectorizer=None,
            columns=cols, raw_rows=raw_rows, csv_source="external",
        )
        if node_id:
            slot = _node_slot(node_id)
            slot.update(columns=cols, raw_rows=raw_rows, dataset_name=f.filename,
                        task=task, csv_source="external", target=tgt,
                        is_tabular=True)

        # Construir resumen tabular (mismo formato que load_tab_dataset)
        n_rows = len(raw_rows)
        col_info = []
        for c in cols:
            vals = df_num[c].dropna()
            try:
                pd.to_numeric(vals)
                ctype = "numeric"
            except (ValueError, TypeError):
                ctype = "categorical"
            col_info.append({"name": c, "type": ctype})

        return jsonify({
            "ok": True,
            "n": n_rows,
            "task": task,
            "dataset_name": f.filename,
            "columns": col_info,
            "target": tgt,
            "is_tabular": True,
            "node_id": node_id,
        })

    else:
        # ── Modo NLP ──────────────────────────────────────────────────────────
        if text_col not in cols:
            return jsonify({"error": f"Column '{text_col}' not found. Available: {cols}"}), 400

        texts    = [str(r.get(text_col,"")).strip() for r in raw_rows if str(r.get(text_col,"")).strip()]
        raw_rows = [r for r in raw_rows if str(r.get(text_col,"")).strip()]

        labels, label_names = [], []
        if label_col and label_col in cols:
            raw_lbl = [str(r.get(label_col,"")).strip() for r in raw_rows]
            uniq    = sorted(set(raw_lbl))
            m       = {v: i for i, v in enumerate(uniq)}
            labels, label_names = [m[l] for l in raw_lbl], uniq

        S.update(texts=texts, labels=labels, label_names=label_names, task=task,
                 dataset_name=f.filename, processed_texts=list(texts),
                 results={}, model=None, vectorizer=None, columns=cols, raw_rows=raw_rows,
                 csv_source="external", nlp_text_col=text_col, nlp_label_col=label_col)

        if node_id:
            slot = _node_slot(node_id)
            slot.update(columns=cols, raw_rows=raw_rows, dataset_name=f.filename,
                        task=task, csv_source="external")

        return jsonify({**_dataset_summary(), "columns": cols, "node_id": node_id,
                        "target": label_col if label_col and label_col in cols else "",
                        "nlp_mode": True, "is_tabular": False})

@app.route("/api/dataset_info")
def dataset_info(): return jsonify(_dataset_summary())

@app.route("/api/all_texts")
def all_texts():
    """Return all doc indices, labels and short previews for the preview selector."""
    texts  = S["texts"]
    labels = S["labels"]
    names  = S["label_names"]
    if not texts:
        return jsonify({"docs": []})
    docs = []
    for i, t in enumerate(texts):
        lbl = names[labels[i]] if labels and i < len(labels) else ""
        docs.append({"idx": i, "label": lbl, "preview": t[:120]})
    return jsonify({"docs": docs, "label_names": names, "task": S["task"]})

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

@app.route("/api/model_cv_poll")
def model_cv_poll():
    """Poll CV background thread progress."""
    global _cv_thread
    running = _cv_thread is not None and _cv_thread.is_alive()
    events  = _cv_drain()
    return jsonify({"running": running, "events": events})

@app.route("/api/model_cv_result")
def model_cv_result():
    """Retrieve the final CV result once the thread is done."""
    node_id = request.args.get("node_id","")
    if node_id in _cv_result:
        return jsonify(_cv_result[node_id])
    return jsonify({"error": "CV result not ready"}), 404

@app.route("/api/model_cv_cancel", methods=["POST"])
def model_cv_cancel():
    _cv_cancel.set()
    _cv_push(100, "__error__:Cancelado por el usuario")
    return jsonify({"ok": True})

@app.route("/api/model_cv", methods=["POST"])
def model_cv():
    """Grid Search + K-fold CV — arranca en background y devuelve {started:true}.
    El cliente hace polling a /api/model_cv_poll y recoge el resultado en /api/model_cv_result.
    """
    global _cv_thread
    body        = request.get_json(force=True, silent=True) or {}
    upstream_id = str(body.get("upstream_id", "")) or None
    node_id     = str(body.get("node_id", "")) or None
    target_col  = str(body.get("target_col", "")) or _TAB.get("target_col", "")
    method      = str(body.get("method", "ridge"))
    param_vals  = body.get("param_vals", None)
    k_folds     = int(body.get("k_folds", 5))

    # Limpiar resultado y flags anteriores
    if node_id and node_id in _cv_result:
        del _cv_result[node_id]
    _cv_cancel.clear()
    with _cv_lock:
        _cv_progress.clear()

    def _worker():
        try:
            result = _run_model_cv(upstream_id, node_id, target_col, method, param_vals, k_folds)
            if node_id:
                _cv_result[node_id] = result
            _cv_push(100, "__done__")
        except Exception as e:
            _cv_push(100, f"__error__:{str(e)}")

    _cv_thread = threading.Thread(target=_worker, daemon=True)
    _cv_thread.start()
    return jsonify({"started": True, "node_id": node_id})


def _run_model_cv(upstream_id, node_id, target_col, method, param_vals, k_folds):
    """Cuerpo real del Grid Search + CV. Llamado desde hilo background."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # normalizations to sweep (always all three for grid search)
    norm_sweep  = ["none", "zscore", "minmax"]
    norm_labels = {"none": "Sin norm.", "zscore": "Z-score", "minmax": "Min-Max"}
    # normalizations to sweep (always all three for grid search)
    norm_sweep  = ["none", "zscore", "minmax"]
    norm_labels = {"none": "Sin norm.", "zscore": "Z-score", "minmax": "Min-Max"}

    # ── Resolve data ──────────────────────────────────────────────────────
    D    = _effective_data(upstream_id) if upstream_id else None
    rows = (D["raw_rows"] if D else None) or S["raw_rows"]
    cols = (D["columns"]  if D else None) or S["columns"]
    if not rows:
        raise ValueError("No hay datos cargados")
    if not target_col:
        target_col = (D["target"] if D else None) or _TAB.get("target_col","") or (cols[-1] if cols else "")

    _cv_push(5, "Preparando datos…")

    # Build raw X, y (no scaler — Pipeline handles per fold)
    result_xy = _build_Xy(rows, cols, target_col, normalize="none")
    if result_xy[0] is None:
        raise ValueError(result_xy[1])
    X, y_raw, feat_names = result_xy

    # ── Detect problem type — always driven by the target, regardless of method ──
    tgt_vals = [r.get(target_col,"") for r in rows if not _is_missing(r.get(target_col,""))]
    tgt_type = _col_type(tgt_vals)
    _unique_tgt = set(str(v).strip() for v in tgt_vals)
    # Categorical string → always classification
    # Numeric but ≤10 unique values → classification (e.g. 0/1, risk_label)
    # Numeric with >10 unique values → regression
    if not _is_numeric_type(tgt_type):
        is_cls = True
    elif len(_unique_tgt) <= 10:
        is_cls = True
    else:
        is_cls = False

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
        raise ValueError(f"Pocos datos para {k_folds}-fold CV. Necesitas al menos {k_folds*2} filas.")

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
        # If the target is classification, map regression methods to their cls equivalents
        if is_cls:
            if method in ("ridge", "lasso", "linreg"):
                return LogisticRegression(C=1.0/max(pv, 1e-6), max_iter=5000, random_state=42, solver="saga")
            if method == "logistic": return LogisticRegression(C=pv, max_iter=5000, random_state=42, solver="saga")
            if method == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(n_neighbors=int(pv))
        if method == "ridge":    return Ridge(alpha=pv)
        if method == "lasso":    return Lasso(alpha=pv, max_iter=10000)
        if method == "logistic": return LogisticRegression(C=pv, max_iter=5000, random_state=42, solver="saga")
        if method == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_neighbors=int(pv))
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
    total_combos = len(norm_sweep) * len(effective_pvals)
    done_combos  = 0

    _cv_push(10, f"Grid search: {total_combos} combinaciones × {k_folds} folds…")

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
            done_combos += 1
            pct = 10 + int(done_combos / total_combos * 65)
            norm_lbl = norm_labels.get(norm, norm)
            pv_lbl = f" λ={round(pv,4)}" if pv is not None else ""
            _cv_push(pct, f"{norm_lbl}{pv_lbl} — {done_combos}/{total_combos}")
        results_by_norm[norm] = norm_results

    # Global best combo
    if higher_is_better:
        best_combo = max(all_combos, key=lambda r: r["score_mean"])
    else:
        best_combo = min(all_combos, key=lambda r: r["score_mean"])

    best_norm  = best_combo["norm"]
    best_pv    = best_combo["param_val"]
    best_score = best_combo["score_mean"]

    _cv_push(75, f"Mejor: {norm_labels.get(best_norm, best_norm)} — entrenando modelo final…")
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
        _avg = "binary" if len(classes) == 2 else "macro"
        def _cls_m(yt, yp):
            return {
                "accuracy":  round(float(accuracy_score(yt, yp)), 4),
                "f1":        round(float(f1_score(yt, yp, average=_avg, zero_division=0)), 4),
                "precision": round(float(precision_score(yt, yp, average=_avg, zero_division=0)), 4),
                "recall":    round(float(recall_score(yt, yp, average=_avg, zero_division=0)), 4),
            }
        train_m = _cls_m(y_tr, y_tr_pred)
        test_m  = _cls_m(y_te, y_te_pred)

    coefs = []
    if hasattr(final_est, "coef_"):
        coef_arr = final_est.coef_
        # Multiclass logistic → coef_ is (n_classes, n_features); use mean abs importance
        if hasattr(coef_arr, "ndim") and coef_arr.ndim == 2:
            coef_arr = np.mean(np.abs(coef_arr), axis=0)
        else:
            coef_arr = np.asarray(coef_arr).ravel()
        coefs = list(zip(feat_names, [round(float(c), 6) for c in coef_arr]))
        coefs = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
    if hasattr(final_est, "intercept_"):
        _intercept_raw = final_est.intercept_
        intercept = round(float(np.mean(_intercept_raw)), 6)
    else:
        intercept = 0.0

    # Store in _MODEL_STORE so model_eval can read it
    if node_id:
        _MODEL_STORE[node_id] = {
            "method":        method,
            "normalize":     best_norm,
            "alpha":         best_pv,
            "problem_type":  "classification" if is_cls else "regression",
            "classes":       classes if is_cls else [],
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

    _cv_push(88, "Generando gráfica…")
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
    # When ridge/lasso/linreg is mapped to classification, the sweep is over alpha
    # but the real model parameter is C = 1/alpha — convert before displaying
    _display_pv = best_pv
    _display_label = cfg["label"]
    if is_cls and method in ("ridge", "lasso", "linreg") and best_pv is not None:
        _display_pv = round(1.0 / max(float(best_pv), 1e-6), 6)
        _display_label = "C (inv. reg.)"

    if _display_pv is not None:
        if cfg.get("isInt"):
            best_pv_display = str(int(round(_display_pv)))
        elif _display_pv < 0.01:
            best_pv_display = f"{_display_pv:.2e}"
        else:
            best_pv_display = str(round(_display_pv, 4))
    else:
        best_pv_display = None

    print(f"[model_cv/display] method={method!r} is_cls={is_cls} best_pv={best_pv} _display_pv={_display_pv} best_pv_display={best_pv_display!r} param_label={_display_label!r}")

    # Backfill cv_img + best_pv_display into _MODEL_STORE now that they're available
    if node_id and node_id in _MODEL_STORE:
        _MODEL_STORE[node_id]["cv_img"]          = cv_img
        _MODEL_STORE[node_id]["best_pv_display"] = best_pv_display
        _MODEL_STORE[node_id]["param_label"]     = _display_label

    _cv_push(98, "Finalizando…")
    return {
        "results_by_norm": results_by_norm,
        "best":            best_combo,
        "best_pv_display": best_pv_display,
        "method":          method,
        "param_label":     _display_label,
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
    }

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
        return LogisticRegression(C=float(c), max_iter=5000, random_state=42, solver="saga")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=int(k))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=int(n), max_depth=d, random_state=42, n_jobs=-1)
    if name == "svm":
        return LinearSVC(C=float(c), max_iter=3000, random_state=42)
    return LogisticRegression(max_iter=5000, random_state=42)

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

# ── NLP EDA plots ─────────────────────────────────────────────────────────────
@app.route("/api/plot_nlp", methods=["POST"])
def plot_nlp():
    """EDA plots for NLP corpora. body: {type, corpus, top_n, class_filter}"""
    body         = request.get_json(force=True, silent=True) or {}
    plot_type    = body.get("type", "doc_length")
    corpus_key   = body.get("corpus", "raw")   # "raw" | "processed"
    top_n        = int(body.get("top_n", 20))
    class_filter = body.get("class_filter", "")  # "" = all
    es           = _UI_LANG == "es"

    texts  = S["processed_texts"] if corpus_key == "processed" else S["texts"]
    labels = S["labels"]
    names  = S["label_names"]

    if not texts:
        return jsonify({"error": "No hay corpus cargado" if es else "No corpus loaded"}), 400

    # Optional per-class filter
    if class_filter and names and class_filter in names:
        cidx = names.index(class_filter)
        pairs = [(t, labels[i] if labels else -1) for i, t in enumerate(texts) if labels and labels[i] == cidx]
        texts  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    # ── 1. Distribución de longitud de documentos ──────────────────────────
    if plot_type == "doc_length":
        lengths = [len(t.split()) for t in texts]
        fig, ax = plt.subplots(figsize=(8, 4))
        style_ax(ax)
        ax.hist(lengths, bins=min(40, max(10, len(lengths)//3)),
                color=PALETTE[0], edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_xlabel("Nº de palabras" if es else "Number of words", color=INK, fontsize=11)
        ax.set_ylabel("Nº de documentos" if es else "Number of documents", color=INK, fontsize=11)
        ax.set_title("Distribución de longitud de documentos" if es else "Document length distribution",
                     color=INK, fontsize=13, fontweight="bold", pad=12)
        med = float(np.median(lengths))
        ax.axvline(med, color=PALETTE[2], linewidth=1.5, linestyle="--",
                   label=f"{'Mediana' if es else 'Median'}: {med:.0f} {'palabras' if es else 'words'}")
        ax.legend(fontsize=10, framealpha=0)
        plt.tight_layout()
        return jsonify({"img": fig_b64(fig)})

    # ── 2. Top N palabras más frecuentes ──────────────────────────────────
    if plot_type == "top_words":
        all_words = []
        for t in texts: all_words.extend(t.split())
        top = Counter(all_words).most_common(top_n)
        if not top: return jsonify({"error": "No hay palabras" if es else "No words found"}), 400
        words, counts = zip(*top)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        style_ax(ax)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(words))]
        ax.barh(list(reversed(words)), list(reversed(counts)), color=list(reversed(colors)), height=0.65)
        ax.set_xlabel("Frecuencia" if es else "Frequency", color=INK, fontsize=11)
        ax.set_title(f"Top {top_n} {'palabras más frecuentes' if es else 'most frequent words'}",
                     color=INK, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        return jsonify({"img": fig_b64(fig)})

    # ── 3. Distribución de clases ─────────────────────────────────────────
    if plot_type == "class_dist":
        if not names:
            return jsonify({"error": "Este dataset no tiene columna de categorías" if es else "Dataset has no category column"}), 400
        c = Counter(labels)
        cats   = [names[k] for k in sorted(c.keys()) if k < len(names)]
        counts = [c[k] for k in sorted(c.keys()) if k < len(names)]
        total  = sum(counts) or 1
        fig, ax = plt.subplots(figsize=(7, 4))
        style_ax(ax)
        bars = ax.bar(cats, counts,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(cats))],
                      width=0.55, zorder=3)
        ax.set_ylabel("Nº de documentos" if es else "Number of documents", color=INK, fontsize=11)
        ax.set_title("Distribución de clases" if es else "Class distribution",
                     color=INK, fontsize=13, fontweight="bold", pad=12)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f"{cnt}\n({100*cnt/total:.1f}%)",
                    ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_ylim(0, max(counts) * 1.18)
        plt.tight_layout()
        return jsonify({"img": fig_b64(fig)})

    # ── 4. Longitud media por clase ────────────────────────────────────────
    if plot_type == "length_by_class":
        if not names:
            return jsonify({"error": "Este dataset no tiene columna de categorías" if es else "Dataset has no category column"}), 400
        from collections import defaultdict
        by_class = defaultdict(list)
        for i, t in enumerate(texts):
            if labels and i < len(labels):
                lname = names[labels[i]] if labels[i] < len(names) else str(labels[i])
                by_class[lname].append(len(t.split()))
        if not by_class: return jsonify({"error": "Sin datos por clase" if es else "No data per class"}), 400
        cats   = sorted(by_class.keys())
        means  = [float(np.mean(by_class[c])) for c in cats]
        stds   = [float(np.std(by_class[c]))  for c in cats]
        fig, ax = plt.subplots(figsize=(7, 4))
        style_ax(ax)
        xs = range(len(cats))
        bars = ax.bar(xs, means, yerr=stds, capsize=5,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(cats))],
                      width=0.55, zorder=3, error_kw={"ecolor": "#888", "linewidth": 1.2})
        ax.set_xticks(list(xs)); ax.set_xticklabels(cats, color=INK, fontsize=11)
        ax.set_ylabel("Palabras por documento" if es else "Words per document", color=INK, fontsize=11)
        ax.set_title("Longitud media de documentos por clase" if es else "Average document length by class",
                     color=INK, fontsize=13, fontweight="bold", pad=12)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.05,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
        plt.tight_layout()
        return jsonify({"img": fig_b64(fig)})

    # ── 5. Nube de palabras por clase ──────────────────────────────────────
    if plot_type == "wordcloud_by_class":
        try:
            from wordcloud import WordCloud as WC
        except ImportError:
            return jsonify({"error": "wordcloud no instalado" if es else "wordcloud not installed"}), 500
        if not names:
            return jsonify({"error": "Este dataset no tiene columna de categorías" if es else "Dataset has no category column"}), 400
        from collections import defaultdict
        by_class = defaultdict(list)
        for i, t in enumerate(texts):
            if labels and i < len(labels):
                lname = names[labels[i]] if labels[i] < len(names) else str(labels[i])
                by_class[lname].append(t)
        cats = sorted(by_class.keys())
        n_cls = len(cats)
        if n_cls == 0: return jsonify({"error": "Sin clases" if es else "No classes found"}), 400
        cols_n = min(n_cls, 3)
        rows_n = (n_cls + cols_n - 1) // cols_n
        fig, axes = plt.subplots(rows_n, cols_n,
                                 figsize=(cols_n * 5, rows_n * 3.2))
        if n_cls == 1: axes = [[axes]]
        elif rows_n == 1: axes = [axes] if cols_n > 1 else [[axes]]
        fig.patch.set_facecolor(LIGHT)
        flat_axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
        import random as _rnd
        _rnd.seed(42)
        wc_palette = PALETTE + ["#FF6B9D", "#00B4D8", "#06D6A0"]
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return _rnd.choice(wc_palette)
        for i, cls in enumerate(cats):
            ax = flat_axes[i]
            ax.set_facecolor(LIGHT)
            words = []
            for t in by_class[cls]: words.extend(t.split())
            freq = {w: c for w, c in Counter(words).most_common(60)}
            if freq:
                wc = WC(width=500, height=260, background_color="#f5f5f5",
                        max_words=60, color_func=color_func, margin=4).generate_from_frequencies(freq)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(cls, color=INK, fontsize=12, fontweight="bold", pad=6)
        for j in range(n_cls, len(flat_axes)):
            flat_axes[j].axis("off")
        fig.suptitle("Nube de palabras por clase" if es else "Word cloud by class",
                     color=INK, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        return jsonify({"img": fig_b64(fig)})

    # ── 6. TF-IDF vs BoW comparison ───────────────────────────────────────────
    if plot_type == "tfidf_comparison":
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        import numpy as np

        if not texts:
            return jsonify({"error": "No hay corpus cargado" if es else "No corpus loaded"}), 400

        # Compute BoW top-N
        cv = CountVectorizer(max_features=5000)
        cv.fit(texts)
        bow_matrix = cv.transform(texts)
        bow_totals = np.asarray(bow_matrix.sum(axis=0)).flatten()
        bow_vocab  = cv.get_feature_names_out()
        bow_pairs  = sorted(zip(bow_vocab, bow_totals), key=lambda x: x[1], reverse=True)[:top_n]
        bow_words  = [p[0] for p in bow_pairs]
        bow_counts = [p[1] for p in bow_pairs]

        # Compute TF-IDF top-N (same vocab base for fair comparison)
        tv = TfidfVectorizer(max_features=5000)
        tv.fit(texts)
        tfidf_matrix = tv.transform(texts)
        tfidf_means  = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        tfidf_vocab  = tv.get_feature_names_out()
        tfidf_pairs  = sorted(zip(tfidf_vocab, tfidf_means), key=lambda x: x[1], reverse=True)[:top_n]
        tfidf_words  = [p[0] for p in tfidf_pairs]
        tfidf_scores = [p[1] for p in tfidf_pairs]

        # Determine which words change ranking: in top-N of BoW but not in top-N of TF-IDF (or vice versa)
        bow_set   = set(bow_words)
        tfidf_set = set(tfidf_words)
        changed   = bow_set.symmetric_difference(tfidf_set)

        COLOR_BOW     = "#4A90D9"   # blue  — stable BoW bars
        COLOR_TFIDF   = "#2ecc71"   # green — stable TF-IDF bars
        COLOR_CHANGED = "#e05c5c"   # red   — ranking change (both panels)

        def bar_colors(words, changed_set, default_color):
            return [COLOR_CHANGED if w in changed_set else default_color for w in words]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.32)))
        fig.patch.set_facecolor(LIGHT)

        # BoW panel — blue stable, red changed
        style_ax(ax1)
        colors1 = bar_colors(list(reversed(bow_words)), changed, COLOR_BOW)
        ax1.barh(list(reversed(bow_words)), list(reversed(bow_counts)), color=colors1, height=0.65)
        ax1.set_xlabel("Frecuencia" if es else "Frequency", color=INK, fontsize=11)
        ax1.set_title(f"BoW — Top {top_n} {'palabras' if es else 'words'}",
                      color=INK, fontsize=13, fontweight="bold", pad=10)

        # TF-IDF panel — green stable, red changed
        style_ax(ax2)
        colors2 = bar_colors(list(reversed(tfidf_words)), changed, COLOR_TFIDF)
        ax2.barh(list(reversed(tfidf_words)), list(reversed(tfidf_scores)), color=colors2, height=0.65)
        ax2.set_xlabel("TF-IDF score", color=INK, fontsize=11)
        ax2.set_title(f"TF-IDF — Top {top_n} {'palabras' if es else 'words'}",
                      color=INK, fontsize=13, fontweight="bold", pad=10)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOR_BOW,     label="Estable en BoW" if es else "Stable in BoW"),
            Patch(facecolor=COLOR_TFIDF,   label="Estable en TF-IDF" if es else "Stable in TF-IDF"),
            Patch(facecolor=COLOR_CHANGED, label="Cambia de ranking" if es else "Ranking change"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3,
                   fontsize=10, framealpha=0, bbox_to_anchor=(0.5, -0.03))

        fig.suptitle(
            "Comparativa BoW vs TF-IDF — Las barras en rojo indican palabras cuyo ranking cambia significativamente"
            if es else
            "BoW vs TF-IDF comparison — Red bars indicate words whose ranking changes significantly",
            color=INK, fontsize=12, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        return jsonify({"img": fig_b64(fig)})

    return jsonify({"error": f"Tipo de gráfica desconocido: {plot_type}" if es else f"Unknown plot type: {plot_type}"}), 400

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
        if model_name=="logistic":      return LogisticRegression(C=lr_c,penalty=lr_penalty,max_iter=5000,random_state=42,solver="saga")
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
        tl = res.get("topic_labels", {})
        return jsonify({
            "ready": True,
            "task": "topic_model",
            "results": {
                "topics":           res.get("topics", []),
                "perplexity":       res.get("perplexity"),
                "coherence_cv":     res.get("coherence_cv",    res.get("coherence", [])),
                "coherence_cnpmi":  res.get("coherence_cnpmi", []),
                "topic_diversity":  res.get("topic_diversity"),
                "algorithm":        res.get("algorithm", "lda"),
                "doc_topics":       res.get("doc_topics", []),
                "label_result":     res.get("label_result"),
                "topic_labels":     tl,
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
    C_V coherence approximation (Röder et al. 2015).
    Uses a boolean sliding-window co-occurrence over the whole corpus
    (window = full document, a common approximation).
    For each topic word w_i, compute the indirect confirmation measure:
        cv(w_i, W) = cos_sim( phi(w_i), sum_{w_j in W, j!=i} phi(w_j) )
    where phi(w) is the NPMI vector of w against every other top word.
    This gives scores in [0, 1] that are meaningfully higher than C_NPMI.
    """
    # Build co-occurrence counts (document-level boolean)
    doc_sets = [set(doc) for doc in tokenized_docs]
    N = len(doc_sets)

    def _cooc(wi, wj):
        return sum(1 for ds in doc_sets if wi in ds and wj in ds)

    def _freq(w):
        return sum(1 for ds in doc_sets if w in ds)

    scores = []
    for words in topic_words_list:
        wds = words[:top_n]
        # Pre-compute NPMI of each pair
        npmi_matrix = {}
        for i, wi in enumerate(wds):
            fi = _freq(wi)
            if fi == 0:
                continue
            for j, wj in enumerate(wds):
                if i == j:
                    continue
                fj  = _freq(wj)
                fij = _cooc(wi, wj)
                if fj == 0 or fij == 0:
                    npmi_matrix[(i, j)] = 0.0
                    continue
                p_i  = fi  / N
                p_j  = fj  / N
                p_ij = fij / N
                raw_pmi  = np.log(p_ij / (p_i * p_j) + 1e-10)
                norm_val = -np.log(p_ij + 1e-10)
                npmi_matrix[(i, j)] = float(raw_pmi / (norm_val + 1e-10))

        # Indirect confirmation measure: cosine of phi(wi) vs sum(phi(wj))
        topic_scores = []
        for i in range(len(wds)):
            vec_i   = np.array([npmi_matrix.get((i, j), 0.0) for j in range(len(wds)) if j != i])
            sum_vec = np.zeros(len(wds) - 1)
            for k, j in enumerate(x for x in range(len(wds)) if x != i):
                sum_vec[k] = sum(npmi_matrix.get((j, m), 0.0) for m in range(len(wds)) if m != j)
            norm_i = np.linalg.norm(vec_i)
            norm_s = np.linalg.norm(sum_vec)
            if norm_i < 1e-10 or norm_s < 1e-10:
                topic_scores.append(0.0)
            else:
                cos = float(np.dot(vec_i, sum_vec) / (norm_i * norm_s))
                # Shift from [-1,1] to [0,1]
                topic_scores.append((cos + 1.0) / 2.0)
        scores.append(round(float(np.mean(topic_scores)) if topic_scores else 0.0, 4))
    return scores


def _coherence_cnpmi(topic_words_list, tokenized_docs, top_n=10):
    """
    C_NPMI coherence: average pairwise NPMI over the top-N topic words.
    NPMI ∈ [-1, 1]:  1 = always co-occur, 0 = independent, -1 = never.
    This is the standard definition and will give different (lower) values than C_V.
    """
    doc_sets = [set(doc) for doc in tokenized_docs]
    N = len(doc_sets)

    def _freq(w):
        return sum(1 for ds in doc_sets if w in ds)

    def _cooc(wi, wj):
        return sum(1 for ds in doc_sets if wi in ds and wj in ds)

    scores = []
    for words in topic_words_list:
        wds = words[:top_n]
        pair_scores = []
        for i in range(len(wds)):
            for j in range(i + 1, len(wds)):
                wi, wj = wds[i], wds[j]
                fi  = _freq(wi)
                fj  = _freq(wj)
                fij = _cooc(wi, wj)
                if fi == 0 or fj == 0 or fij == 0:
                    pair_scores.append(-1.0)
                    continue
                p_i  = fi  / N
                p_j  = fj  / N
                p_ij = fij / N
                pmi  = np.log(p_ij / (p_i * p_j) + 1e-10)
                npmi = pmi / (-np.log(p_ij + 1e-10))
                # Clamp to [-1, 1] to handle floating point edge cases
                pair_scores.append(float(max(-1.0, min(1.0, npmi))))
        scores.append(round(float(np.mean(pair_scores)) if pair_scores else -1.0, 4))
    return scores


@app.route("/api/topic_model", methods=["POST"])
def topic_model():
    global _topic_thread
    body       = request.json
    algorithm  = body.get("algorithm", "lda")   # lda | nmf | lsa
    n_topics   = int(body.get("n_topics",  5))
    max_vocab  = int(body.get("max_vocab", 1000))
    top_n_words= int(body.get("top_words", 15))
    user_max_iter = min(int(body.get("max_iter", 0)), 2000)  # 0 = auto, máx 2000

    # ── Advanced vectorizer params ───────────────────────────────────────────
    min_df     = max(1, int(body.get("min_df", 2)))
    max_df     = float(body.get("max_df", 1.0))
    max_df     = max(0.05, min(1.0, max_df))   # clamp to [0.05, 1.0]
    stop_words_opt = body.get("stop_words", None)  # None | "auto" | "english" | "spanish"

    # ── LDA priors ───────────────────────────────────────────────────────────
    _ALPHA_MAP = {"auto": None, "symmetric": "symmetric",
                  "0.001": 0.001, "0.01": 0.01, "0.05": 0.05, "0.1": 0.1,
                  "0.2": 0.2, "0.3": 0.3, "0.5": 0.5, "0.7": 0.7,
                  "1.0": 1.0, "2.0": 2.0, "5.0": 5.0, "10.0": 10.0}
    _BETA_MAP  = {"auto": None,
                  "0.001": 0.001, "0.01": 0.01, "0.05": 0.05, "0.1": 0.1,
                  "0.2": 0.2, "0.3": 0.3, "0.5": 0.5, "1.0": 1.0,
                  "2.0": 2.0, "5.0": 5.0}
    raw_alpha  = str(body.get("doc_topic_prior",  "auto"))
    raw_beta   = str(body.get("topic_word_prior", "auto"))
    lda_alpha  = _ALPHA_MAP.get(raw_alpha, None)   # None → sklearn default ("auto")
    lda_beta   = _BETA_MAP.get(raw_beta,   None)

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
            _es = (_UI_LANG == "es")
            push_progress(3, "Detectando idioma del corpus…" if _es else "Detecting corpus language…")
            corpus_lang = _detect_corpus_lang(proc)
            print(f"[TOPIC] corpus_lang={corpus_lang}", flush=True)

            push_progress(5, f"Vectorizando corpus ({algorithm.upper()})…" if _es else f"Vectorising corpus ({algorithm.upper()})…")
            time.sleep(0.05)

            # Resolve stop_words: "auto" → use corpus language, else explicit or None
            _sw = None
            if stop_words_opt == "auto":
                _sw = "english" if corpus_lang == "en" else None  # sklearn only has english built-in
            elif stop_words_opt in ("english", "spanish"):
                _sw = stop_words_opt if stop_words_opt == "english" else None

            tokenized = [t.lower().split() for t in proc]

            # Choose vectorizer and model
            if algorithm == "lda":
                vec = CountVectorizer(max_features=max_vocab, min_df=min_df,
                                      max_df=max_df, stop_words=_sw)
                X   = vec.fit_transform(proc)
                push_progress(20, "Ajustando LDA…" if _es else "Fitting LDA…")
                n_docs = len(proc)
                if user_max_iter > 0:
                    n_iter = user_max_iter
                elif n_docs < 500:
                    n_iter = 100
                elif n_docs < 5000:
                    n_iter = 200
                else:
                    n_iter = 300
                print(f"[LDA] n_docs={n_docs}, n_topics={n_topics}, max_iter={n_iter}", flush=True)

                # ── Iterative fit with per-iteration progress + ETA + C_V optimisation ──
                _lda_iter_times  = []
                _lda_converged   = False
                _lda_prev_perp   = None
                _lda_min_iter    = max(50, n_iter // 4)

                _lda_kwargs = dict(n_components=n_topics, random_state=42,
                                   max_iter=1, learning_method="batch",
                                   evaluate_every=-1, perp_tol=0)
                if lda_alpha is not None: _lda_kwargs["doc_topic_prior"]  = lda_alpha
                if lda_beta  is not None: _lda_kwargs["topic_word_prior"] = lda_beta
                print(f"[LDA] alpha={lda_alpha or 'auto'} beta={lda_beta or 'auto'} "
                      f"min_df={min_df} max_df={max_df} stop_words={_sw} vocab={max_vocab}", flush=True)
                lda_partial = LatentDirichletAllocation(**_lda_kwargs)

                for _it in range(1, n_iter + 1):
                    _t0 = time.time()
                    if _it == 1:
                        lda_partial.fit(X)
                    else:
                        lda_partial.partial_fit(X)
                    _iter_t = time.time() - _t0
                    _lda_iter_times.append(_iter_t)

                    # ETA based on rolling average of last 5 iterations
                    _recent  = _lda_iter_times[-5:]
                    _avg_t   = sum(_recent) / len(_recent)
                    _remaining = int(_avg_t * (n_iter - _it))
                    _eta_str = (f"{_remaining // 60}m {_remaining % 60}s"
                                if _remaining >= 60 else f"{_remaining}s")

                    # Perplexity convergence check
                    _perp_now = float(lda_partial.perplexity(X))
                    _converged_str = ""
                    if _lda_prev_perp is not None and _it >= _lda_min_iter:
                        _delta = abs(_lda_prev_perp - _perp_now) / max(abs(_lda_prev_perp), 1)
                        if _delta < 1e-4:
                            _lda_converged = True
                            _converged_str = " ✓ converged" if not _es else " ✓ convergido"
                    _lda_prev_perp = _perp_now

                    # Progress: iter range mapped 20→58%
                    _pct = 20 + int(38 * _it / n_iter)
                    _msg = f"LDA iter {_it}/{n_iter} · perp {_perp_now:.1f} · ETA {_eta_str}{_converged_str}"
                    push_progress(_pct, _msg)
                    print(f"[LDA] {_msg}", flush=True)

                    if _lda_converged:
                        print(f"[LDA] convergencia en iter {_it} (mín={_lda_min_iter})", flush=True)
                        push_progress(58, (f"LDA converged at iter {_it}/{n_iter} ✓") if not _es else (f"LDA convergido en iter {_it}/{n_iter} ✓"))
                        break

                model            = lda_partial
                components       = model.components_
                doc_topic_matrix = model.transform(X)
                perplexity       = round(float(model.perplexity(X)), 1)

            elif algorithm == "nmf":
                vec = TfidfVectorizer(max_features=max_vocab, min_df=min_df,
                                      max_df=max_df, stop_words=_sw)
                X   = vec.fit_transform(proc)
                nmf_iter = user_max_iter if user_max_iter > 0 else 400
                print(f"[NMF] n_docs={len(proc)}, n_topics={n_topics}, max_iter={nmf_iter}", flush=True)

                # NMF no tiene partial_fit, pero sí reportamos progreso por bloques
                push_progress(20, f"Ajustando NMF (max {nmf_iter} iter)…" if _es else f"Fitting NMF (max {nmf_iter} iter)…")
                _nmf_block = max(1, nmf_iter // 10)
                _nmf_done  = 0
                _nmf_t0    = time.time()
                _nmf_converged = False

                while _nmf_done < nmf_iter and not _nmf_converged:
                    _block = min(_nmf_block, nmf_iter - _nmf_done)
                    _nmf_model_tmp = NMF(n_components=n_topics, random_state=42,
                                         max_iter=_nmf_done + _block,
                                         init="nndsvda", l1_ratio=0.5)
                    _W_tmp = _nmf_model_tmp.fit_transform(X)
                    _nmf_done += _block

                    _elapsed = time.time() - _nmf_t0
                    _iter_rate = _nmf_done / max(_elapsed, 0.001)
                    _remaining_i = nmf_iter - _nmf_done
                    _eta_s = int(_remaining_i / max(_iter_rate, 0.001))
                    _eta_str = f"{_eta_s // 60}m {_eta_s % 60}s" if _eta_s >= 60 else f"{_eta_s}s"

                    _recon_err = _nmf_model_tmp.reconstruction_err_
                    _pct = 20 + int(38 * _nmf_done / nmf_iter)
                    _msg = f"NMF iter {_nmf_done}/{nmf_iter} · err {_recon_err:.4f} · ETA {_eta_str}" if not _es else f"NMF iter {_nmf_done}/{nmf_iter} · err {_recon_err:.4f} · ETA {_eta_str}"
                    push_progress(_pct, _msg)
                    print(f"[NMF] {_msg}", flush=True)

                    if _nmf_model_tmp.n_iter_ < _nmf_done:
                        _nmf_converged = True
                        push_progress(58, f"NMF converged at iter {_nmf_done}/{nmf_iter} ✓" if not _es else f"NMF convergido en iter {_nmf_done}/{nmf_iter} ✓")
                        print(f"[NMF] convergencia alcanzada", flush=True)

                model = _nmf_model_tmp
                W     = _W_tmp
                components = model.components_
                row_sums   = W.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                doc_topic_matrix = W / row_sums
                perplexity = None

            else:  # lsa / svd
                vec = TfidfVectorizer(max_features=max_vocab, min_df=min_df,
                                      max_df=max_df, stop_words=_sw)
                X   = vec.fit_transform(proc)
                push_progress(20, "Ajustando LSA (SVD)…" if _es else "Fitting LSA (SVD)…")
                print(f"[LSA] n_docs={len(proc)}, n_topics={n_topics} (SVD exacto, sin iteraciones)", flush=True)
                model = TruncatedSVD(n_components=n_topics, random_state=42)
                W = model.fit_transform(X)
                components = model.components_
                components = np.abs(components)
                doc_topic_matrix = np.abs(W)
                row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                doc_topic_matrix = doc_topic_matrix / row_sums
                perplexity = None

            push_progress(60, "Extrayendo tópicos…" if _es else "Extracting topics…")
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

            push_progress(75, "Calculando coherencia C_V…" if _es else "Computing C_V coherence…")
            time.sleep(0.05)
            topic_word_lists = [tp["words"][:10] for tp in topics]
            cv_scores   = _coherence_cv(topic_word_lists, tokenized, top_n=10)

            push_progress(88, "Calculando coherencia C_NPMI…" if _es else "Computing C_NPMI coherence…")
            time.sleep(0.05)
            cnpmi_scores = _coherence_cnpmi(topic_word_lists, tokenized, top_n=10)

            # ── Weak labeling (if requested) ──────────────────────────────
            label_result = None
            if label_classes:
                push_progress(93, "Etiquetando corpus…" if _es else "Labelling corpus…")
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
                "topics":          topics,
                "doc_topics":      doc_topics,
                "perplexity":      perplexity,
                "coherence_cv":    cv_scores,
                "coherence_cnpmi": cnpmi_scores,
                "topic_diversity": _topic_diversity(topics),
                "algorithm":       algorithm,
                "label_result":    label_result,
                "corpus_lang":     corpus_lang,
                "task": "topic_model"
            }

            # ── Auto-label topics via Groq if toggle is active ───────────
            if HF_LABELING_ACTIVE:
                push_progress(95, "Etiquetando tópicos con IA…" if _es else "Labelling topics with AI…")
                auto_labels = _label_topics_with_lang(topics, corpus_lang)
                S["results"]["topic_labels"] = auto_labels

            push_progress(100, "Listo ✓" if _es else "Done ✓")
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
    print(f"[decision_boundary] node_id={node_id!r} feat_x={feat_x!r} feat_y={feat_y!r}")
    print(f"[decision_boundary] _LOADED_TAB_MODELS keys: {list(_LOADED_TAB_MODELS.keys())}")
    print(f"[decision_boundary] _MODEL_STORE keys:       {list(_MODEL_STORE.keys())}")
    print(f"[decision_boundary] _NODE_MODELS keys:       {list(_NODE_MODELS.keys())}")
    if not payload:
        ms  = _MODEL_STORE.get(node_id) or _MODEL_STORE.get(int(node_id) if node_id.isdigit() else None)
        mdl = _NODE_MODELS.get(node_id) or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
        print(f"[decision_boundary] ms found: {ms is not None}, mdl in _NODE_MODELS: {mdl is not None}")
        if ms:
            print(f"[decision_boundary] ms keys: {list(ms.keys())}")
            print(f"[decision_boundary] problem_type={ms.get('problem_type')!r}, model_obj present: {ms.get('model_obj') is not None}")
            payload = dict(ms)
            if mdl:
                payload["sklearn_model"] = mdl
            elif ms.get("model_obj"):
                payload["sklearn_model"] = ms["model_obj"]
    if not payload:
        return jsonify({"error": "No hay modelo cargado."}), 400

    # Ensure _NODE_MODELS is populated for downstream reuse
    if node_id and node_id not in _NODE_MODELS:
        _mdl = payload.get("sklearn_model") or payload.get("model_obj")
        if _mdl is not None:
            _NODE_MODELS[node_id] = _mdl

    problem_type = payload.get("problem_type", "regression")
    print(f"[decision_boundary] problem_type={problem_type!r}, sklearn_model present: {payload.get('sklearn_model') is not None}")
    if problem_type == "regression":
        return jsonify({"error": "La frontera de decisión solo está disponible para clasificación."}), 400

    mdl = (
        payload.get("sklearn_model")
        or payload.get("model_obj")
        or _NODE_MODELS.get(node_id)
        or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
    )
    print(f"[decision_boundary] mdl resolved: {mdl is not None} ({type(mdl).__name__ if mdl is not None else 'None'})")
    if mdl is None or not hasattr(mdl, "predict"):
        return jsonify({"error": "Modelo no disponible. Reentrena el modelo."}), 400

    features = payload.get("features") or payload.get("feat_names", [])
    classes  = payload.get("classes", [])
    # X_test / y_test may be stored under different key names depending on source
    _xte_raw = payload.get("X_test") or payload.get("X_te") or payload.get("_Xte", [])
    _yte_raw = payload.get("y_test") or payload.get("y_te") or payload.get("_yte", [])
    X_test   = np.array(_xte_raw)
    y_test   = np.array(_yte_raw)

    print(f"[decision_boundary] features={features}, feat_x={feat_x!r}, feat_y={feat_y!r}, X_test shape={X_test.shape}")
    if feat_x not in features or feat_y not in features:
        return jsonify({"error": f"Features '{feat_x}' o '{feat_y}' no encontradas. Disponibles: {features}"}), 400
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

# ── Groq client (lazy init) ───────────────────────────────────────────────────
try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient     = None
    _GROQ_AVAILABLE = False

_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Toggle: True = auto-label topics via LLM when topic model runs
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
    Call Groq API with model openai/gpt-oss-20b (configurable via GROQ_MODEL env var).
    Requires GROQ_API_KEY in .env. Returns generated text or empty string on failure.
    """
    if not _GROQ_AVAILABLE:
        print("[LLM] groq no instalado — ejecuta: pip install groq", flush=True)
        return ""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[LLM] GROQ_API_KEY no definida en .env", flush=True)
        return ""
    model = os.environ.get("GROQ_MODEL", _GROQ_MODEL)
    try:
        print(f"[LLM] Groq → modelo={model}", flush=True)
        client = _GroqClient(api_key=api_key)
        resp   = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        choice  = resp.choices[0]
        content = (choice.message.content or "").strip()
        # Modelos de razonamiento (gpt-oss-*) ponen el output en 'reasoning', no en 'content'
        if not content and hasattr(choice.message, "reasoning") and choice.message.reasoning:
            reasoning_text = choice.message.reasoning.strip()
            print(f"[LLM] content vacío, extrayendo JSON de reasoning ({len(reasoning_text)} chars)", flush=True)
            # Buscar el bloque JSON con "labels" lo más cerca del final (donde el modelo suele concluir)
            matches = list(re.finditer(r'\{\s*"labels"\s*:\s*\[[\s\S]*?\]\s*\}', reasoning_text))
            if matches:
                content = matches[-1].group()  # último match = conclusión final
                print(f"[LLM] JSON extraído de reasoning: {repr(content[:200])}", flush=True)
            else:
                # fallback: intentar extraer cualquier JSON con id+label del final
                content = reasoning_text
        print(f"[LLM] Groq OK ({len(content)} chars): {repr(content[:120])}", flush=True)
        return content
    except Exception as e:
        print(f"[LLM] Groq error: {e}", flush=True)
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


def _sanitize_label(label: str) -> str:
    """
    Post-process LLM label to ensure it's a coherent short phrase (2-4 words).
    - Strips quotes, trailing punctuation.
    - If the result has more than 5 words, keeps only the first 4.
    - Capitalises the first letter.
    """
    if not label:
        return label
    # Strip surrounding quotes and whitespace
    label = label.strip().strip('"\'').strip()
    # Remove trailing punctuation
    label = re.sub(r'[.,;:!?]+$', '', label).strip()
    # If too many words (likely a list dump), keep first 4
    words = label.split()
    if len(words) > 5:
        label = " ".join(words[:4])
    # Capitalise first letter, leave rest as-is
    if label:
        label = label[0].upper() + label[1:]
    return label


def _label_topics_with_lang(topics: list, lang: str = "es") -> dict:
    """Build a language-aware labeling prompt and call the LLM.

    Returns a dict {str(topic_id): label_str} with every topic covered
    (falls back to rule-based labels when the LLM is unavailable or fails).
    """
    is_es = (lang == "es")

    topic_list_str = "\n".join(
        f"{'Tópico' if is_es else 'Topic'} {tp['id']+1} (id={tp['id']}): {', '.join(tp['words'][:10])}"
        for tp in topics
    )

    if is_es:
        prompt_lbl = (
            "Eres un experto en análisis de tópicos. Tu tarea es asignar una etiqueta temática a cada tópico.\n\n"
            "REGLAS ESTRICTAS para la etiqueta:\n"
            "- Entre 2 y 4 palabras como máximo.\n"
            "- Debe ser una expresión coherente y natural, como un titular o categoría temática. "
            "Ejemplos correctos: 'Política exterior', 'Salud pública', 'Mercados financieros', 'Cine europeo'.\n"
            "- NO es una lista de palabras sueltas separadas por espacios. "
            "Incorrecto: 'gobierno ley España pp'. Correcto: 'Política española'.\n"
            "- Usa sustantivos con adjetivos o preposición cuando ayude al sentido.\n"
            "- Idioma de las etiquetas: ESPAÑOL.\n\n"
            "Tópicos a etiquetar:\n"
            f"{topic_list_str}\n\n"
            "Responde ÚNICAMENTE con este JSON (sin texto antes ni después):\n"
            '{"labels":[{"id":0,"label":"Ejemplo etiqueta"},{"id":1,"label":"Otra etiqueta"}]}'
        )
    else:
        prompt_lbl = (
            "You are an expert in topic analysis. Your task is to assign a thematic label to each topic.\n\n"
            "STRICT RULES for the label:\n"
            "- 2 to 4 words maximum.\n"
            "- Must be a coherent, natural expression like a headline or thematic category. "
            "Correct examples: 'Foreign policy', 'Public health', 'Financial markets', 'European cinema'.\n"
            "- NOT a list of unrelated words separated by spaces. "
            "Incorrect: 'government law spain party'. Correct: 'Spanish politics'.\n"
            "- Use nouns with adjectives or prepositions where they add meaning.\n"
            "- Label language: ENGLISH.\n\n"
            "Topics to label:\n"
            f"{topic_list_str}\n\n"
            "Respond ONLY with this JSON (no text before or after):\n"
            '{"labels":[{"id":0,"label":"Example label"},{"id":1,"label":"Another label"}]}'
        )

    raw_lbl = _call_llm(prompt_lbl)
    auto_labels: dict = {}

    if raw_lbl:
        jm = re.search(r'\{[\s\S]*\}', raw_lbl)
        if jm:
            try:
                parsed_lbl = json.loads(jm.group())
                for item in parsed_lbl.get("labels", []):
                    lbl = _sanitize_label(item.get("label", ""))
                    if lbl:
                        auto_labels[str(item["id"])] = lbl
            except Exception:
                pass

    # Fallback for any topic the LLM missed
    for tp in topics:
        if str(tp["id"]) not in auto_labels:
            auto_labels[str(tp["id"])] = _rule_label(tp["words"])

    print(f"[LABEL] lang={lang} → {auto_labels}", flush=True)
    return auto_labels


@app.route("/api/llm_label_topics", methods=["POST"])
def llm_label_topics():
    res    = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return jsonify({"error": "Ejecuta primero el Topic Model"}), 400

    # Use the language detected when the model was trained; fall back to detecting now
    corpus_lang = res.get("corpus_lang") or _detect_corpus_lang(S.get("texts", []) or S.get("processed_texts", []))

    label_map = _label_topics_with_lang(topics, corpus_lang)
    labels_out = [{"id": tp["id"], "label": label_map.get(str(tp["id"]), ""), "words": tp["words"]}
                  for tp in topics]
    llm_ok = any(v for v in label_map.values())

    S["results"]["topic_labels"] = label_map
    reasoning = f"Corpus language detected: {corpus_lang}. {'LLM labeling used.' if llm_ok else 'Rule-based fallback.'}"
    return jsonify({"labels": labels_out, "reasoning": reasoning, "llm_used": llm_ok})


@app.route("/api/clear_topic_labels", methods=["POST"])
def clear_topic_labels():
    """Remove topic labels from current results (called when toggle is deactivated)."""
    if S.get("results"):
        S["results"].pop("topic_labels", None)
    return jsonify({"ok": True})


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
    Routes the dialog request through the main-thread queue when running inside
    pywebview (Cocoa/GTK dialogs must run on the main thread).
    Falls back to tkinter when pywebview is not present.
    """
    global _DIALOG_CTR

    # ── 1. Try via main-thread queue (pywebview) ──────────────────────────────
    try:
        import webview  # only available when running inside pywebview
        with _DIALOG_LOCK:
            _DIALOG_CTR += 1
            req_id = _DIALOG_CTR
        # Normalise file_types_wv to a tuple of strings "Desc (*.ext)" as
        # pywebview expects.  Callers sometimes pass a plain string, a tuple of
        # strings, or a tuple of (desc, pattern) pairs — handle all three.
        def _norm_ft(ft):
            if isinstance(ft, str):
                return (ft,)
            result = []
            for item in ft:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    desc, pat = item
                    # build "Desc (*.ext)" from ("Desc", "*.ext")
                    result.append(f"{desc} ({pat})")
            return tuple(result) if result else ("All files (*.*)",)

        ft_normalised = _norm_ft(file_types_wv)
        print(f"[DIALOG] Enviando req_id={req_id} default_name={default_name!r} file_types={ft_normalised}", flush=True)
        _DIALOG_REQ.put((req_id, {
            "default_name": default_name,
            "file_types_wv": ft_normalised,
        }))
        # Block until main thread processes it (timeout 120 s)
        deadline = 120
        while deadline > 0:
            try:
                rid, path = _DIALOG_RES.get(timeout=1)
                print(f"[DIALOG] Respuesta recibida req_id={rid} path={path!r}", flush=True)
                if rid == req_id:
                    return path   # may be None if user cancelled
                # wrong id — put it back and keep waiting
                _DIALOG_RES.put((rid, path))
            except _queue_mod.Empty:
                pass
            deadline -= 1
        print(f"[DIALOG] TIMEOUT esperando req_id={req_id}", flush=True)
        raise RuntimeError("Dialog timed out")
    except ImportError:
        pass   # not inside pywebview, fall through to tkinter
    except Exception:
        pass   # queue failed for any reason, try tkinter

    # ── 2. Fallback: tkinter ──────────────────────────────────────────────────
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            initialfile      = default_name,
            defaultextension = os.path.splitext(default_name)[1] or "",
            filetypes        = filetypes_tk,
            parent           = root,
        )
        root.destroy()
        return path or None
    except Exception as tk_err:
        raise RuntimeError(f"No native dialog available: {tk_err}")


@app.route("/api/save_canvas_json", methods=["POST"])
def save_canvas_json():
    """Open a native Save dialog and write the canvas JSON to the chosen path."""
    body        = request.json or {}
    canvas_json = body.get("canvas", "{}")
    default_name = body.get("default_name", "canvas.json")
    try:
        path = _native_save_dialog(
            default_name,
            ("JSON file (*.json)", "All files (*.*)"),
            [("JSON file", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return jsonify({"cancelled": True})
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(canvas_json)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e), "fallback": True})


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


@app.route("/api/export_topic_bundle", methods=["POST"])
def export_topic_bundle():
    """
    Genera un ZIP con:
      - wordclouds/topico_XX.png  → word cloud de cada tópico
      - plots/topics_chart.png    → gráfica de barras de palabras clave
      - informe_es.html           → informe completo en español
      - informe_en.html           → informe completo en inglés
      - topic_config.json         → configuración de entrenamiento
    """
    import html as _html
    res    = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return jsonify({"error": "Ejecuta primero el Topic Model"}), 400

    topic_labels  = res.get("topic_labels", {})
    algorithm     = res.get("algorithm", "lda").upper()
    perplexity    = res.get("perplexity")
    cv_scores     = res.get("coherence_cv", [])
    cnpmi_scores  = res.get("coherence_cnpmi", [])
    active_steps  = S.get("active_steps", [])
    dataset_name  = S.get("dataset_name", "—")
    n_docs        = len(S.get("texts", []))

    # ── helpers ───────────────────────────────────────────────────────────────
    def safe_lbl(tid):
        return topic_labels.get(str(tid), "")

    def topic_title(tid, lang="es"):
        lbl = safe_lbl(tid)
        prefix = "Tópico" if lang == "es" else "Topic"
        return f"{prefix} {tid+1}" + (f" — {lbl}" if lbl else "")

    def make_wc_png(tp):
        words  = tp.get("words", [])
        scores = tp.get("weights", tp.get("scores", []))
        freq   = {w: float(s) for w, s in zip(words, scores)} if scores else {w: 1.0/(i+1) for i, w in enumerate(words)}
        wc = WordCloud(width=900, height=420, background_color="white",
                       max_words=40, colormap="viridis").generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        lbl = safe_lbl(tp["id"])
        ax.set_title(f"Tópico {tp['id']+1}" + (f" — {lbl}" if lbl else ""),
                     fontsize=13, pad=10, color="#1a1a2e")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def make_bar_chart():
        n   = len(topics)
        cols = min(n, 4); rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows))
        fig.patch.set_facecolor("#f8f8f8")
        axes_flat = list(np.array(axes).flatten()) if n > 1 else [axes]
        for idx, (ax, tp) in enumerate(zip(axes_flat, topics)):
            words   = tp["words"][:10]; weights = tp["weights"][:10]
            color   = PALETTE[idx % len(PALETTE)]
            ax.barh(list(reversed(words)), list(reversed(weights)),
                    color=color, edgecolor="none", alpha=0.85)
            ax.set_title(topic_title(tp["id"]), fontsize=11, fontweight="bold", color="#1a1a2e")
            ax.tick_params(axis="y", labelsize=9)
            ax.set_facecolor("#f8f8f8")
        for ax in axes_flat[n:]: ax.set_visible(False)
        plt.tight_layout(pad=2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def avg(arr):
        return round(sum(arr)/len(arr), 4) if arr else None

    def make_html(lang="es"):
        _ = {
            "es": {
                "title": "Informe de Topic Model",
                "dataset": "Dataset",
                "algorithm": "Algoritmo",
                "n_docs": "Documentos",
                "prepro": "Preprocesado",
                "n_topics": "Nº tópicos",
                "perplexity": "Perplejidad",
                "coherence_cv": "Coherencia C_V (media)",
                "coherence_cnpmi": "Coherencia C_NPMI (media)",
                "top_words": "Palabras clave",
                "wc_section": "Nubes de palabras",
                "bar_section": "Gráfica de palabras clave",
                "topics_section": "Resumen de tópicos",
                "label_col": "Etiqueta IA",
                "generated": "Generado con NLP Flow",
                "topic": "Tópico",
                "none": "(ninguno)",
            },
            "en": {
                "title": "Topic Model Report",
                "dataset": "Dataset",
                "algorithm": "Algorithm",
                "n_docs": "Documents",
                "prepro": "Preprocessing",
                "n_topics": "Nº topics",
                "perplexity": "Perplexity",
                "coherence_cv": "Coherence C_V (avg)",
                "coherence_cnpmi": "Coherence C_NPMI (avg)",
                "top_words": "Top words",
                "wc_section": "Word clouds",
                "bar_section": "Top-words chart",
                "topics_section": "Topics summary",
                "label_col": "AI label",
                "generated": "Generated with NLP Flow",
                "topic": "Topic",
                "none": "(none)",
            },
        }[lang]

        steps_str = ", ".join(active_steps) if active_steps else _.get("none")

        # embed images as base64
        def img_b64(png_bytes):
            return "data:image/png;base64," + base64.b64encode(png_bytes).decode()

        wc_imgs   = [(tp, img_b64(make_wc_png(tp))) for tp in topics]
        bar_bytes = make_bar_chart()
        bar_img   = img_b64(bar_bytes)

        topic_rows = ""
        for i, tp in enumerate(topics):
            lbl  = safe_lbl(tp["id"]) or "—"
            cv   = cv_scores[i] if i < len(cv_scores) else "—"
            cnpm = cnpmi_scores[i] if i < len(cnpmi_scores) else "—"
            words_str = ", ".join(tp["words"][:10])
            topic_rows += f"""
            <tr>
              <td><strong>{_['topic']} {tp['id']+1}</strong></td>
              <td>{_html.escape(lbl)}</td>
              <td>{words_str}</td>
              <td>{cv}</td>
              <td>{cnpm}</td>
            </tr>"""

        wc_cards = ""
        for tp, img in wc_imgs:
            title = _html.escape(topic_title(tp["id"], lang))
            wc_cards += f"""
            <div class="wc-card">
              <div class="wc-title">{title}</div>
              <img src="{img}" alt="{title}">
            </div>"""

        perp_str = str(round(perplexity, 2)) if perplexity is not None else "—"
        cv_avg   = avg(cv_scores) if cv_scores else "—"
        cnpm_avg = avg(cnpmi_scores) if cnpmi_scores else "—"

        return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_['title']} — {_html.escape(dataset_name)}</title>
<style>
  :root {{
    --bg:#f4f4f8; --card:#ffffff; --accent:#6c63ff;
    --ink:#1a1a2e; --sec:#4a4a6a; --border:#e0e0f0;
    --radius:12px; --shadow:0 2px 12px rgba(0,0,0,.08);
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--ink); padding:32px 16px; }}
  .container {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:2rem; color:var(--accent); margin-bottom:6px; }}
  h2 {{ font-size:1.25rem; color:var(--ink); margin:32px 0 14px; border-left:4px solid var(--accent); padding-left:10px; }}
  .subtitle {{ color:var(--sec); font-size:.95rem; margin-bottom:28px; }}
  .meta-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:14px; margin-bottom:8px; }}
  .meta-card {{ background:var(--card); border-radius:var(--radius); padding:16px 18px; box-shadow:var(--shadow); }}
  .meta-card .val {{ font-size:1.4rem; font-weight:700; color:var(--accent); }}
  .meta-card .lbl {{ font-size:.78rem; color:var(--sec); margin-top:3px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow); }}
  th {{ background:var(--accent); color:#fff; padding:10px 14px; text-align:left; font-size:.85rem; }}
  td {{ padding:9px 14px; border-bottom:1px solid var(--border); font-size:.875rem; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#f0efff; }}
  .wc-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:20px; }}
  .wc-card {{ background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden; }}
  .wc-card img {{ width:100%; display:block; }}
  .wc-title {{ padding:10px 14px; font-weight:600; font-size:.9rem; color:var(--ink); background:var(--bg); }}
  .bar-wrap img {{ width:100%; border-radius:var(--radius); box-shadow:var(--shadow); }}
  footer {{ margin-top:48px; text-align:center; color:var(--sec); font-size:.8rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>{_['title']}</h1>
  <p class="subtitle">{_html.escape(dataset_name)}</p>

  <div class="meta-grid">
    <div class="meta-card"><div class="val">{algorithm}</div><div class="lbl">{_['algorithm']}</div></div>
    <div class="meta-card"><div class="val">{len(topics)}</div><div class="lbl">{_['n_topics']}</div></div>
    <div class="meta-card"><div class="val">{n_docs}</div><div class="lbl">{_['n_docs']}</div></div>
    <div class="meta-card"><div class="val">{perp_str}</div><div class="lbl">{_['perplexity']}</div></div>
    <div class="meta-card"><div class="val">{cv_avg}</div><div class="lbl">{_['coherence_cv']}</div></div>
    <div class="meta-card"><div class="val">{cnpm_avg}</div><div class="lbl">{_['coherence_cnpmi']}</div></div>
    <div class="meta-card" style="grid-column:1/-1"><div class="val" style="font-size:1rem">{_html.escape(steps_str)}</div><div class="lbl">{_['prepro']}</div></div>
  </div>

  <h2>{_['topics_section']}</h2>
  <table>
    <thead><tr>
      <th>{_['topic']}</th><th>{_['label_col']}</th>
      <th>{_['top_words']}</th><th>C_V</th><th>C_NPMI</th>
    </tr></thead>
    <tbody>{topic_rows}</tbody>
  </table>

  <h2>{_['bar_section']}</h2>
  <div class="bar-wrap"><img src="{bar_img}" alt="bar chart"></div>

  <h2>{_['wc_section']}</h2>
  <div class="wc-grid">{wc_cards}</div>

  <footer>{_['generated']} · {dataset_name}</footer>
</div>
</body>
</html>"""

    # ── Build ZIP ─────────────────────────────────────────────────────────────
    try:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Word clouds
            for tp in topics:
                png = make_wc_png(tp)
                lbl = safe_lbl(tp["id"]).replace(" ", "_").replace("/", "-")
                fname = f"wordclouds/topico_{tp['id']+1:02d}" + (f"_{lbl}" if lbl else "") + ".png"
                zf.writestr(fname, png)

            # Bar chart
            zf.writestr("plots/topics_chart.png", make_bar_chart())

            # HTML reports
            zf.writestr("informe_es.html", make_html("es").encode("utf-8"))
            zf.writestr("informe_en.html", make_html("en").encode("utf-8"))

            # JSON config
            config = {
                "dataset":      dataset_name,
                "algorithm":    algorithm,
                "n_topics":     len(topics),
                "n_documents":  n_docs,
                "active_steps": active_steps,
                "perplexity":   perplexity,
                "coherence_cv_avg":    avg(cv_scores),
                "coherence_cnpmi_avg": avg(cnpmi_scores),
                "topics": [
                    {
                        "id":      tp["id"],
                        "label":   safe_lbl(tp["id"]),
                        "words":   tp["words"][:15],
                        "weights": tp.get("weights", [])[:15],
                        "coherence_cv":    cv_scores[i] if i < len(cv_scores) else None,
                        "coherence_cnpmi": cnpmi_scores[i] if i < len(cnpmi_scores) else None,
                    }
                    for i, tp in enumerate(topics)
                ]
            }
            zf.writestr("topic_config.json", json.dumps(config, ensure_ascii=False, indent=2))

        zip_buf.seek(0)
        zip_bytes = zip_buf.read()
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    # ── Save via native dialog or fallback download ───────────────────────────
    try:
        safe_name = re.sub(r"[^\w\-]", "_", dataset_name or "topicos")
        default_name = f"topic_report_{safe_name}.zip"
        path = _native_save_dialog(
            default_name,
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
    except Exception:
        # Fallback: send as download
        safe_name = re.sub(r"[^\w\-]", "_", dataset_name or "topicos")
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"topic_report_{safe_name}.zip"
        )


@app.route("/api/export_topic_bundle_download", methods=["GET"])
def export_topic_bundle_download():
    """Fallback GET: streams the topic bundle ZIP directly as a download."""
    return export_topic_bundle()


# ── Topic diversity metric ────────────────────────────────────────────────────
def _topic_diversity(topics, top_n=10):
    """
    Topic Diversity (TD): fraction of unique words across all top-N words of all topics.
    TD = |unique top-N words across all topics| / (N * num_topics)
    Range [0, 1]. 1 = all topics use completely different vocabulary.
    """
    all_words  = [w for tp in topics for w in tp.get("words", [])[:top_n]]
    if not all_words:
        return 0.0
    unique = len(set(all_words))
    total  = len(all_words)
    return round(unique / total, 4)


# ── Shared HTML builder for topic report ─────────────────────────────────────
def _build_topic_html(lang="es"):
    import html as _html
    res    = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return None, "Sin resultados de Topic Model"

    topic_labels  = res.get("topic_labels", {})
    algorithm     = res.get("algorithm", "lda").upper()
    perplexity    = res.get("perplexity")
    cv_scores     = res.get("coherence_cv", [])
    cnpmi_scores  = res.get("coherence_cnpmi", [])
    active_steps  = S.get("active_steps", [])
    dataset_name  = S.get("dataset_name", "—")
    n_docs        = len(S.get("texts", []))
    td            = _topic_diversity(topics)

    def safe_lbl(tid):
        return topic_labels.get(str(tid), "")

    def topic_title_h(tid):
        lbl    = safe_lbl(tid)
        prefix = "Tópico" if lang == "es" else "Topic"
        return f"{prefix} {tid+1}" + (f" — {lbl}" if lbl else "")

    def make_wc_b64(tp):
        words  = tp.get("words", [])
        scores = tp.get("weights", tp.get("scores", []))
        freq   = {w: float(s) for w, s in zip(words, scores)} if scores else {w: 1.0/(i+1) for i, w in enumerate(words)}
        wc  = WordCloud(width=900, height=420, background_color="white",
                        max_words=40, colormap="viridis").generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        lbl = safe_lbl(tp["id"])
        ax.set_title(f"{'Tópico' if lang=='es' else 'Topic'} {tp['id']+1}" + (f" — {lbl}" if lbl else ""),
                     fontsize=13, pad=10)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    def make_bar_b64():
        n    = len(topics)
        cols = min(n, 4); rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows))
        fig.patch.set_facecolor("#f8f8f8")
        axes_flat = list(np.array(axes).flatten()) if n > 1 else [axes]
        for idx, (ax, tp) in enumerate(zip(axes_flat, topics)):
            words   = tp["words"][:10]; weights = tp["weights"][:10]
            ax.barh(list(reversed(words)), list(reversed(weights)),
                    color=PALETTE[idx % len(PALETTE)], edgecolor="none", alpha=0.85)
            ax.set_title(topic_title_h(tp["id"]), fontsize=11, fontweight="bold")
            ax.tick_params(axis="y", labelsize=9)
            ax.set_facecolor("#f8f8f8")
        for ax in axes_flat[n:]: ax.set_visible(False)
        plt.tight_layout(pad=2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    def avg(arr):
        return round(sum(arr)/len(arr), 4) if arr else None

    _ = {
        "es": dict(
            title="Informe de Topic Model", dataset="Dataset", algorithm="Algoritmo",
            n_docs="Documentos", prepro="Preprocesado aplicado", n_topics="Nº tópicos",
            perplexity="Perplejidad ↓", coherence_cv="C_V media ↑",
            coherence_cnpmi="C_NPMI media ↑", topic_div="Topic Diversity ↑",
            top_words="Palabras clave", wc_section="Nubes de palabras",
            bar_section="Palabras clave por tópico", topics_section="Resumen de tópicos",
            label_col="Etiqueta IA", generated="Generado con NLP Flow",
            topic="Tópico", none="(ninguno)",
            td_explain="Fracción de palabras únicas entre todos los tópicos (1=máxima diversidad)",
        ),
        "en": dict(
            title="Topic Model Report", dataset="Dataset", algorithm="Algorithm",
            n_docs="Documents", prepro="Applied preprocessing", n_topics="Nº topics",
            perplexity="Perplexity ↓", coherence_cv="C_V avg ↑",
            coherence_cnpmi="C_NPMI avg ↑", topic_div="Topic Diversity ↑",
            top_words="Top words", wc_section="Word clouds",
            bar_section="Top words per topic", topics_section="Topics summary",
            label_col="AI label", generated="Generated with NLP Flow",
            topic="Topic", none="(none)",
            td_explain="Fraction of unique words across all topics (1=maximum diversity)",
        ),
    }[lang]

    steps_str = ", ".join(active_steps) if active_steps else _.get("none")
    perp_str  = str(round(perplexity, 2)) if perplexity is not None else "—"
    cv_avg    = str(avg(cv_scores))    if cv_scores    else "—"
    cnpm_avg  = str(avg(cnpmi_scores)) if cnpmi_scores else "—"

    topic_rows = ""
    for i, tp in enumerate(topics):
        lbl       = safe_lbl(tp["id"]) or "—"
        cv_val    = cv_scores[i]    if i < len(cv_scores)    else "—"
        cnpm_val  = cnpmi_scores[i] if i < len(cnpmi_scores) else "—"
        words_str = ", ".join(tp["words"][:10])
        topic_rows += (
            f"<tr><td><strong>{_['topic']} {tp['id']+1}</strong></td>"
            f"<td>{_html.escape(lbl)}</td><td>{words_str}</td>"
            f"<td>{cv_val}</td><td>{cnpm_val}</td></tr>"
        )

    wc_cards  = "".join(
        f'<div class="wc-card"><div class="wc-title">{_html.escape(topic_title_h(tp["id"]))}</div>'
        f'<img src="{make_wc_b64(tp)}" alt="{_html.escape(topic_title_h(tp["id"]))}"></div>'
        for tp in topics
    )
    bar_img   = make_bar_b64()

    html_out = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_['title']} — {_html.escape(dataset_name)}</title>
<style>
  :root{{--bg:#f4f4f8;--card:#fff;--accent:#6c63ff;--ink:#1a1a2e;--sec:#4a4a6a;--border:#e0e0f0;--r:12px;--sh:0 2px 12px rgba(0,0,0,.08);}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--ink);padding:32px 16px;}}
  .wrap{{max-width:980px;margin:0 auto;}}
  h1{{font-size:2rem;color:var(--accent);margin-bottom:4px;}}
  h2{{font-size:1.2rem;color:var(--ink);margin:32px 0 12px;border-left:4px solid var(--accent);padding-left:10px;}}
  .sub{{color:var(--sec);font-size:.93rem;margin-bottom:26px;}}
  .mgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:6px;}}
  .mcard{{background:var(--card);border-radius:var(--r);padding:14px 16px;box-shadow:var(--sh);}}
  .mcard .val{{font-size:1.35rem;font-weight:700;color:var(--accent);}}
  .mcard .lbl{{font-size:.75rem;color:var(--sec);margin-top:3px;}}
  .mcard .tip{{font-size:.68rem;color:var(--sec);margin-top:4px;font-style:italic;}}
  table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:var(--r);overflow:hidden;box-shadow:var(--sh);}}
  th{{background:var(--accent);color:#fff;padding:9px 13px;text-align:left;font-size:.83rem;}}
  td{{padding:8px 13px;border-bottom:1px solid var(--border);font-size:.85rem;vertical-align:top;}}
  tr:last-child td{{border-bottom:none;}}tr:hover td{{background:#f0efff;}}
  .wc-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px;}}
  .wc-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);overflow:hidden;}}
  .wc-card img{{width:100%;display:block;}}.wc-title{{padding:9px 13px;font-weight:600;font-size:.88rem;}}
  .bar-wrap img{{width:100%;border-radius:var(--r);box-shadow:var(--sh);}}
  footer{{margin-top:48px;text-align:center;color:var(--sec);font-size:.78rem;}}
</style>
</head>
<body>
<div class="wrap">
  <h1>{_['title']}</h1>
  <p class="sub">{_html.escape(dataset_name)}</p>

  <div class="mgrid">
    <div class="mcard"><div class="val">{algorithm}</div><div class="lbl">{_['algorithm']}</div></div>
    <div class="mcard"><div class="val">{len(topics)}</div><div class="lbl">{_['n_topics']}</div></div>
    <div class="mcard"><div class="val">{n_docs}</div><div class="lbl">{_['n_docs']}</div></div>
    <div class="mcard"><div class="val">{perp_str}</div><div class="lbl">{_['perplexity']}</div></div>
    <div class="mcard"><div class="val">{cv_avg}</div><div class="lbl">{_['coherence_cv']}</div></div>
    <div class="mcard"><div class="val">{cnpm_avg}</div><div class="lbl">{_['coherence_cnpmi']}</div></div>
    <div class="mcard"><div class="val">{td}</div><div class="lbl">{_['topic_div']}</div><div class="tip">{_['td_explain']}</div></div>
    <div class="mcard" style="grid-column:1/-1"><div class="val" style="font-size:.95rem">{_html.escape(steps_str)}</div><div class="lbl">{_['prepro']}</div></div>
  </div>

  <h2>{_['topics_section']}</h2>
  <table>
    <thead><tr><th>{_['topic']}</th><th>{_['label_col']}</th><th>{_['top_words']}</th><th>C_V</th><th>C_NPMI</th></tr></thead>
    <tbody>{topic_rows}</tbody>
  </table>

  <h2>{_['bar_section']}</h2>
  <div class="bar-wrap"><img src="{bar_img}" alt="bar chart"></div>

  <h2>{_['wc_section']}</h2>
  <div class="wc-grid">{wc_cards}</div>

  <footer>{_['generated']} · {_html.escape(dataset_name)}</footer>
</div>
</body>
</html>"""
    return html_out, None


@app.route("/api/export_topic_selection", methods=["POST"])
def export_topic_selection():
    """
    Recibe items=["tm__bundle_es","tm__wordclouds",...].
    Si es un único ítem, devuelve el fichero directamente.
    Si son varios, los empaqueta en un ZIP y lo devuelve.
    """
    body  = request.get_json(force=True, silent=True) or {}
    items = body.get("items", [])
    print(f"[SAVE] export_topic_selection llamado → items={items}", flush=True)
    if not items:
        return jsonify({"error": "Sin ítems seleccionados"}), 400

    res    = S.get("results", {})
    topics = res.get("topics", [])
    print(f"[SAVE] topics en S['results']: {len(topics)}", flush=True)
    if not topics:
        return jsonify({"error": "Ejecuta primero el Topic Model"}), 400

    files = []   # list of (filename, bytes)

    for item in items:
        try:
            if item in ("tm__bundle_es", "tm__bundle_en"):
                lg        = "es" if item == "tm__bundle_es" else "en"
                html_out, err = _build_topic_html(lg)
                if err: continue
                files.append((f"informe_topicos_{lg}.html", html_out.encode("utf-8")))

            elif item == "tm__wordclouds":
                topic_labels = res.get("topic_labels", {})
                for tp in topics:
                    words  = tp.get("words", [])
                    scores = tp.get("weights", tp.get("scores", []))
                    freq   = {w: float(s) for w, s in zip(words, scores)} if scores else {w: 1.0/(i+1) for i, w in enumerate(words)}
                    wc  = WordCloud(width=900, height=420, background_color="white",
                                    max_words=40, colormap="viridis").generate_from_frequencies(freq)
                    fig, ax = plt.subplots(figsize=(9, 4.2))
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    lbl = topic_labels.get(str(tp["id"]), "")
                    ax.set_title(f"Tópico {tp['id']+1}" + (f" — {lbl}" if lbl else ""), fontsize=13, pad=10)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    safe = lbl.replace(" ", "_").replace("/", "-") if lbl else ""
                    fname_wc = "wordclouds/" + f"topico_{tp['id']+1:02d}" + (f"_{safe}" if safe else "") + ".png"
                    files.append((fname_wc, buf.read()))

            elif item == "tm__chart":
                n    = len(topics)
                cols = min(n, 4); rows = max(1, (n + cols - 1) // cols)
                fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows))
                fig.patch.set_facecolor("#f8f8f8")
                axes_flat = list(np.array(axes).flatten()) if n > 1 else [axes]
                topic_labels = res.get("topic_labels", {})
                for idx, (ax, tp) in enumerate(zip(axes_flat, topics)):
                    words   = tp["words"][:10]; weights = tp["weights"][:10]
                    ax.barh(list(reversed(words)), list(reversed(weights)),
                            color=PALETTE[idx % len(PALETTE)], edgecolor="none", alpha=0.85)
                    lbl = topic_labels.get(str(tp["id"]), "")
                    ax.set_title(f"Tópico {tp['id']+1}" + (f" — {lbl}" if lbl else ""), fontsize=11, fontweight="bold")
                    ax.tick_params(axis="y", labelsize=9)
                for ax in axes_flat[n:]: ax.set_visible(False)
                plt.tight_layout(pad=2)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig); buf.seek(0)
                files.append(("topics_chart.png", buf.read()))

            elif item == "tm__json":
                cv_scores    = res.get("coherence_cv", [])
                cnpmi_scores = res.get("coherence_cnpmi", [])
                topic_labels = res.get("topic_labels", {})
                def _avg(arr): return round(sum(arr)/len(arr), 4) if arr else None
                config = {
                    "dataset": S.get("dataset_name", "—"),
                    "algorithm": res.get("algorithm", "lda").upper(),
                    "n_topics": len(topics), "n_documents": len(S.get("texts", [])),
                    "active_steps": S.get("active_steps", []),
                    "perplexity": res.get("perplexity"),
                    "coherence_cv_avg": _avg(cv_scores),
                    "coherence_cnpmi_avg": _avg(cnpmi_scores),
                    "topic_diversity": _topic_diversity(topics),
                    "topics": [
                        {"id": tp["id"], "label": topic_labels.get(str(tp["id"]), ""),
                         "words": tp["words"][:15], "weights": tp.get("weights", [])[:15],
                         "coherence_cv": cv_scores[ix] if ix < len(cv_scores) else None,
                         "coherence_cnpmi": cnpmi_scores[ix] if ix < len(cnpmi_scores) else None}
                        for ix, tp in enumerate(topics)
                    ]
                }
                files.append(("topic_config.json", json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")))
        except Exception as e:
            import traceback
            print(f"[SAVE] ERROR en item={item}: {e}", flush=True)
            traceback.print_exc()
            continue

    print(f"[SAVE] ficheros generados: {[f[0] for f in files]}", flush=True)
    if not files:
        return jsonify({"error": "No se pudo generar ningún fichero"}), 500

    safe_ds = re.sub(r"[^\w\-]", "_", S.get("dataset_name", "topicos"))

    # Un fichero → diálogo nativo para ese fichero (solo si es un único archivo sin subcarpeta)
    if len(files) == 1 and "/" not in files[0][0]:
        fname, data = files[0]
        ext = fname.rsplit(".", 1)[-1].lower()
        type_map = {
            "html": ("HTML File (*.html)", [("HTML File", "*.html"), ("All files", "*.*")]),
            "json": ("JSON File (*.json)", [("JSON File", "*.json"), ("All files", "*.*")]),
            "png":  ("PNG Image (*.png)",  [("PNG Image", "*.png"),  ("All files", "*.*")]),
            "zip":  ("ZIP Archive (*.zip)",[("ZIP Archive","*.zip"), ("All files", "*.*")]),
        }
        ft_wv, ft_tk = type_map.get(ext, ("All files (*.*)", [("All files", "*.*")]))
        print(f"[SAVE] 1 fichero → abriendo diálogo para {fname!r}", flush=True)
        try:
            path = _native_save_dialog(fname, ft_wv, ft_tk)
            print(f"[SAVE] diálogo devuelve path={path!r}", flush=True)
            if not path:
                print("[SAVE] usuario canceló o path vacío", flush=True)
                return jsonify({"cancelled": True})
            if not path.lower().endswith("." + ext):
                path += "." + ext
            with open(path, "wb") as f:
                f.write(data)
            print(f"[SAVE] fichero escrito OK en {path!r}", flush=True)
            return jsonify({"path": path, "name": fname})
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # Varios ficheros → ZIP con diálogo nativo
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files:
            zf.writestr(fname, data)
    zip_bytes = zip_buf.getvalue()
    zip_name  = f"topic_export_{safe_ds}.zip"
    print(f"[SAVE] {len(files)} ficheros → ZIP {zip_name!r} ({len(zip_bytes)} bytes) → abriendo diálogo", flush=True)
    try:
        path = _native_save_dialog(
            zip_name,
            "ZIP Archive (*.zip)",
            [("ZIP Archive", "*.zip"), ("All files", "*.*")]
        )
        print(f"[SAVE] diálogo devuelve path={path!r}", flush=True)
        if not path:
            print("[SAVE] usuario canceló o path vacío", flush=True)
            return jsonify({"cancelled": True})
        if not path.lower().endswith(".zip"):
            path += ".zip"
        with open(path, "wb") as f:
            f.write(zip_bytes)
        print(f"[SAVE] ZIP escrito OK en {path!r}", flush=True)
        return jsonify({"path": path, "name": zip_name})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_topic_html", methods=["POST"])
def export_topic_html():
    body = request.get_json(force=True, silent=True) or {}
    lang = body.get("lang", "es")
    html_out, err = _build_topic_html(lang)
    if err:
        return jsonify({"error": err}), 400
    fname = f"informe_topicos_{lang}.html"
    return send_file(
        io.BytesIO(html_out.encode("utf-8")),
        mimetype="text/html",
        as_attachment=True,
        download_name=fname
    )


@app.route("/api/export_topic_wc_zip", methods=["POST"])
def export_topic_wc_zip():
    res    = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return jsonify({"error": "Sin resultados de Topic Model"}), 400
    topic_labels = res.get("topic_labels", {})
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for tp in topics:
            words  = tp.get("words", [])
            scores = tp.get("weights", tp.get("scores", []))
            freq   = {w: float(s) for w, s in zip(words, scores)} if scores else {w: 1.0/(i+1) for i, w in enumerate(words)}
            wc  = WordCloud(width=900, height=420, background_color="white",
                            max_words=40, colormap="viridis").generate_from_frequencies(freq)
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            lbl = topic_labels.get(str(tp["id"]), "")
            ax.set_title(f"Tópico {tp['id']+1}" + (f" — {lbl}" if lbl else ""), fontsize=13, pad=10)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig); buf.seek(0)
            safe = lbl.replace(" ", "_").replace("/", "-") if lbl else ""
            fname = f"topico_{tp['id']+1:02d}" + (f"_{safe}" if safe else "") + ".png"
            zf.writestr(fname, buf.read())
    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip",
                     as_attachment=True, download_name="wordclouds_topicos.zip")


@app.route("/api/export_topic_json", methods=["POST"])
def export_topic_json():
    res    = S.get("results", {})
    topics = res.get("topics", [])
    if not topics:
        return jsonify({"error": "Sin resultados de Topic Model"}), 400
    cv_scores    = res.get("coherence_cv", [])
    cnpmi_scores = res.get("coherence_cnpmi", [])
    topic_labels = res.get("topic_labels", {})
    def avg(arr): return round(sum(arr)/len(arr), 4) if arr else None
    config = {
        "dataset":             S.get("dataset_name", "—"),
        "algorithm":           res.get("algorithm", "lda").upper(),
        "n_topics":            len(topics),
        "n_documents":         len(S.get("texts", [])),
        "active_steps":        S.get("active_steps", []),
        "perplexity":          res.get("perplexity"),
        "coherence_cv_avg":    avg(cv_scores),
        "coherence_cnpmi_avg": avg(cnpmi_scores),
        "topic_diversity":     _topic_diversity(topics),
        "topics": [
            {
                "id":              tp["id"],
                "label":           topic_labels.get(str(tp["id"]), ""),
                "words":           tp["words"][:15],
                "weights":         tp.get("weights", [])[:15],
                "coherence_cv":    cv_scores[i]    if i < len(cv_scores)    else None,
                "coherence_cnpmi": cnpmi_scores[i] if i < len(cnpmi_scores) else None,
            }
            for i, tp in enumerate(topics)
        ]
    }
    return send_file(
        io.BytesIO(json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")),
        mimetype="application/json",
        as_attachment=True,
        download_name="topic_config.json"
    )


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

        elif endpoint == "/api/rag_export_html":
            extra = body.get("extra", {})
            with app.test_client() as _tc:
                import json as _js2
                _r = _tc.post(
                    "/api/rag_export_html",
                    data=_js2.dumps(extra),
                    content_type="application/json"
                )
                if _r.status_code != 200:
                    return jsonify({"error": "Error generando informe RAG"}), 500
                content_bytes = _r.data
            _en2         = extra.get("lang", "es") == "en"
            default_name = "rag_report.html" if _en2 else "informe_rag.html"
            file_types   = ("HTML File (*.html)", "All files (*.*)")

        elif endpoint == "/api/export_topic_html":
            extra      = body.get("extra", {})
            _lang      = extra.get("lang", "es")
            html_out, _err = _build_topic_html(_lang)
            if _err:
                return jsonify({"error": _err}), 400
            content_bytes = html_out.encode("utf-8")
            default_name  = f"informe_topicos_{_lang}.html"
            file_types    = ("HTML File (*.html)", "All files (*.*)")

        elif endpoint == "/api/export_topic_wc_zip":
            with app.test_client() as _tc:
                _r = _tc.post("/api/export_topic_wc_zip", content_type="application/json", data="{}")
                if _r.status_code != 200:
                    return jsonify({"error": "Error generando wordclouds"}), 500
                content_bytes = _r.data
            default_name = "wordclouds_topicos.zip"
            file_types   = ("ZIP Archive (*.zip)", "All files (*.*)")

        elif endpoint == "/api/export_topic_json":
            with app.test_client() as _tc:
                _r = _tc.post("/api/export_topic_json", content_type="application/json", data="{}")
                if _r.status_code != 200:
                    return jsonify({"error": "Error generando JSON"}), 500
                content_bytes = _r.data
            default_name = "topic_config.json"
            file_types   = ("JSON File (*.json)", "All files (*.*)")

        elif endpoint == "/api/cnn_report":
            extra       = body.get("extra", {})
            _node_id    = str(extra.get("node", "cnn_default"))
            _lang       = extra.get("lang", "es")
            _img_hist   = extra.get("img_history", [])
            import json as _json
            fake_body = {"node": _node_id, "lang": _lang, "img_history": _img_hist}
            slot      = _cnn_slot(_node_id)
            runs      = slot.get("runs", [])
            if not runs:
                return jsonify({"error": "No runs to report"}), 400
            with app.test_client() as _tc:
                _r = _tc.post("/api/cnn_report",
                              content_type="application/json",
                              data=_json.dumps(fake_body))
                if _r.status_code != 200:
                    return jsonify({"error": "Error generating CNN report"}), 500
                content_bytes = _r.data
            _fname = f"cnn_report_{_lang}.html"
            default_name  = _fname
            file_types    = ("HTML File (*.html)", "All files (*.*)")

        elif endpoint == "/api/ae_report":
            import json as _json
            extra     = body.get("extra", {})
            _node_id  = str(extra.get("node", "ae_default"))
            _lang     = extra.get("lang", "es")
            _uimgs    = extra.get("user_imgs", [])
            _preinfo  = extra.get("pre_info", {})
            if not _ae_slot(_node_id).get("runs"):
                return jsonify({"error": "No AE runs to report"}), 400
            fake_body = {"node": _node_id, "lang": _lang, "user_imgs": _uimgs, "pre_info": _preinfo}
            with app.test_client() as _tc:
                _r = _tc.post("/api/ae_report",
                              content_type="application/json",
                              data=_json.dumps(fake_body))
                if _r.status_code != 200:
                    return jsonify({"error": "Error generating AE report"}), 500
                content_bytes = _r.data
            _fname       = f"ae_report_{_lang}.html"
            default_name = _fname
            file_types   = ("HTML File (*.html)", "All files (*.*)")

        else:
            return jsonify({"error": "Unknown endpoint"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Recalculate ext in case default_name was overwritten inside an elif branch
    ext = os.path.splitext(default_name)[1] or ext

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
_mtrain_cancel = threading.Event()  # set() to request cancellation

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

    # Detect problem type — always driven by the target, regardless of method
    _tgt_vals = [r.get(target_col, "") for r in snap_rows if not _is_missing(r.get(target_col, ""))]
    _tgt_type = _col_type(_tgt_vals)
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
                    mdl = LogisticRegression(C=1.0/max(_alpha,1e-6), max_iter=5000, random_state=42, solver="saga")
                else:
                    mdl = LogisticRegression(max_iter=5000, random_state=42)
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

    _mtrain_cancel.clear()
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

@app.route("/api/model_train_cancel", methods=["POST"])
def model_train_cancel():
    _mtrain_cancel.set()
    with _mtrain_lock:
        _mtrain_progress.append({"pct": 100, "msg": "__done__:cancelled"})
    return jsonify({"ok": True})


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

    # Move the classifier/regressor into _NODE_MODELS — never the scaler
    if node_id:
        _mdl = result.get("model_obj") or result.get("sklearn_model")
        if _mdl is not None:
            _NODE_MODELS[node_id] = _mdl

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

    # Ensure the classifier/regressor (never the scaler) is in _NODE_MODELS
    if node_id:
        _mdl_obj = result.get("model_obj") or result.get("sklearn_model")
        if _mdl_obj is not None:
            _NODE_MODELS[node_id] = _mdl_obj

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
        classes  = result.get("classes") or sorted(set(str(v) for v in np.unique(np.concatenate([yte, yte_pred])).tolist()))
        avg = "binary" if len(classes) == 2 else "macro"
        labels_idx = list(range(len(classes))) if classes else None

        cm = confusion_matrix(yte, yte_pred, labels=labels_idx).tolist()

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
        # ROC curve — look in _NODE_MODELS first, then fall back to model_obj in _MODEL_STORE
        roc_img_cls = None
        live_mdl = (
            _NODE_MODELS.get(node_id)
            or _NODE_MODELS.get(int(node_id) if node_id.isdigit() else None)
            or result.get("model_obj")
        )
        print(f"[model_evaluate/ROC] node_id={node_id!r} _NODE_MODELS keys={list(_NODE_MODELS.keys())} live_mdl={type(live_mdl).__name__ if live_mdl else None} _Xte={'yes' if _Xte else 'None'}")
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


# ── RAG Pipeline: state ───────────────────────────────────────────────────────
# Keyed by node_id of the embed block so multiple embed nodes can coexist.
_RAG: dict = {}   # { embed_node_id: { chunks, embeddings, meta, source_node_id, chunk_cfg, embed_cfg } }

_EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_BATCH      = 64    # increased from 32 — HF handles up to 64 per request fine
_EMBED_MAX_CHUNKS = 5000  # default cap to avoid runaway embed jobs

_rag_thread  = None   # single background thread for embedding

try:
    from huggingface_hub import InferenceClient as _HFClient
    _HF_AVAILABLE = True
except ImportError:
    _HFClient     = None
    _HF_AVAILABLE = False

def _rag_slot(node_id: str) -> dict:
    if node_id not in _RAG:
        _RAG[node_id] = {}
    return _RAG[node_id]


# ── RAG helpers ───────────────────────────────────────────────────────────────

def _cosine_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity of query_vec (dim,) against every row in matrix (N x dim)."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normed = matrix / norms
    return (normed @ q).astype(float)


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-level splitter. chunk_size and overlap are measured in words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    stride = max(1, chunk_size - overlap)
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        piece = " ".join(words[start:end])
        if piece.strip():
            chunks.append(piece)
        if end >= len(words):
            break
        start += stride
    return chunks


# ── /api/rag_chunk ────────────────────────────────────────────────────────────
@app.route("/api/rag_chunk", methods=["POST"])
def rag_chunk():
    """
    Split the text column (or processed_texts) into chunks.
    Body: { node_id, source_node_id, text_col, chunk_size, overlap, use_processed }
    use_processed=true → use S["processed_texts"] instead of raw_rows column.
    Returns: { n_chunks, n_docs, avg_chunk_len, examples, text_col, use_processed }
    """
    body          = request.get_json(force=True, silent=True) or {}
    node_id       = str(body.get("node_id", ""))
    src_id        = str(body.get("source_node_id", ""))
    text_col      = body.get("text_col", "")
    chunk_size    = int(body.get("chunk_size", 100))   # now in words
    overlap       = int(body.get("overlap", 0))          # now in words
    use_processed = bool(body.get("use_processed", False))
    chunk_mode    = body.get("chunk_mode", "auto")   # "auto" | "one_per_doc"

    if chunk_mode == "auto" and overlap >= chunk_size:
        return jsonify({"error": "El solapamiento debe ser menor que el tamaño del chunk."}), 400

    # ── Resolve texts ────────────────────────────────────────────────────────
    if use_processed and S.get("processed_texts"):
        # Use preprocessed NLP texts — these are already cleaned/tokenized
        texts_list   = [t for t in S["processed_texts"] if t and t.strip()]
        text_col_out = "(preprocesado NLP)"
    else:
        slot = _get_data(src_id) if src_id else S
        rows = slot.get("raw_rows", [])
        if not rows:
            return jsonify({"error": "No hay datos cargados en el bloque origen."}), 400
        if not text_col or text_col not in rows[0]:
            return jsonify({"error": f"Columna '{text_col}' no encontrada."}), 400
        texts_list   = [str(row.get(text_col, "") or "").strip() for row in rows]
        text_col_out = text_col

    if not texts_list:
        return jsonify({"error": "No hay textos para dividir en chunks."}), 400

    # ── Chunk ────────────────────────────────────────────────────────────────
    all_chunks: list = []
    for doc_idx, text in enumerate(texts_list):
        if chunk_mode == "one_per_doc":
            # Each document becomes exactly one chunk
            if text.strip():
                all_chunks.append({"doc_idx": doc_idx, "chunk_idx": 0, "text": text})
        else:
            for ci, piece in enumerate(_split_into_chunks(text, chunk_size, overlap)):
                all_chunks.append({"doc_idx": doc_idx, "chunk_idx": ci, "text": piece})

    if not all_chunks:
        return jsonify({"error": "El chunking no produjo ningún fragmento."}), 400

    avg_len = sum(len(c["text"].split()) for c in all_chunks) / len(all_chunks)

    # ── Persist ──────────────────────────────────────────────────────────────
    slot_rag = _rag_slot(node_id)
    slot_rag["chunks"]        = all_chunks
    slot_rag["chunk_cfg"]     = {"chunk_size": chunk_size, "overlap": overlap,
                                  "text_col": text_col_out, "use_processed": use_processed,
                                  "chunk_mode": chunk_mode}
    slot_rag["source_node"]   = src_id
    slot_rag["embeddings"]    = None   # reset embeddings when chunks change

    examples = all_chunks[:3] + ([all_chunks[-1]] if len(all_chunks) > 3 else [])

    return jsonify({
        "n_chunks":      len(all_chunks),
        "n_docs":        len(texts_list),
        "avg_chunk_len": round(avg_len),
        "examples":      examples,
        "text_col":      text_col_out,
        "use_processed": use_processed,
    })


# ── /api/rag_embed  (background + SSE) ───────────────────────────────────────
_rag_progress: list  = []
_rag_embed_thread    = None
_rag_cancel_flag: list = [False]   # [0] = True means "stop after current batch"

def _push_rag(pct, msg):
    _rag_progress.append({"pct": pct, "msg": msg})

@app.route("/api/rag_embed", methods=["POST"])
def rag_embed():
    """
    Generate embeddings for chunks using HF InferenceClient.
    Body: { node_id, max_chunks? }
    max_chunks caps the number of chunks to embed (default _EMBED_MAX_CHUNKS).
    Starts a background thread; progress polled via /api/rag_embed_poll.
    """
    global _rag_embed_thread
    body       = request.get_json(force=True, silent=True) or {}
    node_id    = str(body.get("node_id", ""))
    max_chunks = int(body.get("max_chunks", _EMBED_MAX_CHUNKS))

    slot = _rag_slot(node_id)
    chunks = slot.get("chunks")
    if not chunks:
        return jsonify({"error": "Ejecuta primero el bloque Chunking."}), 400

    if _rag_embed_thread and _rag_embed_thread.is_alive():
        return jsonify({"error": "Embeddings ya en curso."}), 429

    # Cap chunks — sample evenly across the corpus to keep representativeness
    if len(chunks) > max_chunks:
        step = len(chunks) / max_chunks
        chunks_to_embed = [chunks[int(i * step)] for i in range(max_chunks)]
    else:
        chunks_to_embed = chunks

    _rag_progress.clear()
    _rag_cancel_flag[0] = False

    def _worker():
        try:
            if not _HF_AVAILABLE:
                _push_rag(0, "huggingface_hub no está instalado.")
                _push_rag(-1, "ERROR")
                return

            client = _HFClient()

            texts = [c["text"] for c in chunks_to_embed]
            total = len(texts)
            all_embs = []
            processed = 0
            _push_rag(1, f"Iniciando — {total} chunks…")

            for i in range(0, total, _EMBED_BATCH):
                if _rag_cancel_flag[0]:
                    _push_rag(-2, f"Cancelado tras {processed}/{total} chunks")
                    return
                batch = texts[i: i + _EMBED_BATCH]
                emb   = client.feature_extraction(batch, model=_EMBED_MODEL)
                all_embs.append(np.asarray(emb, dtype=np.float32))
                processed += len(batch)
                pct = int(processed / total * 95) + 2
                _push_rag(pct, f"Batch {i // _EMBED_BATCH + 1} — {processed}/{total} chunks")

            matrix = np.vstack(all_embs)
            slot["embeddings"]      = matrix
            slot["embedded_chunks"] = chunks_to_embed
            slot["embed_cfg"]       = {
                "model": _EMBED_MODEL, "dim": matrix.shape[1],
                "n": matrix.shape[0], "capped": len(chunks) > max_chunks,
                "total_chunks": len(chunks),
            }
            _push_rag(100, f"Listo — {matrix.shape[0]} embeddings de dim {matrix.shape[1]} ✓")

        except Exception as e:
            _push_rag(-1, f"ERROR: {e}")

    _rag_embed_thread = threading.Thread(target=_worker, daemon=True)
    _rag_embed_thread.start()
    return jsonify({"ok": True, "started": True})


@app.route("/api/rag_embed_progress")
def rag_embed_progress():
    """SSE stream of { pct, msg } events."""
    def gen():
        sent = 0
        while True:
            while sent < len(_rag_progress):
                ev = _rag_progress[sent]
                yield f"data: {json.dumps(ev)}\n\n"
                sent += 1
                if ev.get("pct") in (100, -1):
                    return
            time.sleep(0.4)
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/rag_embed_poll")
def rag_embed_poll():
    """Polling fallback — returns all progress events and running status."""
    running = _rag_embed_thread is not None and _rag_embed_thread.is_alive()
    return jsonify({"events": list(_rag_progress), "running": running})

@app.route("/api/rag_embed_cancel", methods=["POST"])
def rag_embed_cancel():
    """Signal the embed worker to stop after the current batch."""
    _rag_cancel_flag[0] = True
    return jsonify({"ok": True, "cancelled": True})


# ── /api/rag_status ───────────────────────────────────────────────────────────
@app.route("/api/rag_status")
def rag_status():
    """Return summary of what's been computed for a node."""
    node_id = request.args.get("node_id", "")
    slot    = _RAG.get(node_id, {})
    chunks  = slot.get("chunks")
    embs    = slot.get("embeddings")
    return jsonify({
        "has_chunks":    chunks is not None,
        "n_chunks":      len(chunks) if chunks else 0,
        "has_embeddings": embs is not None,
        "n_embeddings":  int(embs.shape[0]) if embs is not None else 0,
        "embed_cfg":     slot.get("embed_cfg", {}),
        "chunk_cfg":     slot.get("chunk_cfg", {}),
    })


# ── /api/rag_retrieve ─────────────────────────────────────────────────────────
@app.route("/api/rag_retrieve", methods=["POST"])
def rag_retrieve():
    """
    Given a query string, embed it and return the top-k most similar chunks.
    Body: { node_id, query, top_k }
    Returns: { results: [{rank, score, doc_idx, chunk_idx, text}], query_embedding_dim }
    """
    body    = request.get_json(force=True, silent=True) or {}
    node_id = str(body.get("node_id", ""))
    query   = str(body.get("query", "")).strip()
    top_k   = int(body.get("top_k", 5))

    slot  = _RAG.get(node_id, {})
    embs   = slot.get("embeddings")
    # Use embedded_chunks if available (may be a capped subset of all chunks)
    chunks = slot.get("embedded_chunks") or slot.get("chunks")

    if embs is None or chunks is None:
        return jsonify({"error": "Genera los embeddings primero."}), 400
    if not query:
        return jsonify({"error": "Escribe una consulta."}), 400

    if not _HF_AVAILABLE:
        return jsonify({"error": "huggingface_hub no está instalado. Ejecuta: pip install huggingface_hub"}), 500

    try:
        client = _HFClient()
        q_emb  = np.asarray(
            client.feature_extraction([query], model=_EMBED_MODEL),
            dtype=np.float32
        )
        # feature_extraction returns (1, dim) or (dim,) depending on version
        if q_emb.ndim == 2:
            q_emb = q_emb[0]
    except Exception as e:
        return jsonify({"error": f"Error al embeber la consulta: {e}"}), 500

    scores = _cosine_matrix(q_emb, embs)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        c = chunks[int(idx)]
        results.append({
            "rank":      rank + 1,
            "score":     round(float(scores[idx]), 4),
            "doc_idx":   c["doc_idx"],
            "chunk_idx": c["chunk_idx"],
            "text":      c["text"],
        })

    return jsonify({"results": results, "query": query})


# ── /api/rag_columns ──────────────────────────────────────────────────────────
@app.route("/api/rag_generate", methods=["POST"])
def rag_generate():
    """
    Generate LLM responses for a query, with and without RAG context.

    Body: { query, context_chunks: [{text, score}], max_new_tokens, system_prompt }

    Returns: { response_no_rag, response_with_rag, model, error? }

    Future: set HF_TOKEN env var (or load from .env) to use authenticated HF endpoints
    and unlock larger / rate-limit-free models.
    """
    import os, concurrent.futures

    body          = request.get_json(force=True, silent=True) or {}
    query         = (body.get("query") or "").strip()
    ctx_chunks    = body.get("context_chunks") or []
    max_tokens    = int(body.get("max_new_tokens") or 256)
    _lang = body.get("lang", _UI_LANG)
    _en   = (_lang == "en")

    _default_system = (
        "You are a helpful assistant. Be concise and direct."
        if _en else
        "Eres un asistente útil. Sé conciso y directo."
    )
    system_prompt = (body.get("system_prompt") or _default_system)

    if not query:
        return jsonify({"error": "Falta la pregunta (query)."}), 400

    # ── LLM selection ────────────────────────────────────────────────────────
    # Model priority:
    # Model selection:
    #   Default → meta-llama/Llama-3.1-8B-Instruct
    #     - Consistently available on HF serverless free tier, strong RAG quality.
    #   Override → set HF_LLM_MODEL in .env to use another model.
    #   Token   → set HF_TOKEN in .env for authenticated requests (higher rate limits).
    hf_token   = os.environ.get("HF_TOKEN", "").strip()
    llm_model  = os.environ.get("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    if not _HF_AVAILABLE:
        return jsonify({"error": "huggingface_hub no está instalado. Ejecuta: pip install huggingface_hub"}), 500

    # ── Context budget ────────────────────────────────────────────────────────
    # Llama-3.1-8B-Instruct context window: 128k tokens (≈ 4 chars/token)
    # Reserve ~1000 tokens for system + question + response
    MAX_CTX_CHARS = 12000   # conservative: ~3000 tokens, leaves plenty of headroom

    def _truncate_chunks(chunks: list, max_chars: int) -> list:
        """Keep as many chunks as fit within max_chars, truncating the last one if needed."""
        result = []
        used = 0
        for c in chunks:
            text = c.get("text", "")
            if used + len(text) <= max_chars:
                result.append(c)
                used += len(text)
            else:
                remaining = max_chars - used
                if remaining > 200:   # only include if meaningful amount left
                    truncated = dict(c)
                    truncated["text"] = text[:remaining] + "…"
                    result.append(truncated)
                break
        return result

    ctx_chunks_safe = _truncate_chunks(ctx_chunks, MAX_CTX_CHARS)
    print(f"[rag_generate] chunks originales={len(ctx_chunks)} → tras truncar={len(ctx_chunks_safe)} ({sum(len(c['text']) for c in ctx_chunks_safe)} chars)", flush=True)

    # ── Build prompts ─────────────────────────────────────────────────────────
    def _fmt_prompt(with_context: bool) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        if with_context and ctx_chunks_safe:
            context_text = "\n\n---\n\n".join(
                ("[Fragment {i}]\n{t}" if _en else "[Fragmento {i}]\n{t}").format(i=idx+1, t=c["text"])
                for idx, c in enumerate(ctx_chunks_safe)
            )
            if _en:
                user_content = (
                    "Use the following fragments as context to answer the question.\n\n"
                    "=== CONTEXT ===\n{ctx}\n=== END ===\n\n"
                    "Question: {q}"
                ).format(ctx=context_text, q=query)
            else:
                user_content = (
                    "Usa los siguientes fragmentos como contexto para responder.\n\n"
                    "=== CONTEXTO ===\n{ctx}\n=== FIN ===\n\n"
                    "Pregunta: {q}"
                ).format(ctx=context_text, q=query)
        else:
            user_content = ("Question: " if _en else "Pregunta: ") + query
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── Call HF in parallel (no-RAG and with-RAG simultaneously) ─────────────
    import traceback

    def _call_hf(messages, label=""):
        max_retries = 4
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[rag_generate] {label} intento {attempt} — modelo={llm_model}", flush=True)
                client = _HFClient(model=llm_model, token=hf_token if hf_token else None)
                resp   = client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                print(f"[rag_generate] {label} OK — {str(resp)[:120]}", flush=True)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                tb = traceback.format_exc()
                print(f"[rag_generate] {label} ERROR intento {attempt}:\n{tb}", flush=True)
                # Cold start: HF returns "Model ... is currently loading" with estimated_time
                if "loading" in err_str.lower() or "estimated_time" in err_str.lower():
                    wait = min(20 * attempt, 60)
                    print(f"[rag_generate] {label} modelo cargando — esperando {wait}s…", flush=True)
                    time.sleep(wait)
                    continue
                # Rate limit or server error — short retry
                if "503" in err_str or "429" in err_str or "502" in err_str:
                    wait = 10 * attempt
                    print(f"[rag_generate] {label} {err_str[:60]} — reintentando en {wait}s…", flush=True)
                    time.sleep(wait)
                    continue
                # Any other error — fail immediately
                return "__ERROR__:" + err_str
        return "__ERROR__:El modelo tardó demasiado en cargar. Inténtalo de nuevo en unos segundos."

    msgs_no_rag   = _fmt_prompt(with_context=False)
    msgs_with_rag = _fmt_prompt(with_context=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_no  = ex.submit(_call_hf, msgs_no_rag,   "sin-RAG")
        fut_yes = ex.submit(_call_hf, msgs_with_rag, "con-RAG")
        resp_no  = fut_no.result(timeout=90)
        resp_yes = fut_yes.result(timeout=90)

    error = None
    if resp_no.startswith("__ERROR__:") and resp_yes.startswith("__ERROR__:"):
        error = resp_no[len("__ERROR__:"):]

    print(f"[rag_generate] resultado final — no_rag={resp_no[:80] if resp_no else None} | with_rag={resp_yes[:80] if resp_yes else None}", flush=True)

    return jsonify({
        "response_no_rag":   resp_no  if not resp_no.startswith("__ERROR__:")  else None,
        "response_with_rag": resp_yes if not resp_yes.startswith("__ERROR__:") else None,
        "model":    llm_model,
        "n_chunks": len(ctx_chunks),
        "error":    error,
    })


@app.route("/api/rag_export_html", methods=["POST"])
def rag_export_html():
    """Generate a self-contained HTML report of the RAG pipeline results."""
    import html as _html, re as _re
    from datetime import datetime

    body             = request.get_json(force=True, silent=True) or {}
    query            = body.get("query", "")
    resp_no_rag      = body.get("response_no_rag", "")
    resp_with_rag    = body.get("response_with_rag", "")
    llm_model        = body.get("llm_model", "")
    llm_time         = body.get("llm_time", "")
    embed_time       = body.get("embed_time", "")
    system_prompt    = body.get("system_prompt", "")
    ctx_chunks       = body.get("context_chunks", [])
    chunk_cfg        = body.get("chunk_cfg", {})
    corpus_name      = body.get("corpus_name", "")
    n_docs           = body.get("n_docs", 0)
    n_embeddings     = body.get("n_embeddings", 0)
    top_k            = body.get("top_k", len(ctx_chunks))
    embed_model      = body.get("embed_model", "")
    report_lang      = body.get("lang", "es")
    now              = datetime.now().strftime("%d/%m/%Y %H:%M")

    _en = report_lang == "en"

    def _e(t): return _html.escape(str(t or ""))
    def _md(t):
        s = _e(t)
        s = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        s = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>', s)
        s = _re.sub(r'^### (.+)$', r'<h4>\1</h4>', s, flags=_re.MULTILINE)
        s = _re.sub(r'^## (.+)$',  r'<h3>\1</h3>', s, flags=_re.MULTILINE)
        s = _re.sub(r'^# (.+)$',   r'<h3>\1</h3>', s, flags=_re.MULTILINE)
        s = _re.sub(r'^\s*[-*] (.+)$', r'<li>\1</li>', s, flags=_re.MULTILINE)
        s = _re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', s, flags=_re.DOTALL)
        s = s.replace('\n', '<br>')
        return s

    # ── chunk mode label ─────────────────────────────────────────────────────
    one_per = "1 chunk per document" if _en else "1 chunk por documento"
    chunk_sz_lbl = "words, overlap" if _en else "palabras, solapamiento"
    chunk_mode_lbl = one_per if chunk_cfg.get("chunk_mode") == "one_per_doc" \
        else f"{chunk_cfg.get('chunk_size','?')} {chunk_sz_lbl} {chunk_cfg.get('overlap',0)}"

    # ── score color helper ────────────────────────────────────────────────────
    def _score_color(s):
        if s >= 0.7: return "#1a7f4b", "#d4f5e2"
        if s >= 0.4: return "#8a6000", "#fff3cd"
        return "#b00020", "#fde8ec"

    # ── fragments cards ───────────────────────────────────────────────────────
    frag_cards = ""
    for i, c in enumerate(ctx_chunks):
        score = float(c.get("score", 0))
        text  = (c.get("text") or "")
        preview = text[:400] + ("…" if len(text) > 400 else "")
        ink, bg = _score_color(score)
        frag_cards += f"""
        <div class="frag-card">
          <div class="frag-hdr">
            <span class="frag-num">#{i+1}</span>
            <span class="frag-score" style="color:{ink};background:{bg}">{score:.4f}</span>
            <span class="frag-doc" style="color:var(--sec)">doc {c.get('doc_idx','?')} · chunk {c.get('chunk_idx','?')}</span>
          </div>
          <div class="frag-text">{_e(preview)}</div>
        </div>"""

    # ── prompt previews ───────────────────────────────────────────────────────
    prompt_no_rag = ("Question: " if _en else "Pregunta: ") + query
    ctx_preview   = ("\n".join(
        f"[{'Fragment' if _en else 'Fragmento'} {i+1}] {c.get('text','')[:180]}{'…' if len(c.get('text',''))>180 else ''}"
        for i, c in enumerate(ctx_chunks[:4])
    ) + ("\n[…]" if len(ctx_chunks) > 4 else ""))
    prompt_rag = (("=== CONTEXT ===" if _en else "=== CONTEXTO ===") + "\n" +
                  ctx_preview + "\n\n" +
                  ("Question: " if _en else "Pregunta: ") + query)

    html_lang = "en" if _en else "es"
    T = lambda es, en: en if _en else es

    html_out = f"""<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{T("Informe RAG", "RAG Report")} — NLP Flow</title>
<style>
  :root{{
    --bg:#f4f4f8; --card:#fff; --accent:#1db954; --accent2:#0d7a3a;
    --ink:#1a1a2e; --sec:#4a4a6a; --border:#e0e0f0;
    --r:12px; --sh:0 2px 14px rgba(0,0,0,.08);
    --rag-bg:#f0fff6; --rag-border:#b2f5cc;
    --norag-bg:#f8f8fc; --norag-border:#d0d0e8;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--ink);padding:32px 16px;line-height:1.65;}}
  .wrap{{max-width:1000px;margin:0 auto;}}
  /* ── header ── */
  .report-hdr{{background:linear-gradient(135deg,#0d7a3a 0%,#1db954 100%);border-radius:var(--r);padding:28px 32px;color:#fff;margin-bottom:28px;}}
  .report-hdr h1{{font-size:1.8rem;font-weight:800;margin-bottom:4px;}}
  .report-hdr .sub{{opacity:.8;font-size:.93rem;}}
  /* ── sections ── */
  h2{{font-size:1rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em;color:var(--sec);margin:28px 0 12px;border-left:4px solid var(--accent);padding-left:10px;}}
  /* ── metric cards ── */
  .mgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:8px;}}
  .mcard{{background:var(--card);border-radius:var(--r);padding:14px 16px;box-shadow:var(--sh);}}
  .mcard .val{{font-size:1.25rem;font-weight:700;color:var(--accent2);word-break:break-all;}}
  .mcard .lbl{{font-size:.72rem;color:var(--sec);margin-top:3px;text-transform:uppercase;letter-spacing:.04em;}}
  .mcard.wide{{grid-column:1/-1;}}
  .mcard.wide .val{{font-size:.88rem;font-weight:500;word-break:break-word;}}
  /* ── query ── */
  .query-box{{background:var(--card);border:2px solid var(--accent);border-radius:var(--r);padding:16px 20px;font-size:1.05rem;font-weight:600;color:var(--ink);box-shadow:var(--sh);}}
  /* ── fragments ── */
  .frag-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,.05);}}
  .frag-hdr{{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;}}
  .frag-num{{font-weight:700;font-size:.85rem;color:var(--sec);}}
  .frag-score{{padding:2px 9px;border-radius:20px;font-size:.8rem;font-weight:700;font-family:monospace;}}
  .frag-doc{{font-size:.75rem;}}
  .frag-text{{font-size:.82rem;color:#333;line-height:1.6;font-family:'Georgia',serif;}}
  /* ── response boxes ── */
  .resp-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:6px;}}
  @media(max-width:700px){{.resp-wrap{{grid-template-columns:1fr;}}}}
  .resp-panel{{border-radius:var(--r);padding:18px 20px;box-shadow:var(--sh);}}
  .resp-panel.norag{{background:var(--norag-bg);border:1px solid var(--norag-border);}}
  .resp-panel.rag{{background:var(--rag-bg);border:2px solid var(--rag-border);}}
  .resp-panel .resp-label{{font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;}}
  .resp-panel.norag .resp-label{{color:#555;}}
  .resp-panel.rag .resp-label{{color:var(--accent2);}}
  .resp-text{{font-size:.88rem;line-height:1.75;}}
  .resp-text h3,.resp-text h4{{margin:8px 0 4px;font-size:.92rem;color:var(--ink);}}
  .resp-text ul{{margin:4px 0 4px 18px;}}
  .resp-text li{{margin-bottom:2px;}}
  /* ── prompt box ── */
  .prompt-box{{background:#1e1e2e;border-radius:var(--r);padding:14px 16px;font-size:.75rem;font-family:'Fira Mono','Consolas',monospace;white-space:pre-wrap;word-break:break-word;color:#cdd6f4;line-height:1.6;overflow-x:auto;}}
  .prompt-label{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--sec);margin-bottom:6px;}}
  /* ── insight note ── */
  .insight{{background:linear-gradient(135deg,#e8f5fe,#f0fff6);border:1px solid #b3dff5;border-radius:var(--r);padding:16px 20px;font-size:.88rem;color:#1a3a5c;line-height:1.7;margin-top:8px;}}
  /* ── footer ── */
  footer{{margin-top:48px;padding-top:14px;border-top:1px solid var(--border);text-align:center;color:var(--sec);font-size:.75rem;}}
</style>
</head>
<body>
<div class="wrap">

  <!-- Header -->
  <div class="report-hdr">
    <h1>{'RAG Results Report' if _en else 'Informe de Resultados RAG'}</h1>
    <div class="sub">{_e(corpus_name) or '—'} &nbsp;·&nbsp; {_e(now)} &nbsp;·&nbsp; NLP Flow</div>
  </div>

  <!-- Pipeline config cards -->
  <h2>{'Pipeline Configuration' if _en else 'Configuración de la Pipeline'}</h2>
  <div class="mgrid">
    <div class="mcard"><div class="val">{_e(str(n_docs)) or '—'}</div><div class="lbl">{'Documents' if _en else 'Documentos'}</div></div>
    <div class="mcard"><div class="val">{_e(str(n_embeddings)) if n_embeddings else '—'}</div><div class="lbl">Embeddings</div></div>
    <div class="mcard"><div class="val">{_e(str(top_k or len(ctx_chunks)))}</div><div class="lbl">Top-K {'retrieved' if _en else 'recuperados'}</div></div>
    <div class="mcard"><div class="val">{_e(embed_time) or '—'}</div><div class="lbl">{'Embedding time' if _en else 'Tiempo embeddings'}</div></div>
    <div class="mcard"><div class="val">{_e(llm_time) or '—'}</div><div class="lbl">{'LLM time' if _en else 'Tiempo LLM'}</div></div>
    <div class="mcard wide"><div class="val">✂️ {_e(chunk_mode_lbl)}</div><div class="lbl">Chunking</div></div>
    <div class="mcard wide"><div class="val">🤖 {_e(llm_model) or '—'}</div><div class="lbl">{'Model' if _en else 'Modelo'} LLM</div></div>
    {('<div class="mcard wide"><div class="val">🧬 ' + _e(embed_model) + '</div><div class="lbl">Embedding model</div></div>') if embed_model else ''}
  </div>

  <!-- Query -->
  <h2>{'1 — Query' if _en else '1 — Consulta'}</h2>
  <div class="query-box">{_e(query) or ('(no query)' if _en else '(sin consulta)')}</div>

  <!-- Fragments -->
  <h2>{'2 — Retrieved Fragments' if _en else '2 — Fragmentos Recuperados'} ({len(ctx_chunks)})</h2>
  {frag_cards if frag_cards else ('<p style="color:var(--sec);font-size:.88rem">' + ('No fragments retrieved.' if _en else 'No hay fragmentos recuperados.') + '</p>')}

  <!-- Responses side by side -->
  <h2>{'3 — Responses' if _en else '3 — Respuestas'}</h2>
  <div class="resp-wrap">
    <div class="resp-panel norag">
      <div class="resp-label">{'Without RAG' if _en else 'Sin RAG'}</div>
      <div class="resp-text">{_md(resp_no_rag) if resp_no_rag else '<em style="color:#aaa">' + ('No response.' if _en else 'Sin respuesta.') + '</em>'}</div>
    </div>
    <div class="resp-panel rag">
      <div class="resp-label">{'With RAG ✓' if _en else 'Con RAG ✓'}</div>
      <div class="resp-text">{_md(resp_with_rag) if resp_with_rag else '<em style="color:#aaa">' + ('No response.' if _en else 'Sin respuesta.') + '</em>'}</div>
    </div>
  </div>

  <!-- Prompts -->
  <h2 style="margin-top:32px">{'4 — Prompts sent to the LLM' if _en else '4 — Prompts enviados al LLM'}</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div>
      <div class="prompt-label">{'Without RAG' if _en else 'Sin RAG'}</div>
      <div class="prompt-box">{_e(prompt_no_rag)}</div>
    </div>
    <div>
      <div class="prompt-label">{'With RAG' if _en else 'Con RAG'}</div>
      <div class="prompt-box">{_e(prompt_rag)}</div>
    </div>
  </div>

  <!-- Insight note -->
  <div class="insight" style="margin-top:24px">
    {'💡 <strong>RAG in one sentence:</strong> the only difference between the two responses is that the LLM received the retrieved fragments as context in its prompt. It is not magic — it is literally more text in the input.' if _en else
     '💡 <strong>RAG en una frase:</strong> la única diferencia entre las dos respuestas es que el LLM recibió los fragmentos recuperados como contexto en el prompt. No es magia — es literalmente más texto en el input.'}
  </div>

  <footer>{'Generated with NLP Flow' if _en else 'Generado con NLP Flow'} &nbsp;·&nbsp; {_e(now)}</footer>
</div>
</body>
</html>"""

    from flask import Response as _Resp
    fname = "rag_report.html" if _en else "informe_rag.html"
    return _Resp(
        html_out.encode("utf-8"),
        mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )


@app.route("/api/rag_columns")
def rag_columns():
    """Return column names and whether preprocessed NLP texts are available."""
    node_id        = request.args.get("node_id", "")
    slot           = _get_data(node_id) if node_id else S
    rows           = slot.get("raw_rows", [])
    has_processed  = bool(S.get("processed_texts"))
    nlp_text_col   = ""

    # Try to detect which column was used as the NLP text column
    # (upload_csv stores it in S; best proxy is the column whose values match S["texts"])
    if has_processed and S.get("texts") and rows:
        first_text = S["texts"][0][:50] if S["texts"] else ""
        for col in (rows[0].keys() if rows else []):
            sample_val = str(rows[0].get(col, ""))[:50]
            if sample_val == first_text:
                nlp_text_col = col
                break

    if not rows:
        return jsonify({"columns": [], "text_cols": [], "has_processed": has_processed,
                        "nlp_text_col": nlp_text_col, "avg_text_len": 0, "n_docs": 0})

    text_like = []
    for col in rows[0].keys():
        sample = [str(r.get(col, "") or "") for r in rows[:20]]
        avg    = sum(len(v) for v in sample) / max(len(sample), 1)
        if avg > 30:
            text_like.append(col)

    # Avg length of the primary text column (use processed if available)
    if has_processed and S.get("processed_texts"):
        sample_texts = S["processed_texts"][:50]
    elif text_like:
        sample_texts = [str(r.get(text_like[0], "")) for r in rows[:50]]
    else:
        sample_texts = []
    avg_text_len = int(sum(len(t) for t in sample_texts) / max(len(sample_texts), 1))
    max_text_len = int(max((len(t) for t in sample_texts), default=0))

    return jsonify({
        "columns":      list(rows[0].keys()),
        "text_cols":    text_like,
        "has_processed": has_processed,
        "nlp_text_col": nlp_text_col,
        "avg_text_len": avg_text_len,
        "max_text_len": max_text_len,
        "n_docs":       len(rows),
    })


# ════════════════════════════════════════════════════════════════════════════
# IMAGE DATA BLOCK  — load MNIST or Dogs-vs-Muffins, return info + class chart
# ════════════════════════════════════════════════════════════════════════════

_IMAGE_LOAD_LOCK = threading.Lock()
_IMAGE_LOAD_PROGRESS: dict = {}   # { node_id: {pct, msg, done, error} }


def _img_progress(node_id, pct, msg, done=False, error=None):
    _IMAGE_LOAD_PROGRESS[node_id] = {"pct": pct, "msg": msg, "done": done, "error": error}


def _class_dist_b64(class_names, counts, title="Class distribution"):
    """Bar chart (horizontal) of class counts — returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Use the same app-wide palette so colours match every other chart
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(5, max(2.2, len(class_names) * 0.55)))
    bars = ax.barh(class_names, counts, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Images", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8, color="#555")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/api/image_reset", methods=["POST"])
def image_reset():
    """
    Clears the loaded image state for a node so the picker UI is shown again.
    Body: { node: <node_id> }
    """
    data    = request.get_json(force=True)
    node_id = data.get("node", "img_default")
    if node_id in _IMAGE_NODE_DATA:
        _IMAGE_NODE_DATA[node_id] = {
            "loaded": False, "dataset": None, "class_names": [],
            "n_train": 0, "n_test": 0, "n_classes": 0, "img_size": [0, 0],
            "train_data": [], "test_data": []
        }
    _node_slot(node_id).pop("task", None)
    _node_slot(node_id).pop("dataset_name", None)
    return jsonify({"ok": True})


@app.route("/api/image_load", methods=["POST"])
def image_load():
    """
    Body: { node: <node_id>, dataset: "mnist" | "dogs_muffins" | "cats_vs_dogs" }
    Spawns a background thread; poll /api/image_load_progress?node=<id> for updates.
    """
    body    = request.json or {}
    node_id = str(body.get("node", "img_default"))
    dataset = body.get("dataset", "mnist")
    cap     = int(body.get("cap", 0))   # 0 = no cap; >0 = max images to load (dogs_muffins only)

    if _IMAGE_LOAD_LOCK.locked():
        return jsonify({"error": "Another dataset is already loading. Please wait."}), 429

    slot = _image_slot(node_id)
    slot["loaded"] = False

    def _worker():
        with _IMAGE_LOAD_LOCK:
            try:
                from PIL import Image as PILImage
                _img_progress(node_id, 2, "Initializing…")

                if dataset == "mnist":
                    # ── MNIST via torchvision ──────────────────────────────
                    try:
                        import torchvision
                        import torchvision.transforms as T_tv
                    except ImportError:
                        _img_progress(node_id, 0, "", done=True,
                                      error="torchvision not installed. Run: pip install torchvision pillow")
                        return

                    _img_progress(node_id, 10, "Downloading MNIST (first time only)…")
                    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nlpflow_images")
                    os.makedirs(cache_dir, exist_ok=True)

                    tf = T_tv.Compose([T_tv.ToTensor()])
                    train_ds = torchvision.datasets.MNIST(cache_dir, train=True,  download=True, transform=tf)
                    test_ds  = torchvision.datasets.MNIST(cache_dir, train=False, download=True, transform=tf)

                    _img_progress(node_id, 70, "Preparing samples…")
                    class_names = [str(i) for i in range(10)]

                    # Store a compact sample (2000 train + 500 test PIL images) to keep RAM low
                    import random as _rnd
                    _rnd.seed(42)
                    sample_train_idx = _rnd.sample(range(len(train_ds)), min(2000, len(train_ds)))
                    sample_test_idx  = _rnd.sample(range(len(test_ds)),  min(500,  len(test_ds)))

                    def _to_pil(ds, idx):
                        img_t, lbl = ds[idx]
                        # img_t shape: (1, H, W) for MNIST
                        arr = (img_t.numpy().squeeze() * 255).astype("uint8")
                        return PILImage.fromarray(arr, mode="L").convert("RGB"), int(lbl)

                    train_data = [_to_pil(train_ds, i) for i in sample_train_idx]
                    test_data  = [_to_pil(test_ds,  i) for i in sample_test_idx]
                    all_data   = train_data + test_data   # combined for dynamic test split
                    img_size   = (28, 28)
                    n_train_total = len(train_ds)
                    n_test_total  = len(test_ds)

                elif dataset == "dogs_muffins":
                    # ── Dogs vs Muffins (Kaggle or HuggingFace fallback) ───
                    import random as _rnd
                    _rnd.seed(42)

                    # ── Try Kaggle first (if credentials are available in .env) ──────────
                    kaggle_user = os.environ.get("KAGGLE_USERNAME", "").strip()
                    kaggle_key  = os.environ.get("KAGGLE_KEY", "").strip()
                    kaggle_ok   = False
                    combined    = []
                    class_names = ["muffin", "chihuahua"]

                    if kaggle_user and kaggle_key:
                        try:
                            import kaggle as _kaggle_api
                            # Set credentials programmatically (avoids needing kaggle.json on disk)
                            import kaggle.api as _kapi
                            os.environ["KAGGLE_USERNAME"] = kaggle_user
                            os.environ["KAGGLE_KEY"]      = kaggle_key
                            _kapi.authenticate()

                            _KAGGLE_DATASET  = "samuelcortinhas/muffin-vs-chihuahua-image-classification"
                            _KAGGLE_CACHE    = os.path.join(os.path.expanduser("~"), ".cache", "nlpflow_images", "chihuahua_muffin_kaggle")

                            if not os.path.isdir(_KAGGLE_CACHE):
                                _img_progress(node_id, 8, "Downloading Chihuahua vs Muffin from Kaggle (~500 MB, first time only)…")
                                os.makedirs(_KAGGLE_CACHE, exist_ok=True)
                                _kapi.dataset_download_files(_KAGGLE_DATASET, path=_KAGGLE_CACHE, unzip=True, quiet=False)
                                _img_progress(node_id, 35, "Download complete. Unzipping…")
                            else:
                                _img_progress(node_id, 35, "Kaggle dataset already cached. Reading images…")

                            # Walk the downloaded folder: expects train/muffin, train/chihuahua, test/muffin, test/chihuahua
                            _img_progress(node_id, 40, "Scanning image folders…")
                            _all_files = []
                            for _root, _dirs, _files in os.walk(_KAGGLE_CACHE):
                                _folder = os.path.basename(_root).lower()
                                if _folder in ("muffin", "muffins"):
                                    _lbl = 0
                                elif _folder in ("chihuahua", "chihuahuas"):
                                    _lbl = 1
                                else:
                                    continue
                                for _fname in _files:
                                    if _fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                                        _all_files.append((os.path.join(_root, _fname), _lbl))

                            _total_files = len(_all_files)
                            print(f"[IMG] Kaggle: found {_total_files} image files", flush=True)
                            for _ii, (_fpath, _lbl) in enumerate(_all_files):
                                if _ii % 300 == 0:
                                    _pct = 40 + int(30 * _ii / max(_total_files, 1))
                                    _img_progress(node_id, _pct, f"Loading images {_ii}/{_total_files}…")
                                try:
                                    _img = PILImage.open(_fpath).convert("RGB").resize((64, 64))
                                    combined.append((_img, _lbl))
                                except Exception:
                                    continue

                            if combined:
                                kaggle_ok = True
                                _img_progress(node_id, 72, f"Kaggle dataset ready — {len(combined)} images loaded.")
                                print(f"[IMG] Kaggle dataset loaded: {len(combined)} images", flush=True)
                            else:
                                print("[IMG] Kaggle dataset downloaded but no images found — falling back to HuggingFace", flush=True)

                        except Exception as _ke:
                            print(f"[IMG] Kaggle download failed ({_ke}) — falling back to HuggingFace", flush=True)

                    # ── Fallback: combine sasha/chihuahua-muffin + VatsaDev/cnn_muffins ──
                    if not kaggle_ok:
                        try:
                            from datasets import load_dataset as hf_load
                        except ImportError:
                            _img_progress(node_id, 0, "", done=True,
                                          error="'datasets' package not installed. Run: pip install datasets pillow")
                            return

                        def _load_hf_img_label(row, img_col, label_col, target_size=(64, 64)):
                            img = row[img_col]
                            lbl = row[label_col]
                            if not isinstance(img, PILImage.Image):
                                try:
                                    img = PILImage.fromarray(img)
                                except Exception:
                                    return None
                            return (img.convert("RGB").resize(target_size), int(lbl))

                        _img_progress(node_id, 10, "Downloading Chihuahua vs Muffin from Hugging Face (first time only)…")

                        # Source 1: sasha/chihuahua-muffin (labels: 0=muffin, 1=chihuahua)
                        try:
                            ds1 = hf_load("sasha/chihuahua-muffin")["train"]
                            _img_progress(node_id, 35, f"Loaded sasha/chihuahua-muffin ({len(ds1)} images)…")
                            for i in range(len(ds1)):
                                item = _load_hf_img_label(ds1[i], "image", "label")
                                if item:
                                    combined.append(item)
                        except Exception as e:
                            print(f"[IMG] sasha/chihuahua-muffin failed: {e}", flush=True)

                        # Source 2: VatsaDev/cnn_muffins (labels: 0=chihuahua→1, 1=muffin→0 — remap)
                        try:
                            ds2 = hf_load("VatsaDev/cnn_muffins")
                            splits2 = list(ds2.keys())
                            _img_progress(node_id, 55, f"Loaded VatsaDev/cnn_muffins ({splits2})…")
                            for sname in splits2:
                                if sname == "hard_16":
                                    continue
                                sp = ds2[sname]
                                img_col2   = "image" if "image" in sp.features else list(sp.features.keys())[0]
                                label_col2 = "label" if "label" in sp.features else "labels"
                                for i in range(len(sp)):
                                    item = _load_hf_img_label(sp[i], img_col2, label_col2)
                                    if item:
                                        remapped_lbl = 1 - item[1]  # VatsaDev: 0=chihuahua→1, 1=muffin→0
                                        combined.append((item[0], remapped_lbl))
                        except Exception as e:
                            print(f"[IMG] VatsaDev/cnn_muffins failed: {e}", flush=True)

                        _img_progress(node_id, 70, f"HuggingFace fallback — {len(combined)} total images.")

                    if not combined:
                        _img_progress(node_id, 0, "", done=True, error="Could not load any chihuahua-muffin images.")
                        return

                    _rnd.shuffle(combined)

                    # Apply optional cap — balanced across classes
                    # Cache the capped result so next load is instant
                    _cap_cache_path = None
                    if cap and cap > 0:
                        _cap_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nlpflow_images")
                        os.makedirs(_cap_cache_dir, exist_ok=True)
                        _cap_cache_path = os.path.join(_cap_cache_dir, f"dogs_muffins_cap{cap}.pkl")

                    if _cap_cache_path and os.path.exists(_cap_cache_path):
                        # ── Fast path: load from cache ──────────────────────
                        _img_progress(node_id, 74, f"Loading {cap} cached images…")
                        with open(_cap_cache_path, "rb") as _f:
                            combined = pickle.load(_f)
                        print(f"[IMG] Cap cache hit: {len(combined)} images loaded from {_cap_cache_path}", flush=True)
                    elif cap and cap > 0 and len(combined) > cap:
                        # ── Slow path: apply cap and cache result ───────────
                        _img_progress(node_id, 74, f"Applying cap: {cap} images from {len(combined)} total…")
                        _by_cls_cap = {}
                        for _item in combined:
                            _by_cls_cap.setdefault(_item[1], []).append(_item)
                        _per_cls_cap = cap // len(_by_cls_cap)
                        combined = []
                        for _lbl_items in _by_cls_cap.values():
                            combined.extend(_lbl_items[:_per_cls_cap])
                        _rnd.shuffle(combined)
                        print(f"[IMG] Cap applied: {len(combined)} images kept", flush=True)
                        if _cap_cache_path:
                            try:
                                with open(_cap_cache_path, "wb") as _f:
                                    pickle.dump(combined, _f)
                                print(f"[IMG] Cap cached to {_cap_cache_path}", flush=True)
                            except Exception as _ce:
                                print(f"[IMG] Cap cache write failed: {_ce}", flush=True)

                    split_at = int(len(combined) * 0.8)
                    all_data   = combined
                    train_data = combined[:split_at]
                    test_data  = combined[split_at:]
                    img_size   = (64, 64)
                    n_train_total = len(train_data)
                    n_test_total  = len(test_data)

                elif dataset == "cats_vs_dogs":
                    # microsoft/cats_vs_dogs — ~23K images, ~722MB download (full dataset, no cap)
                    _img_progress(node_id, 5, "Downloading Cats vs Dogs (~720 MB, first time only)…")
                    from datasets import load_dataset as hf_load
                    hf_ds = hf_load("microsoft/cats_vs_dogs")
                    full = hf_ds["train"]   # only split available
                    img_col   = "image"
                    label_col = "labels"
                    feat = full.features
                    if hasattr(feat.get(label_col, None), "names"):
                        class_names = feat[label_col].names
                    else:
                        class_names = ["cat", "dog"]

                    import random as _rnd2
                    _rnd2.seed(42)

                    # Load ALL images (no cap) — resize to 64×64 to keep RAM manageable
                    total_cvd = len(full)
                    all_items = []
                    for ii in range(total_cvd):
                        if ii % 500 == 0:
                            _img_progress(node_id, 10 + int(75 * ii / total_cvd), f"Loading images {ii}/{total_cvd}…")
                        row = full[ii]
                        img = row[img_col]
                        lbl = row[label_col]
                        if not isinstance(img, PILImage.Image):
                            try:
                                img = PILImage.fromarray(img)
                            except Exception:
                                continue
                        all_items.append((img.convert("RGB").resize((64, 64)), int(lbl)))

                    _rnd2.shuffle(all_items)
                    split_at = int(len(all_items) * 0.8)
                    all_data   = all_items
                    train_data = all_items[:split_at]
                    test_data  = all_items[split_at:]
                    img_size   = (64, 64)
                    n_train_total = len(train_data)
                    n_test_total  = len(test_data)

                else:
                    _img_progress(node_id, 0, "", done=True, error=f"Unknown dataset: {dataset}")
                    return

                _img_progress(node_id, 90, "Finalizing…")
                slot["loaded"]      = True
                slot["dataset"]     = dataset
                slot["class_names"] = list(class_names)
                slot["n_classes"]   = len(class_names)
                slot["img_size"]    = img_size
                slot["all_data"]    = all_data if 'all_data' in dir() else (train_data + test_data)
                slot["train_data"]  = train_data
                slot["test_data"]   = test_data
                slot["n_train"]     = n_train_total
                slot["n_test"]      = n_test_total
                # Also write a lightweight marker into _NODE_DATA so inspect/preprocess
                # blocks know this node carries image data (task = "image")
                nd = _node_slot(node_id)
                nd["task"]         = "image"
                _dsname_map = {"mnist": "MNIST", "dogs_muffins": "Chihuahua vs Muffin", "cats_vs_dogs": "Cats vs Dogs"}
                nd["dataset_name"] = _dsname_map.get(dataset, dataset)
                nd["raw_rows"]     = []   # no CSV rows for image datasets
                _img_progress(node_id, 100, "Done", done=True)

            except Exception as exc:
                import traceback
                traceback.print_exc()
                _img_progress(node_id, 0, "", done=True, error=str(exc))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/api/image_load_progress")
def image_load_progress():
    node_id = request.args.get("node", "img_default")
    info = _IMAGE_LOAD_PROGRESS.get(node_id, {"pct": 0, "msg": "Waiting…", "done": False, "error": None})
    return jsonify(info)


@app.route("/api/image_info")
def image_info():
    """
    Returns summary info + base64 class-distribution chart for a loaded image node.
    Query: ?node=<node_id>
    """
    node_id = request.args.get("node", "img_default")
    slot    = _IMAGE_NODE_DATA.get(node_id)
    if not slot or not slot.get("loaded"):
        return jsonify({"error": "No image dataset loaded for this node"}), 400

    _es = (_UI_LANG == "es")
    class_names = slot["class_names"]
    train_data  = slot["train_data"]
    test_data   = slot["test_data"]

    # Per-class counts in our sample
    from collections import Counter as _Counter
    train_counts = _Counter(lbl for _, lbl in train_data)
    test_counts  = _Counter(lbl for _, lbl in test_data)

    train_count_list = [train_counts.get(i, 0) for i in range(len(class_names))]
    test_count_list  = [test_counts.get(i,  0) for i in range(len(class_names))]

    # Generate distribution chart (train split)
    title = ("Distribución de clases (train)" if _es else "Class distribution (train)")
    img_b64 = _class_dist_b64(class_names, train_count_list, title=title)

    # Thumbnail grid — up to 3 examples per class at 96px
    THUMBS_PER_CLASS = 3
    import random as _rnd_thumb
    shuffled = list(train_data)
    _rnd_thumb.shuffle(shuffled)   # random sample each time the block is opened
    thumbs = {}  # {class_idx: [b64, ...]}
    for img, lbl in shuffled:
        bucket = thumbs.setdefault(lbl, [])
        if len(bucket) < THUMBS_PER_CLASS:
            buf = io.BytesIO()
            img.resize((160, 160)).save(buf, format="PNG")
            bucket.append(base64.b64encode(buf.getvalue()).decode())
        if all(len(thumbs.get(i, [])) >= THUMBS_PER_CLASS for i in range(len(class_names))):
            break
    # Flatten: [[b64, b64, b64], [b64, b64, b64], ...] — one list per class
    thumbs_list = [thumbs.get(i, []) for i in range(len(class_names))]

    return jsonify({
        "dataset":       slot["dataset"],
        "dataset_name":  {"mnist": "MNIST", "dogs_muffins": "Chihuahua vs Muffin", "cats_vs_dogs": "Cats vs Dogs"}.get(slot.get("dataset",""), slot.get("dataset","")),
        "class_names":   class_names,
        "n_classes":     slot["n_classes"],
        "img_size":      list(slot["img_size"]),
        "n_train":       slot["n_train"],
        "n_test":        slot["n_test"],
        "train_counts":  train_count_list,
        "test_counts":   test_count_list,
        "dist_img":      img_b64,
        "thumbs":        thumbs_list,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# CNN CLASSIFICATION BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

# Per-node CNN state ─────────────────────────────────────────────────────────
_CNN_STATE: dict = {}   # node_id → state dict
_CNN_THREADS: dict = {} # node_id → Thread

def _cnn_slot(node_id: str) -> dict:
    if node_id not in _CNN_STATE:
        _CNN_STATE[node_id] = {
            "status": "idle",       # idle | training | done | error
            "cfg": {},
            "log": [],
            "result": None,
            "error": None,
            "epoch": 0,
            "total_epochs": 0,
            "pct": 0,
            "msg": "",
            "runs": [],            # list of completed run dicts (name, cfg, metrics, model_state, class_names, norm_mode, img_size, n_ch)
            "run_counter": 0,      # increments for default names
        }
    return _CNN_STATE[node_id]


def _cnn_train_worker(node_id: str, cfg: dict, train_data: list, test_data: list,
                       class_names: list):
    """Background thread: trains a small CNN on the pre-loaded image data."""
    slot = _cnn_slot(node_id)
    print(f"[CNN] worker started  node={node_id}  train={len(train_data)}  test={len(test_data)}  classes={class_names}", flush=True)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
        from PIL import Image as PILImage

        epochs     = int(cfg.get("epochs", 5))
        lr         = float(cfg.get("lr", 0.001))
        n_conv     = int(cfg.get("n_conv", 2))
        img_size   = int(cfg.get("img_size", 64))
        batch_size = int(cfg.get("batch_size", 32))
        norm_mode  = cfg.get("norm_mode", "01")
        aug_flip   = bool(cfg.get("aug_flip", False))
        aug_rotate = bool(cfg.get("aug_rotate", False))
        aug_crop   = bool(cfg.get("aug_crop", False))
        aug_n_cfg  = max(0, int(cfg.get("aug_factor", 0)))  # absolute number of extra augmented images to add
        n_classes  = len(class_names)
        n_ch       = 1 if (len(train_data) > 0 and train_data[0][0].mode == "L") else 3

        print(f"[CNN] cfg → epochs={epochs} lr={lr} n_conv={n_conv} img_size={img_size} bs={batch_size} norm={norm_mode} n_classes={n_classes} n_ch={n_ch}", flush=True)
        slot["total_epochs"] = epochs
        slot["log"] = []

        def _augment_one(img):
            """Apply random augmentations to a single PIL image."""
            import random
            if aug_flip and random.random() > 0.5:
                img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
            if aug_rotate:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, resample=PILImage.BILINEAR, expand=False)
            if aug_crop:
                w, h = img.size
                scale = random.uniform(0.75, 0.95)
                nw, nh = int(w * scale), int(h * scale)
                x0 = random.randint(0, w - nw)
                y0 = random.randint(0, h - nh)
                img = img.crop((x0, y0, x0 + nw, y0 + nh)).resize((img_size, img_size), PILImage.BILINEAR)
            return img

        def _img_to_arr(img):
            arr = np.array(img, dtype=np.float32)
            if n_ch == 1:
                arr = arr[..., np.newaxis]
            if norm_mode == "01":
                arr = arr / 255.0
            elif norm_mode == "meanstd":
                arr = (arr / 255.0 - 0.5) / 0.5
            return arr.transpose(2, 0, 1)  # HWC → CHW

        def _to_tensor(data, augment=False, aug_n_extra=0):
            """Convert (PIL, label) list to TensorDataset.
            aug_n_extra: total extra augmented samples to add across the dataset,
                         distributed proportionally across classes.
            """
            xs, ys = [], []
            do_aug = augment and (aug_flip or aug_rotate or aug_crop) and aug_n_extra > 0
            # Always add all originals first
            base_imgs = []
            for img, lbl in data:
                img = img.resize((img_size, img_size), PILImage.BILINEAR)
                img = img.convert("L" if n_ch == 1 else "RGB")
                xs.append(_img_to_arr(img))
                ys.append(int(lbl))
                base_imgs.append((img, int(lbl)))
            # Add augmented extras balanced across classes
            if do_aug and aug_n_extra > 0:
                import random as _aug_rnd
                # Distribute extras proportionally across classes
                by_cls = {}
                for img, lbl in base_imgs:
                    by_cls.setdefault(lbl, []).append(img)
                n_cls = len(by_cls)
                per_cls = max(1, aug_n_extra // n_cls) if n_cls > 0 else 0
                for lbl, imgs in by_cls.items():
                    added = 0
                    while added < per_cls:
                        src = _aug_rnd.choice(imgs)
                        xs.append(_img_to_arr(_augment_one(src)))
                        ys.append(lbl)
                        added += 1
            X = torch.tensor(np.stack(xs), dtype=torch.float32)
            Y = torch.tensor(ys, dtype=torch.long)
            return TensorDataset(X, Y)

        val_pct  = max(0.05, min(0.4, float(cfg.get("val_split", 0.2))))

        slot["msg"] = "Preparing data…"
        slot["pct"] = 2
        print("[CNN] converting images to tensors…", flush=True)

        # Split train_data → actual_train + val (stratified by label)
        import random as _rnd
        by_class = {}
        for item in train_data:
            by_class.setdefault(item[1], []).append(item)
        actual_train, val_data = [], []
        for lbl, items in by_class.items():
            _rnd.shuffle(items)
            n_val = max(1, round(len(items) * val_pct))
            val_data.extend(items[:n_val])
            actual_train.extend(items[n_val:])

        print(f"[CNN] split → train={len(actual_train)}  val={len(val_data)}  test(held-out)={len(test_data)}", flush=True)

        do_aug      = aug_flip or aug_rotate or aug_crop
        aug_n_extra = aug_n_cfg if (do_aug and aug_n_cfg > 0) else 0
        train_ds    = _to_tensor(actual_train, augment=do_aug, aug_n_extra=aug_n_extra)
        val_ds      = _to_tensor(val_data,     augment=False)
        test_ds     = _to_tensor(test_data,    augment=False)
        n_train_aug = len(train_ds)
        print(f"[CNN] tensors ready — train={n_train_aug} (+{aug_n_extra} aug extra)  val={len(val_ds)}  test={len(test_ds)}", flush=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # ── Build CNN ────────────────────────────────────────────────────────
        filters = [32, 64, 128]
        layers = []
        in_ch = n_ch
        for i in range(n_conv):
            out_ch = filters[i]
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        # Compute flatten size
        feat_size = img_size // (2 ** n_conv)
        flatten   = in_ch * feat_size * feat_size

        layers += [
            nn.Flatten(),
            nn.Linear(flatten, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        ]
        model = nn.Sequential(*layers)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        slot["msg"] = "Training…"
        print(f"[CNN] model built — flatten={flatten}  starting training…", flush=True)

        import time as _time
        _train_start = _time.time()
        _epoch_times = []

        def _fmt_seconds(s):
            s = int(s)
            if s < 60:
                return f"{s}s"
            return f"{s // 60}m {s % 60:02d}s"

        for ep in range(1, epochs + 1):
            if slot["status"] == "cancelled":
                return

            _ep_start = _time.time()

            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out  = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss    += loss.item() * len(yb)
                train_correct += (out.argmax(1) == yb).sum().item()
                train_total   += len(yb)

            # Validate on held-out val split
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    out  = model(xb)
                    loss = criterion(out, yb)
                    val_loss    += loss.item() * len(yb)
                    val_correct += (out.argmax(1) == yb).sum().item()
                    val_total   += len(yb)

            _ep_elapsed = _time.time() - _ep_start
            _epoch_times.append(_ep_elapsed)
            _avg_ep_time = sum(_epoch_times) / len(_epoch_times)
            _epochs_left = epochs - ep
            _eta_seconds = _avg_ep_time * _epochs_left
            _elapsed_total = _time.time() - _train_start

            ep_log = {
                "epoch":       ep,
                "loss":        round(train_loss / train_total, 4),
                "acc":         round(train_correct / train_total, 4),
                "val_loss":    round(val_loss / val_total, 4),
                "val_acc":     round(val_correct / val_total, 4),
                "epoch_time":  round(_ep_elapsed, 2),
                "elapsed":     round(_elapsed_total, 2),
                "eta":         round(_eta_seconds, 2),
            }
            slot["log"].append(ep_log)
            slot["epoch"]   = ep
            slot["pct"]     = int(ep / epochs * 90)
            slot["eta"]     = round(_eta_seconds, 2)
            slot["elapsed"] = round(_elapsed_total, 2)
            _eta_str = ("ETA " + _fmt_seconds(_eta_seconds)) if _epochs_left > 0 else ("Total " + _fmt_seconds(_elapsed_total))
            slot["msg"] = f"Epoch {ep}/{epochs} — acc {ep_log['acc']:.1%} val_acc {ep_log['val_acc']:.1%} · {_eta_str}"
            print(f"[CNN] {slot['msg']}", flush=True)

        _total_elapsed = _time.time() - _train_start
        slot["total_elapsed"] = round(_total_elapsed, 2)

        # ── Final evaluation + confusion matrix ──────────────────────────────
        slot["msg"] = "Computing results…"
        slot["pct"] = 92

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        wrong_samples = []  # (img_tensor, true_lbl, pred_lbl, confidence)

        with torch.no_grad():
            for xb, yb in test_dl:
                probs = torch.softmax(model(xb), dim=1)
                preds = probs.argmax(1)
                all_preds.extend(preds.tolist())
                all_labels.extend(yb.tolist())
                all_probs.extend(probs.tolist())

        # Confusion matrix as list of lists
        cm = [[0] * n_classes for _ in range(n_classes)]
        for true, pred in zip(all_labels, all_preds):
            cm[true][pred] += 1

        final_acc = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)

        # Per-class precision, recall, F1
        per_class = []
        for c in range(n_classes):
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(n_classes) if r != c)
            fn = sum(cm[c][r] for r in range(n_classes) if r != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class.append({
                "label":     class_names[c],
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "support":   sum(cm[c]),
            })

        # Macro averages
        macro_prec = round(sum(p["precision"] for p in per_class) / n_classes, 4)
        macro_rec  = round(sum(p["recall"]    for p in per_class) / n_classes, 4)
        macro_f1   = round(sum(p["f1"]        for p in per_class) / n_classes, 4)
        print(f"[CNN] test acc={final_acc:.1%}  macro-F1={macro_f1:.3f}", flush=True)

        # Confusion matrix image
        cm_b64 = _cnn_cm_b64(cm, class_names)

        # Wrong predictions: pick up to 8, sorted by confidence of wrong class
        wrong_idxs = [i for i, (t, p) in enumerate(zip(all_labels, all_preds)) if t != p]
        wrong_idxs.sort(key=lambda i: all_probs[i][all_preds[i]], reverse=True)
        wrong_samples_b64 = []
        for i in wrong_idxs[:8]:
            img_t, lbl_t = test_ds[i]
            # Convert tensor CHW → PIL
            arr = img_t.numpy().transpose(1, 2, 0)
            if norm_mode == "01":
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            elif norm_mode == "meanstd":
                arr = ((arr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            pil = PILImage.fromarray(arr).resize((96, 96))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            wrong_samples_b64.append({
                "img":        base64.b64encode(buf.getvalue()).decode(),
                "true_label": class_names[all_labels[i]],
                "pred_label": class_names[all_preds[i]],
                "confidence": round(all_probs[i][all_preds[i]], 3),
            })

        # Loss/acc curve image
        curve_b64 = _cnn_curve_b64(slot["log"])

        # Save model state dict in memory for prediction later
        import copy
        model_state = copy.deepcopy(model.state_dict())

        slot["run_counter"] = slot.get("run_counter", 0) + 1
        run_idx  = slot["run_counter"]
        run_name = f"Run {run_idx}"

        # Serialisable result (safe for jsonify)
        result_dict = {
            "run_idx":        run_idx,
            "run_name":       run_name,
            "final_acc":      round(final_acc, 4),
            "macro_f1":       macro_f1,
            "macro_prec":     macro_prec,
            "macro_rec":      macro_rec,
            "per_class":      per_class,
            "log":            slot["log"],
            "cm":             cm,
            "class_names":    class_names,
            "cm_img":         cm_b64,
            "curve_img":      curve_b64,
            "wrong_samples":  wrong_samples_b64,
            "n_val":          len(val_data),
            "n_test":         len(test_data),
            "cfg":            cfg,
            "total_elapsed":  slot.get("total_elapsed", 0),
        }
        # Non-serialisable model data stored separately under _ keys
        result_dict["_model_state"] = model_state
        result_dict["_arch"] = {
            "n_conv":    n_conv,
            "n_ch":      n_ch,
            "img_size":  img_size,
            "n_classes": n_classes,
            "filters":   [32, 64, 128],
        }
        result_dict["_norm_mode"] = norm_mode

        slot["result"] = result_dict
        slot["runs"].append(result_dict)
        slot["status"] = "done"
        slot["pct"]    = 100
        slot["msg"]    = f"Done — test accuracy {final_acc:.1%}"
        print(f"[CNN] run '{run_name}' saved. Total runs: {len(slot['runs'])}", flush=True)

    except Exception as exc:
        import traceback
        slot["status"] = "error"
        slot["error"]  = str(exc)
        slot["msg"]    = f"Error: {exc}"
        print(f"[CNN] error: {traceback.format_exc()}", flush=True)


def _cnn_cm_b64(cm, class_names):
    """Render confusion matrix as base64 PNG."""
    import numpy as np
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(3.5, n * 1.1), max(3, n * 1.0)))
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion matrix", fontsize=11, fontweight="bold", pad=8)
    thresh = arr.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    fontsize=10, color="white" if arr[i, j] > thresh else "black")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _cnn_curve_b64(log):
    """Render loss+accuracy curves as base64 PNG."""
    if not log:
        return ""
    epochs   = [e["epoch"]   for e in log]
    tr_loss  = [e["loss"]    for e in log]
    val_loss = [e["val_loss"] for e in log]
    tr_acc   = [e["acc"]     for e in log]
    val_acc  = [e["val_acc"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(epochs, tr_loss,  color=PALETTE[0], label="Train",      linewidth=1.8)
    ax1.plot(epochs, val_loss, color=PALETTE[1], label="Validation", linewidth=1.8, linestyle="--")
    ax1.set_title("Loss", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.legend(fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax2.plot(epochs, tr_acc,  color=PALETTE[0], label="Train",      linewidth=1.8)
    ax2.plot(epochs, val_acc, color=PALETTE[1], label="Validation", linewidth=1.8, linestyle="--")
    ax2.set_title("Accuracy", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/api/cnn_train", methods=["POST"])
def cnn_train():
    """
    Start CNN training in background.
    Body: { node, pre_node, img_data_node, img_cfg: {...}, cfg: {...} }
    img_data_node and img_cfg are sent from the frontend (JS node.data).
    """
    body             = request.get_json(force=True)
    node_id          = body.get("node", "cnn_default")
    cfg              = body.get("cfg", {})
    img_data_node_id = body.get("img_data_node") or cfg.get("imgDataNodeId")
    img_cfg          = body.get("img_cfg") or {}

    print(f"[CNN] /api/cnn_train  node={node_id}  img_data_node={img_data_node_id}  cfg={cfg}  img_cfg={img_cfg}", flush=True)

    # Check a thread isn't already running for this node
    existing = _CNN_THREADS.get(node_id)
    if existing and existing.is_alive():
        print("[CNN] already training, rejecting", flush=True)
        return jsonify({"error": "Already training"}), 409

    img_slot = _IMAGE_NODE_DATA.get(img_data_node_id) if img_data_node_id else None
    print(f"[CNN] img_slot keys={list(img_slot.keys()) if img_slot else None}  loaded={img_slot.get('loaded') if img_slot else None}", flush=True)

    if not img_slot or not img_slot.get("loaded"):
        print(f"[CNN] ERROR: no image dataset. _IMAGE_NODE_DATA keys={list(_IMAGE_NODE_DATA.keys())}", flush=True)
        return jsonify({"error": "No image dataset loaded. Connect and load an Image Data block."}), 400

    merged_cfg = {**img_cfg, **cfg}

    # Apply configurable test split from preprocess config
    test_split_pct = max(0.05, min(0.4, float(merged_cfg.get("testSplit", 0.2))))
    all_data    = img_slot.get("all_data") or (img_slot["train_data"] + img_slot["test_data"])
    class_names = img_slot["class_names"]
    n_classes_total = len(class_names)

    import random as _rnd_split
    by_class_split = {}
    for item in all_data:
        by_class_split.setdefault(item[1], []).append(item)
    train_data, test_data = [], []
    _rnd_split.seed(42)
    for lbl in sorted(by_class_split.keys()):
        items = by_class_split[lbl][:]
        _rnd_split.shuffle(items)
        n_test = max(1, round(len(items) * test_split_pct))
        test_data.extend(items[:n_test])
        train_data.extend(items[n_test:])

    print(f"[CNN] test_split={test_split_pct:.0%} → train={len(train_data)}  test={len(test_data)}", flush=True)

    # Reset slot
    slot = _cnn_slot(node_id)
    slot.update({"status": "training", "cfg": merged_cfg, "log": [], "result": None,
                  "error": None, "epoch": 0, "total_epochs": int(merged_cfg.get("epochs", 5)),
                  "pct": 0, "msg": "Starting…"})

    t = threading.Thread(
        target=_cnn_train_worker,
        args=(node_id, merged_cfg, train_data, test_data, class_names),
        daemon=True
    )
    _CNN_THREADS[node_id] = t
    t.start()
    return jsonify({"started": True})


@app.route("/api/cnn_poll")
def cnn_poll():
    """Poll training progress. Query: ?node=<node_id>"""
    node_id = request.args.get("node", "cnn_default")
    slot    = _cnn_slot(node_id)
    running = _CNN_THREADS.get(node_id) is not None and _CNN_THREADS[node_id].is_alive()
    return jsonify({
        "status":       slot["status"],
        "pct":          slot["pct"],
        "msg":          slot["msg"],
        "epoch":        slot["epoch"],
        "total_epochs": slot["total_epochs"],
        "log":          slot["log"],
        "running":      running,
    })


@app.route("/api/cnn_result")
def cnn_result():
    """Return final result once training is done. Query: ?node=<node_id>"""
    node_id = request.args.get("node", "cnn_default")
    slot    = _cnn_slot(node_id)
    if slot["status"] != "done":
        return jsonify({"error": "Not ready", "status": slot["status"]}), 400
    # Strip non-serialisable keys (PyTorch tensors in _model_state, _arch, _norm_mode)
    safe = {k: v for k, v in slot["result"].items() if not k.startswith("_")}
    return jsonify(safe)


@app.route("/api/cnn_cancel", methods=["POST"])
def cnn_cancel():
    """Cancel an ongoing training run."""
    body    = request.get_json(force=True)
    node_id = body.get("node", "cnn_default")
    slot    = _cnn_slot(node_id)
    slot["status"] = "cancelled"
    return jsonify({"ok": True})


# ── CNN: list runs ────────────────────────────────────────────────────────────
@app.route("/api/cnn_runs")
def cnn_runs():
    """Return all completed runs for a CNN node (without model state / large images)."""
    node_id = request.args.get("node", "cnn_default")
    slot    = _cnn_slot(node_id)
    safe = []
    for r in slot.get("runs", []):
        _log = r.get("log", [])
        _epoch_times = [e.get("epoch_time") for e in _log if e.get("epoch_time") is not None]
        _avg_epoch = round(sum(_epoch_times) / len(_epoch_times), 2) if _epoch_times else None
        safe.append({
            "run_idx":       r["run_idx"],
            "run_name":      r["run_name"],
            "final_acc":     r["final_acc"],
            "macro_f1":      r["macro_f1"],
            "macro_prec":    r["macro_prec"],
            "macro_rec":     r["macro_rec"],
            "per_class":     r["per_class"],
            "cm_img":        r["cm_img"],
            "curve_img":     r["curve_img"],
            "wrong_samples": r["wrong_samples"],
            "log":           _log,
            "class_names":   r["class_names"],
            "n_val":         r["n_val"],
            "n_test":        r["n_test"],
            "cfg":           r["cfg"],
            "total_elapsed": r.get("total_elapsed", 0),
            "avg_epoch_time": _avg_epoch,
        })
    return jsonify({"runs": safe, "status": slot["status"]})


# ── CNN: rename a run ─────────────────────────────────────────────────────────
@app.route("/api/cnn_rename_run", methods=["POST"])
def cnn_rename_run():
    body     = request.get_json(force=True)
    node_id  = body.get("node", "cnn_default")
    run_idx  = int(body.get("run_idx", -1))
    new_name = str(body.get("name", "")).strip()[:60]
    slot     = _cnn_slot(node_id)
    for r in slot.get("runs", []):
        if r["run_idx"] == run_idx:
            r["run_name"] = new_name or r["run_name"]
            print(f"[CNN] run {run_idx} renamed → '{r['run_name']}'", flush=True)
            return jsonify({"ok": True, "name": r["run_name"]})
    return jsonify({"error": "Run not found"}), 404


# ── CNN: predict a single image with a saved run's model ─────────────────────
@app.route("/api/cnn_predict", methods=["POST"])
def cnn_predict():
    """
    Body: { node, run_idx, image_b64 }
    Returns: { class_names, probs, pred_class, pred_prob }
    """
    body      = request.get_json(force=True)
    node_id   = body.get("node", "cnn_default")
    run_idx   = int(body.get("run_idx", -1))
    img_b64   = body.get("image_b64", "")

    slot = _cnn_slot(node_id)
    run  = next((r for r in slot.get("runs", []) if r["run_idx"] == run_idx), None)
    if not run:
        # fallback: use last run
        run = slot["runs"][-1] if slot.get("runs") else None
    if not run or "_model_state" not in run:
        return jsonify({"error": "No trained model found for this run"}), 400

    try:
        import torch, torch.nn as nn
        import numpy as np
        from PIL import Image as PILImage
        import base64, io as _io

        arch      = run["_arch"]
        n_conv    = arch["n_conv"]
        n_ch      = arch["n_ch"]
        img_size  = arch["img_size"]
        n_classes = arch["n_classes"]
        filters   = arch["filters"]
        norm_mode = run["_norm_mode"]

        # Rebuild model
        layers = []
        in_ch  = n_ch
        for i in range(n_conv):
            out_ch = filters[i]
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch  = out_ch
        feat_size = img_size // (2 ** n_conv)
        flatten   = in_ch * feat_size * feat_size
        layers   += [nn.Flatten(), nn.Linear(flatten, 128), nn.ReLU(),
                     nn.Dropout(0.3), nn.Linear(128, n_classes)]
        model = nn.Sequential(*layers)
        model.load_state_dict(run["_model_state"])
        model.eval()

        # Decode image
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        img = PILImage.open(_io.BytesIO(base64.b64decode(img_b64)))
        img = img.convert("L" if n_ch == 1 else "RGB").resize((img_size, img_size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if n_ch == 1:
            arr = arr[..., np.newaxis]
        if norm_mode == "01":
            arr = arr / 255.0
        elif norm_mode == "meanstd":
            arr = (arr / 255.0 - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)
        x   = torch.tensor(arr[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].tolist()

        pred_idx = int(np.argmax(probs))
        return jsonify({
            "class_names": run["class_names"],
            "probs":       [round(p, 4) for p in probs],
            "pred_class":  run["class_names"][pred_idx],
            "pred_prob":   round(probs[pred_idx], 4),
        })

    except Exception as exc:
        import traceback
        print(f"[CNN predict] {traceback.format_exc()}", flush=True)
        return jsonify({"error": str(exc)}), 500


# ── CNN: export evaluation report as HTML (POST: node, lang, img_history) ────
@app.route("/api/cnn_report", methods=["GET", "POST"])
def cnn_report():
    """Generate a self-contained HTML report for all runs of a CNN node.
    POST body: { node, lang: 'es'|'en', img_history: [{b64, filename, results, class_names}] }
    GET (legacy): ?node=... (no images, lang=es)
    """
    if request.method == "POST":
        body    = request.get_json(force=True) or {}
        node_id = body.get("node", "cnn_default")
        lang    = body.get("lang", "es")
        img_history = body.get("img_history", [])
    else:
        node_id     = request.args.get("node", "cnn_default")
        lang        = request.args.get("lang", "es")
        img_history = []

    en = (lang == "en")
    slot = _cnn_slot(node_id)
    runs = slot.get("runs", [])
    if not runs:
        return jsonify({"error": "No runs to report"}), 400

    # ── Labels ───────────────────────────────────────────────────────────────
    L = {
        "title":      "CNN Classification — Informe de Evaluación" if not en else "CNN Classification — Evaluation Report",
        "generated":  f"Generado por NLP Flow · {len(runs)} run(s)" if not en else f"Generated by NLP Flow · {len(runs)} run(s)",
        "comparison": "Comparativa de runs" if not en else "Run comparison",
        "name":       "Nombre" if not en else "Name",
        "epochs":     "Epochs",
        "lr":         "Learning rate",
        "conv":       "Capas conv" if not en else "Conv layers",
        "valsplit":   "Val split",
        "testsplit":  "Test split",
        "acc":        "Accuracy",
        "total_time": "Tiempo total" if not en else "Total time",
        "avg_epoch":  "Media/epoch" if not en else "Avg/epoch",
        "detail":     "Detalle por run" if not en else "Run details",
        "perclass":   "Métricas por clase" if not en else "Per-class metrics",
        "cls":        "Clase" if not en else "Class",
        "prec":       "Precisión" if not en else "Precision",
        "recall":     "Recall",
        "f1":         "F1",
        "curves":     "Curvas de entrenamiento" if not en else "Training curves",
        "cm":         "Matriz de confusión" if not en else "Confusion matrix",
        "images":     "Imágenes predichas por el usuario" if not en else "User-uploaded predictions",
        "pred":       "Predicción" if not en else "Prediction",
        "conf":       "Confianza" if not en else "Confidence",
        "show":       "Ver detalles ▾" if not en else "Show details ▾",
        "hide":       "Ocultar ▴" if not en else "Hide ▴",
        "disclaimer": "",
        "testimg":    "Imágenes de test" if not en else "Test images",
        "val_note":   ("El val split se extrae del train, por lo que el % global efectivo sobre el total de datos es menor."
                       if not en else
                       "The val split is taken from the train set, so the effective global percentage over total data is lower."),
    }

    def _fmt_secs(s):
        s = round(s or 0)
        if s < 60: return f"{s}s"
        return f"{s // 60}m {s % 60:02d}s"

    # ── Comparison table ─────────────────────────────────────────────────────
    best_f1  = max(r["macro_f1"] for r in runs)
    comp_rows = ""
    for r in runs:
        cfg  = r.get("cfg", {})
        best = r["macro_f1"] == best_f1
        vs   = f"{float(cfg.get('val_split',0.2))*100:.0f}%" if cfg.get("val_split") else "—"
        ts   = f"{float(cfg.get('testSplit',0.2))*100:.0f}%" if cfg.get("testSplit") else "—"
        log  = r.get("log", [])
        ep_times = [e.get("epoch_time") for e in log if e.get("epoch_time") is not None]
        avg_ep   = _fmt_secs(sum(ep_times) / len(ep_times)) if ep_times else "—"
        total_t  = _fmt_secs(r.get("total_elapsed", 0)) if r.get("total_elapsed") else "—"
        comp_rows += f"""<tr{'style="background:#f0f4ff"' if best else ''}>
          <td>{'★ ' if best else ''}<b>{r['run_name']}</b></td>
          <td>{cfg.get('epochs','—')}</td><td>{cfg.get('lr','—')}</td>
          <td>{cfg.get('n_conv','—')}</td><td>{vs}</td><td>{ts}</td>
          <td>{r['final_acc']*100:.1f}%</td>
          <td><b>{r['macro_f1']*100:.1f}%</b></td>
          <td>{r['macro_prec']*100:.1f}%</td>
          <td>{r['macro_rec']*100:.1f}%</td>
          <td>{total_t}</td>
          <td>{avg_ep}/ep</td>
        </tr>"""

    # ── Run detail panels (pill selector + single visible panel) ─────────────
    run_pills = ""
    run_panels = ""
    for idx, r in enumerate(runs):
        cfg      = r.get("cfg", {})
        is_best  = r['macro_f1'] == best_f1
        is_first = idx == 0
        pc_rows  = "".join(
            f"<tr><td>{p['label']}</td><td>{p['precision']*100:.1f}%</td>"
            f"<td>{p['recall']*100:.1f}%</td><td><b>{p['f1']*100:.1f}%</b></td><td>{p['support']}</td></tr>"
            for p in r.get("per_class", [])
        )
        pc_macro = (
            f"<tr style='border-top:2px solid #ddd;color:#666'><td><i>macro avg</i></td>"
            f"<td>{r['macro_prec']*100:.1f}%</td><td>{r['macro_rec']*100:.1f}%</td>"
            f"<td><b>{r['macro_f1']*100:.1f}%</b></td><td>{r.get('n_test','—')}</td></tr>"
        )
        curve_tag = (f'<img src="data:image/png;base64,{r["curve_img"]}" '
                     f'style="max-width:640px;width:100%;border-radius:8px;margin:6px 0">')  if r.get("curve_img") else ""
        cm_tag    = (f'<img src="data:image/png;base64,{r["cm_img"]}" '
                     f'style="max-width:340px;border-radius:8px;margin:6px 0">')              if r.get("cm_img")    else ""
        vs = f"{float(cfg.get('val_split', 0.2))*100:.0f}%" if cfg.get("val_split") else "—"
        ts = f"{float(cfg.get('testSplit', 0.2))*100:.0f}%" if cfg.get("testSplit") else "—"
        best_star = "★ " if is_best else ""

        active_pill  = "run-pill-active"  if is_first else "run-pill"
        active_panel = "block"            if is_first else "none"

        run_pills += (
            f'<button class="{active_pill}" id="rpill-{idx}" onclick="selectRun({idx})">'
            f'{best_star}{r["run_name"]}</button>'
        )
        r_log     = r.get("log", [])
        r_eptimes = [e.get("epoch_time") for e in r_log if e.get("epoch_time") is not None]
        r_avg_ep  = _fmt_secs(sum(r_eptimes) / len(r_eptimes)) if r_eptimes else "—"
        r_total_t = _fmt_secs(r.get("total_elapsed", 0)) if r.get("total_elapsed") else "—"
        # Effective val% over total data: val_split is taken from train, so global = (1-test)*(val)
        try:
            _vs_f  = float(cfg.get("val_split", 0.2))
            _ts_f  = float(cfg.get("testSplit", 0.2))
            _eff_v = f"{(1 - _ts_f) * _vs_f * 100:.1f}%"
        except Exception:
            _eff_v = vs

        run_panels += f"""
        <div class="run-panel" id="rpanel-{idx}" style="display:{active_panel}">
          <div class="cfg-pills" style="margin-bottom:14px">
            <span class="pill">Epochs: {cfg.get('epochs','—')}</span>
            <span class="pill">LR: {cfg.get('lr','—')}</span>
            <span class="pill">Conv: {cfg.get('n_conv','—')}</span>
            <span class="pill">Val: {vs} <span style="color:#888;font-size:10px">({_eff_v} {"global" if en else "global"})</span></span>
            <span class="pill">Test: {ts}</span>
            <span class="pill">{L['testimg']}: {r.get('n_test','—')}</span>
          </div>
          <p style="font-size:11px;color:#888;margin:0 0 10px">{L['val_note']}</p>
          <div class="metric-row">
            <div class="metric-card {('metric-card-best' if is_best else '')}">
              <div class="metric-val">{r['final_acc']*100:.1f}%</div>
              <div class="metric-lbl">{L['acc']}</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r['macro_f1']*100:.1f}%</div>
              <div class="metric-lbl">Macro F1</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r['macro_prec']*100:.1f}%</div>
              <div class="metric-lbl">{L['prec']}</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r['macro_rec']*100:.1f}%</div>
              <div class="metric-lbl">{L['recall']}</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r_total_t}</div>
              <div class="metric-lbl">{L['total_time']}</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r_avg_ep}</div>
              <div class="metric-lbl">{L['avg_epoch']}</div>
            </div>
          </div>
          <h3>{L['perclass']}</h3>
          <table><thead><tr>
            <th>{L['cls']}</th><th>{L['prec']}</th><th>{L['recall']}</th><th>{L['f1']}</th><th>N</th>
          </tr></thead><tbody>{pc_rows}{pc_macro}</tbody></table>
          <h3>{L['curves']}</h3>{curve_tag if curve_tag else '<p style="color:#aaa;font-size:12px">—</p>'}
          <h3>{L['cm']}</h3>{cm_tag if cm_tag else '<p style="color:#aaa;font-size:12px">—</p>'}
        </div>"""

    detail_block = f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px">
      <div class="run-pills" id="run-pill-bar">{run_pills}</div>
      {run_panels}
    </div>"""

    # ── User image predictions — clickable rows open prob modal ──────────────
    img_section = ""
    if img_history:
        # Serialise all prob data as JS so the modal can read it
        import json as _json
        modal_data_js = "var IMG_DATA = " + _json.dumps([
            {
                "b64":      h.get("b64", ""),
                "filename": h.get("filename", ""),
                "results":  [
                    {
                        "run_name":   rr.get("run_name", ""),
                        "pred_class": rr.get("pred_class", ""),
                        "pred_prob":  rr.get("pred_prob", 0),
                        "probs":      rr.get("probs", []),
                    }
                    for rr in h.get("results", [])
                ],
                "class_names": h.get("class_names", []),
            }
            for h in img_history
        ]) + ";"

        img_rows = ""
        for hi, h in enumerate(img_history):
            top_run = h["results"][0] if h.get("results") else {}
            img_rows += (
                f'<tr class="img-row" onclick="openImgModal({hi})" style="cursor:pointer">'
                f'<td><img src="{h["b64"]}" style="width:52px;height:52px;object-fit:cover;border-radius:6px;vertical-align:middle"></td>'
                f'<td>{h.get("filename","")}</td>'
                f'<td><b>{top_run.get("pred_class","—")}</b></td>'
                f'<td>{top_run.get("pred_prob",0)*100:.1f}%</td>'
                f'<td style="color:#5B7FDB;font-size:12px">{len(h.get("results",[]))} run(s) →</td>'
                f'</tr>'
            )

        img_section = f"""
<h2>{L['images']}</h2>
<table>
  <thead><tr>
    <th></th>
    <th>{L.get('file','Archivo') if not en else 'File'}</th>
    <th>{L['pred']}</th>
    <th>{L['conf']}</th>
    <th></th>
  </tr></thead>
  <tbody>{img_rows}</tbody>
</table>

<!-- Modal overlay -->
<div id="img-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;align-items:center;justify-content:center">
  <div style="background:#fff;border-radius:14px;max-width:520px;width:90%;padding:24px;position:relative;max-height:85vh;overflow-y:auto">
    <button onclick="closeImgModal()" style="position:absolute;top:12px;right:14px;background:none;border:none;font-size:20px;cursor:pointer;color:#666">✕</button>
    <div id="modal-content"></div>
  </div>
</div>"""

    # ── Inline JS for run selector + image modal ──────────────────────────────
    n_runs = len(runs)
    report_js = f"""
{modal_data_js if img_history else ''}
function selectRun(idx) {{
  for (var i = 0; i < {n_runs}; i++) {{
    var p = document.getElementById('rpanel-' + i);
    var b = document.getElementById('rpill-'  + i);
    if (p) p.style.display = 'none';
    if (b) b.className = 'run-pill';
  }}
  var active = document.getElementById('rpanel-' + idx);
  var activePill = document.getElementById('rpill-' + idx);
  if (active) {{
    active.style.display = 'block';
    void active.offsetWidth; /* force reflow so grid recalculates */
  }}
  if (activePill) activePill.className = 'run-pill-active';
}}
function openImgModal(hi) {{
  var h = IMG_DATA[hi];
  var cls = h.class_names;
  var html = '<img src="' + h.b64 + '" style="width:90px;height:90px;object-fit:cover;border-radius:8px;display:block;margin:0 auto 14px">';
  html += '<div style="font-size:12px;color:#888;text-align:center;margin-bottom:14px">' + (h.filename || '') + '</div>';
  h.results.forEach(function(rr) {{
    var probRows = cls.map(function(c, i) {{
      var pct = (rr.probs[i] * 100).toFixed(1);
      var win = c === rr.pred_class;
      return '<tr style="' + (win ? 'background:#f0f4ff' : '') + '">'
        + '<td style="padding:5px 10px;font-weight:' + (win ? '700' : '400') + '">' + c + '</td>'
        + '<td style="padding:5px 10px;width:90px"><div style="height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden">'
        + '<div style="height:100%;width:' + pct + '%;background:' + (win ? '#5B7FDB' : '#94a3b8') + ';border-radius:4px"></div></div></td>'
        + '<td style="padding:5px 10px;text-align:right;font-weight:' + (win ? '700' : '400') + '">' + pct + '%</td></tr>';
    }}).join('');
    html += '<div style="margin-bottom:12px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden">'
      + '<div style="padding:7px 12px;background:#f8faff;font-size:12px;font-weight:700;display:flex;justify-content:space-between">'
      + '<span>' + rr.run_name + '</span>'
      + '<span style="color:#5B7FDB">' + rr.pred_class + ' ' + (rr.pred_prob * 100).toFixed(1) + '%</span></div>'
      + '<table style="margin:0"><tbody>' + probRows + '</tbody></table></div>';
  }});
  document.getElementById('modal-content').innerHTML = html;
  var m = document.getElementById('img-modal');
  m.style.display = 'flex';
}}
function closeImgModal() {{
  document.getElementById('img-modal').style.display = 'none';
}}
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeImgModal(); }});
"""

    html = f"""<!DOCTYPE html>
<html lang="{lang}"><head><meta charset="utf-8">
<title>{L['title']}</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:940px;margin:40px auto;padding:0 20px;color:#1a1a1a;background:#fafafa}}
  h1{{font-size:22px;border-bottom:2px solid #5B7FDB;padding-bottom:10px;color:#1a1a1a}}
  h2{{font-size:16px;margin:32px 0 12px;color:#5B7FDB;font-weight:700}}
  h3{{font-size:13px;color:#666;margin:14px 0 6px;font-weight:600}}
  table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:14px;background:#fff;border-radius:8px;overflow:hidden}}
  th{{background:#f0f4ff;padding:8px 10px;text-align:left;font-weight:600;font-size:12px}}
  td{{padding:7px 10px;border-bottom:1px solid #f0f0f0}}
  img{{display:block}}
  .img-row:hover{{background:#f8faff}}
  .run-pills{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}}
  .run-pill{{background:#f0f4ff;color:#5B7FDB;border:1.5px solid #c7d4f7;border-radius:8px;padding:5px 14px;font-size:13px;font-weight:600;cursor:pointer}}
  .run-pill:hover{{background:#e0eaff}}
  .run-pill-active{{background:#5B7FDB;color:#fff;border:1.5px solid #5B7FDB;border-radius:8px;padding:5px 14px;font-size:13px;font-weight:600;cursor:pointer}}
  .cfg-pills{{display:flex;flex-wrap:wrap;gap:6px}}
  .pill{{background:#f0f4ff;color:#5B7FDB;border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600}}
  .run-panel{{width:100%;box-sizing:border-box}}
  .metric-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px;width:100%}}
  .metric-card{{background:#f8faff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;text-align:center;min-width:0}}
  .metric-card-best{{border-color:#5B7FDB;background:#f0f4ff}}
  .metric-val{{font-size:20px;font-weight:800;color:#1a1a1a}}
  .metric-lbl{{font-size:11px;color:#888;margin-top:2px}}
</style>
</head><body>
<h1>{L['title']}</h1>
<p style="color:#666;font-size:13px">{L['generated']}</p>

<h2>{L['comparison']}</h2>
<table><thead><tr>
  <th>{L['name']}</th><th>{L['epochs']}</th><th>{L['lr']}</th><th>{L['conv']}</th>
  <th>{L['valsplit']}</th><th>{L['testsplit']}</th>
  <th>{L['acc']}</th><th>Macro F1</th><th>{L['prec']}</th><th>{L['recall']}</th>
  <th>{L['total_time']}</th><th>{L['avg_epoch']}</th>
</tr></thead><tbody>{comp_rows}</tbody></table>

<h2>{L['detail']}</h2>
{detail_block}
{img_section}
<script>{report_js}</script>
</body></html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOENCODER — backend
# ═══════════════════════════════════════════════════════════════════════════════

_AE_SLOTS: dict = {}   # { node_id: { model, encoder, decoder, slot data … } }

def _ae_slot(node_id: str) -> dict:
    if node_id not in _AE_SLOTS:
        _AE_SLOTS[node_id] = {
            "training": False, "cancel": False, "done": False, "error": None,
            "progress": 0, "status": "", "log": [],
            "runs": [],   # list of run dicts stored server-side
            "model": None, "encoder": None, "decoder": None,
            "sample_originals": [], "sample_latents": [],
            "bottleneck": 32,
        }
    return _AE_SLOTS[node_id]


def _ae_train_worker(node_id: str, img_data_node: str, cfg: dict):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np, io, base64
    from PIL import Image

    slot = _ae_slot(node_id)
    slot["training"]    = True
    slot["cancel"]      = False
    slot["done"]        = False
    slot["error"]       = None
    slot["log"]         = []
    slot["progress"]    = 0
    slot["cfg_epochs"]  = int(cfg.get("epochs", 10))   # available to poller during training

    bottleneck   = int(cfg.get("bottleneck", 32))
    epochs       = int(cfg.get("epochs", 10))
    lr           = float(cfg.get("lr", 0.001))
    loss_fn      = cfg.get("loss_fn", "mse").lower()
    batch_size   = int(cfg.get("batch_size", 128))
    hidden_mode  = cfg.get("hidden_mode", "auto")   # "auto" | "wide" | "deep"

    try:
        # ── Load data ────────────────────────────────────────────────────────
        slot["status"]   = "Cargando imágenes…"
        slot["progress"] = 5

        img_slot = _IMAGE_NODE_DATA.get(img_data_node) or {}
        if not img_slot.get("train_data"):
            raise RuntimeError("train_data not found — reload the Image Data block.")

        # ── Dataset properties ───────────────────────────────────────────────
        dataset_name = img_slot.get("dataset", "mnist")
        is_color     = dataset_name in ("dogs_muffins", "cats_vs_dogs")
        # 64×64 for color (more detail → better reconstruction quality)
        # 28×28 for MNIST (standard, trains fast)
        target_size  = 64 if is_color else 28
        n_channels   = 3 if is_color else 1
        input_dim    = n_channels * target_size * target_size

        slot["status"]   = "Preparando tensores…"
        slot["progress"] = 10

        def pil_to_tensor(pil_img):
            if is_color:
                img = pil_img.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                return arr.transpose(2, 0, 1).flatten()   # CHW flat
            else:
                img = pil_img.convert("L").resize((target_size, target_size), Image.BILINEAR)
                return np.array(img, dtype=np.float32).flatten() / 255.0

        # AE uses ALL available images (train + test) — no labels needed,
        # no held-out set required for the autoencoder objective.
        all_pairs = img_slot["train_data"] + img_slot.get("test_data", [])
        all_tensors = [pil_to_tensor(p) for p, _ in all_pairs]
        X_all = torch.tensor(np.array(all_tensors), dtype=torch.float32)

        # Keep a small test split just for MSE reporting (last 10% or 200 images)
        n_test   = max(20, min(200, len(X_all) // 10))
        X_test   = X_all[-n_test:]
        X_train  = X_all[:-n_test]

        train_loader = DataLoader(TensorDataset(X_train, X_train),
                                  batch_size=batch_size, shuffle=True)

        slot["progress"]    = 20
        slot["input_dim"]   = input_dim
        slot["is_color"]    = is_color
        slot["target_size"] = target_size
        slot["n_channels"]  = n_channels
        slot["dataset_name"] = dataset_name

        # ── Architecture ──────────────────────────────────────────────────────
        # Always compress (never expand beyond input_dim).
        # hidden_mode controls the intermediate layer sizes:
        #   auto: h1 ≈ 2/3 input, h2 ≈ 1/3 input  (balanced descent)
        #   wide: h1 ≈ input/2,  h2 ≈ input/4      (wider, more capacity)
        #   deep: h1, h2, h3 equally spaced         (extra compression step)

        def _snap(v, mult):
            return max(1, (int(v) // mult) * mult) or mult

        if hidden_mode == "wide":
            h1 = min(1024, max(bottleneck * 2 + 1, _snap(input_dim / 2, 64)))
            h2 = min(512,  max(bottleneck + 1,      _snap(input_dim / 4, 32)))
            h3 = None
        elif hidden_mode == "deep":
            h1 = min(512,  max(bottleneck * 3 + 2, _snap(input_dim * 3 / 4, 64)))
            h2 = min(256,  max(bottleneck * 2 + 1, _snap(input_dim / 2,     32)))
            h3 = min(128,  max(bottleneck + 1,      _snap(input_dim / 4,     16)))
        else:  # auto
            h1 = min(512, max(bottleneck * 2 + 1, _snap(input_dim * 2 / 3, 64)))
            h2 = min(256, max(bottleneck + 1,      _snap(input_dim / 3,     32)))
            h3 = None

        # Enforce strict descent through all layers
        h1 = min(h1, input_dim - 1)
        h2 = min(h2, h1 - 1)
        h2 = max(h2, bottleneck + 1)
        if h3 is not None:
            h3 = min(h3, h2 - 1)
            h3 = max(h3, bottleneck + 1)

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                if h3 is not None:
                    enc = [nn.Linear(input_dim, h1), nn.ReLU(),
                           nn.Linear(h1, h2),         nn.ReLU(),
                           nn.Linear(h2, h3),         nn.ReLU(),
                           nn.Linear(h3, bottleneck)]
                    dec = [nn.Linear(bottleneck, h3), nn.ReLU(),
                           nn.Linear(h3, h2),         nn.ReLU(),
                           nn.Linear(h2, h1),         nn.ReLU(),
                           nn.Linear(h1, input_dim),  nn.Sigmoid()]
                else:
                    enc = [nn.Linear(input_dim, h1), nn.ReLU(),
                           nn.Linear(h1, h2),         nn.ReLU(),
                           nn.Linear(h2, bottleneck)]
                    dec = [nn.Linear(bottleneck, h2), nn.ReLU(),
                           nn.Linear(h2, h1),         nn.ReLU(),
                           nn.Linear(h1, input_dim),  nn.Sigmoid()]
                self.encoder = nn.Sequential(*enc)
                self.decoder = nn.Sequential(*dec)
            def forward(self, x):
                return self.decoder(self.encoder(x))

        model     = Autoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss() if loss_fn == "mse" else nn.BCELoss()

        arch_str = f"{input_dim}→{h1}→{h2}" + (f"→{h3}" if h3 else "") + f"→{bottleneck}"
        print(f"[AE] arch ({hidden_mode}): {arch_str}→...→{input_dim}  batch={batch_size}", flush=True)

        # ── Training loop ────────────────────────────────────────────────────
        slot["status"] = "Entrenando…"
        log = []
        for ep in range(1, epochs + 1):
            if slot["cancel"]:
                slot["status"]   = "Cancelado"
                slot["training"] = False
                return
            model.train()
            ep_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            ep_loss /= len(train_loader)
            log.append({"epoch": ep, "loss": round(ep_loss, 6)})
            slot["log"]      = log
            slot["progress"] = 20 + int(60 * ep / epochs)
            slot["status"]   = f"Epoch {ep}/{epochs} — loss {ep_loss:.4f}"

        # ── Evaluation ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            xtest    = X_test[:min(512, len(X_test))]
            mse_test = float(nn.MSELoss()(model(xtest), xtest).item())

        slot["progress"] = 85
        slot["status"]   = "Generando visualizaciones…"

        # ── Helpers ──────────────────────────────────────────────────────────
        def tensor_to_b64(t):
            """Convert a flat/CHW float tensor [0,1] to a 112×112 PNG base64 string."""
            arr = (t.detach().numpy() * 255).clip(0, 255).astype(np.uint8)
            if is_color:
                arr = arr.reshape(n_channels, target_size, target_size).transpose(1, 2, 0)
                img = Image.fromarray(arr, "RGB").resize((112, 112), Image.NEAREST)
            else:
                img = Image.fromarray(arr.reshape(target_size, target_size), "L").resize((112, 112), Image.NEAREST)
            buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        def vec_to_b64(v_tensor):
            """Render a 1D activation as a square greyscale heatmap PNG."""
            v = v_tensor.squeeze().detach().numpy()
            vmin, vmax = float(v.min()), float(v.max())
            v_norm = (v - vmin) / max(vmax - vmin, 1e-8)
            side    = max(8, int(np.ceil(np.sqrt(len(v_norm)))))
            pad     = side * side - len(v_norm)
            v_pad   = np.concatenate([v_norm, np.zeros(pad)])
            bar     = (v_pad.reshape(side, side) * 255).astype(np.uint8)
            img_bar = Image.fromarray(bar, "L").resize((112, 112), Image.NEAREST)
            buf = io.BytesIO(); img_bar.save(buf, "PNG"); buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        # ── Sample images ─────────────────────────────────────────────────────
        n_samples = min(20, len(X_test))
        sample_x  = X_test[:n_samples]
        with torch.no_grad():
            sample_recon  = model(sample_x)
            sample_latent = model.encoder(sample_x)

        sample_originals = [tensor_to_b64(sample_x[i])    for i in range(n_samples)]
        sample_recons    = [tensor_to_b64(sample_recon[i]) for i in range(n_samples)]
        sample_latents   = sample_latent.detach().numpy().tolist()

        # ── Layer progression (Compare tab) ───────────────────────────────────
        def layer_progression(x_single):
            imgs = [tensor_to_b64(x_single)]   # original
            with torch.no_grad():
                h = x_single.unsqueeze(0)
                for layer in model.encoder:
                    h = layer(h)
                    if isinstance(layer, nn.ReLU):
                        imgs.append(vec_to_b64(h))
                imgs.append(vec_to_b64(h))          # bottleneck
                h2 = h
                for layer in model.decoder:
                    h2 = layer(h2)
                    if isinstance(layer, nn.ReLU):
                        imgs.append(vec_to_b64(h2))
                imgs.append(tensor_to_b64(model(x_single.unsqueeze(0)).squeeze(0)))  # reconstructed
            return imgs

        progression_imgs = layer_progression(sample_x[0])

        slot["progress"] = 98

        # ── Persist run ───────────────────────────────────────────────────────
        run_idx  = len(slot.get("runs", []))
        run_name = f"Run {run_idx + 1}"
        run_record = {
            "run_idx":          run_idx,
            "run_name":         run_name,
            "name":             run_name,
            "bottleneck":       bottleneck,
            "epochs":           epochs,
            "lr":               lr,
            "loss_fn":          loss_fn,
            "hidden_mode":      hidden_mode,
            "batch_size":       batch_size,
            "mse_test":         round(mse_test, 6),
            "h1":               h1,
            "h2":               h2,
            "h3":               h3,
            "input_dim":        input_dim,
            "is_color":         is_color,
            "target_size":      target_size,
            "n_channels":       n_channels,
            "dataset_name":     dataset_name,
            "log":              log,
            "sample_originals": sample_originals,
            "sample_recons":    sample_recons,
            "latent_vectors":   sample_latents,
            "progression_imgs": progression_imgs,
            "model":            model,           # kept in RAM only
        }
        slot.setdefault("runs", []).append(run_record)

        # Keep slot-level shortcuts for backwards-compat
        slot["model"]            = model
        slot["bottleneck"]       = bottleneck
        slot["sample_originals"] = sample_originals
        slot["sample_latents"]   = sample_latents

        slot["status"]   = f"Listo — MSE test: {mse_test:.4f}"
        slot["progress"] = 100
        slot["done"]     = True
        slot["training"] = False

        # Poller payload — only serialisable fields
        slot["_last_result"] = {
            "bottleneck":       bottleneck,
            "epochs":           epochs,
            "lr":               lr,
            "loss_fn":          loss_fn,
            "hidden_mode":      hidden_mode,
            "batch_size":       batch_size,
            "mse_test":         round(mse_test, 6),
            "h1":               h1,
            "h2":               h2,
            "h3":               h3,
            "input_dim":        input_dim,
            "dataset_name":     dataset_name,
            "is_color":         is_color,
            "log":              log,
            "sample_originals": sample_originals,
            "sample_recons":    sample_recons,
            "latent_vectors":   sample_latents,
            "run_index":        len(slot["runs"]) - 1,
        }

    except Exception as exc:
        import traceback; traceback.print_exc()
        slot["error"]    = str(exc)
        slot["done"]     = True
        slot["training"] = False
        slot["status"]   = f"Error: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# AE EVALUATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ae_eval_data is now a thin wrapper around _ae_eval_runs (defined above)


@app.route("/api/ae_eval_reconstruct", methods=["POST"])
def ae_eval_reconstruct():
    """Reconstruct a user-uploaded image through all AE runs."""
    import torch, numpy as np, io, base64
    from PIL import Image

    body    = request.get_json(force=True) or {}
    node_id = str(body.get("node", "ae_default"))
    img_b64 = body.get("image_b64", "")   # data URI or raw base64

    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    slot = _ae_slot(node_id)
    runs = slot.get("runs", [])
    if not runs:
        return jsonify(error="No runs"), 404

    out = []
    for run_idx, run in enumerate(runs):
        model       = run.get("model")
        is_color    = run.get("is_color", False)
        target_size = run.get("target_size", 28)
        n_channels  = run.get("n_channels", 1)
        if model is None:
            out.append({"run_idx": run_idx, "error": "no model"})
            continue
        try:
            pil = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            if is_color:
                pil = pil.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                x   = torch.tensor(arr.transpose(2,0,1).flatten(), dtype=torch.float32).unsqueeze(0)
            else:
                pil = pil.convert("L").resize((target_size, target_size), Image.BILINEAR)
                x   = torch.tensor(np.array(pil, dtype=np.float32).flatten() / 255.0, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                recon = model(x).squeeze(0)
                lv    = model.encoder(x).squeeze(0)

            def to_b64(t):
                arr2 = (t.detach().numpy() * 255).clip(0,255).astype(np.uint8)
                if is_color:
                    img2 = Image.fromarray(arr2.reshape(n_channels, target_size, target_size).transpose(1,2,0), "RGB")
                else:
                    img2 = Image.fromarray(arr2.reshape(target_size, target_size), "L")
                img2 = img2.resize((112,112), Image.NEAREST)
                buf = io.BytesIO(); img2.save(buf,"PNG"); buf.seek(0)
                return base64.b64encode(buf.read()).decode()

            # Also return the original resized to same display size
            pil_disp = pil.resize((112,112), Image.NEAREST)
            buf = io.BytesIO(); pil_disp.save(buf,"PNG"); buf.seek(0)
            orig_b64 = base64.b64encode(buf.read()).decode()

            out.append({
                "run_idx":    run_idx,
                "name":       run.get("name") or f"Run {run_idx+1}",
                "bottleneck": run.get("bottleneck"),
                "mse_test":   run.get("mse_test"),
                "original":   orig_b64,
                "recon":      to_b64(recon),
                "latent_dim": len(lv),
                "mse_img":    round(float(torch.nn.MSELoss()(recon, x.squeeze(0)).item()), 5),
            })
        except Exception as e:
            out.append({"run_idx": run_idx, "error": str(e)})

    return jsonify(results=out)


def _ae_eval_runs(node_id: str) -> list:
    """Compute evaluation data for all runs of an AE node (shared by ae_eval_data and ae_report)."""
    import torch, torch.nn as nn, numpy as np, io, base64
    from PIL import Image

    slot = _ae_slot(node_id)
    results = []
    for run_idx, run in enumerate(slot.get("runs", [])):
        model       = run.get("model")
        is_color    = run.get("is_color", False)
        target_size = run.get("target_size", 28)
        n_channels  = run.get("n_channels", 1)
        bottleneck  = run.get("bottleneck", 32)
        mse_test    = run.get("mse_test", 0)

        entry = {
            "run_idx":      run_idx,
            "name":         run.get("name") or run.get("run_name") or f"Run {run_idx+1}",
            "bottleneck":   bottleneck,
            "epochs":       run.get("epochs"),
            "lr":           run.get("lr"),
            "loss_fn":      run.get("loss_fn"),
            "hidden_mode":  run.get("hidden_mode", "auto"),
            "batch_size":   run.get("batch_size"),
            "mse_test":     mse_test,
            "dataset_name": run.get("dataset_name", "mnist"),
            "is_color":     is_color,
            "input_dim":    run.get("input_dim"),
            "h1": run.get("h1"), "h2": run.get("h2"), "h3": run.get("h3"),
            "log":          run.get("log", []),
        }

        if model is None:
            results.append(entry)
            continue

        sample_originals = run.get("sample_originals", [])
        sample_recons    = run.get("sample_recons", [])
        latent_vectors   = run.get("latent_vectors", [])
        n = min(len(sample_originals), len(sample_recons))

        def _dec(b64):
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            if is_color:
                img = img.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                return torch.tensor(arr.transpose(2, 0, 1).flatten(), dtype=torch.float32)
            else:
                img = img.convert("L").resize((target_size, target_size), Image.BILINEAR)
                return torch.tensor(np.array(img, dtype=np.float32).flatten() / 255.0, dtype=torch.float32)

        per_img_mse = []
        for i in range(n):
            try:
                mse_i = float(nn.MSELoss()(_dec(sample_recons[i]), _dec(sample_originals[i])).item())
                per_img_mse.append(mse_i)
            except Exception:
                per_img_mse.append(0.0)

        ranked   = sorted(range(len(per_img_mse)), key=lambda i: per_img_mse[i])
        n_show   = min(6, len(ranked))
        best_idx  = ranked[:n_show]
        worst_idx = ranked[-n_show:][::-1]

        entry["best_originals"]  = [sample_originals[i] for i in best_idx]
        entry["best_recons"]     = [sample_recons[i]    for i in best_idx]
        entry["best_mses"]       = [round(per_img_mse[i], 5) for i in best_idx]
        entry["worst_originals"] = [sample_originals[i] for i in worst_idx]
        entry["worst_recons"]    = [sample_recons[i]    for i in worst_idx]
        entry["worst_mses"]      = [round(per_img_mse[i], 5) for i in worst_idx]
        entry["per_img_mse"]     = [round(v, 5) for v in per_img_mse]

        if latent_vectors and len(latent_vectors) >= 4:
            try:
                lv = np.array(latent_vectors, dtype=np.float32)
                if lv.shape[1] == 2:
                    entry["latent_2d"] = lv.tolist()
                else:
                    lv_c  = lv - lv.mean(axis=0)
                    cov   = np.cov(lv_c.T)
                    if cov.ndim < 2: cov = np.array([[float(cov), 0], [0, 0]])
                    evals, evecs = np.linalg.eigh(cov)
                    idx2  = np.argsort(evals)[::-1][:2]
                    proj  = lv_c @ evecs[:, idx2]
                    entry["latent_2d"] = proj.tolist()
            except Exception:
                entry["latent_2d"] = []

        results.append(entry)
    return results


@app.route("/api/ae_eval_data")
def ae_eval_data():
    """Compute evaluation metrics for all runs of an AE node."""
    node_id = request.args.get("node", "ae_default")
    results = _ae_eval_runs(node_id)
    if not results:
        return jsonify(error="No runs"), 404
    mse_bn = [{"bottleneck": r["bottleneck"], "mse": r["mse_test"], "name": r["name"]} for r in results]
    return jsonify(runs=results, mse_vs_bottleneck=mse_bn)


def _svg_bar_chart(labels, values, colors, width=460, height=200, y_label="MSE"):
    """Generate a self-contained SVG bar chart. No external dependencies."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 48
    inner_w = width  - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(values)
    if n == 0:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="12">—</text></svg>'
    max_v = max(values) if max(values) > 0 else 1
    min_v = min(values) * 0.95
    span  = max_v - min_v if max_v != min_v else max_v * 0.1 or 0.001
    bar_w = max(4, inner_w // n - 6)
    step  = inner_w / n

    # Y-axis ticks (5 ticks)
    tick_vals = [min_v + span * i / 4 for i in range(5)]
    grid_lines = ""
    for tv in tick_vals:
        y = pad_t + inner_h - (tv - min_v) / span * inner_h
        label_txt = f"{tv:.4f}" if max_v < 0.01 else f"{tv:.3f}"
        grid_lines += (f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+inner_w}" y2="{y:.1f}" '
                       f'stroke="#f1f5f9" stroke-width="1"/>'
                       f'<text x="{pad_l-4}" y="{y+4:.1f}" text-anchor="end" fill="#94a3b8" '
                       f'font-size="9">{label_txt}</text>')

    bars = ""
    for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
        x = pad_l + i * step + (step - bar_w) / 2
        bar_h = max(2, (val - min_v) / span * inner_h)
        y = pad_t + inner_h - bar_h
        short = lbl[:10] + "…" if len(lbl) > 10 else lbl
        bars += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" '
                 f'fill="{col}" rx="3"/>'
                 f'<text x="{x + bar_w/2:.1f}" y="{pad_t+inner_h+14}" text-anchor="middle" '
                 f'fill="#64748b" font-size="9">{short}</text>'
                 f'<text x="{x + bar_w/2:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
                 f'fill="#374151" font-size="9">{val:.4f}</text>')

    return (f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="overflow:visible">'
            f'<text x="{pad_l}" y="12" fill="#64748b" font-size="9" font-weight="700">{y_label}</text>'
            f'{grid_lines}{bars}'
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+inner_h}" stroke="#e2e8f0" stroke-width="1"/>'
            f'</svg>')


def _svg_line_chart(datasets, n_epochs, width=460, height=200):
    """Generate a self-contained SVG multi-line chart for loss curves."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 40
    inner_w = width  - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    if not datasets or n_epochs < 2:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="12">No loss data</text></svg>'

    all_vals = [v for ds in datasets for v in ds["data"] if v is not None]
    if not all_vals:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="12">No loss data</text></svg>'

    min_v = min(all_vals)
    max_v = max(all_vals)
    span  = max_v - min_v if max_v != min_v else max_v * 0.1 or 0.001

    # Grid lines
    grid = ""
    for i in range(5):
        tv = min_v + span * i / 4
        y  = pad_t + inner_h - (tv - min_v) / span * inner_h
        grid += (f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+inner_w}" y2="{y:.1f}" '
                 f'stroke="#f1f5f9" stroke-width="1"/>'
                 f'<text x="{pad_l-4}" y="{y+4:.1f}" text-anchor="end" fill="#94a3b8" '
                 f'font-size="9">{tv:.4f}</text>')

    lines = ""
    legend = ""
    for di, ds in enumerate(datasets):
        vals   = ds["data"]
        color  = ds.get("borderColor", "#60a5fa")
        lbl    = ds.get("label", f"Run {di+1}")
        pts    = []
        for xi, v in enumerate(vals):
            if v is None: continue
            x = pad_l + xi / max(n_epochs - 1, 1) * inner_w
            y = pad_t + inner_h - (v - min_v) / span * inner_h
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            lines += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        ly = 12 + di * 14
        legend += (f'<rect x="{pad_l+inner_w-90}" y="{pad_t + ly - 6}" width="10" height="4" fill="{color}" rx="2"/>'
                   f'<text x="{pad_l+inner_w-77}" y="{pad_t + ly - 1:.0f}" fill="#374151" font-size="9">{lbl[:14]}</text>')

    # X-axis labels
    x_labels = ""
    tick_n = min(n_epochs, 6)
    for i in range(tick_n):
        xi = i * (n_epochs - 1) // max(tick_n - 1, 1)
        x  = pad_l + xi / max(n_epochs - 1, 1) * inner_w
        x_labels += (f'<text x="{x:.1f}" y="{pad_t+inner_h+12}" text-anchor="middle" '
                     f'fill="#94a3b8" font-size="9">{xi+1}</text>')

    return (f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="overflow:visible">'
            f'<text x="{pad_l}" y="11" fill="#64748b" font-size="9" font-weight="700">Loss</text>'
            f'{grid}{lines}{legend}{x_labels}'
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+inner_h}" stroke="#e2e8f0" stroke-width="1"/>'
            f'</svg>')


@app.route("/api/ae_report", methods=["POST"])
def ae_report():
    """Generate a self-contained interactive HTML report for all AE runs (CNN-style)."""
    import datetime, json as _json
    body      = request.get_json(force=True) or {}
    node_id   = body.get("node", "ae_default")
    lang      = body.get("lang", "es")
    user_imgs = body.get("user_imgs", [])
    pre_info  = body.get("pre_info", {})
    en = (lang == "en")

    slot = _ae_slot(node_id)
    if not slot.get("runs"):
        return jsonify(error="No runs"), 400
    # Use the shared evaluator so best/worst grids are always populated
    runs = _ae_eval_runs(node_id)

    ds_map = {
        "mnist":       "MNIST (28×28, greyscale)",
        "dogs_muffins":"Chihuahua vs Muffin (64×64, RGB)",
        "cats_vs_dogs":"Cats vs Dogs (64×64, RGB)",
    }
    dataset_name = runs[0].get("dataset_name", "mnist")
    pre_size     = pre_info.get("imgSize", runs[0].get("target_size", "?"))
    pre_norm     = pre_info.get("normMode", "[0,1]")

    t_now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    best_mse = min((r.get("mse_test") or 1e9) for r in runs)

    def img_tag(b64, w=72, label=""):
        lbl = f'<div style="font-size:9px;color:#888;margin-top:3px">{label}</div>' if label else ""
        return (f'<div style="display:inline-block;text-align:center">'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:{w}px;height:{w}px;border-radius:5px;border:1.5px solid #e5e7eb;'
                f'image-rendering:pixelated;display:block">{lbl}</div>')

    # ── Runs table rows ───────────────────────────────────────────────────────
    runs_rows = ""
    for i, r in enumerate(runs):
        is_best  = abs((r.get("mse_test") or 1e9) - best_mse) < 1e-9
        row_bg   = "background:#f0fdf4" if is_best else ""
        arch     = f"{r.get('input_dim','?')}→{r.get('h1','?')}→{r.get('h2','?')}"
        if r.get("h3"): arch += f"→{r.get('h3')}"
        arch += f"→{r.get('bottleneck','?')}"
        runs_rows += f"""
<tr class="run-row" data-idx="{i}" style="cursor:pointer;{row_bg}" onclick="toggleDetail({i})">
  <td style="padding:9px 12px;font-weight:{'700' if is_best else '400'}">
    {r.get('name') or f'Run {i+1}'} {'<span class="star">★</span>' if is_best else ''}
  </td>
  <td style="text-align:right;padding:9px 12px">{r.get('bottleneck','—')}</td>
  <td style="text-align:right;padding:9px 12px">{r.get('hidden_mode','auto')}</td>
  <td style="text-align:right;padding:9px 12px">{r.get('epochs','—')}</td>
  <td style="text-align:right;padding:9px 12px">{r.get('lr','—')}</td>
  <td style="text-align:right;padding:9px 12px">{(r.get('loss_fn') or '—').upper()}</td>
  <td style="text-align:right;padding:9px 12px;font-weight:700;color:{'#16a34a' if is_best else '#111'}">
    {f"{r.get('mse_test',0):.5f}"}
  </td>
</tr>
<tr id="detail-{i}" class="detail-row" style="display:none;background:#fafafa">
  <td colspan="7" style="padding:12px 20px">
    <div style="display:flex;gap:32px;flex-wrap:wrap;font-size:12px">
      <div><span class="dlabel">{'Arquitectura' if not en else 'Architecture'}</span><br><code style="font-size:11px">{arch}</code></div>
      <div><span class="dlabel">MSE test</span><br><b>{f"{r.get('mse_test',0):.5f}"}</b></div>
      <div><span class="dlabel">Loss fn</span><br>{(r.get('loss_fn') or '—').upper()}</div>
      <div><span class="dlabel">LR</span><br>{r.get('lr','—')}</div>
      <div><span class="dlabel">Epochs</span><br>{r.get('epochs','—')}</div>
      <div><span class="dlabel">Batch</span><br>{r.get('batch_size','—')}</div>
    </div>
  </td>
</tr>"""

    # ── Grids: table rows + JS data for modal ────────────────────────────────
    _best_run_mse = min((r.get("mse_test") or 1e9) for r in runs)
    recon_rows    = ""
    recon_data    = []   # serialised as JS array for the modal

    def _thumb(b64, w=48):
        if not b64:
            return f'<div style="width:{w}px;height:{w}px;background:#f1f5f9;border-radius:4px"></div>'
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="width:{w}px;height:{w}px;border-radius:4px;border:1px solid #e5e7eb;'
                f'image-rendering:pixelated;display:block;object-fit:contain">')

    for i, r in enumerate(runs):
        best_o  = r.get("best_originals",  [])
        best_rc = r.get("best_recons",     [])
        best_m  = r.get("best_mses",       [])
        worst_o = r.get("worst_originals", [])
        worst_rc= r.get("worst_recons",    [])
        worst_m = r.get("worst_mses",      [])
        if not best_o: continue

        name = r.get("name") or f"Run {i+1}"
        _is_best_run = abs((r.get("mse_test") or 1e9) - _best_run_mse) < 1e-9
        _star_html   = ' <span style="color:#16a34a">★</span>' if _is_best_run else ''

        # Preview: first best + first worst thumbnail pair shown inline
        preview_best  = (_thumb(best_rc[0])  if best_rc  else "") + (_thumb(best_o[0])  if best_o  else "")
        preview_worst = (_thumb(worst_rc[0]) if worst_rc else "") + (_thumb(worst_o[0]) if worst_o else "")

        recon_rows += (
            f'<tr class="img-row" style="cursor:pointer" onclick="openReconModal({i})">'
            f'<td style="font-weight:{"700" if _is_best_run else "400"}">'
            f'  {name}{_star_html}</td>'
            f'<td>{r.get("bottleneck","—")}</td>'
            f'<td style="text-align:right">{r.get("mse_test",0):.5f}</td>'
            f'<td><div style="display:flex;gap:4px;align-items:center">{preview_best}</div></td>'
            f'<td><div style="display:flex;gap:4px;align-items:center">{preview_worst}</div></td>'
            f'<td style="color:#16a34a;font-size:12px">{'ver →' if not en else 'view →'}</td>'
            f'</tr>'
        )

        # Build data entry for the modal
        best_pairs  = [{"orig": best_o[j],  "recon": best_rc[j],  "mse": round(best_m[j],  5)}
                       for j in range(len(best_o))]
        worst_pairs = [{"orig": worst_o[j], "recon": worst_rc[j], "mse": round(worst_m[j], 5)}
                       for j in range(len(worst_o))]
        recon_data.append({
            "name":       name,
            "bottleneck": r.get("bottleneck", "?"),
            "mse_test":   round(r.get("mse_test") or 0, 5),
            "is_best":    _is_best_run,
            "best":       best_pairs,
            "worst":      worst_pairs,
        })

    grids_html = recon_rows  # used in HTML template below

    # ── User images (interactive table + modal, same pattern as CNN report) ──
    user_html        = ""
    ae_modal_data_js = ""
    ae_modal_entries = []
    if user_imgs:
        # Build JS data array for the modal
        ae_modal_entries = []
        img_rows_ae      = ""
        for hi, uimg in enumerate(user_imgs):
            fname   = uimg.get("filename", "image")
            b64_src = uimg.get("b64", "")
            results = uimg.get("results", [])
            if not results:
                continue
            # Best single run by lowest MSE for the summary column
            best_r2  = min(results, key=lambda x: x.get("mse_img", 1e9)) if results else {}
            best_mse_img = best_r2.get("mse_img", 0)
            ae_modal_entries.append({
                "b64":      b64_src,
                "filename": fname,
                "results":  results,
            })
            img_rows_ae += (
                f'<tr class="img-row" style="cursor:pointer" onclick="openAEImgModal({hi})">'
                f'<td><img src="{b64_src}" style="width:52px;height:52px;object-fit:contain;'
                f'border-radius:6px;vertical-align:middle;image-rendering:pixelated"></td>'
                f'<td>{fname}</td>'
                f'<td style="text-align:right">{best_mse_img:.4f}</td>'
                f'<td style="color:#16a34a;font-size:12px">{len(results)} run(s) →</td>'
                f'</tr>'
            )

        ae_modal_data_js = f"var AE_IMG_DATA = {_json.dumps(ae_modal_entries)};"

        user_html = f"""
<table>
  <thead><tr>
    <th></th>
    <th>{'Archivo' if not en else 'File'}</th>
    <th style="text-align:right">{'Mejor MSE' if not en else 'Best MSE'}</th>
    <th></th>
  </tr></thead>
  <tbody>{img_rows_ae}</tbody>
</table>

<!-- AE image modal -->
<div id="ae-img-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;align-items:center;justify-content:center">
  <div style="background:#fff;border-radius:14px;max-width:580px;width:92%;padding:24px;position:relative;max-height:88vh;overflow-y:auto">
    <button onclick="closeAEImgModal()" style="position:absolute;top:12px;right:14px;background:none;border:none;font-size:20px;cursor:pointer;color:#666">✕</button>
    <div id="ae-modal-content"></div>
  </div>
</div>"""

    # ── Chart data ────────────────────────────────────────────────────────────
    # ── SVG charts (no CDN dependency — works with file://) ──────────────────
    palette = ["#4ade80","#60a5fa","#f59e0b","#f87171","#a78bfa","#34d399"]

    # Bar chart: MSE per run
    _bn_labels = [r.get("name") or f"Run {i+1}" for i, r in enumerate(runs)]
    _bn_values = [round(r.get("mse_test") or 0, 5) for r in runs]
    _best_mse_val = min(_bn_values) if _bn_values else 0
    _bn_colors  = ["#4ade80" if abs(v - _best_mse_val) < 1e-9 else "#60a5fa" for v in _bn_values]
    svg_bar = _svg_bar_chart(_bn_labels, _bn_values, _bn_colors,
                             width=460, height=200,
                             y_label="MSE test")

    # Loss curves
    loss_datasets = []
    for i, r in enumerate(runs):
        log  = r.get("log", [])
        vals = [e.get("loss") for e in log if e.get("loss") is not None]
        if vals:
            color = palette[i % len(palette)]
            loss_datasets.append({"label": (r.get("name") or f"Run {i+1}"),
                                  "data": vals, "borderColor": color})
    n_epochs_max = max((len(r.get("log",[])) for r in runs), default=1)
    svg_loss = _svg_line_chart(loss_datasets, n_epochs_max, width=460, height=200)

    html = f"""<!DOCTYPE html>
<html lang="{'es' if not en else 'en'}">
<head>
<meta charset="UTF-8">
<title>{'Autoencoder — Informe' if not en else 'Autoencoder — Report'}</title>

<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#111;min-height:100vh}}
  .header{{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);color:#fff;padding:28px 40px}}
  .header h1{{font-size:24px;font-weight:800;letter-spacing:-.5px}}
  .header p{{font-size:12px;color:#94a3b8;margin-top:6px}}
  .container{{max-width:1080px;margin:0 auto;padding:32px 24px}}
  .card{{background:#fff;border-radius:12px;border:1px solid #e2e8f0;margin-bottom:24px;overflow:hidden}}
  .card-header{{padding:14px 20px;border-bottom:1px solid #f1f5f9;display:flex;align-items:center;gap:8px}}
  .card-header h2{{font-size:14px;font-weight:700;color:#111}}
  .card-body{{padding:20px}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:0}}
  .stat{{background:#f8fafc;border-radius:10px;padding:14px 16px;border:1px solid #e2e8f0}}
  .stat-label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:4px}}
  .stat-val{{font-size:22px;font-weight:800}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead th{{padding:8px 12px;background:#f8fafc;border-bottom:2px solid #e2e8f0;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#64748b;text-align:right}}
  thead th:first-child{{text-align:left}}
  tbody td{{padding:9px 12px;border-bottom:1px solid #f1f5f9;text-align:right}}
  tbody td:first-child{{text-align:left}}
  .run-row:hover{{background:#f0fdf4!important}}
  .detail-row td{{padding:12px 20px}}
  .star{{color:#16a34a;font-size:14px}}
  .dlabel{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8}}
  code{{background:#f1f5f9;border-radius:4px;padding:2px 6px;font-size:11px}}
  details summary{{list-style:none}} details summary::-webkit-details-marker{{display:none}}
  .disclaimer{{display:none}}
  .chart-wrap{{overflow-x:auto;padding:4px 0}}
</style>
</head>
<body>
<div class="header">
  <h1>🔄 {'Autoencoder — Informe de Evaluación' if not en else 'Autoencoder — Evaluation Report'}</h1>
  <p>{'Generado por NLP Flow' if not en else 'Generated by NLP Flow'} · {len(runs)} run(s) · {t_now}</p>
</div>

<div class="container">

<!-- Stat cards -->
<div class="stat-grid" style="margin-bottom:24px">
  <div class="stat"><div class="stat-label">{'Mejor MSE' if not en else 'Best MSE'}</div>
    <div class="stat-val" style="color:#16a34a">{best_mse:.5f}</div></div>
  <div class="stat"><div class="stat-label">Runs</div>
    <div class="stat-val" style="color:#3b82f6">{len(runs)}</div></div>
  <div class="stat"><div class="stat-label">Dataset</div>
    <div class="stat-val" style="font-size:13px;font-weight:700;color:#111">{ds_map.get(dataset_name,dataset_name)}</div></div>
  <div class="stat"><div class="stat-label">{'Tamaño imagen' if not en else 'Image size'}</div>
    <div class="stat-val" style="font-size:16px">{pre_size}×{pre_size} px</div></div>
</div>

<!-- Runs table -->
<div class="card">
  <div class="card-header"><h2>📊 {'Comparativa de runs' if not en else 'Run comparison'}</h2>
    <span style="font-size:11px;color:#94a3b8">({'haz clic para ver detalles' if not en else 'click to expand details'})</span></div>
  <div class="card-body" style="padding:0">
    <table>
      <thead><tr>
        <th style="text-align:left">Run</th>
        <th>Bottleneck</th><th>{'Capas' if not en else 'Hidden'}</th>
        <th>Epochs</th><th>LR</th><th>Loss fn</th><th>MSE test</th>
      </tr></thead>
      <tbody>{runs_rows}</tbody>
    </table>
  </div>
</div>

<!-- Charts -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">
  <div class="card">
    <div class="card-header"><h2>📉 MSE vs Bottleneck</h2></div>
    <div class="card-body"><div class="chart-wrap">{svg_bar}</div></div>
  </div>
  <div class="card">
    <div class="card-header"><h2>📈 {'Curvas de pérdida' if not en else 'Loss curves'}</h2></div>
    <div class="card-body"><div class="chart-wrap">{svg_loss if loss_datasets else '<p style="font-size:11px;color:#94a3b8">Sin datos de log.</p>'}</div>
    </div>
  </div>
</div>

<!-- Reconstructions table -->
<div class="card">
  <div class="card-header">
    <h2>🖼️ {'Reconstrucciones' if not en else 'Reconstructions'}</h2>
    <span style="font-size:11px;color:#94a3b8">({'haz clic para ver todas' if not en else 'click to see all'})</span>
  </div>
  <div class="card-body" style="padding:0">
    {'<table><thead><tr>'
     '<th style="text-align:left">Run</th>'
     '<th>Bottleneck</th>'
     '<th style="text-align:right">MSE test</th>'
     '<th>' + ('✅ Mejor' if not en else '✅ Best') + '</th>'
     '<th>' + ('❌ Peor'  if not en else '❌ Worst') + '</th>'
     '<th></th>'
     '</tr></thead><tbody>' + grids_html + '</tbody></table>'
     if grids_html else '<p style="color:#94a3b8;padding:20px">—</p>'}
  </div>
</div>

<!-- Recon modal -->
<div id="recon-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;align-items:center;justify-content:center">
  <div style="background:#fff;border-radius:14px;max-width:640px;width:94%;padding:24px;position:relative;max-height:88vh;overflow-y:auto">
    <button onclick="closeReconModal()" style="position:absolute;top:12px;right:14px;background:none;border:none;font-size:20px;cursor:pointer;color:#666">✕</button>
    <div id="recon-modal-content"></div>
  </div>
</div>

{'<!-- User images --><div class="card"><div class="card-header"><h2>👤 ' + ("Imágenes del usuario" if not en else "User images") + '</h2></div><div class="card-body">' + user_html + '</div></div>' if user_html else ''}


</div>

<script>
// Runs table collapsible
function toggleDetail(i) {{
  var d = document.getElementById('detail-'+i);
  if (d) d.style.display = d.style.display === 'none' ? 'table-row' : 'none';
}}

// Reconstructions modal
var RECON_DATA = {_json.dumps(recon_data)};
var _r_lbl_best  = '{'Mejores reconstrucciones' if not en else 'Best reconstructions'}';
var _r_lbl_worst = '{'Peores reconstrucciones'  if not en else 'Worst reconstructions'}';
var _r_lbl_orig  = 'original';
var _r_lbl_recon = '{'reconstruida' if not en else 'reconstructed'}';
function openReconModal(ri) {{
  var d = RECON_DATA[ri];
  if (!d) return;
  var star = d.is_best ? ' <span style="color:#16a34a">★</span>' : '';
  var out = '<h3 style="font-size:15px;font-weight:800;margin:0 0 4px">' + d.name + star + '</h3>';
  out += '<div style="font-size:12px;color:#64748b;margin-bottom:16px">Bottleneck ' + d.bottleneck + ' &nbsp;·&nbsp; MSE ' + d.mse_test.toFixed(5) + '</div>';

  function pairRow(pairs, label, color) {{
    if (!pairs || !pairs.length) return '';
    var cells = pairs.map(function(p) {{
      return '<div style="display:flex;flex-direction:column;align-items:center;gap:3px;'
        + 'padding:8px;border-radius:8px;border:1px solid #f0f0f0;background:#fff">'
        + (p.orig  ? '<img src="data:image/png;base64,' + p.orig  + '" style="width:64px;height:64px;border-radius:4px;image-rendering:pixelated;display:block">' : '')
        + '<div style="font-size:9px;color:#888">' + _r_lbl_orig + '</div>'
        + (p.recon ? '<img src="data:image/png;base64,' + p.recon + '" style="width:64px;height:64px;border-radius:4px;border:1.5px solid ' + color + ';image-rendering:pixelated;display:block">' : '')
        + '<div style="font-size:9px;color:' + color + '">' + _r_lbl_recon + '</div>'
        + '<div style="font-size:9px;color:#94a3b8">MSE ' + p.mse.toFixed(4) + '</div>'
        + '</div>';
    }}).join('');
    return '<div style="font-size:11px;font-weight:700;color:' + color + ';margin:12px 0 8px">' + label + '</div>'
      + '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">' + cells + '</div>';
  }}

  out += pairRow(d.best,  '✅ ' + _r_lbl_best,  '#16a34a');
  out += pairRow(d.worst, '❌ ' + _r_lbl_worst, '#dc2626');
  document.getElementById('recon-modal-content').innerHTML = out;
  document.getElementById('recon-modal').style.display = 'flex';
}}
function closeReconModal() {{
  document.getElementById('recon-modal').style.display = 'none';
}}
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') {{ closeReconModal(); closeAEImgModal(); }} }});

// AE user-image modal
var AE_IMG_DATA = {_json.dumps(ae_modal_entries)};
var _lbl_recon = '{'reconstruida' if not en else 'reconstructed'}';
var _lbl_orig  = 'original';
function openAEImgModal(hi) {{
  var h = AE_IMG_DATA[hi];
  if (!h) return;
  var out = '';
  if (h.b64) {{
    out += '<img src="' + h.b64 + '" style="width:90px;height:90px;object-fit:contain;border-radius:8px;display:block;margin:0 auto 10px;image-rendering:pixelated">';
  }}
  out += '<div style="font-size:12px;color:#888;text-align:center;margin-bottom:14px">' + (h.filename||'') + '</div>';
  var results = h.results || [];
  var bestMse = Infinity;
  results.forEach(function(r) {{ if ((r.mse_img||Infinity) < bestMse) bestMse = r.mse_img||Infinity; }});
  results.forEach(function(r) {{
    var isBest = Math.abs((r.mse_img||Infinity) - bestMse) < 1e-9;
    var orig  = r.original ? '<img src="data:image/png;base64,' + r.original + '" style="width:64px;height:64px;border-radius:5px;border:1.5px solid #e5e7eb;image-rendering:pixelated;display:block">' : '';
    var recon = r.recon    ? '<img src="data:image/png;base64,' + r.recon    + '" style="width:64px;height:64px;border-radius:5px;border:1.5px solid #16a34a;image-rendering:pixelated;display:block">' : '';
    var starHtml = isBest ? ' <span style="color:#16a34a;font-size:13px">★</span>' : '';
    out += '<div style="margin-bottom:12px;border:1px solid ' + (isBest?'#16a34a':'#e5e7eb') + ';border-radius:8px;overflow:hidden">'
      + '<div style="padding:8px 12px;background:' + (isBest?'#f0fdf4':'#f8fafc') + ';font-size:12px;font-weight:700;display:flex;justify-content:space-between;align-items:center">'
      + '<span>' + (r.name||'Run') + ' — BN ' + (r.bottleneck||'?') + starHtml + '</span>'
      + '<span style="color:#16a34a">MSE ' + (r.mse_img||0).toFixed(4) + '</span></div>'
      + '<div style="padding:12px;display:flex;gap:16px;align-items:flex-start">'
      + '<div style="display:flex;flex-direction:column;align-items:center;gap:4px">' + orig
      + '<div style="font-size:9px;color:#888;margin-top:2px">' + _lbl_orig + '</div></div>'
      + '<div style="display:flex;flex-direction:column;align-items:center;gap:4px">' + recon
      + '<div style="font-size:9px;color:#16a34a;margin-top:2px">' + _lbl_recon + '</div></div>'
      + '</div></div>';
  }});
  document.getElementById('ae-modal-content').innerHTML = out;
  document.getElementById('ae-img-modal').style.display = 'flex';
}}
function closeAEImgModal() {{
  document.getElementById('ae-img-modal').style.display = 'none';
}}
</script>
</body>
</html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/ae_rename_run", methods=["POST"])
def ae_rename_run():
    body     = request.get_json(force=True) or {}
    node_id  = str(body.get("node", "ae_default"))
    run_idx  = int(body.get("run_idx", -1))
    new_name = str(body.get("name", "")).strip()[:60]
    slot     = _ae_slot(node_id)
    for r in slot.get("runs", []):
        if r.get("run_idx") == run_idx:
            r["run_name"] = new_name or r.get("run_name", f"Run {run_idx+1}")
            r["name"]     = r["run_name"]
            return jsonify({"ok": True, "name": r["run_name"]})
    return jsonify({"error": "Run not found"}), 404


@app.route("/api/ae_runs")
def ae_runs():
    """Return all completed AE runs for a node (summary, no model weights)."""
    node_id = request.args.get("node", "ae_default")
    slot    = _ae_slot(node_id)
    safe = []
    for r in slot.get("runs", []):
        _log = r.get("log", [])
        safe.append({
            "run_idx":    r.get("run_idx", 0),
            "run_name":   r.get("run_name") or r.get("name") or f"Run {r.get('run_idx',0)+1}",
            "bottleneck": r.get("bottleneck"),
            "epochs":     r.get("epochs"),
            "lr":         r.get("lr"),
            "loss_fn":    r.get("loss_fn"),
            "hidden_mode":r.get("hidden_mode"),
            "batch_size": r.get("batch_size"),
            "mse_test":   r.get("mse_test"),
            "h1":         r.get("h1"),
            "h2":         r.get("h2"),
            "h3":         r.get("h3"),
            "input_dim":  r.get("input_dim"),
            "dataset_name": r.get("dataset_name"),
            "is_color":   r.get("is_color"),
            "log":        _log,
        })
    return jsonify({"runs": safe, "status": slot.get("status","")})


@app.route("/api/ae_train", methods=["POST"])
def ae_train():
    body          = request.get_json(force=True) or {}
    node_id       = str(body.get("node", "ae_default"))
    img_data_node = body.get("img_data_node")
    cfg           = body.get("cfg", {})

    slot = _ae_slot(node_id)
    if slot["training"]:
        return jsonify(error="Already training"), 409

    # Validate image slot before launching worker (same pattern as CNN)
    img_slot = _IMAGE_NODE_DATA.get(str(img_data_node)) if img_data_node else None
    print(f"[AE] ae_train node={node_id} img_data_node={img_data_node} "
          f"img_slot_loaded={img_slot.get('loaded') if img_slot else None} "
          f"_IMAGE_NODE_DATA keys={list(_IMAGE_NODE_DATA.keys())}", flush=True)

    if not img_slot or not img_slot.get("loaded"):
        return jsonify(error="No image dataset loaded. Connect and load an Image Data block first."), 400

    slot["done"]     = False
    slot["training"] = True   # set here so UI updates immediately
    slot["cancel"]   = False
    slot["error"]    = None
    slot["progress"] = 0
    slot["status"]   = "Iniciando…"
    slot["log"]      = []
    t = threading.Thread(
        target=_ae_train_worker,
        args=(node_id, str(img_data_node), cfg),
        daemon=True
    )
    t.start()
    return jsonify(ok=True, job_id=node_id)


@app.route("/api/ae_poll")
def ae_poll():
    node_id = request.args.get("node", "ae_default")
    slot    = _ae_slot(node_id)

    resp = {
        "training": slot["training"],
        "done":     slot["done"],
        "progress": slot["progress"],
        "status":   slot["status"],
        "log":      slot["log"],
        "error":    slot["error"],
        "epochs":   slot.get("cfg_epochs"),   # total epochs, available during training
    }
    if slot["done"] and not slot["error"]:
        resp.update(slot.get("_last_result", {}))
    return jsonify(resp)


@app.route("/api/ae_run_data")
def ae_run_data():
    """Return sample images and full layer progression for a specific run+image (Compare tab)."""
    import torch, torch.nn as nn, numpy as np, io, base64
    from PIL import Image

    node_id = request.args.get("node", "ae_default")
    run_idx = int(request.args.get("run_idx", -1))
    img_idx = int(request.args.get("img_idx", 0))

    slot = _ae_slot(node_id)
    runs = slot.get("runs", [])
    if not runs:
        return jsonify(error="No runs"), 404

    run = runs[run_idx] if 0 <= run_idx < len(runs) else runs[-1]
    n   = len(run.get("sample_originals", []))
    if img_idx >= n:
        img_idx = 0

    model       = run.get("model")
    is_color    = run.get("is_color", False)
    target_size = run.get("target_size", 28)
    n_channels  = run.get("n_channels", 1)

    # ── Helpers (same as worker) ─────────────────────────────────────────────
    def tensor_to_b64(t):
        arr = (t.detach().numpy() * 255).clip(0, 255).astype(np.uint8)
        if is_color:
            arr = arr.reshape(n_channels, target_size, target_size).transpose(1, 2, 0)
            img = Image.fromarray(arr, "RGB").resize((112, 112), Image.NEAREST)
        else:
            img = Image.fromarray(arr.reshape(target_size, target_size), "L").resize((112, 112), Image.NEAREST)
        buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def vec_to_b64(v_tensor):
        v = v_tensor.squeeze().detach().numpy()
        vmin, vmax = float(v.min()), float(v.max())
        v_norm = (v - vmin) / max(vmax - vmin, 1e-8)
        side   = max(8, int(np.ceil(np.sqrt(len(v_norm)))))   # ceil — never negative pad
        pad    = side * side - len(v_norm)
        v_pad  = np.concatenate([v_norm, np.zeros(pad)])
        bar    = (v_pad.reshape(side, side) * 255).astype(np.uint8)
        img_b  = Image.fromarray(bar, "L").resize((112, 112), Image.NEAREST)
        buf = io.BytesIO(); img_b.save(buf, "PNG"); buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ── Build progression for the requested image ────────────────────────────
    # For img_idx==0 the stored progression is already there; for any other index
    # we re-run the model layer-by-layer (identical logic to the worker).
    if img_idx == 0 and run.get("progression_imgs"):
        progression = run["progression_imgs"]
    elif model is not None:
        latent_vecs = run.get("latent_vectors", [])
        # Reconstruct the original flat tensor from the stored latent + original image
        # We decode the stored original PNG back to a tensor so the encoder sees the
        # exact same input that was used during training (already preprocessed).
        orig_bytes = base64.b64decode(run["sample_originals"][img_idx])
        pil_orig   = Image.open(io.BytesIO(orig_bytes))
        if is_color:
            pil_orig = pil_orig.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
            x_arr    = np.array(pil_orig, dtype=np.float32) / 255.0
            x_flat   = torch.tensor(x_arr.transpose(2, 0, 1).flatten(), dtype=torch.float32)
        else:
            pil_orig = pil_orig.convert("L").resize((target_size, target_size), Image.BILINEAR)
            x_flat   = torch.tensor(np.array(pil_orig, dtype=np.float32).flatten() / 255.0,
                                    dtype=torch.float32)

        progression = [tensor_to_b64(x_flat)]   # original
        with torch.no_grad():
            h = x_flat.unsqueeze(0)
            for layer in model.encoder:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    progression.append(vec_to_b64(h))
            progression.append(vec_to_b64(h))   # bottleneck
            h2 = h
            for layer in model.decoder:
                h2 = layer(h2)
                if isinstance(layer, nn.ReLU):
                    progression.append(vec_to_b64(h2))
            progression.append(tensor_to_b64(model(x_flat.unsqueeze(0)).squeeze(0)))  # output
    else:
        # Model not in RAM (e.g. server restarted) — fall back to stored originals/recons
        orig  = run["sample_originals"][img_idx]
        recon = (run.get("sample_recons") or [])[img_idx] if img_idx < len(run.get("sample_recons") or []) else None
        progression = [orig] + ([recon] if recon else [])

    return jsonify(
        sample_originals = run.get("sample_originals", []),
        sample_recons    = run.get("sample_recons", []),
        progression_imgs = progression,
        n_samples        = n,
        dataset_name     = run.get("dataset_name", "mnist"),
        is_color         = run.get("is_color", False),
    )


@app.route("/api/ae_cancel", methods=["POST"])
def ae_cancel():
    node_id = str((request.get_json(force=True) or {}).get("node", "ae_default"))
    slot = _ae_slot(node_id)
    slot["cancel"]   = True
    slot["training"] = False   # force-unlock so a new run can start
    slot["done"]     = True
    slot["status"]   = "Cancelado"
    return jsonify(ok=True)

@app.route("/api/ae_reset", methods=["POST"])
def ae_reset():
    """Clear all runs and state for an AE node (called when upstream dataset changes)."""
    node_id = str((request.get_json(force=True) or {}).get("node", "ae_default"))
    slot = _ae_slot(node_id)
    slot["cancel"] = True   # stop any running worker
    # Re-initialise to clean state
    _AE_SLOTS[node_id] = {
        "training": False, "cancel": False, "done": False, "error": None,
        "progress": 0, "status": "", "log": [],
        "runs": [], "model": None,
        "sample_originals": [], "sample_latents": [], "bottleneck": 32,
    }
    return jsonify(ok=True)


@app.route("/api/ae_reconstruct", methods=["POST"])
def ae_reconstruct():
    import torch, numpy as np, io, base64
    from PIL import Image

    body    = request.get_json(force=True) or {}
    node_id = str(body.get("node", "ae_default"))
    run_idx = int(body.get("run_idx", -1))
    img_idx = int(body.get("img_idx", 0))
    noise   = float(body.get("noise", 0.0))

    slot = _ae_slot(node_id)
    runs = slot.get("runs", [])
    if not runs:
        return jsonify(error="No trained runs"), 404

    # Always read model AND samples from the specific run — not from the slot globals
    run = runs[run_idx] if 0 <= run_idx < len(runs) else runs[-1]
    model         = run.get("model")
    is_color      = run.get("is_color", False)
    target_size   = run.get("target_size", 28)
    n_channels    = run.get("n_channels", 1)
    latent_vectors = run.get("latent_vectors", [])
    sample_originals = run.get("sample_originals", [])

    if model is None:
        return jsonify(error="Model not in memory — retrain"), 404

    if img_idx >= len(sample_originals):
        img_idx = 0

    # Use stored latent vector for this image (already encoded by this run's model)
    if latent_vectors and img_idx < len(latent_vectors):
        lv = torch.tensor(latent_vectors[img_idx], dtype=torch.float32)
    else:
        # Fallback: re-encode from the stored original PNG
        orig_bytes = base64.b64decode(sample_originals[img_idx])
        pil_img = Image.open(io.BytesIO(orig_bytes))
        if is_color:
            pil_img = pil_img.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            x = torch.tensor(arr.transpose(2,0,1).flatten(), dtype=torch.float32).unsqueeze(0)
        else:
            pil_img = pil_img.convert("L").resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            x = torch.tensor(arr.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lv = model.encoder(x).squeeze(0)

    # Add noise to latent vector
    if noise > 0:
        lv = lv + torch.randn_like(lv) * noise

    with torch.no_grad():
        recon = model.decoder(lv.unsqueeze(0)).squeeze(0)

    # Encode reconstructed to PNG
    recon_np = (recon.detach().numpy() * 255).clip(0, 255).astype(np.uint8)
    if is_color:
        recon_img = Image.fromarray(
            recon_np.reshape(n_channels, target_size, target_size).transpose(1,2,0), mode="RGB"
        ).resize((112, 112), Image.NEAREST)
    else:
        recon_img = Image.fromarray(
            recon_np.reshape(target_size, target_size), mode="L"
        ).resize((112, 112), Image.NEAREST)

    buf = io.BytesIO(); recon_img.save(buf, "PNG"); buf.seek(0)
    recon_b64 = base64.b64encode(buf.read()).decode()

    # Also return the stored pre-noised original (from this run) for the UI
    orig_b64 = sample_originals[img_idx] if img_idx < len(sample_originals) else None

    return jsonify(
        reconstructed=recon_b64,
        original=orig_b64,
        latent_vector=[round(float(v), 3) for v in lv.tolist()],
        latent_dim=len(lv)
    )
