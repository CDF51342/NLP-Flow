"""NLP Flow 3 — Flask API with SSE progress streaming"""
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import re, io, base64, csv, json, time, threading
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.decomposition import LatentDirichletAllocation

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

# ── Global state ─────────────────────────────────────────────────────────────
S = dict(
    texts=[], labels=[], label_names=[], task="classification",
    processed_texts=[], active_steps=[],
    model=None, vectorizer=None, results={}, dataset_name="",
    columns=[], raw_rows=[],
)

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
LIGHT="#ffffff"; BG="#f5f5f5"; INK="#121212"; SEC="#656565"; MINT="#19e68c"; BORDER="#dedede"

# Vibrant colour palette for charts
PALETTE = ["#6C63FF","#FF6B6B","#FFD93D","#4ECDC4","#FF8E53","#A8E6CF","#C77DFF","#F72585"]

def style_ax(ax):
    ax.set_facecolor(LIGHT)
    ax.tick_params(colors=SEC, labelsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=LIGHT)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data

def _dataset_summary():
    texts = S["texts"]
    if not texts: return {"loaded": False}
    wc = [len(t.split()) for t in texts]
    c  = Counter(S["labels"])
    return {
        "loaded": True, "name": S["dataset_name"], "task": S["task"],
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
             results={}, model=None, vectorizer=None, columns=cols, raw_rows=raw_rows)
    return jsonify({**_dataset_summary(), "columns": cols})

@app.route("/api/dataset_info")
def dataset_info(): return jsonify(_dataset_summary())

# ── Table endpoint (paginated) ────────────────────────────────────────────────
@app.route("/api/table")
def table():
    q      = request.args.get("q","").lower()
    page   = int(request.args.get("page",1))
    per    = 20
    rows   = S["raw_rows"]
    if q:
        rows = [r for r in rows if any(q in str(v).lower() for v in r.values())]
    total  = len(rows)
    start  = (page-1)*per
    return jsonify({"columns": S["columns"], "rows": rows[start:start+per],
                    "total": total, "page": page, "pages": max(1,(total+per-1)//per)})

# ── Explore ───────────────────────────────────────────────────────────────────
@app.route("/api/plot_explore")
def plot_explore():
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
    ax.set_xlabel("Words per text",color=SEC,fontsize=11)
    ax.set_ylabel("Frequency",color=SEC,fontsize=11)
    ax.set_title("Length distribution",color=INK,fontsize=13,fontweight="bold")
    ax.yaxis.grid(True,color=BORDER,linestyle="--",linewidth=0.6,zorder=0)
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
    return jsonify({"img": fig_b64(fig)})

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
    texts=S["texts"]; proc=S["processed_texts"]
    if not texts: return jsonify({"error":"No data"}),400
    fig,axes=plt.subplots(1,2,figsize=(12,4.5)); fig.patch.set_facecolor(LIGHT)
    def top_words(corpus,n=12):
        all_w=[]
        for t in corpus: all_w.extend(t.lower().split())
        return Counter(all_w).most_common(n)
    for ax_i,(ax,corpus,title,pal) in enumerate([(axes[0],texts,"Top words — Raw",PALETTE[1]),(axes[1],proc,"Top words — Preprocessed",PALETTE[3])]):
        style_ax(ax)
        wf=top_words(corpus)
        if wf:
            words,freqs=zip(*wf)
            bar_colors=[pal if j==0 else PALETTE[(ax_i*3+j)%len(PALETTE)] for j in range(len(words))]
            ax.barh(list(reversed(words)),list(reversed(freqs)),
                    color=list(reversed(bar_colors)),edgecolor="none",alpha=0.88)
        ax.xaxis.grid(True,color=BORDER,linestyle="--",linewidth=0.6,zorder=0)
        ax.set_title(title,color=INK,fontsize=13,fontweight="bold")
    plt.tight_layout(pad=1.8)
    return jsonify({"img":fig_b64(fig)})

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
                "topics":    res.get("topics", []),
                "perplexity": res.get("perplexity"),
                "coherence":  res.get("coherence", []),
                "doc_topics": res.get("doc_topics", []),
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
    n=len(topics)
    cols=min(n,4); rows=max(1,(n+cols-1)//cols)
    fig,axes=plt.subplots(rows,cols,figsize=(4.5*cols,4.5*rows))
    fig.patch.set_facecolor(LIGHT)
    axes_flat=np.array(axes).flatten() if n>1 else [axes]
    for idx,(ax,tp) in enumerate(zip(axes_flat,topics)):
        style_ax(ax)
        words=tp["words"][:10]; weights=tp["weights"][:10]
        color=PALETTE[idx % len(PALETTE)]
        bars=ax.barh(list(reversed(words)),list(reversed(weights)),
                     color=color,edgecolor="none",alpha=0.85)
        ax.set_title(f"Topic {tp['id']+1}",color=INK,fontsize=13,fontweight="bold")
        ax.tick_params(axis="y",labelsize=10)
    # hide unused subplots
    for ax in axes_flat[n:]: ax.set_visible(False)
    plt.tight_layout(pad=2)
    return jsonify({"img":fig_b64(fig),"topics":topics})

# ── Topic model ───────────────────────────────────────────────────────────────
_topic_thread = None

@app.route("/api/topic_model", methods=["POST"])
def topic_model():
    global _topic_thread
    body=request.json
    n_topics  = int(body.get("n_topics",  5))
    max_vocab  = int(body.get("max_vocab", 500))
    top_words  = int(body.get("top_words", 15))
    proc=S["processed_texts"] or S["texts"]
    if not proc: return jsonify({"error":"No texts loaded — connect a Data block first"}),400

    def topic_worker():
        try:
            reset_progress()
            push_progress(5,"Vectorizing corpus…")
            time.sleep(0.05)
            min_df = 2 if len(proc) > 50 else 1
            vec=CountVectorizer(max_features=max_vocab, min_df=min_df)
            X=vec.fit_transform(proc)
            push_progress(25,"Fitting LDA model…")
            time.sleep(0.05)
            n_iter = min(40, max(10, len(proc)//8))
            lda=LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                          max_iter=n_iter, learning_method="batch",
                                          evaluate_every=5)
            lda.fit(X)
            push_progress(75,"Extracting topics…")
            time.sleep(0.05)
            feat=vec.get_feature_names_out()
            topics=[{"id":i,"words":[feat[j] for j in c.argsort()[-top_words:][::-1]],
                     "weights":[round(float(c[j]),4) for j in c.argsort()[-top_words:][::-1]]}
                    for i,c in enumerate(lda.components_)]
            doc_topics=lda.transform(X).argmax(axis=1).tolist()
            perplexity=round(float(lda.perplexity(X)),1)
            push_progress(95,"Computing coherence…")
            time.sleep(0.05)

            # Coherence proxy: avg top-word co-occurrence (UMass style, fast)
            # We compute token overlap between topic top words
            topic_sets = [set(tp["words"][:10]) for tp in topics]
            coherence_scores = []
            for ts in topic_sets:
                doc_counts = sum(1 for t in proc if any(w in t.split() for w in ts))
                coherence_scores.append(round(doc_counts / max(len(proc), 1), 3))

            S["results"]={
                "topics": topics,
                "doc_topics": doc_topics,
                "perplexity": perplexity,
                "coherence": coherence_scores,
                "task": "topic_model"
            }
            push_progress(100,"Done ✓")
        except Exception as e:
            push_progress(100, f"Error: {str(e)}")

    if _topic_thread and _topic_thread.is_alive():
        return jsonify({"error":"Already running"}), 429

    _topic_thread = threading.Thread(target=topic_worker, daemon=True)
    _topic_thread.start()
    return jsonify({"ok":True,"started":True})

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
