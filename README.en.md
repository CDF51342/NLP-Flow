# NLP Flow

> [🇪🇸 Español](README.md) | 🇬🇧 English | [📖 Interactive docs](https://cdf51342.github.io/NLP-flow/doc.html)

Visual machine learning environment based on pipelines. Connect blocks on a canvas, configure each one with a double-click, and run the full flow with a single button. **No coding required.**

---

## Requirements & startup

Python 3.10 or higher. Install dependencies:

```bash
pip install -r requirements.txt
```

**Desktop mode** (native window 1440×900):

```bash
python main.py
```

**Browser only** (no pywebview):

```bash
python server.py
# Open http://localhost:5053
```

---

## Project structure

```
nlp-flow4/
├── main.py              # Launcher: Flask + native window
├── server.py            # Flask backend — 80+ ML routes
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (GROQ_API_KEY, HF_TOKEN, etc.)
├── static/
│   ├── index.html       # SPA frontend — canvas, blocks, modals
│   ├── favicon.png      # App icon
│   └── style.css        # Styles
└── docs/
    ├── index.html       # Documentation landing page
    ├── doc.html         # Interactive block reference
    ├── pipelines.html   # Typical pipeline guide
    └── favicon.svg      # Icon for the documentation site
```

---

## Available blocks

Each block has **input** (●—) and/or **output** (—●) ports. Connect blocks by dragging from port to port.

### Data blocks

| Block | Ports | Description |
|-------|-------|-------------|
| 📂 **Data** | output only | Upload your own CSV (text or tabular). Configure columns, target and task type. Automatic NLP vs tabular detection. |
| 🔬 **Analysis** | input only | Dataset info: rows, columns, types, missing values, statistics and distributions. |
| 📊 **Plots** | input + output | **Tabular:** histogram, boxplot, scatter, correlation, bar chart by class, pie, line. **NLP / Topic model:** document length, top N frequent words, class distribution, average length by class, wordcloud by class. Plot history with side-by-side comparison. |

### Shared blocks (tabular, NLP and image)

| Block | Ports | Description |
|-------|-------|-------------|
| 🔧 **Preprocessing** | input + output | Adapts to the upstream data type. **Tabular:** impute nulls (median/mode), scale with StandardScaler, encode categoricals with OneHotEncoder. **NLP:** lowercase, remove punctuation and numbers, EN/ES stopwords, stemming or lemmatisation, live preview. **Image:** resize (28–128 px), normalisation ([0,1] or mean/std), batch size, test split (10/15/20/25%), data augmentation on train only (horizontal flip, ±15° rotation, random crop) with multiplication factor x1–x5, real-time preview. |
| 📋 **Evaluation** | input + output | Adapts to the upstream model. **Tabular/NLP:** accuracy, F1, RMSE, R², confusion matrix, ROC curve, 2D decision boundary, cross-validation tab. **Image (CNN):** four tabs — Compare (runs table with ★ best), Report (per-class metrics + curves + CM per run), Predict (own image → per-class probabilities for all runs, history of 12 images), Errors (misclassified images). |
| 💾 **Save** | input only | Adapts to what is connected upstream. **Tabular/NLP:** exports model (.pkl), metrics (.json/.csv) and plots (.png/.zip). **RAG:** HTML report with query, retrieved fragments, prompt and responses. **Image (CNN):** generates a full HTML report in ES or EN — run comparison, per-class metrics, training curves, confusion matrix and user-uploaded image predictions. Native OS save dialog in all cases. |
| 📂 **Load Model** | output only | Load a previously saved .pkl model for evaluation or classifying new samples. |

### Tabular blocks

| Block | Ports | Description |
|-------|-------|-------------|
| 📉 **Linear Regression** | input + output | OLS, Ridge (L2) and LASSO (L1). Auto-detects classification vs regression from target type. CV grid search for lambda/C with curve visualisation. |

### NLP blocks

| Block | Ports | Description |
|-------|-------|-------------|
| 🧠 **NLP Training** | input + output | TF-IDF + classifier pipeline (Naive Bayes, Logistic Regression, KNN, Random Forest, SVM). Cross-validation and hyperparameter grid search. |
| 🗂️ **Topic Model** | input + output | LDA, NMF and LSA to discover latent topics in a text corpus without labels. Configurable: number of topics, vocabulary, iterations, min_df, max_df, alpha and beta (LDA). C_V and C_NPMI coherence per topic, topic diversity. Automatic topic labelling with LLM (Groq). |
| 🎯 **Classify NLP** | input only | Classify new text samples using a trained NLP model. Shows predicted class and probability distribution. |

### RAG pipeline

| Block | Ports | Description |
|-------|-------|-------------|
| ✂️ **Chunking** | input + output | Split documents into overlapping fragments. Configurable: chunk size and overlap. |
| 🧬 **Embeddings** | input + output | Vectorise chunks with `all-MiniLM-L6-v2`. Real-time progress. |
| 🔎 **Retriever** | input + output | Semantic search by cosine similarity. Shows the most relevant chunks with score. |
| 🤖 **LLM (RAG)** | input only | Generates a response without RAG and with RAG in parallel using Groq. Prompt adapts to the active language (ES/EN). |

### Vision blocks (image)

| Block | Ports | Description |
|-------|-------|-------------|
| 📷 **Image Data** | output only | Automatic download of MNIST (70,000 images, digits 0–9), Chihuahua vs Muffin (299 colour images) or Cats vs Dogs (~23,000 images, ~720 MB, 2,000-image sample). Shows class distribution and per-class examples. |
| 🧠 **CNN Classification** | input + output | CNN trained from scratch with PyTorch on CPU. Configurable: epochs (1–20), learning rate, conv layers (1–3), val split. Real-time progress with loss/accuracy curves. Runs system: multiple configurations comparable and renameable. Train button locks during training. |
| 🔄 **Autoencoder** | input + output | Convolutional autoencoder trained from scratch with PyTorch. Learns to compress and reconstruct images without labels. Configurable: bottleneck (neurons in the compression layer), epochs (1–30), learning rate, architecture (auto/wide/deep), loss function (MSE or BCE). Interactive latent space exploration with noise slider. Run comparison by test MSE. Exportable HTML report. |

---

## Typical pipelines

**Tabular classification / regression:**
```
📂 Data → 🔧 Preprocessing → 📉 Linear Regression → 📋 Evaluation → 💾 Save
```

**Data exploration:**
```
📂 Data → 🔬 Analysis
📂 Data → 📊 Plots
📂 Data → 🔧 Preprocessing → 📊 Plots
```

**NLP — text classification:**
```
📂 Data → 🔧 Preprocessing → 🧠 NLP Training → 🎯 Classify NLP
```

**Topic Model:**
```
📂 Data → 🔧 Preprocessing → 🗂️ Topic Model
```

**RAG (Retrieval-Augmented Generation):**
```
📂 Data → ✂️ Chunking → 🧬 Embeddings → 🔎 Retriever → 🤖 LLM (RAG)
📂 Data → 🔧 Preprocessing → ✂️ Chunking → 🧬 Embeddings → 🔎 Retriever → 🤖 LLM (RAG)
```

**Image classification (CNN):**
```
📷 Image Data → 🔧 Preprocessing → 🧠 CNN Classification → 📋 Evaluation → 💾 Save
```

**Autoencoder (compression and reconstruction):**
```
📷 Image Data → 🔧 Preprocessing → 🔄 Autoencoder → 💾 Save
```

---

## Loading data

The **📂 Data** block only accepts **user-uploaded CSV files**. There are no built-in datasets.

1. Drag your CSV to the upload area or use the file picker.
2. The system automatically detects whether the dataset is NLP (few columns, long text) or tabular (multiple numeric columns).
3. Configure the text column, target column and task type.

Column and target configuration propagates to all connected downstream blocks.

---

## NLP Preprocessing — available steps

| Step | Description |
|------|-------------|
| Lowercase | Converts all text to lowercase |
| Remove punctuation | Removes punctuation marks (preserves accented characters) |
| Remove numbers | Strips numeric characters |
| Stop words (EN) | Removes common English stop words |
| Stop words (ES) | Removes common Spanish stop words |
| Stemming | Reduces words to their root (EN/ES). Mutually exclusive with Lemmatisation |
| Lemmatisation | Reduces words to their morphological lemma. Mutually exclusive with Stemming |
| Normalize whitespace | Collapses multiple spaces |

The document selector in the preview panel updates automatically when you change the selected document. If the dataset has a category column, a **class filter** appears so you can browse only the documents from that class — even in topic model mode.

---

## NLP Plots

When the Plots block is connected to an upstream with text data (directly from Data, or from Preprocessing), the panel automatically detects the type and shows text EDA charts:

| Chart | Description | Requires classes |
|-------|-------------|-----------------|
| 📏 Document length | Histogram of words per doc with median line | No |
| 🔤 Frequent words | Horizontal bars with top N words (N configurable) | No |
| 📊 Class distribution | Bars with count and % per category | Yes |
| 📐 Length by class | Mean ± std of words per document and class | Yes |
| ☁️ Wordcloud by class | Grid of word clouds, one per category | Yes |

You can choose between the original or preprocessed corpus (if a Preprocessing block has been applied upstream).

---

## Language

The app supports **Spanish and English**. Switch language from the selector in the top bar. All blocks, labels, chart axes and error messages respect the active language, including the LLM RAG block prompts.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Double-click on block | Open block popup |
| `⌘/Ctrl` + click | Add/remove block from selection |
| `⌘/Ctrl` + `A` | Select all (select mode) |
| `⌘/Ctrl` + `C` | Copy selection (with internal connections) |
| `⌘/Ctrl` + `V` | Paste (restores internal connections) |
| `⌘/Ctrl` + `Z` | Undo |
| `⌘/Ctrl` + `Shift` + `Z` | Redo |
| `Del` / `Backspace` | Delete selected blocks |
| Mouse wheel | Zoom in/out |

---

## Saving and loading the canvas

- **💾 Save session (.zip)** — exports canvas + full configuration + block state
- **📂 Load session (.zip)** — restores a saved canvas (server data must be reloaded manually)
- **🗺 Export canvas (.json)** — block structure and connections only

> Data loaded on the server is not persisted between sessions. When restoring a canvas, reload the CSV in the Data block.

---

## Environment setup (RAG / LLM)

The LLM (RAG) block supports two providers. Create a `.env` file in the project root:

```
# Groq provider (default)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant   # optional, this is the default

# Hugging Face provider (alternative)
HF_TOKEN=your_hf_token
HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct   # optional, this is the default HF model
```

Embeddings (🧬 Embeddings block) work without a key — they use `all-MiniLM-L6-v2` from Sentence Transformers.

---

## Technical notes

- Server listens on `localhost:5053`. No external ports are exposed.
- Trained models live in memory. Restarting the server clears them.
- Plots are generated with matplotlib and returned as base64 PNG.
- Each Data node has its own slot `_NODE_DATA[node_id]`: you can run parallel pipelines with different datasets.
- The frontend is a framework-free SPA. All canvas state lives in JS memory.
- Block state is persisted in `node.data` and serialised when saving a session, including the Plots history images.
