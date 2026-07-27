# NLP Flow v4

> [🇪🇸 Español](README.md) | 🇬🇧 English | [📖 Interactive docs](https://TUUSUARIO.github.io/NOMBRE-REPO/)

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
├── server.py            # Flask backend — 55+ ML routes
├── requirements.txt     # Python dependencies
└── static/
    ├── index.html       # SPA frontend — canvas, blocks, modals
    └── style.css        # Styles
```

---

## Available blocks

Each block has **input** (●—) and/or **output** (—●) ports. Connect blocks by dragging from port to port.

| Block | Ports | Description |
|-------|-------|-------------|
| 📂 **Data** | output only | Load built-in datasets or your own CSV. Configure columns and target. |
| 🔬 **Analysis** | input only | Dataset info: rows, columns, types, missing values and statistics. |
| 📊 **Plots** | input + output | Histogram, boxplot, scatter, correlation, bar chart, pie, line. Plot history. |
| 🔧 **Preprocessing** | input + output | Imputation, normalisation, encoding, train/test split, feature selection. |
| 📉 **Linear Regression** | input + output | OLS, Ridge (L2) and LASSO (L1). Auto-detects classification vs regression from target type. CV grid search for lambda/C. |
| 📋 **Evaluation** | input + output | Metrics, confusion matrix, ROC curve, decision boundary, regression assumptions, CV tab. |
| 💾 **Save** | input only | Export model (.pkl), metrics (.json/.csv) and plots (.png). |
| 📂 **Load Model** | output only | Load a previously saved .pkl model for evaluation or classifying new samples. |
| 🧠 **NLP Training** | input + output | Text pipeline: TF-IDF + classifier. |
| 🗂️ **Topic Model** | input only | LDA to discover latent topics in a text corpus. |
| 🎯 **Classify NLP** | input only | Classify new text samples using a trained NLP model. |

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
```

**NLP — text:**
```
📂 Data → 🔧 Preprocessing → 🧠 Training → 🎯 Classify
```

---

## Included datasets

### Text

| Dataset | Task | Description |
|---------|------|-------------|
| 🎬 Movie Reviews | Classification | Positive and negative film reviews |
| 📧 Spam vs Ham | Classification | Spam and legitimate messages |
| 😊 Twitter Sentiment | Classification | Positive, negative and neutral tweets |
| 📰 News | Topic model | News articles by topic |

### Tabular

| Dataset | Task | Description |
|---------|------|-------------|
| 🏥 Patient Health Risk | Classification | Health variables → high/low risk |
| 🏠 California Housing | Regression | Housing features → median price |

You can also upload your own CSV from the **Data** block.

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

- **💾 Save session (.zip)** — exports canvas + full configuration
- **📂 Load session (.zip)** — restores a saved canvas
- **🗺 Export canvas (.json)** — block structure and connections only

> Data loaded on the server is not persisted between sessions. When restoring a canvas, reload the dataset in the Data block.

---

## Technical notes

- Server listens on `localhost:5053`. No external ports are exposed.
- Trained models live in memory. Restarting the server clears them.
- Plots are generated with matplotlib and returned as base64 PNG.
- Each Data node has its own slot `_NODE_DATA[node_id]`: you can run parallel pipelines with different datasets.
- The frontend is a framework-free SPA. All canvas state lives in JS memory.
