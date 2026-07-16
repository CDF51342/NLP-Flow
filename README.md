# NLP Flow 2 — Native window canvas

Visual NLP node canvas that opens as a native desktop app (pywebview), no browser needed.
Closing the window kills the Python server automatically.

## Run

```bash
cd nlp-flow2
uv venv
uv pip install -r requirements.txt
uv run python main.py
```

## Node types

| Node | Accepts | Emits |
|------|---------|-------|
| 📂 Data | — | data |
| ⚙️ Preprocessing | data | processed |
| 🤖 Training | data / processed | model |
| 📊 Results | model | — |
| 🔍 Classify | model | — |
| 🗂️ Topic Model | data / processed | topics |

## Canvas rules

- **Drag** nodes from the left panel anywhere on the canvas
- **Connect** by dragging from the right port (output) to another node's left port (input) — or drop the cable anywhere on the target node body
- **Move** any node at any time, even connected ones — cables redraw automatically
- **Open dashboard** by clicking a node — a large modal with charts, live preview, and a Run button
- **Delete** a node by double-clicking it
- **EN/ES** toggle in the top bar

## Export as native app

```bash
uv pip install pyinstaller
pyinstaller --onefile --windowed --add-data "static:static" --name "NLP-Flow" main.py
```
