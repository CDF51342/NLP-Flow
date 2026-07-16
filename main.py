"""NLP Flow 4 — Native window launcher"""
import threading, time, os
import webview
from server import app, PORT

ICON = os.path.join(os.path.dirname(__file__), "icon.png")

def start_server():
    app.run(port=PORT, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(1.0)
    window_kwargs = dict(
        title="NLP Flow",
        url=f"http://localhost:{PORT}",
        width=1440,
        height=900,
        min_size=(1024, 660),
        resizable=True,
        background_color="#f5f5f5",
    )
    window = webview.create_window(**window_kwargs)
    start_kwargs = dict(debug=False)
    # pywebview ≥ 4.x accepts icon= in start(); older versions ignore it gracefully
    if os.path.exists(ICON):
        start_kwargs["icon"] = ICON
    webview.start(**start_kwargs)
