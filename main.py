"""NLP Flow — Native window launcher"""
import threading, time, os, queue as _queue_mod
import webview
from server import app, PORT, cleanup_hf_cache, _DIALOG_REQ, _DIALOG_RES

ICON = os.path.join(os.path.dirname(__file__), "static", "icon-nlpflow.png")

def start_server():
    app.run(port=PORT, debug=False, use_reloader=False, threaded=True)

def on_closed():
    """Called by pywebview when the window is closed. Clean up HF cache."""
    cleanup_hf_cache()

def _drain_dialog_queue(window):
    """
    Drain one pending dialog request (if any) and open the native Save dialog
    on the main Cocoa/GTK thread.  Called repeatedly via window.evaluate_js
    scheduling trick — but we use a simpler approach: a dedicated daemon thread
    that calls window.create_file_dialog directly after the webview loop is up.
    """
    try:
        req_id, kwargs = _DIALOG_REQ.get_nowait()
    except _queue_mod.Empty:
        return

    try:
        result = window.create_file_dialog(
            webview.SAVE_DIALOG,
            directory     = os.path.expanduser("~"),
            save_filename = kwargs["default_name"],
            file_types    = kwargs["file_types_wv"],
        )
        if result is None:
            path = None
        else:
            path = result[0] if isinstance(result, (list, tuple)) else result
            path = path or None
    except Exception:
        path = None

    _DIALOG_RES.put((req_id, path))


def _dialog_pump(window):
    """Background thread: poll the request queue and dispatch to main thread."""
    print("[PUMP] dialog pump arrancado", flush=True)
    while True:
        try:
            req_id, kwargs = _DIALOG_REQ.get(timeout=0.2)
        except _queue_mod.Empty:
            continue

        print(f"[PUMP] petición recibida req_id={req_id} default_name={kwargs['default_name']!r}", flush=True)
        # create_file_dialog blocks until the user picks a path or cancels.
        try:
            result = window.create_file_dialog(
                webview.SAVE_DIALOG,
                directory     = os.path.expanduser("~"),
                save_filename = kwargs["default_name"],
                file_types    = kwargs["file_types_wv"],
            )
            print(f"[PUMP] create_file_dialog devolvió: {result!r}", flush=True)
            if result is None:
                path = None
            else:
                path = result[0] if isinstance(result, (list, tuple)) else result
                path = path or None
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[PUMP] ERROR en create_file_dialog: {e}", flush=True)
            path = None

        print(f"[PUMP] enviando respuesta req_id={req_id} path={path!r}", flush=True)
        _DIALOG_RES.put((req_id, path))


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
    window.events.closed += on_closed

    # Start the dialog pump after webview is ready
    def _on_loaded():
        pump = threading.Thread(target=_dialog_pump, args=(window,), daemon=True)
        pump.start()

    window.events.loaded += _on_loaded

    import inspect as _inspect
    start_kwargs = dict(debug=False)
    # icon= was added in pywebview 4.5 — check signature before passing it
    _start_params = _inspect.signature(webview.start).parameters
    if os.path.exists(ICON) and "icon" in _start_params:
        start_kwargs["icon"] = ICON
    webview.start(**start_kwargs)
