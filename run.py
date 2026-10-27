"""Atalho de execução: python run.py  → abre o HUD em http://127.0.0.1:8050"""

import os
import threading
import webbrowser

import uvicorn

PORT = int(os.getenv("CREDITAI_PORT", "8050"))

if __name__ == "__main__":
    if os.getenv("CREDITAI_NO_BROWSER") is None:
        threading.Timer(2.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=False)
