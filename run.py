"""Launch the built local application; stdlib mode works without Python packages."""
import argparse
import importlib.util
import socket
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / "dist" / "index.html").exists():
        raise SystemExit("Frontend build missing. Run npm install, then npm run build.")
    port = options.port
    for candidate in range(port, port + 20):
        with socket.socket() as sock:
            try: sock.bind(("127.0.0.1", candidate)); port = candidate; break
            except OSError: continue
    else: raise SystemExit("No free port found")
    url = f"http://127.0.0.1:{port}"
    print(f"TensorCraft3D · Build and Explore Neural Networks in 3D: {url}", flush=True)
    if not options.no_browser: webbrowser.open(url)
    if importlib.util.find_spec("uvicorn") and importlib.util.find_spec("fastapi"):
        import uvicorn
        print("Python API enabled: /api/health, /api/import/pytorch, /api/train", flush=True)
        uvicorn.run("backend.app:app", host="127.0.0.1", port=port, ws_max_size=10_000_000)
    else:
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        import functools
        print("Browser-only mode: Python API unavailable. Run .\\install.ps1, then restart .\\start.ps1", flush=True)
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root / "dist"))
        ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__": main()
