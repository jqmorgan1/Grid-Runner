"""
Serve the Grid Runner web interface over HTTP (stdlib only).

Usage:
    python serve_interface.py

Then open http://127.0.0.1:8765/index.html in your browser.
"""

from __future__ import annotations

import http.server
import socketserver
import pathlib

PORT = 8765
ROOT = pathlib.Path(__file__).resolve().parent


class _RootHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)


def main() -> None:
    with socketserver.TCPServer(("", PORT), _RootHandler) as httpd:
        print(f"Serving {ROOT} at http://127.0.0.1:{PORT}/index.html")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
