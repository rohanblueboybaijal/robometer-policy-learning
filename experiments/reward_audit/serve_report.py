"""Serve a local result directory, refreshing the HTML on each page request."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from .html_report import generate_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    root = args.directory.resolve()
    lock = Lock()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path in ("/", "/index.html"):
                with lock:
                    generate_html(root)
            super().do_GET()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=str(root)))
    print(f"Serving {root} on {args.bind}:{args.port}; refresh to update results", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
