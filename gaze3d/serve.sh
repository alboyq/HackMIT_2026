#!/usr/bin/env bash
# The camera only works on a secure origin, and localhost counts as one.
# Opening index.html as a file:// path will fail.
#
# Serves with Cache-Control: no-store — python's stock http.server sends no
# cache headers at all, which lets Chrome hold on to stale js/css after an edit.
set -euo pipefail
PORT="${1:-8000}"
cd "$(dirname "$0")"

echo "Gaze Accuracy Bench → http://localhost:$PORT"
echo "ctrl-c to stop"

exec python3 - "$PORT" <<'PY'
import sys, functools, http.server, socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "304" not in fmt % args:
            super().log_message(fmt, *args)

class Server(socketserver.TCPServer):
    allow_reuse_address = True

with Server(("127.0.0.1", int(sys.argv[1])), Handler) as httpd:
    httpd.serve_forever()
PY
