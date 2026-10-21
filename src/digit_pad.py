import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / 'digit_pad.html'
INPUTS = Path.cwd() / 'inputs'
PORT = 8000


class Handler(BaseHTTPRequestHandler):
    """Serves digit_pad.html and saves drawings posted to /save into inputs/."""

    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_error(404)
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/save':
            self.send_error(404)
            return

        # Keep only the file name (no folders), and allow only simple .png names.
        name = Path(self.headers.get('X-Filename', 'digit.png')).name
        if not re.fullmatch(r'[\w.-]+\.png', name):
            self.reply(400, {'error': 'Use letters, numbers, dots, dashes or underscores, ending in .png'})
            return

        data = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if not data.startswith(b'\x89PNG'):
            self.reply(400, {'error': 'The drawing was not a PNG image'})
            return

        INPUTS.mkdir(exist_ok=True)
        path = INPUTS / name
        if path.exists():
            self.reply(409, {'error': f'inputs/{name} already exists. Choose a different file name'})
            return
        path.write_bytes(data)
        self.reply(200, {'path': f'inputs/{name}'})

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the terminal quiet


def main():
    server = HTTPServer(('127.0.0.1', PORT), Handler)
    url = f'http://localhost:{PORT}/'
    print(f"Digit pad running at {url} (press Ctrl+C to stop)")
    print(f"Drawings are saved to {INPUTS}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
