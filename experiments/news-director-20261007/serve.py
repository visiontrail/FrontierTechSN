"""Loopback-only review server with byte-range support for reliable media seeks."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.media_range = None
        path = Path(self.translate_path(self.path))
        header = self.headers.get('Range')
        if not header or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
        if not match or (not match[1] and not match[2]):
            self.send_error(416)
            return None
        if match[1]:
            start = int(match[1]); end = min(int(match[2]) if match[2] else size - 1, size - 1)
        else:
            start = max(0, size - int(match[2])); end = size - 1
        if start > end or start >= size:
            self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers()
            return None
        handle = path.open('rb'); handle.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        self.media_range = end - start + 1
        return handle

    def copyfile(self, source, outputfile):
        if self.media_range is None:
            try:super().copyfile(source, outputfile)
            except (BrokenPipeError, ConnectionResetError):pass
            return
        remaining = self.media_range
        try:
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:break
                outputfile.write(block);remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8127)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(args.directory.resolve())))
    print(f'http://127.0.0.1:{args.port}/review.html', flush=True)
    server.serve_forever()
