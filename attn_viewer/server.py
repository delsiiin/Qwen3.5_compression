import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


STATIC_DIR = Path(__file__).resolve().parent / "static"


def read_json(path):
    with open(path, encoding="utf-8") as fin:
        return json.load(fin)


class HeatmapStore:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()

    def list_runs(self):
        runs = []
        if not self.root_dir.exists():
            return runs
        for run_dir in sorted(self.root_dir.iterdir()):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            runs.append(
                {
                    "run_id": run_dir.name,
                    "model_name": manifest.get("model_name"),
                    "result_path": manifest.get("result_path"),
                    "sample_count": manifest.get("sample_count", 0),
                    "full_attention_layers": manifest.get("full_attention_layers", []),
                }
            )
        return runs

    def get_run(self, run_id):
        run_dir = self.root_dir / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run not found: {run_id}")
        manifest = read_json(manifest_path)
        manifest["run_id"] = run_id
        return manifest

    def get_sample(self, run_id, sample_id):
        sample_path = self.root_dir / run_id / "samples" / sample_id / "sample.json"
        if not sample_path.is_file():
            raise FileNotFoundError(f"Sample not found: {sample_id}")
        sample = read_json(sample_path)
        sample["run_id"] = run_id
        return sample

    def get_matrix(self, run_id, sample_id, prefill_index, layer_idx, head_idx):
        sample_dir = self.root_dir / run_id / "samples" / sample_id
        sample = self.get_sample(run_id, sample_id)
        try:
            prefill = sample["prefills"][int(prefill_index)]
        except (IndexError, ValueError):
            raise FileNotFoundError(f"Prefill not found: {prefill_index}")
        if prefill.get("status") != "saved":
            raise ValueError(f"Prefill {prefill_index} is not available: {prefill.get('status')}")
        attn_path = sample_dir / prefill["attention_file"]
        with np.load(attn_path) as data:
            attn = data["attn"]
            layer_indices = data["layer_indices"].astype(int).tolist()
        if int(layer_idx) not in layer_indices:
            raise FileNotFoundError(f"Layer {layer_idx} not found in prefill {prefill_index}")
        layer_pos = layer_indices.index(int(layer_idx))
        head_idx = int(head_idx)
        if head_idx < 0 or head_idx >= attn.shape[1]:
            raise FileNotFoundError(f"Head {head_idx} out of range")
        matrix = attn[layer_pos, head_idx].astype(np.float32)
        return {
            "run_id": run_id,
            "sample_id": sample_id,
            "prefill_index": int(prefill_index),
            "layer_index": int(layer_idx),
            "head_index": head_idx,
            "shape": list(matrix.shape),
            "matrix": matrix.tolist(),
        }


class HeatmapRequestHandler(BaseHTTPRequestHandler):
    store = None

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                return self._serve_static("index.html", "text/html; charset=utf-8")
            if parsed.path.startswith("/static/"):
                return self._serve_static(parsed.path[len("/static/"):], self._guess_type(parsed.path))
            if parsed.path == "/healthz":
                return self._send_json({"ok": True})
            if parsed.path == "/api/runs":
                return self._send_json({"runs": self.store.list_runs()})
            if parsed.path.startswith("/api/runs/"):
                return self._handle_api(parsed)
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format, *args):
        return

    def _handle_api(self, parsed):
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) == 3:
            return self._send_json(self.store.get_run(path_parts[2]))
        if len(path_parts) == 5 and path_parts[3] == "samples":
            return self._send_json(self.store.get_sample(path_parts[2], path_parts[4]))
        if len(path_parts) == 8 and path_parts[3] == "samples" and path_parts[5] == "prefills" and path_parts[7] == "matrix":
            query = parse_qs(parsed.query)
            layer_idx = query.get("layer", [None])[0]
            head_idx = query.get("head", [None])[0]
            if layer_idx is None or head_idx is None:
                raise ValueError("Missing required query params: layer, head")
            payload = self.store.get_matrix(
                run_id=path_parts[2],
                sample_id=path_parts[4],
                prefill_index=path_parts[6],
                layer_idx=layer_idx,
                head_idx=head_idx,
            )
            return self._send_json(payload)
        self.send_error(HTTPStatus.NOT_FOUND, "API route not found")

    def _serve_static(self, relative_path, content_type):
        file_path = (STATIC_DIR / relative_path).resolve()
        if STATIC_DIR not in file_path.parents and file_path != STATIC_DIR:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        with open(file_path, "rb") as fin:
            data = fin.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _guess_type(path):
        if path.endswith(".js"):
            return "application/javascript; charset=utf-8"
        if path.endswith(".css"):
            return "text/css; charset=utf-8"
        return "text/plain; charset=utf-8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results/attn_heatmaps")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    store = HeatmapStore(args.root)
    HeatmapRequestHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), HeatmapRequestHandler)
    print(f"Serving attention heatmaps from {os.path.abspath(args.root)}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
