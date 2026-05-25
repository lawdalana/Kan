from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from codes.common import measure_latency_ms


PredictMany = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def parse_prediction_body(body: bytes) -> list[dict[str, Any]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON") from exc

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        if not isinstance(samples, list) or not all(isinstance(sample, dict) for sample in samples):
            raise ValueError("'samples' must be a list of JSON objects")
        return samples

    if isinstance(payload, dict):
        return [payload]

    raise ValueError("Request body must be a JSON object or an object with a 'samples' list")


def success_response(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}


def error_response(message: str) -> dict[str, Any]:
    return {"success": False, "error": {"message": message}}


def run_prediction_server(model_name: str, predict_many: PredictMany, default_port: int) -> None:
    port = int(os.environ.get("PORT", default_port))

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self._send(success_response({"ok": True}))

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(error_response("Not found"), status=404)
                return
            self._send(success_response({"model": model_name, "status": "ok"}))

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(error_response("Not found"), status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                samples = parse_prediction_body(self.rfile.read(length))
                predictions, latency_ms = measure_latency_ms(lambda: predict_many(samples))
                self._send(
                    success_response(
                        {
                            "model": model_name,
                            "predictions": predictions,
                            "latency_ms": latency_ms,
                        }
                    )
                )
            except ValueError as exc:
                self._send(error_response(str(exc)), status=400)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._send(error_response(f"Prediction failed: {exc}"), status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            content = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(content)

    print(f"{model_name} API listening on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
