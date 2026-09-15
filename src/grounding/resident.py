"""Node-local inference transport. Task processes retain ALL checkpoint writes.

No pickle or tensors cross the process boundary. Images retain their original
encoded bytes; masks use lossless NumPy serialization. Without the explicit
runtime environment variable the original standalone adapter is unchanged.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import zlib

MAX_BODY = 128 << 20
NAMES = {"egm": "egm", "rex": "rex_omni", "sam31": "sam31", "groundingdino": "groundingdino"}


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def service_signature(model, config, environment=None):
    environment = os.environ if environment is None else environment
    settings = {}
    if model == 'rex':
        settings = {key: int(environment.get(key, config.get(option, default))) for key, option, default in (
            ('REX_MAX_OUTPUT_TOKENS', 'max_output_tokens', 256), ('REX_BATCH_SIZE', 'batch_size', 2))}
    return fingerprint(dict(model=model, config=config, inference_environment=settings))


def alive(pid, start=None):
    try:
        os.kill(int(pid), 0)
        if start is not None:
            fields = Path(f'/proc/{int(pid)}/stat').read_text().split(') ', 1)[1].split()
            return fields[0] != 'Z' and fields[19] == str(start)
        return True
    except (OSError, ValueError, TypeError):
        return False


def rpc(endpoint, token, payload, timeout=1800):
    data = json.dumps(payload, separators=(",", ":")).encode()
    if len(data) > MAX_BODY:
        raise ValueError("Resident inference request exceeds bounded transport size")
    request = urllib.request.Request(endpoint + "/infer", data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(MAX_BODY + 1)
    if len(raw) > MAX_BODY:
        raise ValueError("Resident inference response exceeds bounded transport size")
    value = json.loads(raw)
    if value.get("error"):
        raise RuntimeError(value["error"])
    return value


def encode(value):
    import numpy as np
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        return encode(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        # Sparse masks otherwise explode JSON size. Lossless level-1 compression
        # applies to masks only; source JPEG/PNG bytes are never recompressed.
        raw = stream.getvalue()
        return {"__ndarray__": base64.b64encode(zlib.compress(raw, 1)).decode(),
                "codec": "zlib", "bytes": len(raw)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"__ndarray__", "codec", "bytes"}:
            import numpy as np
            expected = int(value['bytes'])
            if value['codec'] != 'zlib' or not 0 < expected <= (1 << 30):
                raise ValueError('Invalid array transport size or codec')
            decoder = zlib.decompressobj()
            raw = decoder.decompress(base64.b64decode(value["__ndarray__"], validate=True), expected + 1)
            if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('Array transport length mismatch')
            return np.load(io.BytesIO(raw), allow_pickle=False)
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


class Proxy:
    def __init__(self, model, config):
        self.model, self.signature = model, service_signature(model, config)
        self.root = Path(os.environ["OPD_RESIDENT_POOL"])
        self.name = NAMES[model]
        # Filled by the ACTUAL executing model, not the task's original GPU.
        self.provenance = {}
        self.image_cache = OrderedDict()
        self.batch_size = (max(1, int(os.environ.get("REX_BATCH_SIZE", config.get("batch_size", 2))))
                           if model == "rex" else 1)

    def ground_batch(self, requests):
        from src.utils.config import resolve_path
        images, items = {}, []
        for path, phrase in requests:
            path = resolve_path(path)
            stat = path.stat()
            identity = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if identity not in self.image_cache:
                raw = path.read_bytes()  # Inside the TASK's private RAM namespace.
                key = hashlib.sha256(raw).hexdigest()
                self.image_cache[identity] = {"key": key, "suffix": path.suffix.lower(), "data": base64.b64encode(raw).decode()}
            self.image_cache.move_to_end(identity)
            image = self.image_cache[identity]
            key = image['key']
            if key not in images:
                images[key] = image
            items.append([key, phrase])
        while len(self.image_cache) > 8 or (len(self.image_cache) > 1 and sum(len(v['data']) for v in self.image_cache.values()) > (64 << 20)):
            self.image_cache.popitem(last=False)
        payload = dict(model=self.model, signature=self.signature, images=list(images.values()), requests=items)
        # A bounded scan/reservation is local RAM only, never Ceph. A busy GPU
        # returns 429; there is no unbounded queue inside any model process.
        deadline = time.monotonic() + 1800
        excluded, errors = set(), []
        while time.monotonic() < deadline:
            candidates = []
            for path in (self.root / "services").glob("*.json"):
                try:
                    service = json.loads(path.read_text())
                    if (service["status"] != "ready" or not alive(service["pid"], service.get('owner_start'))
                            or service["signatures"].get(self.model) != self.signature
                            or service["endpoint"] in excluded):
                        continue
                    candidates.append(service)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            # Prefer available warm models, then other free GPUs, then busy ones.
            candidates.sort(key=lambda s: (s.get("busy", False) or self.model in s.get('active_models', []),
                                           self.model not in s.get("loaded", []), s["endpoint"]))
            for service in candidates:
                try:
                    value = rpc(service["endpoint"], service["token"], payload,
                                timeout=max(1, deadline - time.monotonic()))
                    results = decode(value["results"])
                    self.provenance = value['provenance']
                    if len(results) != len(requests):
                        raise RuntimeError("Resident grounder changed result cardinality")
                    print(f"[resident] model={self.model} gpu={service['device']} infer_s={value.get('seconds', 0):.3f}", flush=True)
                    return results
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:
                        continue
                    excluded.add(service["endpoint"])
                    errors.append(exc.read(8192).decode(errors="replace"))
                except (OSError, ValueError, RuntimeError) as exc:
                    excluded.add(service["endpoint"])
                    errors.append(str(exc))
            if errors and not candidates:
                raise RuntimeError("No healthy compatible resident grounder: " + " | ".join(errors))
            time.sleep(0.1)
        raise TimeoutError("Waiting for compatible resident grounder exceeded 1800s: " + " | ".join(errors))

    def ground(self, path, phrase):
        return self.ground_batch([(path, phrase)])[0]

    def close(self):
        pass  # Only the owning GPU worker can shut down a service.


def adapter(model, config, factory):
    return Proxy(model, config) if os.environ.get("OPD_RESIDENT_POOL") else factory(config)


class ImageCache:
    def __init__(self, root, capacity=8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = OrderedDict()
        self.capacity = capacity

    def materialize(self, images):
        current = {}
        for image in images:
            data = base64.b64decode(image["data"], validate=True)
            key = hashlib.sha256(data).hexdigest()
            suffix = image["suffix"]
            if key != image["key"] or suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
                raise ValueError("Invalid resident image identity or format")
            path = self.root / (key + suffix)
            if key not in self.paths:
                path.write_bytes(data)
                self.paths[key] = path
            self.paths.move_to_end(key)
            current[key] = str(self.paths[key])
        # Never evict inputs of the current batch.
        for key in list(self.paths):
            if len(self.paths) <= self.capacity:
                break
            if key not in current:
                self.paths.pop(key).unlink()
        return current


def memory_preflight(config):
    """Conservative startup guard, not a claim that every inference peak fits."""
    import subprocess
    from src.utils.config import resolve_path
    path = resolve_path(config["local_path"])
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            weights = int(json.loads(index.read_text())["metadata"]["total_size"])
        else:
            files = list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
            if not files:
                raise RuntimeError("Cannot estimate resident weight footprint: " + str(path))
            weights = sum(p.stat().st_size for p in files)
    else:
        weights = path.stat().st_size
    result = subprocess.run(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    free = int(result.stdout.strip()) << 20
    # Account for deserialization/loading overhead and retain workspace room.
    required = int(weights * 1.5) + (8 << 30)
    if free < required:
        raise RuntimeError(f"Resident VRAM budget insufficient: free={free >> 20} MiB, conservative load+reserve={required >> 20} MiB")
    print(f"[resident] VRAM preflight free={free >> 20} MiB, load+reserve={required >> 20} MiB", flush=True)


def serve(spec_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import importlib
    spec = json.loads(Path(spec_path).read_text())
    # Exit even during a slow import/load if the owning worker disappears.
    def watchdog():
        while alive(spec["owner"]):
            try:
                if Path(f"/proc/{spec['owner']}/stat").read_text().split(") ", 1)[1].split()[19] != spec["owner_start"]:
                    break
            except OSError:
                break
            time.sleep(0.5)
        os._exit(75)
    threading.Thread(target=watchdog, daemon=True).start()
    factories = {"egm": ("egm", "EGMGrounder"), "rex": ("rex", "RexGrounder"), "sam31": ("sam31", "Sam31Grounder"),
                 "groundingdino": ("groundingdino", "GroundingDinoGrounder")}
    memory_preflight(spec["config"])
    module, name = factories[spec["model"]]
    model = getattr(importlib.import_module("src.grounding." + module), name)(spec["config"])
    cache = ImageCache(spec["images"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/infer" or self.headers.get("Authorization") != "Bearer " + spec["token"]:
                self.send_error(403)
                return
            fatal = False
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if request["signature"] != service_signature(spec['model'], spec["config"]) or request["model"] != spec["model"]:
                    raise ValueError("Resident model/config identity mismatch")
                paths = cache.materialize(request["images"])
                started = time.monotonic()
                results = model.ground_batch([(paths[key], phrase) for key, phrase in request["requests"]])
                data = json.dumps({"results": encode(results), "provenance": model.provenance,
                                   "seconds": time.monotonic() - started}).encode()
                if len(data) > MAX_BODY:
                    raise ValueError("Inference result exceeds transport bound")
                code = 200
            except Exception as exc:
                import traceback
                traceback.print_exc()
                code, fatal = 500, True
                data = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            finally:
                if fatal:
                    os._exit(76)  # Never reuse uncertain CUDA/model state after failure.

    server = HTTPServer(("127.0.0.1", 0), Handler)
    ready = Path(spec["ready"])
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(json.dumps({"endpoint": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()}))
    temporary.replace(ready)
    print(f"[resident] {spec['model']} ready; retained across task/stage boundaries", flush=True)
    try:
        server.serve_forever()
    finally:
        model.close()
        server.server_close()


if __name__ == "__main__":
    import sys
    serve(sys.argv[1])
