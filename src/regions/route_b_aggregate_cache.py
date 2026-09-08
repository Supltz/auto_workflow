"""Per-image content-validated aggregation cache and bounded parallel scheduling."""

import hashlib
import json
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.utils.io import atomic_write_json, read_jsonl, rewrite_jsonl_atomic
from src.utils.progress import completed_with_progress
from src.utils.provenance import sha256_file


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def aggregate_cached(serial, *, entities_path, raw_grounding_path, candidates_path,
                     rejections_path, output_root, manifest, selected_ids, config, overwrite):
    started = time.monotonic()
    print("[aggregate] Checking content signatures (images, masks, inputs, code)...", flush=True)
    rows = {r["image_id"]: r for r in read_jsonl(entities_path) if r["image_id"] in selected_ids}
    raw = defaultdict(list)
    for r in read_jsonl(raw_grounding_path):
        if r["image_id"] in selected_ids:
            raw[r["image_id"]].append(r)
    code = {p.name: sha256_file(p) for p in [
        Path(__file__), Path(__file__).with_name("route_b.py"),
        Path(__file__).with_name("route_b_artifacts.py"),
        Path(__file__).with_name("consensus.py"),
        Path(__file__).parents[1] / "utils" / "geometry.py",
        Path(__file__).parents[1] / "utils" / "images.py",
    ]}
    cache_root = candidates_path.parent / "aggregate_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(config.get("aggregation_workers", 2)), 4))

    def worker(image_id):
        source = manifest[image_id]
        masks = {r["mask_path"] for r in raw[image_id] if r.get("mask_path")}
        signature = fingerprint({
            "entity": rows[image_id], "raw": raw[image_id], "manifest": source,
            "image_sha256": sha256_file(Path(source["image_path"])),
            "masks": {p: sha256_file(Path(p)) if Path(p).is_file() else None
                      for p in sorted(masks)},
            "rules": config["aggregation"], "promotion": config.get("candidate_promotion"),
            "output_root": str(output_root.resolve()), "code": code,
        })
        cache_path = cache_root / f"{signature}.json"
        if not overwrite and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                payload = cached["payload"]
                if cached["signature"] == signature and cached["checksum"] == fingerprint(payload):
                    return image_id, payload, True
            except (OSError, ValueError, KeyError, TypeError):
                pass
        # Isolate per-image intermediate writes; published aggregate outputs are atomic.
        with tempfile.TemporaryDirectory(prefix="image-", dir=cache_root) as directory:
            root = Path(directory)
            ep, rp, cp, jp = [root / name for name in ("entities.jsonl", "raw.jsonl",
                                                      "candidates.jsonl", "rejections.jsonl")]
            rewrite_jsonl_atomic(ep, [rows[image_id]])
            rewrite_jsonl_atomic(rp, raw[image_id])
            serial(entities_path=ep, raw_grounding_path=rp, candidates_path=cp,
                   rejections_path=jp, output_root=output_root, manifest={image_id: source},
                   selected_ids={image_id}, config=config, overwrite=False)
            payload = {"candidates": list(read_jsonl(cp)), "rejections": list(read_jsonl(jp))}
        atomic_write_json(cache_path, {"signature": signature, "payload": payload,
                                      "checksum": fingerprint(payload)})
        return image_id, payload, False

    results = {}
    hits = 0
    print(f"[aggregate] images={len(rows)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, image_id) for image_id in sorted(rows)]
        for future in completed_with_progress(futures, "aggregate_images"):
            image_id, payload, hit = future.result()
            results[image_id] = payload
            hits += int(hit)
            print(f"[aggregate] image={image_id} cache={'hit' if hit else 'computed'}", flush=True)
    candidates = [r for key in sorted(results) for r in results[key]["candidates"]]
    rejections = [r for key in sorted(results) for r in results[key]["rejections"]]
    pool = {}
    for r in candidates:
        entry = pool.setdefault(r["shared_instance_id"], {
            "shared_instance_id": r["shared_instance_id"], "image_id": r["image_id"],
            "bbox_xyxy": r["bbox_xyxy"], "source_image": r["source_image"],
            "verifier_overlay_path": r["verifier_overlay_path"],
            "tight_crop_path": r["tight_crop_path"], "context_crop_path": r["context_crop_path"],
            "references": [],
        })
        entry["references"].append({k: r[k] for k in (
            "entity_id", "instance_id", "region_id", "query_types", "grounding_queries",
            "discovery_category_only")})
    rewrite_jsonl_atomic(candidates_path, candidates)
    rewrite_jsonl_atomic(rejections_path, rejections)
    rewrite_jsonl_atomic(candidates_path.with_name("shared_instances.jsonl"), list(pool.values()))
    print(f"[aggregate] DONE images={len(rows)} cache_hits={hits} "
          f"candidate_references={len(candidates)} shared_instances={len(pool)} "
          f"size_pass={sum(r['size_pass'] for r in candidates)} "
          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
