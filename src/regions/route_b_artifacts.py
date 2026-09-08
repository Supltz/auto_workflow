"""Exact-geometry shared target artifacts; no approximate bbox merging."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore, RLock

from PIL import Image

from src.utils.images import open_rgb, padded_box, save_bbox_only_overlay, save_crop
from src.utils.provenance import sha256_file

_LOCKS = [RLock() for _ in range(32)]
_WRITERS = BoundedSemaphore(2)


@lru_cache(maxsize=256)
def _image_hash(path, size, mtime_ns, ctime_ns):
    return sha256_file(Path(path))


def plan_artifacts(candidate, root, padding):
    source = Path(candidate["source_image"])
    stat = source.stat()
    digest = _image_hash(str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    identity = [candidate["image_id"], digest, candidate["bbox_xyxy"], padding, "jpeg_v1"]
    shared_id = "shared_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    candidate.update(shared_instance_id=shared_id, artifact_context_padding=padding)
    for field, folder in [("verifier_overlay_path", "overlays"),
                          ("tight_crop_path", "tight_crops"),
                          ("context_crop_path", "context_crops")]:
        candidate[field] = str(root / "shared" / folder / f"{shared_id}.jpg")


def ensure_artifacts(candidate):
    """Repair missing/corrupt artifacts too; serialize writers to a shared target."""
    fields = ["verifier_overlay_path", "tight_crop_path", "context_crop_path"]
    lock_index = int(hashlib.sha256(str(candidate[fields[0]]).encode()).hexdigest(), 16) % len(_LOCKS)
    with _LOCKS[lock_index], _WRITERS:
        missing = []
        for field in fields:
            try:
                with Image.open(candidate[field]) as image:
                    image.verify()
            except (OSError, TypeError, ValueError):
                missing.append(field)
        if not missing:
            return
        image = open_rgb(candidate["source_image"])
        try:
            box = tuple(candidate["bbox_xyxy"])
            if fields[0] in missing:
                save_bbox_only_overlay(image, box, Path(candidate[fields[0]]))
            if fields[1] in missing:
                save_crop(image, box, Path(candidate[fields[1]]))
            if fields[2] in missing:
                save_crop(image, padded_box(box, image.width, image.height,
                          candidate.get("artifact_context_padding", 0.20)),
                          Path(candidate[fields[2]]))
        finally:
            image.close()
