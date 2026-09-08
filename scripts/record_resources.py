#!/usr/bin/env python3
"""Record local repository commits, model file hashes, and availability status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import resolve_path
from src.utils.io import atomic_write_json
from src.utils.provenance import git_commit, sha256_file

REPOSITORIES = {
    "rex_omni": "third_party/Rex-Omni",
    "sam3": "third_party/sam3",
    "groundingdino": "third_party/GroundingDINO",
}
HUGGINGFACE_REVISIONS = {
    "qwen38_27b": {
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    },
    "rex_omni": {
        "repo_id": "IDEA-Research/Rex-Omni",
        "revision": "0e5693d24657f6c0e091008dd6809bb1bd28988c",
    },
    "sam31": {
        "repo_id": "facebook/sam3.1",
        "revision": "daa63191845a41281374e725f4c9e51c7a824460",
        "gated": "manual",
    },
}
MODEL_FILES = {
    "qwen_config": "models/qwen38_27b/config.json",
    "qwen_index": "models/qwen38_27b/model.safetensors.index.json",
    "rex_config": "models/rex_omni/config.json",
    "rex_index": "models/rex_omni/model.safetensors.index.json",
    "sam31_checkpoint": "models/sam3_1/sam3.1_multiplex.pt",
    "groundingdino_checkpoint": "models/grounding_dino/groundingdino_swinb_cogcoor.pth",
}


def _indexed_shards(model_key: str, model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {"available": False, "shards": []}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()))
    shards = []
    for name in names:
        path = model_dir / name
        metadata_path = model_dir / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        metadata = (
            metadata_path.read_text(encoding="utf-8").splitlines() if metadata_path.exists() else []
        )
        expected_revision = HUGGINGFACE_REVISIONS[model_key]["revision"]
        revision = metadata[0] if metadata else None
        digest = metadata[1] if len(metadata) > 1 and len(metadata[1]) == 64 else None
        shards.append(
            {
                "name": name,
                "path": str(path),
                "available": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": digest,
                "hash_source": "huggingface_lfs_etag" if digest else None,
                "revision": revision,
                "revision_matches": revision == expected_revision,
            }
        )
    return {
        "available": all(item["available"] for item in shards),
        "expected_shards": len(names),
        "total_size_bytes": sum(item["size_bytes"] or 0 for item in shards),
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    repos = {
        name: {"path": str(resolve_path(path)), "git_commit": git_commit(resolve_path(path))}
        for name, path in REPOSITORIES.items()
    }
    files = {}
    for name, configured_path in MODEL_FILES.items():
        path = resolve_path(configured_path)
        files[name] = {
            "path": str(path),
            "available": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    indexed_weights = {
        "qwen38_27b": _indexed_shards("qwen38_27b", resolve_path("models/qwen38_27b")),
        "rex_omni": _indexed_shards("rex_omni", resolve_path("models/rex_omni")),
    }
    atomic_write_json(
        Path(args.output_dir) / "resources.json",
        {
            "repositories": repos,
            "huggingface_revisions": HUGGINGFACE_REVISIONS,
            "model_files": files,
            "indexed_weights": indexed_weights,
            "notes": {
                "sam31_checkpoint": "Gated by Meta license and Hugging Face authorization.",
                "rex_score": "Official Rex-Omni parser returns boxes without confidence scores.",
                "indexed_weight_hashes": (
                    "Per-shard SHA-256 values are immutable Hugging Face LFS object etags "
                    "recorded by hf download at the pinned revision."
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
