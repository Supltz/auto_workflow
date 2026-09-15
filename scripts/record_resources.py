#!/usr/bin/env python3
"""Record local repository commits, model file hashes, and availability status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import resolve_path, load_yaml
from src.utils.io import atomic_write_json
from src.utils.provenance import git_commit, sha256_file

MODELS = load_yaml("configs/models.yaml")
REPOSITORIES = {name: value['repo_path'] for name, value in MODELS.items() if value.get('repo_path')}
HUGGINGFACE_REVISIONS = {name: dict(repo_id=value['name'], revision=value['revision'])
                        for name, value in MODELS.items() if value.get('revision')}
MODEL_FILES = {}
for name, value in MODELS.items():
    path=resolve_path(value['local_path'])
    if name=='sam31':MODEL_FILES[name+'_checkpoint']=str(path)
    else:
        MODEL_FILES[name+'_config']=str(path/'config.json')
        MODEL_FILES[name+'_index']=str(path/'model.safetensors.index.json')


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
    indexed_weights = {name: _indexed_shards(name, resolve_path(model['local_path']))
                       for name, model in MODELS.items() if name in ('qwen','egm')}
    atomic_write_json(
        Path(args.output_dir) / "resources.json",
        {
            "repositories": repos,
            "huggingface_revisions": HUGGINGFACE_REVISIONS,
            "model_files": files,
            "indexed_weights": indexed_weights,
            "notes": {
                "sam31_checkpoint": "Gated by Meta license and Hugging Face authorization.",
                "egm_score": "Locator evidence has no detector confidence score.",
                "indexed_weight_hashes": (
                    "Per-shard SHA-256 values are immutable Hugging Face LFS object etags "
                    "recorded by hf download at the pinned revision."
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
