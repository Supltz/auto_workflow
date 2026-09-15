#!/usr/bin/env python3
"""Perform non-inference acceptance checks for Route B and its model resources."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.schema import DatasetRecord
from src.utils.config import PROJECT_ROOT, load_yaml, resolve_path
from src.utils.io import add_job_arguments, atomic_write_json, read_jsonl

SOURCE_MODULES = [
    "src.schema",
    "src.utils.config",
    "src.utils.io",
    "src.utils.geometry",
    "src.utils.images",
    "src.utils.provenance",
    "src.models.qwen38_client",
    "src.models.region_ocr",
    "src.grounding.base",
    "src.grounding.cache",
    "src.grounding.egm",
    "src.grounding.sam31",
    "src.routes.common",
    "src.routes.route_b_caption",
    "src.routes.route_b_stages",
    "src.routes.route_b_checkpoint",
    "src.routes.route_b_discovery",
    "src.regions.consensus",
    "src.regions.route_b",
    "src.visualization.route_b_review",
]
ENV_IMPORTS = {
    "qwen": ["vllm", "transformers"],
    "egm": ["torch", "transformers.models.qwen3_vl"],
    "sam31": ["sam3"],
}
EXPECTED_DIRECT_HASHES = {
    "sam31": (
        "sam31_checkpoint",
        "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
    ),
    "groundingdino": (
        "groundingdino_checkpoint",
        "46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7",
    ),
}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _weight_index(model_dir: Path) -> dict[str, Any]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    missing = [name for name in files if not (model_dir / name).is_file()]
    _check(not missing, f"missing weight shards in {model_dir}: {missing}")
    return {
        "expected_shards": len(files),
        "total_bytes": sum((model_dir / name).stat().st_size for name in files),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_job_arguments(parser, "configs/experiment.yaml")
    parser.add_argument("--strict-gated", action="store_true")
    parser.add_argument("--env-import-timeout", type=int, default=120)
    args = parser.parse_args()
    _check(args.env_import_timeout>0, "positive import timeout required")
    experiment = load_yaml(args.config)
    models = load_yaml("configs/models.yaml")
    route_config = load_yaml("configs/route_b.yaml")
    report: dict[str, Any] = {"checks": {}, "warnings": []}

    for module in SOURCE_MODULES:
        importlib.import_module(module)
    report["checks"]["source_imports"] = len(SOURCE_MODULES)

    for key in (
        "models_config",
        "entity_prompt",
        "alignment_prompt",
        "bbox_verifier_prompt",
        "ocr_prompt",
        "expression_prompt",
        "expression_verifier_prompt",
        "refinement_prompt",
    ):
        _check(
            resolve_path(route_config[key]).is_file(),
            f"missing {key}: {route_config[key]}",
        )
    if route_config.get('grounding_contract')=='role-grounding-v1':
        from src.routes.role_schedule import stage_plan
        _check(set(models)=={'qwen','sam31','egm'}, 'Unexpected active model set')
        _check(route_config['grounders']==['sam31','egm'], 'Unexpected grounding roles')
        for name in ('scene','object','identity','phrase','verify','adjudicate','ocr'):
            _check(resolve_path('prompts/role_'+name+'.txt').is_file(), 'Missing role prompt')
        report['checks']['stages']=[row[1] for row in stage_plan(route_config)]
    report["checks"]["route_configs"] = 1

    manifest = [
        DatasetRecord.model_validate(item)
        for item in read_jsonl(resolve_path(experiment["manifest"]))
    ]
    _check(
        len(manifest) == int(experiment["num_images"]), "manifest count does not equal num_images"
    )
    _check(len({item.image_id for item in manifest}) == len(manifest), "duplicate image IDs")
    _check(len({item.image_sha256 for item in manifest}) == len(manifest), "duplicate image hashes")
    _check(
        all(resolve_path(item.image_path).is_file() for item in manifest),
        "manifest image missing",
    )
    sample_ids = resolve_path(experiment["sample_ids"]).read_text(encoding="utf-8").splitlines()
    _check(
        sample_ids == [item.image_id for item in manifest],
        "sample_ids order differs from manifest",
    )
    report["checks"]["dataset_records"] = len(manifest)

    report["checks"]["qwen_weights"] = _weight_index(resolve_path(models["qwen"]["local_path"]))
    report["checks"]["egm_weights"] = _weight_index(resolve_path(models["egm"]["local_path"]))
    resources = json.loads(resolve_path("artifacts/resources.json").read_text(encoding="utf-8"))
    for resource_key, model_key in (
        ("qwen", "qwen"),
        ("egm", "egm"),
        ("sam31", "sam31"),
    ):
        _check(
            resources["huggingface_revisions"][resource_key]["revision"]
            == models[model_key]["revision"],
            f"{resource_key} revision record differs from models config",
        )
    for model_key in ("qwen", "egm"):
        indexed = resources["indexed_weights"][model_key]
        _check(indexed["available"], f"{model_key} indexed weights are incomplete")
        _check(
            all(item["sha256"] and item["revision_matches"] for item in indexed["shards"]),
            f"{model_key} shard hash/revision record is incomplete",
        )
        _check(
            all(
                Path(item["path"]).is_file()
                and Path(item["path"]).stat().st_size == item["size_bytes"]
                for item in indexed["shards"]
            ),
            f"{model_key} shard size record is stale",
        )
    report["checks"]["resource_hash_manifest"] = True
    direct_hashes = {}
    for model_key in ("sam31",):
        path = resolve_path(models[model_key]["local_path"])
        _check(path.is_file(), f"missing {model_key} checkpoint")
        resource_key, expected_hash = EXPECTED_DIRECT_HASHES[model_key]
        recorded = resources["model_files"][resource_key]
        _check(recorded["sha256"] == expected_hash, f"unexpected {model_key} SHA-256")
        _check(recorded["size_bytes"] == path.stat().st_size, f"stale {model_key} size record")
        actual_hash = _sha256(path)
        _check(actual_hash == expected_hash, f"current {model_key} file SHA-256 mismatch")
        direct_hashes[model_key] = actual_hash
    report["checks"]["direct_checkpoint_hashes"] = direct_hashes
    sam31_available = resolve_path(models["sam31"]["local_path"]).is_file()
    if args.strict_gated:
        _check(sam31_available, "SAM3.1 checkpoint missing; gated access is required")
    elif not sam31_available:
        report["warnings"].append(
            "SAM3.1 checkpoint unavailable: accept Meta license and set HF_TOKEN"
        )
    report["checks"]["sam31_checkpoint"] = sam31_available

    environment_results = {}
    for model_key, modules in ENV_IMPORTS.items():
        python = Path(models[model_key]["env_python"])
        if not python.exists():
            environment_results[model_key] = {"ok": False, "message": f"missing {python}"}
            continue
        command = [str(python), "-c", ";".join(f"import {module}" for module in modules)]
        print(f"[static] checking environment {model_key}", flush=True)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=args.env_import_timeout,
            check=False,
        )
        environment_results[model_key] = {
            "ok": completed.returncode == 0,
            "message": completed.stderr[-2000:],
        }
    report["checks"]["environment_imports"] = environment_results

    destination = resolve_path(args.output_dir or "artifacts") / "static_validation.json"
    atomic_write_json(destination, report)
    failed_envs = [name for name, result in environment_results.items() if not result["ok"]]
    _check(not failed_envs, f"environment import failures: {failed_envs}; see {destination}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
