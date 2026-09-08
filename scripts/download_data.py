#!/usr/bin/env python3
"""Download the fixed ModelScope shard and record its physical Parquet schema."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from src.utils.config import load_yaml, resolve_path
from src.utils.io import add_job_arguments, atomic_write_json
from src.utils.provenance import sha256_file

CAPTION_CANDIDATES = ("global_caption", "dense_caption", "caption", "text", "description")
IMAGE_CANDIDATES = ("opensource_url", "image", "image_url", "url", "path", "image_path")
ID_CANDIDATES = ("id", "image_id", "sa_id", "key")
WIDTH_CANDIDATES = ("width", "image_width", "w")
HEIGHT_CANDIDATES = ("height", "image_height", "h")


def _first_present(names: list[str], candidates: tuple[str, ...]) -> str | None:
    folded = {name.casefold(): name for name in names}
    return next((folded[item] for item in candidates if item in folded), None)


def _json_preview(value):
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_job_arguments(parser, "configs/experiment.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)["dataset"]
    download_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(config["download_dir"])
    )
    shard = download_dir / config["shard"]
    if shard.exists() and args.overwrite:
        shard.unlink()
    if not shard.exists():
        executable = shutil.which("modelscope")
        if not executable:
            raise FileNotFoundError("modelscope CLI is absent from the active environment")
        subprocess.run(
            [
                executable,
                "download",
                "--repo-type",
                "dataset",
                "--revision",
                config["revision"],
                config["repo_id"],
                config["shard"],
                "dataset_infos.json",
                "README.md",
                "--local-dir",
                str(download_dir),
            ],
            check=True,
        )
    parquet = pq.ParquetFile(shard)
    names = parquet.schema_arrow.names
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in parquet.schema_arrow
    ]
    row_groups = [
        {
            "index": index,
            "rows": parquet.metadata.row_group(index).num_rows,
            "total_byte_size": parquet.metadata.row_group(index).total_byte_size,
        }
        for index in range(parquet.num_row_groups)
    ]
    preview_batch = next(parquet.iter_batches(batch_size=5))
    preview = [
        {key: _json_preview(value) for key, value in record.items()}
        for record in preview_batch.to_pylist()
    ]
    image_column = _first_present(names, IMAGE_CANDIDATES)
    id_column = _first_present(names, ID_CANDIDATES)
    column_roles = {
        "image_id": id_column or f"derived from basename of {image_column}",
        "image": image_column,
        "dense_caption": _first_present(names, CAPTION_CANDIDATES),
        "width": _first_present(names, WIDTH_CANDIDATES) or "decoded image width",
        "height": _first_present(names, HEIGHT_CANDIDATES) or "decoded image height",
        "source_index": "physical parquet row index",
    }
    atomic_write_json(
        resolve_path(config["schema_output"]),
        {
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "shard": config["shard"],
            "local_path": str(shard),
            "sha256": sha256_file(shard),
            "num_rows": parquet.metadata.num_rows,
            "num_row_groups": parquet.num_row_groups,
            "fields": fields,
            "column_roles": column_roles,
            "preview_records": preview,
            "row_groups": row_groups,
        },
    )


if __name__ == "__main__":
    main()
