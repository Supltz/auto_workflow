"""Common phrase-grounder protocol and JSONL worker driver."""

from __future__ import annotations

import argparse
import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.schema import GroundingRequest, RegionRecord
from src.utils.ram_checkpoint import checkpoint_boundary
from src.utils.geometry import clip_box, valid_box
from src.utils.images import open_rgb, save_crop, save_mask, save_overlay
from src.utils.io import SessionJsonlWriter, failure_writer, read_jsonl


class PhraseGrounder(ABC):
    name: str
    provenance: dict[str, Any]

    @abstractmethod
    def ground(self, image_path: str, phrase: str) -> list[dict[str, Any]]:
        """Return detections with bbox_xyxy, optional score, and optional mask."""

    def ground_batch(self, requests: list[tuple[str, str]]) -> list[list[dict[str, Any]]]:
        """Semantics-preserving default; adapters may vectorize independent calls."""
        return [self.ground(image_path, phrase) for image_path, phrase in requests]

    def close(self) -> None:
        """Release optional per-image state."""


def worker_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--failures-file", default="outputs/failures.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--context-signature", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_phrase_worker(adapter: PhraseGrounder, args: argparse.Namespace) -> None:
    output = Path(args.output_jsonl)
    if args.overwrite and output.exists():
        output.unlink()
    progress_path = output.with_name(f"{output.stem}.progress.jsonl")
    if args.overwrite and progress_path.exists():
        progress_path.unlink()
    done = (
        {item["request_id"] for item in read_jsonl(progress_path)}
        if args.resume and progress_path.exists()
        else set()
    )
    artifact_root = Path(args.output_dir)

    requests = list(read_jsonl(args.input_jsonl))
    print(f"[ground_{adapter.name}] input_requests={len(requests)} cached_ids={len(done)}", flush=True)
    selected = []
    for index, raw_request in enumerate(requests):
        if index < args.start_index or (args.end_index is not None and index >= args.end_index):continue
        request = GroundingRequest.model_validate(raw_request)
        if request.request_id not in done:selected.append((index,request))
    batch_size=max(1,int(getattr(adapter,"batch_size",1)))
    try:
        with SessionJsonlWriter(output) as writer, SessionJsonlWriter(progress_path) as progress:
            for offset in range(0,len(selected),batch_size):
                batch=selected[offset:offset+batch_size]
                for index,request in batch:
                    print(f"[ground_{adapter.name}] request={index + 1}/{len(requests)} image={request.image_id} START",flush=True)
                batch_started=time.perf_counter()
                try:
                    results=adapter.ground_batch([(request.image_path,request.phrase) for _,request in batch])
                    if len(results)!=len(batch):raise RuntimeError("ground_batch result count changed")
                except Exception as exc:
                    index,request=batch[0]
                    failure_writer(args.failures_file,image_id=request.image_id,route=request.route,
                        stage=f"ground_{adapter.name}",error_type=type(exc).__name__,message=str(exc),
                        request_id=request.request_id)
                    raise RuntimeError(f"ground_{adapter.name}: fatal batch at {request.request_id}: {exc}") from exc
                infer_each=(time.perf_counter()-batch_started)/len(batch)
                for (index,request),detections in zip(batch,results):
                    started=time.perf_counter();image=None
                    artifact_seconds=commit_seconds=0.0
                    try:
                        artifact_started=time.perf_counter()
                        if args.save_crops or args.save_overlays:image=open_rgb(request.image_path)
                        saved_count=0;records=[]
                        for detection_index,detection in enumerate(detections):
                            box=clip_box(detection["bbox_xyxy"],request.image_width,request.image_height)
                            if not valid_box(box,request.image_width,request.image_height):continue
                            request_tag=hashlib.sha1(request.request_id.encode("utf-8")).hexdigest()[:10]
                            stem=f"{request.image_id}_{request.rank:02d}_{request_tag}_{adapter.name}_{detection_index:02d}"
                            crop_path=overlay_path=mask_path=None
                            if args.save_crops:
                                crop_path=artifact_root/"crops"/request.route/f"{stem}.jpg";save_crop(image,box,crop_path)
                            if args.save_overlays:
                                overlay_path=artifact_root/"overlays"/request.route/f"{stem}.jpg";save_overlay(image,box,request.phrase,overlay_path)
                            if args.save_masks and detection.get("mask") is not None:
                                mask_path=artifact_root/"masks"/request.route/f"{stem}.png";save_mask(np.asarray(detection["mask"]),mask_path)
                            metadata={**request.metadata,**adapter.provenance,**detection.get("metadata",{}),
                                "request_id":request.request_id,"detection_index":detection_index,
                                "route_variant":f"{request.route}_{adapter.name}"}
                            result=RegionRecord(image_id=request.image_id,route=request.route,rank=request.rank,
                                phrase=request.phrase,bbox_xyxy=box,mask_path=str(mask_path) if mask_path else None,
                                crop_path=str(crop_path) if crop_path else None,
                                overlay_path=str(overlay_path) if overlay_path else None,
                                source_method=request.metadata.get("source_method","phrase_grounding"),
                                grounder=adapter.name,grounding_score=detection.get("score"),
                                image_width=request.image_width,image_height=request.image_height,
                                runtime_ms=(infer_each+time.perf_counter()-started)*1000,metadata=metadata)
                            records.append(result.model_dump(mode="json"));saved_count+=1
                        artifact_seconds=time.perf_counter()-artifact_started
                        commit_started=time.perf_counter();writer.append_many(records)
                        progress.append({"request_id":request.request_id,"image_id":request.image_id,
                            "detections":saved_count,"context_signature":getattr(args,"context_signature","")})
                        commit_seconds=time.perf_counter()-commit_started;done.add(request.request_id);checkpoint_boundary()
                    except Exception as exc:
                        failure_writer(args.failures_file,image_id=request.image_id,route=request.route,
                            stage=f"ground_{adapter.name}",error_type=type(exc).__name__,message=str(exc),
                            request_id=request.request_id)
                        raise RuntimeError(f"ground_{adapter.name}: fatal request {request.request_id}: {exc}") from exc
                    finally:
                        print(f"[ground_{adapter.name}] request={index + 1}/{len(requests)} saved={request.request_id in done} "
                              f"elapsed={infer_each+time.perf_counter()-started:.1f}s infer={infer_each:.3f}s "
                              f"artifacts={artifact_seconds:.3f}s commit={commit_seconds:.3f}s",flush=True)
                        if image is not None:image.close()
    finally:adapter.close()
