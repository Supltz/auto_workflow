"""SAM 3.1 phrase-to-mask worker via the official single-image session API."""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any

from src.grounding.base import PhraseGrounder, run_phrase_worker, worker_parser
from src.utils.config import load_yaml, resolve_path
from src.utils.geometry import normalized_xywh_to_xyxy
from src.utils.provenance import model_metadata


class Sam31Grounder(PhraseGrounder):
    name = "sam31"

    def __init__(self, config: dict[str, Any]) -> None:
        import sys

        self.config = config
        repo = str(resolve_path(config["repo_path"]))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import sam3.model_builder as sam3_builder

        checkpoint = resolve_path(config["local_path"])
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM3.1 gated checkpoint missing: {checkpoint}; accept the Meta license and set HF_TOKEN"
            )
        # The public SAM 3.1 builder loads the merged detector+tracker
        # checkpoint once into a tracker-only intermediate model and then again
        # into the assembled model.  The first load cannot match the merged key
        # hierarchy and emits thousands of misleading missing-key warnings; the
        # second load is the one that initializes the predictor.  Build that
        # intermediate model without a checkpoint and retain the final load.
        original_tracker_builder = sam3_builder.build_sam3_multiplex_video_model

        def build_uninitialized_tracker(*args, **kwargs):
            kwargs["checkpoint_path"] = None
            kwargs["load_from_HF"] = False
            return original_tracker_builder(*args, **kwargs)

        sam3_builder.build_sam3_multiplex_video_model = build_uninitialized_tracker
        try:
            self.predictor = sam3_builder.build_sam3_predictor(
                checkpoint_path=str(checkpoint),
                version="sam3.1",
                compile=False,
                use_fa3=bool(config.get("use_flash_attention_3", False)),
                use_rope_real=bool(config.get("use_rope_real", False)),
            )
        finally:
            sam3_builder.build_sam3_multiplex_video_model = original_tracker_builder
        self.provenance = model_metadata(config, config, "v1")
        self._session_id: str | None = None
        self._session_path: str | None = None
        self._image_size: tuple[int,int] | None = None
        # SAM3 clears every cache for a new text prompt. Integer keys are the
        # source-inspected, prompt-independent frame/backbone cache; text and
        # tracking state keep their original reset behavior.
        original_reset=self.predictor.model.reset_state
        def reset_preserving_image_features(state):
            visual={key:value for key,value in state.get("feature_cache",{}).items()
                    if isinstance(key,int)}
            original_reset(state)
            state.setdefault("feature_cache",{}).update(visual)
        self.predictor.model.reset_state=reset_preserving_image_features

    def _start_session(self, resource_path: str) -> str:
        """Start a session while tolerating SAM3/SAM3.1 init-state API drift."""
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
        }
        if hasattr(self.predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.predictor.async_loading_frames
        if hasattr(self.predictor, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.predictor.video_loader_type

        signature = inspect.signature(self.predictor.model.init_state)
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            init_kwargs = {
                key: value for key, value in init_kwargs.items() if key in signature.parameters
            }
        inference_state = self.predictor.model.init_state(**init_kwargs)
        session_id = str(uuid.uuid4())
        now = time.time()
        self.predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": now,
            "last_use_time": now,
        }
        return session_id

    def ground(self, image_path: str, phrase: str) -> list[dict[str, Any]]:
        from PIL import Image
        resolved=str(resolve_path(image_path))
        if self._session_path!=resolved:
            self.close()
            with Image.open(image_path) as image:self._image_size=image.size
            self._session_id=self._start_session(resolved);self._session_path=resolved
        width,height=self._image_size
        response=self.predictor.handle_request({"type":"add_prompt","session_id":self._session_id,
            "frame_index":0,"text":phrase,
            "output_prob_thresh":float(self.config.get("score_threshold",0.5))})
        outputs=response["outputs"]
        scores=outputs.get("out_probs",[None]*len(outputs["out_boxes_xywh"]))
        return [{"bbox_xyxy":normalized_xywh_to_xyxy(box,width,height),
            "score":None if score is None else float(score),"mask":mask,
            "metadata":{"sam_version":"sam3.1","object_id":int(object_id)}}
            for box,score,mask,object_id in zip(outputs["out_boxes_xywh"],scores,
                outputs["out_binary_masks"],outputs["out_obj_ids"])]

    def close(self) -> None:
        if self._session_id is not None:
            self.predictor.handle_request({"type":"close_session","session_id":self._session_id})
        self._session_id=None;self._session_path=None;self._image_size=None


def main() -> None:
    args = worker_parser(__doc__ or "SAM3.1 worker").parse_args()
    config = load_yaml(args.model_config)["sam31"]
    from src.grounding.resident import adapter
    run_phrase_worker(adapter("sam31", config, Sam31Grounder), args)


if __name__ == "__main__":
    main()
