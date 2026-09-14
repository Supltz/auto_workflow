"""Rex-Omni phrase-to-box worker using the official wrapper."""

from __future__ import annotations

import os
from typing import Any

from src.grounding.base import PhraseGrounder, run_phrase_worker, worker_parser
from src.utils.config import load_yaml, resolve_path
from src.utils.images import open_rgb
from src.utils.provenance import model_metadata


class RexGrounder(PhraseGrounder):
    name = "rex_omni"

    def __init__(self, config: dict[str, Any]) -> None:
        import sys

        repo = str(resolve_path(config["repo_path"]))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from rex_omni import RexOmniWrapper

        self.model = RexOmniWrapper(
            model_path=str(resolve_path(config["local_path"])),
            backend="transformers",
            # The official wrapper defaults to FlashAttention 2, but that optional
            # extension is not part of the isolated Rex environment. PyTorch SDPA
            # is supported by Qwen2.5-VL and avoids a brittle local CUDA build.
            attn_implementation=config.get("attn_implementation", "sdpa"),
            # A normal single-phrase detection is only a few dozen tokens. On
            # difficult prompts Rex can otherwise repeat near-identical boxes
            # until the 2048-token limit, wasting time and producing junk rows.
            max_tokens=int(
                os.environ.get(
                    "REX_MAX_OUTPUT_TOKENS", config.get("max_output_tokens", 256)
                )
            ),
            temperature=0.0,
            top_p=0.05,
            top_k=1,
            repetition_penalty=1.05,
        )
        self.provenance = model_metadata(config, config, "v1")
        self.batch_size = max(1, int(os.environ.get("REX_BATCH_SIZE", config.get("batch_size", 2))))

    @staticmethod
    def _detections(result: dict[str, Any], phrase: str) -> list[dict[str, Any]]:
        extracted = result.get("extracted_predictions", {})
        predictions = extracted.get(phrase, [])
        if not predictions and len(extracted) == 1:
            predictions = next(iter(extracted.values()))
        valid_predictions = [prediction for prediction in predictions if prediction.get("type") == "box"]
        return [{"bbox_xyxy":prediction["coords"],"score":None,"metadata":{
            "score_available":False,"raw_output":result.get("raw_output"),
            "raw_prediction_count":len(valid_predictions),"predictions_truncated":False,
        }} for prediction in valid_predictions]

    def ground(self, image_path: str, phrase: str) -> list[dict[str, Any]]:
        return self.ground_batch([(image_path,phrase)])[0]

    def ground_batch(self, requests: list[tuple[str,str]]) -> list[list[dict[str,Any]]]:
        images=[open_rgb(image_path) for image_path,_ in requests]
        try:
            # Rex's official wrapper treats each list item as an independent
            # prompt/sample. This batches tensor work without combining phrases.
            results=self.model.inference(images=images,task=["detection"]*len(images),
                categories=[[phrase] for _,phrase in requests])
            if len(results)!=len(requests):raise RuntimeError("Rex batch result count changed")
            return [self._detections(result,phrase) for result,(_,phrase) in zip(results,requests)]
        finally:
            for image in images:image.close()


def main() -> None:
    parser = worker_parser(__doc__ or "Rex worker")
    args = parser.parse_args()
    config = load_yaml(args.model_config)["rex"]
    from src.grounding.resident import adapter
    run_phrase_worker(adapter("rex", config, RexGrounder), args)


if __name__ == "__main__":
    main()
