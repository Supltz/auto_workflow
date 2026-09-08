"""GroundingDINO Swin-B phrase-to-box worker using official inference utilities."""

from __future__ import annotations

from typing import Any

from src.grounding.base import PhraseGrounder, run_phrase_worker, worker_parser
from src.utils.config import load_yaml, resolve_path
from src.utils.provenance import model_metadata


class GroundingDinoGrounder(PhraseGrounder):
    name = "groundingdino"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        repo = resolve_path(config["repo_path"])
        # GroundingDINO is installed with its compiled CUDA extension in this
        # environment.  Prepending the source checkout shadows that installed
        # package with a directory that has no _C shared object.
        import torch  # noqa: F401 - loads the libraries required by groundingdino._C

        try:
            from groundingdino import _C  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "GroundingDINO CUDA extension failed to load; rebuild the "
                "groundingdino environment with CUDA_HOME set"
            ) from exc
        from groundingdino.util.inference import load_model

        self._load_image = __import__(
            "groundingdino.util.inference", fromlist=["load_image"]
        ).load_image
        self._predict = __import__("groundingdino.util.inference", fromlist=["predict"]).predict
        self.model = load_model(
            str(repo / config["model_config"]), str(resolve_path(config["local_path"]))
        )
        self.provenance = model_metadata(config, config, "v1")

    def ground(self, image_path: str, phrase: str) -> list[dict[str, Any]]:
        _, tensor = self._load_image(image_path)
        boxes, logits, detected_phrases = self._predict(
            model=self.model,
            image=tensor,
            caption=phrase,
            box_threshold=float(self.config.get("box_threshold", 0.25)),
            text_threshold=float(self.config.get("text_threshold", 0.20)),
        )
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        detections = []
        for box, score, detected_phrase in zip(boxes.tolist(), logits.tolist(), detected_phrases):
            center_x, center_y, box_width, box_height = box
            detections.append(
                {
                    "bbox_xyxy": [
                        (center_x - box_width / 2) * width,
                        (center_y - box_height / 2) * height,
                        (center_x + box_width / 2) * width,
                        (center_y + box_height / 2) * height,
                    ],
                    "score": float(score),
                    "metadata": {"detected_phrase": detected_phrase},
                }
            )
        return sorted(detections, key=lambda item: item["score"], reverse=True)


def main() -> None:
    args = worker_parser(__doc__ or "GroundingDINO worker").parse_args()
    config = load_yaml(args.model_config)["groundingdino"]
    run_phrase_worker(GroundingDinoGrounder(config), args)


if __name__ == "__main__":
    main()
