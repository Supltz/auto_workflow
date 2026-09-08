# Route B model and environment setup

Route B requires four isolated model runtimes:

- Qwen: `Qwen/Qwen3.8-27B`
- Rex: `IDEA-Research/Rex-Omni`
- SAM3.1: `facebookresearch/sam3` with gated `facebook/sam3.1` checkpoint
- GroundingDINO: `IDEA-Research/GroundingDINO` Swin-B CogCoor checkpoint

Download or verify the Route B resources and record their provenance:

```bash
hf auth login
bash scripts/download_models.sh
python scripts/record_resources.py
```

Create or repair the isolated environments:

```bash
bash scripts/setup_envs.sh
```

The configured executables are in `configs/models.yaml`. On a configured GPU runtime, verify the
device and run the non-inference checks before running Route B:

```bash
nvidia-smi
python scripts/static_validate.py
python scripts/check_route_b.py
```

SAM3.1 uses the official single-image session flow and returns normalized XYWH boxes,
probabilities, and binary masks. GroundingDINO requires its compiled CUDA extension.
