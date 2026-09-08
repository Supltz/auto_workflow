# Isolated model environments

Default executables are `envs/{qwen,rex,sam3,groundingdino}/bin/python`.
Override executable paths through external `.local/config.yaml` settings.
Dependency specifications live in `requirements/environment_*.yml`.
The setup script accepts `ENV_ROOT`, `MAMBA` and `REGION_BENCHMARK_STORE`.
