import os
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.config import PROJECT_ROOT, load_yaml
from src.grounding.cache import legacy_contract


class RuntimeConfigTests(unittest.TestCase):
    def test_standalone_server_supports_role_review_images_and_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scripts=root/'scripts';scripts.mkdir()
            for name in ('runtime.sh','serve_qwen38.sh'):
                shutil.copyfile(PROJECT_ROOT/'scripts'/name,scripts/name)
            binary=root/'env/bin/vllm';binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');binary.chmod(0o755)
            env={k:v for k,v in os.environ.items() if k!='QWEN_IMAGE_LIMIT'}
            env['QWEN_ENV']=str(root/'env')
            for limit in (None,'4'):
                with self.subTest(limit=limit):
                    if limit:env['QWEN_IMAGE_LIMIT']=limit
                    args=subprocess.check_output(['bash',str(scripts/'serve_qwen38.sh')],env=env,text=True).splitlines()
                    self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt')+1])['image'],int(limit or 6))

    def test_override_preserves_algorithm_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / "config.yaml"
            local.write_text("configs/models.yaml:\n  egm:\n    env_python: /example/python\n")
            with patch.dict(os.environ, {"WORKFLOW_LOCAL_CONFIG": str(local)}):
                config = load_yaml("configs/models.yaml")
                self.assertEqual(config["egm"]["env_python"], "/example/python")
                self.assertEqual(config["egm"]["name"], "nvidia/EGM-8B")
                self.assertEqual(config["egm"]["dtype"], "bfloat16")
                self.assertEqual(config["sam31"]["version"], "sam3.1")

    def test_explicit_missing_config_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"WORKFLOW_LOCAL_CONFIG": folder + "/missing"}):
                with self.assertRaises(FileNotFoundError):
                    load_yaml("configs/models.yaml")

    def test_invalid_override_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / "config.yaml"
            local.write_text("configs/models.yaml: invalid\n")
            with patch.dict(os.environ, {"WORKFLOW_LOCAL_CONFIG": str(local)}):
                with self.assertRaises(TypeError):
                    load_yaml("configs/models.yaml")

    def test_missing_legacy_metadata_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"WORKFLOW_LEGACY_ROOT": folder}):
                legacy_contract.cache_clear()
                self.assertEqual(legacy_contract(), ({}, {}, {}))
        legacy_contract.cache_clear()


if __name__ == "__main__":
    unittest.main()
