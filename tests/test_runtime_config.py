import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.config import PROJECT_ROOT, load_yaml
from src.grounding.cache import legacy_contract


class RuntimeConfigTests(unittest.TestCase):
    def test_override_preserves_algorithm_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / "config.yaml"
            local.write_text("configs/models.yaml:\n  rex:\n    env_python: /example/python\n")
            with patch.dict(os.environ, {"WORKFLOW_LOCAL_CONFIG": str(local)}):
                config = load_yaml("configs/models.yaml")
                self.assertEqual(config["rex"]["env_python"], "/example/python")
                self.assertEqual(config["rex"]["name"], "IDEA-Research/Rex-Omni")
                self.assertEqual(config["rex"]["dtype"], "bfloat16")
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
