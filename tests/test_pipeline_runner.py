import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("setsid"), "requires Linux process groups")
class PipelineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / ".local").mkdir()
        (self.root / "bin").mkdir()
        for name in ["runtime.sh", "run_pipeline.sh", "run_route_b.sh"]:
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.log = self.root / "calls"
        python = self.root / "bin" / "fake-python"
        python.write_text("""#!/usr/bin/env bash
if [[ " $* " == *" --check-only "* ]]; then exit 3; fi
if [[ "$1" == "-" ]]; then cat >/dev/null; echo acceptance >>"$CALL_LOG"; exit 0; fi
while (( $# )); do
  if [[ "$1" == "--stage" ]]; then echo "$2" >>"$CALL_LOG"; break; fi
  shift
done
if [[ "${BLOCK_STAGE:-0}" == 1 ]]; then exec sleep 300; fi
""")
        python.chmod(0o755)
        (self.root / ".local/runtime.sh").write_text(
            'export WORKFLOW_PYTHON="' + str(python) + '"\n')
        (self.root / "scripts/serve_qwen38.sh").write_text(
            "#!/usr/bin/env bash\nexec sleep 300\n")
        for name in ["curl", "nvidia-smi"]:
            tool = self.root / "bin" / name
            tool.write_text("#!/usr/bin/env bash\nexit 0\n")
            tool.chmod(0o755)
        self.env = dict(os.environ, CALL_LOG=str(self.log))
        self.env["PATH"] = str(self.root / "bin") + ":" + os.environ["PATH"]

    def run_process(self):
        return subprocess.Popen(
            ["bash", "scripts/run_pipeline.sh"], cwd=self.root, env=self.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            start_new_session=True)

    def test_stage_order_and_acceptance(self):
        process = self.run_process()
        _, errors = process.communicate(timeout=90)
        self.assertEqual(process.returncode, 0, errors)
        self.assertEqual(self.log.read_text().splitlines(), [
            "entities", "ground", "aggregate", "align", "promote", "bbox_verify",
            "ocr", "describe", "reground", "expression_verify",
            "refine_generate", "refine_reground", "refine_verify",
            "refine_generate", "refine_reground", "refine_verify",
            "finalize", "review", "acceptance"])

    def test_termination_reports_resumable_exit(self):
        self.env["BLOCK_STAGE"] = "1"
        process = self.run_process()
        try:
            deadline = time.monotonic() + 15
            while not self.log.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(self.log.exists(), "stage did not start")
            process.send_signal(signal.SIGTERM)
            _, errors = process.communicate(timeout=45)
            self.assertEqual(process.returncode, 75, errors)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
