import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.utils.acceptance import source_hashes,checked_acceptance


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ('src/adapter.py','.local/config.yaml','.local/multi_node/scheduler.py'):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
        (self.root/'tests.log').write_text('test result')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        hashes=source_hashes(self.root)
        (self.root/'gpu.json').write_text(json.dumps(dict(source_hashes=hashes,status='passed',objects=2,
                                                        verified=1,resident_models=['sam31','egm'])))
        self.receipt=dict(status='passed',platform='linux',source_hashes=hashes,
                          tests={n:dict(log='tests.log',returncode=0,sha256=sha(self.root/'tests.log'))
                                 for n in ('workflow','multi_node','multi_gpu')},
                          gpu=[dict(result='gpu.json',sha256=sha(self.root/'gpu.json'))])
        self.write()

    def write(self):(self.root/'acceptance.json').write_text(json.dumps(self.receipt))

    def test_exact_tested_tree_is_accepted(self):
        self.assertEqual(checked_acceptance(self.root),self.receipt)

    def test_adapter_infra_and_private_config_changes_invalidate_acceptance(self):
        for name in ('src/adapter.py','.local/config.yaml','.local/multi_node/scheduler.py'):
            p=self.root/name
            with self.subTest(name=name):
                p.write_text('changed')
                with self.assertRaisesRegex(ValueError,'Source changed'):checked_acceptance(self.root)
                p.write_text('original')

    def test_old_or_failed_gpu_log_cannot_be_reused(self):
        (self.root/'gpu.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'GPU receipt changed'):checked_acceptance(self.root)

    def test_failed_suite_and_missing_suite_are_rejected(self):
        self.receipt['tests']['workflow']['returncode']=1;self.write()
        with self.assertRaisesRegex(ValueError,'failed'):checked_acceptance(self.root)
        self.receipt['tests'].pop('workflow');self.write()
        with self.assertRaisesRegex(ValueError,'Missing'):checked_acceptance(self.root)


if __name__=='__main__':unittest.main()
