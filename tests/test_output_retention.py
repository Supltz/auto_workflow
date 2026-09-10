"""Output-retention regressions without model inference."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from src.routes import route_b_checkpoint as checkpoint
from src.routes import route_b_discovery as discovery
from src.routes import route_b_stages as stages
from src.utils.io import read_jsonl, rewrite_jsonl_atomic


class OutputRetentionTests(unittest.TestCase):
    def test_history_is_opt_in_and_existing_history_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stage.jsonl'
            checkpoint.archive_and_replace(path, [{'value': 1}], history_limit=1)
            checkpoint.archive_and_replace(path, [{'value': 2}], history_limit=1)
            history = list((path.parent / 'checkpoint_archive').iterdir())
            with patch.dict(os.environ, {'ROUTE_B_CHECKPOINT_HISTORY_LIMIT': '0'}):
                checkpoint.archive_and_replace(path, [{'value': 3}])
            self.assertEqual(list(read_jsonl(path)), [{'value': 3}])
            self.assertEqual(list((path.parent / 'checkpoint_archive').iterdir()), history)
            summary = path.parent / 'checkpoint_updates/stage.jsonl.json'
            before = summary.read_bytes()
            checkpoint.archive_and_replace(path, [{'value': 3}], history_limit=0)
            self.assertEqual(summary.read_bytes(), before)

    def test_history_limit_does_not_remove_other_stages_or_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stage.jsonl'
            history = path.parent / 'checkpoint_archive'
            history.mkdir()
            unrelated = history / 'other_0000000000000000.jsonl'
            unrelated.write_text('keep')
            link = history / 'stage_0000000000000000.jsonl'
            link.symlink_to(unrelated)
            for i in range(6):
                checkpoint.archive_and_replace(path, [{'value': i}], history_limit=2)
            retained = [p for p in history.glob('stage_*.jsonl') if not p.is_symlink()]
            self.assertEqual(len(retained), 2)
            self.assertIn([{'value': 4}], [list(read_jsonl(p)) for p in retained])
            self.assertEqual(unrelated.read_text(), 'keep')
            self.assertTrue(link.is_symlink())

    def test_failed_replace_keeps_active_checkpoint_and_does_not_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stage.jsonl'
            checkpoint.archive_and_replace(path, [{'value': 1}], history_limit=1)
            checkpoint.archive_and_replace(path, [{'value': 2}], history_limit=1)
            history = list((path.parent / 'checkpoint_archive').iterdir())
            def fail_active(target, records):
                if target == path:
                    raise OSError('simulated write failure')
                rewrite_jsonl_atomic(target, records)
            with patch.object(checkpoint, 'rewrite_jsonl_atomic', side_effect=fail_active):
                with self.assertRaises(OSError):
                    checkpoint.archive_and_replace(path, [{'value': 3}], history_limit=1)
            self.assertEqual(list(read_jsonl(path)), [{'value': 2}])
            self.assertTrue(all(p.exists() for p in history))
            with self.assertRaises(ValueError):
                checkpoint.archive_and_replace(path, [], history_limit=-1)
            self.assertEqual(list(read_jsonl(path)), [{'value': 2}])

    def test_promotion_only_writes_numbered_view_for_accepted_targets(self):
        for accepted in (False, True):
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'source.jpg'
                Image.new('RGB', (16, 16), 'red').save(source)
                prompt = root / 'prompt.txt'
                prompt.write_text('promotion')
                candidate = dict(instance_id='i', bbox_xyxy=[0, 0, 8, 8], category_query='object',
                                 verifier_overlay_path=str(source), tight_crop_path=str(source),
                                 context_crop_path=str(source))
                entity = dict(entity_id='e', category_query='object', caption_supported=False,
                              caption_span='')
                decision = SimpleNamespace(accepted=accepted, reject_reason=None if accepted else 'no match',
                                           proposal=Mock(model_dump=Mock(return_value=entity)) if accepted else None,
                                           model_dump=lambda **kwargs: {'accepted': accepted})
                with patch.object(discovery, 'ensure_artifacts'), patch.object(
                    stages, '_compact_candidate', side_effect=lambda c: c
                ), patch.object(discovery, 'Qwen38Client') as client:
                    client.return_value.generate_json.return_value = (decision, '{}')
                    discovery._promote_tasks.__wrapped__(
                        tasks=[dict(image_id='image', entity_id='e', rank=1, candidate=candidate,
                                    peers=[candidate])], output_path=root / 'out.jsonl',
                        failures_path=root / 'failures.jsonl', prompt_path=prompt, output_root=root,
                        manifest={'image': {'image_path': str(source), 'caption': ''}},
                        qwen_config={}, config={'route': 'B_caption'}, workers=1, resume=False,
                        overwrite=False)
                    self.assertEqual(len(client.return_value.generate_json.call_args.args[1]), 5)
                record = list(read_jsonl(root / 'out.jsonl'))[0]
                self.assertEqual(record['accepted'], accepted)
                numbered = Path(record['target_numbered_overlay_path'])
                self.assertEqual(numbered.exists(), accepted)
                if accepted:
                    expected = root / 'expected.jpg'
                    with Image.open(source) as image:
                        discovery.save_numbered_bbox_overlay(image, [candidate], expected, highlight_id='i')
                    self.assertEqual(numbered.read_bytes(), expected.read_bytes())

    def test_final_rejections_keep_identity_and_reason_without_nested_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = dict(image_id='image', entity_id='e', entity_rank=1, region_id='r',
                           accepted=False, metadata={'raw_response': 'large'},
                           candidate_instances=[{'tight_crop_path': 'temporary.jpg'}])
            stages.materialize_final_route_b_outputs(
                initial_verifications=[dict(payload, base_expression_id='expr', revision=0,
                                            reject_reason='expression failed')],
                refinement_verifications=[],
                grounding_rejections=[dict(payload, stage='grounding', reject_reason='no match')],
                alignments=[dict(payload, alignment={'reject_reason': 'alignment failed'})],
                bbox_records=[dict(payload, bbox_verification={'reject_reason': 'bbox failed'})],
                verified_path=root / 'verified.jsonl', rejected_path=root / 'rejected.jsonl')
            self.assertEqual(list(read_jsonl(root / 'verified.jsonl')), [])
            records = list(read_jsonl(root / 'rejected.jsonl'))
            self.assertEqual({r['reject_reason'] for r in records},
                             {'expression failed', 'no match', 'alignment failed', 'bbox failed'})
            for record in records:
                self.assertEqual(record['region_id'], 'r')
                self.assertNotIn('metadata', record)
                self.assertNotIn('candidate_instances', record)
