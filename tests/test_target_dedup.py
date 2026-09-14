import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.models.qwen38_client import QwenValidationError
from src.routes.route_b_checkpoint import CHECK_ONLY, StagePending
from src.routes.route_b_discovery import promotion_tasks, unique_alignments
from src.routes.route_b_stages import _deduplicate_final_records
from src.routes.target_dedup import TargetDeduplicator, TargetIdentityDecision

ROOT = Path(__file__).resolve().parents[1]
VAN_A = [1109.8936032503843, 1191.033040624112, 1223.728196695447, 1244.3597587533295]
VAN_B = [1111.111111111111, 1198.5425425425424, 1220.7207207207207, 1244.964964964965]


class TargetDedupTests(unittest.TestCase):
    def test_project_relative_inputs_reuse_cache_from_task_cwd(self):
        from src.utils.config import resolve_path
        project=self.root/'project';(project/'data/images').mkdir(parents=True)
        (project/'prompts').mkdir()
        source=project/'data/images/source.png';source.write_bytes(self.source.read_bytes())
        prompt=project/'prompts/identity.txt';prompt.write_text('same prompt')
        task=self.root/'task';task.mkdir()
        config={**self.config,'target_dedup_prompt':'prompts/identity.txt'}
        first={**self.a,'source_image':'data/images/source.png'}
        second={**self.b,'source_image':'data/images/source.png'}
        before=copy.deepcopy([first,second]);cwd=Path.cwd()
        try:
            with patch('src.routes.target_dedup.resolve_path',side_effect=lambda p:resolve_path(p,root=project)):
                os.chdir(project)
                matcher=TargetDeduplicator(config=config,qwen_config={'name':'test'},output_root=task/'outputs')
                with self.answer('different_objects') as model:
                    original=matcher.compare(first,second)
                    model.assert_called_once()
                cache=Path(original['audit_path']);saved=cache.read_bytes()
                os.chdir(task)
                matcher=TargetDeduplicator(config=config,qwen_config={'name':'test'},output_root=Path('outputs'))
                token=CHECK_ONLY.set(True)
                try:
                    with self.answer() as model:
                        resumed=matcher.compare(first,second)
                        model.assert_not_called()
                finally:CHECK_ONLY.reset(token)
                self.assertEqual(Path(resumed['audit_path']).resolve(),cache.resolve())
                self.assertEqual(cache.read_bytes(),saved)
                self.assertFalse(resumed['same_object'])
                self.assertEqual([first,second],before)
                source.unlink()
                with self.assertRaises(FileNotFoundError):matcher.compare(first,second)
        finally:os.chdir(cwd)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'
        Image.new('RGB', (1500, 2108)).save(self.source)
        self.config = {'target_dedup_prompt': str(ROOT / 'prompts/route_b_target_identity.txt'),
                       'aggregation': {'min_grounder_support': 2}}
        self.matcher = self.new_matcher()
        self.a = self.record('a', VAN_A, 1)
        self.b = self.record('b', VAN_B, 2)

    def new_matcher(self, model='test'):
        return TargetDeduplicator(config=self.config, qwen_config={'name': model},
                                  output_root=self.root)

    def record(self, name, box, rank):
        return {'image_id': 'van_image', 'source_image': str(self.source),
                'region_id': name, 'entity_id': name, 'instance_id': name,
                'bbox_xyxy': box, 'category': 'van', 'category_query': 'van',
                'scope': 'whole_object', 'entity_rank': rank, 'accepted': True,
                'grounder_support': 2, 'median_pairwise_iou': .9,
                'discovery_category_only': True, 'size_pass': True,
                'reground_audit': {'expression_grounder_support': 2, 'reground_iou': .9},
                'bbox_grounder_support': 2, 'bbox_median_pairwise_iou': .9, 'revision': 0,
                'final_referring_expression': 'the van ' + name}

    def alignment(self, record):
        return {**record, 'selected_candidate': copy.deepcopy(record),
                'entity': {'category_query': 'van', 'scope': 'whole_object'},
                'alignment': {'reject_reason': None}}

    def answer(self, decision='same_object'):
        return patch('src.routes.target_dedup.Qwen38Client.generate_json',
                     return_value=(TargetIdentityDecision(decision=decision, reason='visual evidence'), '{}'))

    def test_van_regression_and_persistent_symmetric_cache(self):
        with self.answer() as model:
            audit = self.matcher.compare(self.a, self.b)
            self.assertTrue(audit['same_object'])
            self.assertAlmostEqual(audit['iou'], .8183480852)
            self.assertEqual(len(model.call_args.args[1]), 3)
            self.assertTrue(self.new_matcher().compare(self.b, self.a)['same_object'])
            model.assert_called_once()
            self.assertTrue(Path(audit['audit_path']).exists())

    def test_adjacent_overlapping_objects_and_uncertain_are_retained(self):
        for decision in ['different_objects', 'uncertain']:
            with self.subTest(decision=decision), self.answer(decision):
                matcher = self.new_matcher(model=decision)
                kept, rejected = _deduplicate_final_records([self.a, self.b], .98,
                                                            deduplicator=matcher)
                self.assertEqual(len(kept), 2)
                self.assertEqual(rejected, [])

    def test_both_discovery_paths_and_finalization_share_identity(self):
        with self.answer() as model:
            tasks = promotion_tasks([self.b], [self.alignment(self.a)], self.config,
                                    deduplicator=self.matcher)
            self.assertEqual(tasks, [])
            alignments = unique_alignments([self.alignment(self.b), self.alignment(self.a)],
                                           deduplicator=self.matcher)
            self.assertEqual(sum(r['accepted'] for r in alignments), 1)
            kept, rejected = _deduplicate_final_records([self.b, self.a], .98,
                                                        deduplicator=self.matcher)
            self.assertEqual([r['region_id'] for r in kept], ['a'])
            self.assertEqual(rejected[0]['duplicate_of_region_id'], 'a')
            self.assertEqual(rejected[0]['target_dedup_audit']['method'], 'vlm')
            self.assertFalse(rejected[0]['accepted'])
            model.assert_called_once()

    def test_scope_category_image_and_disjoint_boxes_are_not_merged(self):
        variants = [{'scope': 'object_part'},
                    {'image_id': 'other'}, {'bbox_xyxy': [0, 0, 110, 50]}]
        with self.answer() as model:
            for changes in variants:
                self.assertFalse(self.matcher.compare(self.a, {**self.b, **changes})['same_object'])
            model.assert_not_called()

    def test_cross_category_real_duplicate_regressions(self):
        cases = [
            ('branch', 'tree', [897, 1287, 2248, 1500], [861, 1286, 2248, 1500],
             'the leafy green branches in the foreground that are slightly out of focus',
             'the out-of-focus green leafy tree branches in the foreground obscuring the city buildings'),
            ('tower', 'ruin', [876, 532, 1049, 967], [875, 532, 1049, 967],
             'the tall rectangular stone tower with a crenellated top standing to the left of the ruined walls',
             'the tall, rectangular stone tower with a crenellated top, standing to the left of the lower ruined walls'),
        ]
        Image.new('RGB', (2248, 1500)).save(self.source)
        for cat_a, cat_b, box_a, box_b, text_a, text_b in cases:
            with self.subTest(categories=(cat_a, cat_b)), self.answer() as model:
                a = {**self.record(cat_a, box_a, 1), 'category': cat_a,
                     'category_query': cat_a, 'final_referring_expression': text_a}
                b = {**self.record(cat_b, box_b, 2), 'category': cat_b,
                     'category_query': cat_b, 'final_referring_expression': text_b}
                kept, rejected = _deduplicate_final_records([a, b], .98,
                                                            deduplicator=self.matcher)
                self.assertEqual(len(kept), 1)
                self.assertEqual(rejected[0]['target_dedup_audit']['method'], 'vlm')
                model.assert_called_once()
                payload = json.loads(model.call_args.kwargs['extra_text'])
                self.assertEqual({t['expression'] for t in payload['targets']}, {text_a, text_b})
                alignment = {**a, 'selected_candidate': a,
                             'entity': {'category_query': cat_a, 'scope': 'whole_object'}}
                self.assertEqual(promotion_tasks([b], [alignment], self.config,
                                                 deduplicator=self.matcher), [])

    def test_cross_category_identical_boxes_require_visual_evidence(self):
        for decision in ['different_objects', 'uncertain']:
            with self.subTest(decision=decision), self.answer(decision) as model:
                b = {**self.b, 'category_query': 'truck', 'bbox_xyxy': VAN_A}
                audit = self.new_matcher(model=decision).compare(self.a, b)
                self.assertFalse(audit['same_object'])
                self.assertEqual(audit['method'], 'vlm')
                model.assert_called_once()

    def test_expression_change_invalidates_identity_cache(self):
        with self.answer() as model:
            self.matcher.compare(self.a, self.b)
            self.matcher.compare(self.a, {**self.b, 'final_referring_expression': 'a revised description'})
            self.assertEqual(model.call_count, 2)

    def test_identical_expression_deduplicates_disjoint_boxes_without_model(self):
        b = {**self.b, 'bbox_xyxy': [0, 0, 110, 50],
             'final_referring_expression': self.a['final_referring_expression']}
        b['final_referring_expression'] = '  ' + b['final_referring_expression'].upper() + '  '
        with self.answer() as model:
            kept, rejected = _deduplicate_final_records([self.a, b], .98, deduplicator=self.matcher)
            model.assert_not_called()
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected[0]['reject_reason'], 'duplicate_referring_expression')
        b['image_id'] = 'another_image'
        self.assertFalse(self.matcher.compare(self.a, b)['same_object'])

    def test_exact_text_takes_priority_over_other_geometric_matches(self):
        b = {**self.b, 'bbox_xyxy': [0, 0, 110, 50]}
        c = {**self.record('c', VAN_B, 3),
             'final_referring_expression': b['final_referring_expression']}
        with self.answer() as model:
            kept, rejected = _deduplicate_final_records([self.a, b, c], .98,
                                                        deduplicator=self.matcher)
            model.assert_not_called()
        self.assertEqual(len(kept), 2)
        self.assertEqual(rejected[0]['duplicate_of_region_id'], 'b')
        self.assertEqual(rejected[0]['reject_reason'], 'duplicate_referring_expression')

    def test_near_identical_geometry_needs_no_model(self):
        with self.answer() as model:
            self.assertTrue(self.matcher.compare(self.a, {**self.b, 'bbox_xyxy': VAN_A})['same_object'])
            model.assert_not_called()

    def test_part_within_whole_fails_area_filter(self):
        self.matcher.suspect_iou = .4
        b = {**self.b, 'bbox_xyxy': [1110, 1192, 1165, 1244]}
        with self.answer() as model:
            self.assertFalse(self.matcher.compare(self.a, b)['same_object'])
            model.assert_not_called()

    def test_read_only_probe_pending_then_cached(self):
        token = CHECK_ONLY.set(True)
        try:
            with self.assertRaises(StagePending), self.answer() as model:
                self.matcher.compare(self.a, self.b)
            model.assert_not_called()
            self.assertFalse((self.root / 'route_b').exists())
        finally:
            CHECK_ONLY.reset(token)
        with self.answer():
            self.matcher.compare(self.a, self.b)
        token = CHECK_ONLY.set(True)
        try:
            self.assertTrue(self.new_matcher().compare(self.a, self.b)['same_object'])
        finally:
            CHECK_ONLY.reset(token)

    def test_model_and_box_changes_invalidate_cache(self):
        with self.answer() as model:
            self.matcher.compare(self.a, self.b)
            self.new_matcher(model='changed').compare(self.a, self.b)
            changed = {**self.b, 'bbox_xyxy': [1112, *VAN_B[1:]]}
            self.matcher.compare(self.a, changed)
            self.assertEqual(model.call_count, 3)

    def test_model_failure_does_not_cache_or_silently_merge(self):
        with patch('src.routes.target_dedup.Qwen38Client.generate_json',
                   side_effect=QwenValidationError('invalid')), self.assertRaises(QwenValidationError):
            self.matcher.compare(self.a, self.b)
        self.assertEqual(list(self.root.glob('route_b/target_dedup/decisions/*')), [])

    def test_no_transitive_merge_through_rejected_bridge(self):
        c = self.record('c', [1100, 1200, 1220, 1250], 3)
        class PairMatcher:
            def compare(self, a, b):
                return {'same_object': {a['region_id'], b['region_id']} in [{'a', 'b'}, {'b', 'c'}],
                        'iou': .8}
        kept, rejected = _deduplicate_final_records([c, self.b, self.a], .98,
                                                    deduplicator=PairMatcher())
        self.assertEqual([r['region_id'] for r in kept], ['a', 'c'])
        self.assertEqual(len(rejected), 1)


if __name__ == '__main__':
    unittest.main()
