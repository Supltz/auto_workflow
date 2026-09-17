"""CPU-only adversarial policy and transport tests; no model/GPU required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from test_role_grounding import Fake, phrase_decision
from src.routes.role_engine import Engine, initial_state
from src.routes.role_backend import Backend
from src.routes.role_contract import BlindDecision, PhraseDecision, phrase_verdict
from src.routes.role_context import context_views, competitors, expand, blind_gate
from src.routes.role_schedule import stage_plan
from src.routes.role_pipeline import signature, validate_config, run
from src.routes import role_outputs
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, rewrite_jsonl_atomic


def match(x):
    return dict(bbox_normalized=[x,10,x+10,40],referent='pole',evidence='visible original-image position')


def observation(outcome='unique'):
    return dict(outcome=outcome,matches=[match(10)] if outcome=='unique' else
                [match(10),match(35)] if outcome=='multiple' else [],requested_regions=[],
                comparison_scope='all visible poles',reason='independent search')


class Controlled(Fake):
    def __init__(self,blind='unique',change=None,n=2,disagreement=False):
        super().__init__(n,disagreement);self.blind_outcome=blind;self.change=change
    def ask(self,name,state,card,schema,boxes):
        v,raw=super().ask(name,state,card,schema,boxes)
        if name=='blind' and self.blind_outcome!='unique':v=observation(self.blind_outcome)
        if name in ('verify','adjudicate') and self.change:self.change(v)
        return schema.model_validate(v).model_dump(),raw


class PhraseContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'source.png';Image.new('RGB',(1000,1000),(23,42,61)).save(self.source)
        self.row=dict(image_id='image',image_path=str(self.source),width=1000,height=1000)
        self.config=load_yaml('configs/route_b.yaml');self.state=initial_state(self.row,'sig')
    def execute(self,fake):
        engine=Engine(self.state,self.config,fake,lambda:None)
        for _,_,action,_ in stage_plan(self.config):engine.stage(action)
        return engine
    def test_tiny_and_edge_context_have_minimum_scene_extent(self):
        for box in ([1,1,5,8],[990,990,999,999]):
            r=expand(box,1000,1000,3,.15)
            self.assertGreaterEqual(r[2]-r[0],150);self.assertGreaterEqual(r[3]-r[1],150)
            self.assertTrue(0<=r[0]<r[2]<=1000 and 0<=r[1]<r[3]<=1000)
    def test_group_view_precedes_local_and_contains_anchor(self):
        card=dict(context_regions=[dict(region=[100,100,800,800],reason='whole group and anchor')],
                  peers=[dict(bbox=[300,300,330,330])])
        views=context_views(self.state,card,[[400,400,410,410],[600,600,615,615]],self.config)
        self.assertLessEqual(len(views),2)
        self.assertLessEqual(views[0][0],100);self.assertGreaterEqual(views[0][2],800)
    def test_competitors_include_synonyms_containment_and_all_category_peers(self):
        self.execute(Fake(35));obj=self.state['objects'][0]
        self.state['objects'][1].update(query='striped post',category_ids=['different'])
        rows=competitors(self.state,obj,self.config)
        self.assertEqual(len(rows),34)
        self.assertIn(self.state['objects'][1]['id'],[r['id'] for r in rows])
    def test_blind_multiple_cannot_be_overridden_even_when_egm_succeeds(self):
        self.execute(Controlled('multiple'))
        self.assertEqual(len(self.state['objects']),2)
        self.assertTrue(all(o['phrase']['status']=='unresolved' for o in self.state['objects']))
        self.assertTrue(all(o['phrase']['reground']['passed'] for o in self.state['objects']))
    def test_blind_missing_or_uncertain_never_passes(self):
        for outcome in ('none','uncertain'):
            with self.subTest(outcome=outcome):
                self.state=initial_state(self.row,'sig');self.execute(Controlled(outcome))
                self.assertTrue(all(o['phrase']['status']=='unresolved' for o in self.state['objects']))
    def test_wrong_blind_referent_cannot_be_certified(self):
        self.execute(Controlled(change=lambda d:d.update(blind_target_relation='different_object')))
        self.assertTrue(all(o['phrase']['status']=='unresolved' for o in self.state['objects']))
    def test_missing_peer_comparison_fails_closed(self):
        self.execute(Controlled(change=lambda d:d.update(comparisons=[])))
        self.assertTrue(all(o['phrase']['status']=='unresolved' for o in self.state['objects']))
    def test_positive_or_unknown_competitor_blocks_acceptance(self):
        for value in (True,None):
            with self.subTest(value=value):
                self.state=initial_state(self.row,'sig')
                self.execute(Controlled(change=lambda d:d['comparisons'][0].update(also_matches=value)))
                self.assertTrue(all(o['phrase']['status']=='unresolved' for o in self.state['objects']))
    def test_granularity_scope_and_false_attribute_block_yes_booleans(self):
        for issue in ('granularity','scope','false_attribute'):
            d=phrase_decision();d['issue_type']=issue
            self.assertEqual(phrase_verdict(d),'needs_rewrite')
    def test_egm_failure_can_pass_with_independent_evidence(self):
        self.execute(Controlled(n=3,disagreement=True))
        accepted=[o for o in self.state['objects'] if o['phrase']['status']=='verified']
        self.assertEqual(len(accepted),2)
        self.assertTrue(all(o['phrase']['pass_via']=='semantic_adjudication' for o in accepted))
    def test_blind_card_and_pixels_exclude_target(self):
        backend=Backend.__new__(Backend);backend.config=self.config
        seen=[]
        def generate(prompt,images,schema,extra_text):
            seen.append((json.loads(extra_text.split('\nRequired schema:')[0]),[(i.size,i.getpixel((0,0))) for i in images]))
            return schema.model_validate(observation()),'raw'
        backend.client=SimpleNamespace(generate_json=generate)
        backend.ask('blind',self.state,dict(phrase='the leftmost pole'),BlindDecision)
        self.assertEqual(len(seen[0][1]),1)
        self.assertEqual(seen[0][1][0],((1000,1000),(23,42,61)))
        self.assertNotIn('boxes_xyxy_original_px',seen[0][0])
        for card,boxes in ((dict(phrase='the pole',target_A='secret'),()),(dict(phrase='the pole'),[[1,2,3,4]])):
            with self.assertRaisesRegex(ValueError,'Blind review'):
                backend.ask('blind',self.state,card,BlindDecision,boxes)
        obs=observation('uncertain');obs['requested_regions']=[dict(region=[100,100,500,500],reason='visible group')]
        backend.ask('blind',self.state,dict(phrase='the leftmost pole',observation=obs),BlindDecision)
        self.assertEqual(seen[-1][1],[((1000,1000),(23,42,61)),((400,400),(23,42,61))])
    def test_target_aware_image_budget_remains_six(self):
        backend=Backend.__new__(Backend);backend.config=self.config
        seen=[]
        def generate(prompt,images,schema,extra_text):
            seen.append(len(images));return schema.model_validate(phrase_decision()),'raw'
        backend.client=SimpleNamespace(generate_json=generate)
        backend.ask('adjudicate',self.state,dict(context_regions=[dict(region=[10,10,300,300],reason='group'),dict(region=[500,500,900,900],reason='anchor')]),PhraseDecision,[[20,20,25,25],[600,600,610,610]])
        self.assertEqual(seen,[6])
    def test_group_crop_requests_do_not_certify_unseen_scope(self):
        v=observation();v['requested_regions']=[dict(region=[0,0,500,500],reason='need detail')]
        self.assertEqual(blind_gate(v),'unresolved')
    def test_archive_requires_cue_and_audit_and_v2(self):
        self.execute(Fake(2));root=self.root/'outputs';role_outputs.export([self.state],root)
        role_outputs.validate(root,{'image'},False,2)
        path=root/'verified_regions/route_b.jsonl';original=list(read_jsonl(path))
        for key in ('phrase_audit','locator_cue'):
            rows=copy.deepcopy(original);rows[0].pop(key);rewrite_jsonl_atomic(path,rows)
            with self.assertRaises((ValueError,KeyError)):role_outputs.validate(root,{'image'},False,2)
        rewrite_jsonl_atomic(path,original)
        path=root/'verified_regions/route_b_counts_by_image.jsonl';rows=list(read_jsonl(path));rows[0].pop('phrase_review_version');rewrite_jsonl_atomic(path,rows)
        with self.assertRaises(ValueError):role_outputs.validate(root,{'image'},False,2)
    def test_new_schedule_and_context_invalidate_old_policy(self):
        self.assertEqual(len(stage_plan(self.config)),29)
        self.assertEqual(len(stage_plan({'max_refinement_rounds':2})),25)
        models=load_yaml('configs/models.yaml');a=signature(self.config,models,self.row)
        changed=copy.deepcopy(self.config);changed['phrase_context']['min_image_fraction']=.2
        self.assertNotEqual(a,signature(changed,models,self.row))
        changed.pop('phrase_review_version')
        with self.assertRaisesRegex(ValueError,'phrase_review_version'):validate_config(changed)
    def test_old_checkpoint_rejected_before_any_write(self):
        manifest=self.root/'manifest.jsonl';rewrite_jsonl_atomic(manifest,[self.row])
        root=self.root/'outputs';path=root/'route_b/role_state.jsonl'
        state=copy.deepcopy(self.state);state.pop('phrase_review_version');rewrite_jsonl_atomic(path,[state]);before=path.read_bytes()
        args=SimpleNamespace(output_dir=str(root),start_index=0,end_index=1,stage='all',check_only=False,stage_step=None)
        with self.assertRaisesRegex(ValueError,'checkpoint contract'):
            run(args,{**self.config,'manifest':str(manifest)},lambda *_:Fake())
        self.assertEqual(before,path.read_bytes())

if __name__=='__main__':unittest.main()
