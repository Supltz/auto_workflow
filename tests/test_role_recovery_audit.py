"""Fault injection for resume identity, compact delivery and the discovery drain."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_role_grounding as fixtures
from test_role_grounding import Fake
from src.routes import role_pipeline, role_outputs
from src.routes.role_engine import Engine, initial_state
from src.routes.role_schedule import CONTRACT, stage_plan
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, rewrite_jsonl_atomic


class RecoveryAudit(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.RoleTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.row=self.fixture.row;self.config=self.fixture.config
        self.models=load_yaml('configs/models.yaml')

    def pipeline(self,records=None,end=1,check_only=False):
        manifest=self.root/'manifest.jsonl';rewrite_jsonl_atomic(manifest,[self.row])
        output=self.root/'output'
        if records is not None:rewrite_jsonl_atomic(output/'route_b/role_state.jsonl',records)
        args=SimpleNamespace(output_dir=str(output),start_index=0,end_index=end,
                             stage='category_inventory',check_only=check_only,stage_step=None)
        role_pipeline.run(args,{**self.config,'manifest':str(manifest)},lambda *_:Fake(1))
        return output

    def test_semantic_inputs_invalidate_but_transport_and_batch_paths_do_not(self):
        original=role_pipeline.signature(self.config,self.models,self.row)
        for field,value in [('image_id','another'),('caption','new reference'),('width',999),('height',999)]:
            with self.subTest(field=field):
                self.assertNotEqual(original,role_pipeline.signature(self.config,self.models,{**self.row,field:value}))
        models=copy.deepcopy(self.models);models['qwen']['api_base']='http://another-node/v1'
        config={**self.config,'manifest':'/regrouped/manifest','role_workers':16}
        self.assertEqual(original,role_pipeline.signature(config,models,self.row))

    def test_grounding_adapter_is_part_of_resume_identity(self):
        code=role_pipeline.implementation_fingerprint()
        self.assertIn('src/grounding/egm.py',code)
        original=role_pipeline.signature(self.config,self.models,self.row,code)
        code['src/grounding/egm.py']='changed'
        self.assertNotEqual(original,role_pipeline.signature(self.config,self.models,self.row,code))

    def test_explicit_empty_slice_does_not_launch_default_sample(self):
        with self.assertRaisesRegex(ValueError,'Empty'):self.pipeline(end=0)
        self.assertFalse((self.root/'output/route_b/role_state.jsonl').exists())

    def test_corrupt_checkpoint_fails_before_modifying_it(self):
        state=initial_state(self.row,role_pipeline.signature(self.config,self.models,self.row))
        cases=[([state,state],'Duplicate'),([{**state,'image_id':'foreign'}],'unselected'),
               ([{**state,'contract':'legacy'}],'contract'),([{**state,'step':-1}],'boundary'),
               ([{**state,'step':10000}],'boundary')]
        for records,message in cases:
            with self.subTest(message=message):
                path=self.root/'output/route_b/role_state.jsonl';rewrite_jsonl_atomic(path,records)
                before=path.read_bytes()
                with self.assertRaisesRegex(ValueError,message):self.pipeline(check_only=True)
                self.assertEqual(before,path.read_bytes())

    def test_invalid_search_budget_fails_before_any_model_call(self):
        for field,value in [('category_pages',0),('category_locator_limit',-1),
                            ('local_search_limit',-1),('role_workers',0)]:
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                role_pipeline.validate_config({**self.config,field:value})

    def exported(self):
        self.fixture.run_engine(Fake(2,True));state=self.fixture.state;root=self.root/'delivery'
        role_outputs.export([state],root)
        rewrite_jsonl_atomic(root/'route_b/role_state.jsonl',[state])
        return root,state

    def test_modified_phrase_and_lost_audit_cannot_pass_completion(self):
        root,state=self.exported()
        for name in ('verified_regions/route_b.jsonl','unresolved_regions/route_b.jsonl','search_audit/route_b.jsonl'):
            with self.subTest(file=name):
                role_outputs.export([state],root)
                rows=list(read_jsonl(root/name))
                if name.startswith('verified'):rows[0]['final_referring_expression']='the wrong target'
                elif name.startswith('unresolved'):rows[0]['reason']='fabricated adjudication'
                else:rows=[]
                rewrite_jsonl_atomic(root/name,rows)
                with self.assertRaisesRegex(ValueError,'confirmed state'):role_outputs.validate(root,{'image'},False)

    def test_post_cleanup_counts_detect_joint_object_and_phrase_loss(self):
        root,state=self.exported();(root/'route_b/role_state.jsonl').unlink()
        rows=list(read_jsonl(root/'objects/route_b.jsonl'))
        rewrite_jsonl_atomic(root/'objects/route_b.jsonl',[r for r in rows if r['phrase_status']=='verified'])
        rewrite_jsonl_atomic(root/'unresolved_regions/route_b.jsonl',[])
        with self.assertRaisesRegex(ValueError,'count mismatch'):role_outputs.validate(root,{'image'},False)

    def test_last_discovery_objects_get_a_phrase_attempt(self):
        state=initial_state(self.row,'sig');config={**self.config,'max_refinement_rounds':0}
        engine=Engine(state,config,Fake(1),lambda:None)
        for _,name,action,_ in stage_plan(config):
            engine.stage(action)
            if name=='phrase_reground':
                engine.add_detections(state['categories'][0],{'predictions':[{'bbox_xyxy':[200,10,210,40]}]},'egm')
        self.assertEqual(len(state['objects']),2)
        self.assertTrue(all(o['phrase'] and o['phrase']['status']=='verified' for o in state['objects']))

    def test_final_reground_does_not_start_another_discovery_chain(self):
        engine,fake=self.fixture.run_engine(Fake(1));state=self.fixture.state
        obj=state['objects'][0];obj['phrase']['reground']=None
        before=len(state['candidates'])
        with patch.object(engine,'ground',return_value={'predictions':[{'bbox_xyxy':[200,10,210,40]}]}):
            engine.stage('reground_final')
        self.assertEqual(before,len(state['candidates']))
        self.assertFalse(obj['phrase']['reground']['passed'])
        self.assertEqual(obj['phrase']['reground']['predictions'][0]['bbox_xyxy'],[200,10,210,40])

    def test_context_overflow_is_not_cached_as_semantic_uncertainty(self):
        from src.routes.role_backend import Backend
        from src.routes.role_contract import PhraseDecision
        from src.models.qwen38_client import QwenContextLengthError
        backend=Backend(self.config,self.models)
        state=initial_state(self.row,'sig');engine=Engine(state,self.config,backend,lambda:None)
        with patch.object(backend.client,'generate_json',side_effect=QwenContextLengthError('too many tokens')):
            with self.assertRaises(QwenContextLengthError):
                engine.call('adjudicate',{},PhraseDecision,[[10,10,20,40]])
        self.assertEqual(state['calls'],{})

    def test_rewrite_does_not_resubmit_locator_thinking_and_full_audit_history(self):
        engine,fake=self.fixture.run_engine(Fake(1))
        phrase=self.fixture.state['objects'][0]['phrase'];phrase['status']='needs_rewrite'
        phrase['reground']['predictions'][0]['metadata']={'raw_answer':'long model thinking '*10000}
        engine.describe(rewrite=True)
        card=next(card for name,card in reversed(fake.asked) if name=='phrase')
        self.assertNotIn('long model thinking',json.dumps(card))
        self.assertEqual(card['previous']['text'],phrase['text'])


if __name__=='__main__':unittest.main()
