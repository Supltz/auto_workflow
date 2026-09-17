import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from src.grounding.egm import parse_box
from src.routes.role_engine import Engine,initial_state
from src.routes.role_contract import reserve_search,add_hint,crop_to_original,geometry_audit
from src.routes.role_schedule import stage_plan,phases,STORAGE_CONTRACT
from src.routes import role_outputs,role_pipeline
from src.utils.config import load_yaml
from src.utils.io import read_jsonl
from src.routes.route_b_checkpoint import StagePending

ROOT=Path(__file__).resolve().parents[1]


def object_decision():
    return dict(referent_name='pole',referent_kind='physical_object',boundary_evidence=dict(top='top of pole',bottom='base of pole',left='left edge of pole',right='right edge of pole'),
                box_issues=[],target_identifiable=True,category_correct=True,whole_object=True,
                severe_occlusion_or_truncation=False,tight_and_complete=True,independent_object=True,
                target_attributes=['red'],target_actions=[],supported_relations=[],uncertain_attributes=[],reason='visible')


def phrase_decision(outcome='supports_A'):
    return dict(outcome=outcome,target_kind_matches=True,locator_cue_valid=True,reference_scope_clear=True,
                blind_target_relation='same_object',comparisons=[],issue_type='none',describes_A=True,facts_visible=True,single_whole_target=True,
                grammatical=True,unique_in_full_image=outcome=='supports_A',also_matches_B=False,
                competing_object_ids=[],discriminator='position',evidence=['full image position'],reason=outcome)


class Fake:
    def __init__(self,n=35,disagreement=False):
        self.n=n;self.disagreement=disagreement;self.count=0;self.asked=[];self.grounded=[]
    def prompt(self,name):return name
    def ask(self,name,state,card,schema,boxes):
        self.count+=1;self.asked.append((name,copy.deepcopy(card)))
        if name=='scene':v=dict(categories=[dict(query='pole',visible_evidence='visible poles',locators=[],search_hints=[])],more_categories=False)
        elif name=='object':v=object_decision()
        elif name=='identity':v=dict(same_object=False,certain=True,preferred='A',reason='distinct')
        elif name=='ocr':v=dict(verified_target_text=[])
        elif name=='context':v=dict(regions=[],comparison_scope='all poles in original image',reason='visible group')
        elif name=='blind':
            x=int(card['phrase'].rsplit(' ',1)[1])
            v=dict(outcome='unique',matches=[dict(bbox_normalized=[x,10,x+10,40],referent='pole',evidence='full-image position')],
                   requested_regions=[],comparison_scope='full image',reason='one match')
        elif name=='phrase':v=dict(expression='the pole at '+str(int(boxes[0][0])),locator_cue='at '+str(int(boxes[0][0])),cue_type='spatial',visible_evidence=['position in original'],reason='unique position')
        else:
            outcome=('uncertain' if int(boxes[0][0])==10 else 'supports_A') if self.disagreement else 'supports_A'
            v=phrase_decision(outcome)
            v['comparisons']=[dict(object_id=p['id'],also_matches=False,reason='different position') for p in card.get('peers',[])]
        return schema.model_validate(v).model_dump(), 'raw answer deliberately not archived'
    def ground(self,model,state,query,region=None):
        self.grounded.append((model,query,region))
        if model=='sam31':boxes=[[10+i*25,10,20+i*25,40] for i in range(self.n)] if region is None else []
        elif self.disagreement:boxes=[]
        else:
            x=int(query.rsplit(' ',1)[1]);boxes=[[x,10,x+10,40]]
        return dict(predictions=[dict(bbox_xyxy=b,box_method='sam31_mask_tight' if model=='sam31' else 'detector_box') for b in boxes],view_size=[1000,1000])


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'source.png';Image.new('RGB',(1000,1000)).save(self.source)
        self.row=dict(image_id='image',image_path=str(self.source),width=1000,height=1000)
        self.config=load_yaml('configs/route_b.yaml');self.state=initial_state(self.row,'sig')
    def run_engine(self,fake=None):
        fake=fake or Fake();engine=Engine(self.state,self.config,fake,lambda:None)
        for _,_,action,_ in stage_plan(self.config):engine.stage(action)
        return engine,fake
    def test_no_top30_short_boxes_and_short_phrases(self):
        self.run_engine();root=self.root/'outputs';role_outputs.export([self.state],root);role_outputs.review(root)
        counts=role_outputs.validate(root,{'image'})
        self.assertEqual(counts['objects'],35);self.assertEqual(counts['verified'],35)
        self.assertTrue(all(o['bbox_xyxy'][2]-o['bbox_xyxy'][0]==10 for o in self.state['objects']))
        self.assertNotIn('raw answer', ''.join(p.read_text() for p in root.rglob('*.jsonl')))
    def test_adjudication_and_unresolved_object_survive(self):
        self.run_engine(Fake(3,True));root=self.root/'outputs';role_outputs.export([self.state],root);role_outputs.review(root)
        self.assertEqual(role_outputs.validate(root,{'image'})['unresolved'],1)
        verified=list(read_jsonl(root/'verified_regions/route_b.jsonl'))
        self.assertTrue(all(v['pass_via']=='semantic_adjudication' and not v['reground_passed'] for v in verified))
        self.assertEqual(len(list((root/'human_review/route_b').glob('*.jpg'))),2)
        (root/'objects/route_b.jsonl').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):role_outputs.validate(root,{'image'})
    def test_near_equivalent_egm_then_sam_adopts_sam_box(self):
        engine,fake=self.run_engine(Fake(1));obj=self.state['objects'][0];obj['source']='egm'
        original_id=obj['id'];category=self.state['categories'][0]
        box=obj['bbox_xyxy'][:];box[0]+=.05
        engine.add_detections(category,dict(predictions=[dict(bbox_xyxy=box)]),'sam31');engine.verify_objects()
        self.assertEqual(obj['source'],'sam31');self.assertEqual(obj['id'],original_id)
        self.assertEqual(obj['box_version'],2)

    def test_retry_cached_evidence_and_no_new_budget(self):
        engine,fake=self.run_engine(Fake(3));calls=fake.count;grounded=len(fake.grounded)
        restored=json.loads(json.dumps(self.state));retry=Engine(restored,self.config,fake,lambda:None)
        for _,_,action,_ in stage_plan(self.config):retry.stage(action)
        self.assertEqual(len(restored['objects']),3);self.assertEqual(fake.count,calls);self.assertEqual(len(fake.grounded),grounded)
    def test_fair_slots_zero_detection_and_retry(self):
        categories=[dict(id=str(i),hints=[],local_searches=0) for i in range(2)]
        for c in categories:
            for j in range(4):add_hint(c,[10+j*100,0,50+j*100,40],'visible_region','visible')
        for c in categories:
            first=reserve_search(c,2);self.assertIs(first,reserve_search(c,2));self.assertEqual(c['local_searches'],1)
            first['status']='done';reserve_search(c,2)['status']='done';self.assertIsNone(reserve_search(c,2))
            self.assertEqual(c['local_searches'],2)
    def test_peer_discovery_reopens_uniqueness(self):
        engine,fake=self.run_engine(Fake(1));obj=self.state['objects'][0];old=obj['phrase']['verified_peers'];category=self.state['categories'][0]
        engine.add_detections(category,dict(predictions=[dict(bbox_xyxy=[200,10,210,40])]),'sam31');engine.verify_objects()
        engine.verify_phrases();self.assertNotEqual(old,obj['phrase']['verified_peers'])
        self.assertEqual(obj['id'],self.state['objects'][0]['id'])
    def test_box_change_reaudits_geometry(self):
        engine,_=self.run_engine(Fake(1));obj=self.state['objects'][0]
        obj['bbox_xyxy']=[20,10,30,40];obj['box_version']+=1;engine.verify_phrases()
        self.assertFalse(obj['phrase']['reground']['passed']);self.assertEqual(obj['phrase']['pass_via'],'semantic_adjudication')
    def test_ambiguous_rewrite_keeps_target(self):
        engine,fake=self.run_engine(Fake(1));obj=self.state['objects'][0];identity=obj['id'];box=obj['bbox_xyxy'][:]
        obj['phrase']['status']='needs_rewrite';engine.describe(rewrite=True)
        self.assertEqual(obj['id'],identity);self.assertEqual(obj['bbox_xyxy'],box);self.assertEqual(obj['phrase']['revision'],1)
    def test_phrase_collision_rewrites_before_finalization(self):
        engine,fake=self.run_engine(Fake(2));a,b=self.state['objects'];identity=[a['id'],b['id']]
        b['phrase']['text']=a['phrase']['text'];engine.verify_phrases()
        self.assertEqual([o['phrase']['status'] for o in (a,b)],['needs_rewrite','needs_rewrite'])
        engine.stage('rewrite');engine.reground();engine.blind_review();engine.verify_phrases()
        self.assertEqual([o['phrase']['status'] for o in (a,b)],['verified','verified'])
        self.assertEqual([o['id'] for o in (a,b)],identity)
        self.assertEqual([o['phrase']['revision'] for o in (a,b)],[1,1])
    def test_box_issue_cannot_be_overruled_by_yes_booleans(self):
        from src.routes.role_contract import ObjectDecision
        decision=object_decision();decision['box_issues']=['bottom is far below the actual base']
        self.assertFalse(ObjectDecision.model_validate(decision).accepted())
    def test_crop_mapping(self):
        self.assertEqual(crop_to_original([0,0,50,100],[100,200,200,400],[100,200]),[100,200,150,300])
    def test_egm_invalid_and_thinking(self):
        self.assertEqual(parse_box('<think>[0,0,10,10]</think>{"bbox_2d":[100,200,300,400]}',200,100),[20,20,60,40])
        for text in ('no box','[1,2,3,4] [20,30,40,50]','[-1,2,3,4]','[1,2,NaN,4]','<think>[1,2,3,4]'):
            self.assertIsNone(parse_box(text,100,100))
    def test_schedule_single_source_and_new_names(self):
        plan=stage_plan(self.config);names=[r[1] for r in plan]
        self.assertEqual(len(names),len(set(names)));self.assertNotIn('aggregate',names);self.assertNotIn('promote',names)
        self.assertEqual(names,[name for _,group in phases(self.config) for name in group]);self.assertEqual(names[-1],'review_export')
        self.assertEqual(len(stage_plan(dict(max_refinement_rounds=0,phrase_review_version=2))),17)
    def test_cli_missing_final_export_can_be_rebuilt(self):
        from types import SimpleNamespace
        fake=Fake(1);output=self.root/'pipeline';manifest=self.root/'manifest.jsonl';manifest.write_text(json.dumps(self.row)+'\n')
        config={**self.config,'manifest':str(manifest)}
        args=SimpleNamespace(output_dir=str(output),start_index=0,end_index=1,stage='all',check_only=False,stage_step=None)
        role_pipeline.run(args,config,lambda *_:fake)
        (output/'objects/route_b.jsonl').unlink();args.stage='finalize_objects';args.check_only=True
        with self.assertRaises(StagePending):role_pipeline.run(args,config,lambda *_:fake)
        args.check_only=False;role_pipeline.run(args,config,lambda *_:fake)
        self.assertEqual(role_outputs.validate(output,{'image'},require_review=False)['objects'],1)

if __name__=='__main__':unittest.main()
