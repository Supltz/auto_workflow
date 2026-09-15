"""Model IO for role grounding; task process remains checkpoint owner."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from PIL import ImageDraw
from src.models.qwen38_client import Qwen38Client, QwenValidationError, QwenContextLengthError
from src.routes.common import invoke_grounders
from src.routes.role_contract import key
from src.utils.config import resolve_path
from src.utils.geometry import clip_box
from src.utils.images import open_rgb
from src.utils.io import read_jsonl


class Backend:
    def __init__(self,config,models):
        self.config,self.models=config,models
        self.client=Qwen38Client({**models['qwen'],'max_output_tokens':int(config.get('role_output_tokens',8192))})

    def prompt(self,name):
        return resolve_path('prompts/role_'+name+'.txt').read_text()

    def ask(self,name,state,card,schema,boxes=()):
        original=open_rgb(state['source_image']); images=[original]
        card={**card,'original_image_size_px':list(original.size),
              'image_order':['image 1: original full image, without annotations']}
        try:
            if boxes:
                overlay=original.copy(); draw=ImageDraw.Draw(overlay)
                for index,box in enumerate(boxes):
                    draw.rectangle(box,outline=('red' if index==0 else 'blue'),width=max(3,min(original.size)//300))
                    draw.text((box[0],box[1]),'A' if index==0 else 'B',fill='red' if index==0 else 'blue')
                images.append(overlay)
                a=boxes[0]; images.append(original.crop(a))
                dx=(a[2]-a[0])*.25;dy=(a[3]-a[1])*.25
                images.append(original.crop(clip_box([a[0]-dx,a[1]-dy,a[2]+dx,a[3]+dy],*original.size)))
                if len(boxes)>1:images.append(original.crop(boxes[1]))
                card['image_order'] += ['image 2: full-image overlay, A=red, B=blue if present',
                    'image 3: EXACT pixels inside A; its edges are the proposed box boundaries',
                    'image 4: surrounding context for A; not the proposed box']
                if len(boxes)>1:card['image_order'].append('image 5: EXACT pixels inside B')
                card['boxes_xyxy_original_px']={label:list(box) for label,box in zip(('A','B'),boxes)}
                card['boxes_xyxy_normalized_1000']={label:[box[0]*1000/original.width,box[1]*1000/original.height,
                    box[2]*1000/original.width,box[3]*1000/original.height] for label,box in zip(('A','B'),boxes)}
            try:
                result,raw=self.client.generate_json(self.prompt(name),images,schema,
                    extra_text=json.dumps(card,ensure_ascii=False)+'\nRequired schema: '+json.dumps(schema.model_json_schema()))
                return result.model_dump(mode='json'),raw
            except QwenContextLengthError:
                # Input sizing is an execution/configuration fault, not an
                # uncertain visual judgment. Do not cache it as phrase evidence.
                raise
            except QwenValidationError as exc:
                # Exhausted malformed semantic answers do not delete confirmed
                # objects. Transport/service exceptions propagate for recovery.
                reason='model_output_invalid: '+str(exc)[:1000]
                if name=='phrase':value=dict(expression='',visible_evidence=[],reason=reason)
                elif name in ('verify','adjudicate'):
                    value=dict(outcome='uncertain',describes_A=False,facts_visible=False,
                        single_whole_target=False,grammatical=False,unique_in_full_image=False,
                        also_matches_B=False,competing_object_ids=[],discriminator=None,evidence=[],reason=reason)
                elif name=='ocr':value=dict(verified_target_text=[])
                else:raise
                return schema.model_validate(value).model_dump(mode='json'),reason
        finally:
            for image in images:image.close()

    def ground(self,model,state,query,region=None):
        if model not in ('sam31','egm'):raise ValueError('Role grounding only allows SAM/EGM')
        if model=='egm' and region is not None:raise ValueError('EGM must use the original full image')
        with tempfile.TemporaryDirectory(prefix='role-model-') as temporary:
            root=Path(temporary); source=state['source_image']; width=state['image_width'];height=state['image_height']
            if region is not None:
                with open_rgb(source) as image:
                    crop=image.crop(region);width,height=crop.size
                    source=str(root/'crop.png');crop.save(source);crop.close()
            request={'request_id':'role:'+key([state['image_id'],model,query,region]),
                     'image_id':state['image_id'],'image_path':source,'phrase':query,'rank':1,
                     'route':'B_caption','image_width':width,'image_height':height,
                     'metadata':{'query_type':'category' if model=='sam31' else 'locator',
                                 'role_contract':'role-grounding-v1'}}
            out=root/'raw.jsonl'
            invoke_grounders([request],[model],self.config['models_config'],self.models,out,root,
                root/'failures.jsonl',False,True,save_crops=False,save_overlays=False,
                save_masks=model=='sam31',worker_environment={})
            predictions=[]
            for record in read_jsonl(out):
                box=record['bbox_xyxy']; method='detector_box'
                mask=record.get('mask_path')
                if model=='sam31' and mask:
                    from PIL import Image
                    with Image.open(mask) as image:
                        if image.size!=(width,height):raise ValueError('SAM mask coordinate size mismatch')
                        tight=image.getbbox()
                    if tight:box=list(tight);method='sam31_mask_tight'
                predictions.append({'bbox_xyxy':box,'grounder':model,'score':record.get('grounding_score'),
                                    'box_method':method,'metadata':record.get('metadata',{})})
            progress=root/('raw_'+model+'.progress.jsonl')
            for record in read_jsonl(progress):
                for invalid in record.get('invalid_outputs',[]):
                    predictions.append({'bbox_xyxy':None,'grounder':model,'metadata':invalid})
            # No transient file paths may enter state/final results.
            for item in predictions:
                item['metadata']={k:v for k,v in item['metadata'].items()
                                  if k in ('raw_answer','invalid_locator_output','coordinate_system','sam_version','object_id')}
            return {'predictions':predictions,'view_size':[width,height]}
