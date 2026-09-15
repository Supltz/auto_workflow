"""EGM locator adapter. Original image + phrase only; no target-answer inputs."""
from __future__ import annotations
import json
import math
import re
from typing import Any
from src.grounding.base import PhraseGrounder, run_phrase_worker, worker_parser
from src.utils.config import load_yaml, resolve_path
from src.utils.provenance import model_metadata


def parse_box(text, width, height):
    # Never parse a preliminary box in the thinking segment as the final answer.
    answer=text.rsplit('</think>',1)[-1]
    if '<think>' in answer: return None
    decoder=json.JSONDecoder(); boxes=[]
    def walk(value):
        if isinstance(value,dict):
            if 'bbox_2d' in value: boxes.append(value['bbox_2d'])
            elif 'box' in value: boxes.append(value['box'])
            else:
                for v in value.values(): walk(v)
        elif isinstance(value,list):
            if len(value)==4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in value): boxes.append(value)
            else:
                for v in value: walk(v)
    pos=0
    while pos<len(answer):
        start=re.search(r'[\[{]',answer[pos:])
        if not start: break
        pos+=start.start()
        try:
            value,end=decoder.raw_decode(answer[pos:]); walk(value); pos+=end
        except ValueError: pos+=1
    unique=[]
    for box in boxes:
        if not isinstance(box,list) or len(box)!=4: return None
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box): return None
        if not 0<=box[0]<box[2]<=1000 or not 0<=box[1]<box[3]<=1000: return None
        if box not in unique: unique.append(box)
    if len(unique)!=1: return None
    x1,y1,x2,y2=unique[0]
    return [x1*width/1000,y1*height/1000,x2*width/1000,y2*height/1000]


class EGMGrounder(PhraseGrounder):
    name='egm'
    def __init__(self,config:dict[str,Any]):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.config=config
        path=str(resolve_path(config['local_path']))
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,
                        max_pixels=int(config.get('max_pixels',1048576)))
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
                        torch_dtype=torch.bfloat16,attn_implementation=config.get('attn_implementation','sdpa')).eval().to('cuda')
        self.provenance=model_metadata(config,config,'egm-locator-1000-v1')

    def ground(self,image_path,phrase):
        import torch
        from src.utils.images import open_rgb
        image=open_rgb(image_path)
        try:
            prompt=self.config.get('prompt','Locate {phrase}, output its bbox coordinates using JSON format').replace('{phrase}',phrase)
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
            text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[text],images=[image],return_tensors='pt').to(self.model.device)
            with torch.inference_mode():
                output=self.model.generate(**inputs,max_new_tokens=int(self.config.get('max_output_tokens',4096)),
                                           do_sample=False)
            generated=output[0,inputs['input_ids'].shape[-1]:]
            raw=self.processor.decode(generated,skip_special_tokens=True)
            box=parse_box(raw,*image.size)
            # Empty/invalid answer is recorded by worker progress, not a fake box.
            if box is None: return [{'bbox_xyxy':None,'score':None,'metadata':{'raw_answer':raw,'invalid_locator_output':True}}]
            return [{'bbox_xyxy':box,'score':None,'metadata':{'raw_answer':raw,'coordinate_system':'normalized_1000'}}]
        finally: image.close()



def main():
    args=worker_parser(__doc__).parse_args()
    from src.grounding.resident import adapter
    run_phrase_worker(adapter('egm',load_yaml(args.model_config)['egm'],EGMGrounder),args)

if __name__=='__main__': main()
