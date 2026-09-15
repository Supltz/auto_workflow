"""Role stage execution with per-image durable checkpoints."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from src.routes.role_contract import CONTRACT,key
from src.routes.role_schedule import stage_plan
from src.routes.role_engine import Engine,initial_state
from src.routes.role_backend import Backend
from src.routes import role_outputs
from src.routes.common import sampled_manifest_rows
from src.routes.route_b_checkpoint import StagePending
from src.utils.config import load_yaml,resolve_path
from src.utils.io import read_jsonl,rewrite_jsonl_atomic


def implementation_fingerprint():
    root=Path(__file__).resolve().parents[2]
    # Adapter, image transport and geometry changes can change the evidence even
    # when the state-machine code is unchanged. Hash once per invocation.
    paths=[*root.joinpath('src').rglob('*.py'),*root.joinpath('prompts').glob('role_*.txt')]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def signature(config,models,row,code=None):
    # Logical batch paths and concurrency vary when images regroup across nodes.
    config={k:config.get(k) for k in ('grounding_contract','grounders','max_refinement_rounds',
        'category_pages','category_locator_limit','local_search_limit','local_padding','aggregation','role_output_tokens')}
    models=copy.deepcopy(models)
    models['qwen'].pop('api_base',None) # transport changes do not change inference semantics
    source=Path(row['image_path'])
    if code is None:code=implementation_fingerprint()
    stat=source.stat()
    semantics={k:row.get(k,'') for k in ('image_id','width','height','caption')}
    return key([CONTRACT,config,models,code,semantics,str(source),stat.st_size,stat.st_mtime_ns])


def validate_config(config):
    for name,default,minimum in [('category_pages',3,1),('category_locator_limit',2,0),
                                ('local_search_limit',2,0),('role_workers',4,1)]:
        value=config.get(name,default)
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')


def run(args,config,backend_factory=Backend):
    validate_config(config)
    schedule=stage_plan(config);stages=[row[1] for row in schedule]
    models=load_yaml(config['models_config']);root=resolve_path(args.output_dir or 'outputs')
    end=args.end_index if args.end_index is not None else args.start_index+int(config.get('sample_size',1962))
    rows=sampled_manifest_rows(config['manifest'],args.start_index,end,seed=int(config.get('sample_seed',42)))
    if not rows:raise ValueError('Empty role image selection; check start/end indices')
    for row in rows:row['image_path']=str(resolve_path(row['image_path']))
    selected={r['image_id'] for r in rows}
    path=root/'route_b/role_state.jsonl'
    records=list(read_jsonl(path)) if path.exists() else []
    stored={r['image_id']:r for r in records}
    if len(stored)!=len(records):raise ValueError('Duplicate image IDs in role checkpoint')
    if not set(stored).issubset(selected):raise ValueError('Role checkpoint contains unselected images; use a new run')
    for state in records:
        step=state.get('step')
        if (state.get('contract')!=CONTRACT or isinstance(step,bool)
                or not isinstance(step,int) or not 0<=step<=len(stages)):
            raise ValueError('Invalid role checkpoint contract or stage boundary')
    code=implementation_fingerprint()
    for row in rows:
        sig=signature(config,models,row,code)
        if row['image_id'] not in stored:stored[row['image_id']]=initial_state(row,sig)
        elif stored[row['image_id']]['signature']!=sig:raise ValueError('Role inputs or policy changed; use a new run')
    lock=threading.Lock()
    def execute(step):
        _,stage,action,_=schedule[step-1]
        if any(stored[i]['step']<step-1 for i in selected):raise ValueError('Earlier role stage incomplete: '+stage)
        if args.check_only:
            if any(stored[i]['step']<step for i in selected):raise StagePending(stage)
            if action in ('finalize','review'):
                try:role_outputs.validate(root,selected,require_review=action=='review')
                except (ValueError,FileNotFoundError,KeyError):raise StagePending(stage)
            return
        def worker(image_id):
            state=copy.deepcopy(stored[image_id])
            if state['step']>=step:return
            def save():
                with lock:
                    stored[image_id]=copy.deepcopy(state)
                    rewrite_jsonl_atomic(path,[stored[i] for i in sorted(stored)])
            print(f'[role_{stage}] image={image_id} START',flush=True)
            engine=Engine(state,config,backend_factory(config,models),save)
            engine.stage(action);state['step']=step;save()
            print(f'[role_{stage}] image={image_id} objects={len(state["objects"])} DONE',flush=True)
        with ThreadPoolExecutor(max_workers=int(config.get('role_workers',4))) as pool:
            list(pool.map(worker,sorted(selected)))
        if action in ('finalize','review'):
            role_outputs.export([stored[i] for i in sorted(selected)],root)
            role_outputs.validate(root,selected,require_review=False)
        if action=='review':
            role_outputs.review(root);role_outputs.validate(root,selected)
    if args.stage=='all':
        for step in range(1,len(stages)+1):execute(step)
    else:
        explicit=getattr(args,'stage_step',None)
        if explicit is not None:
            if not 1<=explicit<=len(stages) or stages[explicit-1]!=args.stage:raise ValueError('Stage step/name mismatch')
            step=explicit
        else:
            if args.stage not in stages:raise ValueError('Unknown role stage: '+args.stage)
            step=stages.index(args.stage)+1
        execute(step)
