"""Bind isolated acceptance results to the exact implementation tested."""
import hashlib
import json
from pathlib import Path


def source_hashes(root):
    root=Path(root)
    files={}
    for folder in ('src','prompts','configs','scripts','tests','.local/multi_gpu','.local/multi_node','.local/ram_runtime'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.sh','.txt','.yaml'):
                files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    override=root/'.local/config.yaml'
    if override.is_file():files['.local/config.yaml']=hashlib.sha256(override.read_bytes()).hexdigest()
    return files


def checked_acceptance(root):
    root=Path(root);receipt=json.loads((root/'acceptance.json').read_text())
    if receipt.get('status')!='passed' or receipt.get('platform')!='linux':
        raise ValueError('Linux acceptance has not passed')
    if receipt.get('source_hashes')!=source_hashes(root):
        raise ValueError('Source changed after acceptance')
    if set(receipt.get('tests',{}))!={'workflow','multi_node','multi_gpu'}:
        raise ValueError('Missing acceptance test suite')
    for result in receipt['tests'].values():
        path=root/result['log']
        if result['returncode']!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=result['sha256']:
            raise ValueError('Acceptance test log changed or failed')
    if not receipt.get('gpu'):raise ValueError('Missing GPU acceptance')
    for case in receipt['gpu']:
        path=root/case['result']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=case['sha256']:raise ValueError('GPU receipt changed')
        value=json.loads(path.read_text())
        if (value.get('source_hashes')!=receipt['source_hashes'] or value.get('status')!='passed'
                or value.get('objects',0)<1 or value.get('verified',0)<1
                or set(value.get('resident_models',[]))!={'sam31','egm'}):
            raise ValueError('GPU acceptance failed or tested different source')
    return receipt
