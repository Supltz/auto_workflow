"""Compact final delivery; runtime reasoning/masks/crops are not archived."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path
from src.routes.role_contract import CONTRACT
from src.routes.role_context import REVIEW_VERSION, blind_gate
from src.utils.io import read_jsonl,rewrite_jsonl_atomic,atomic_write_text
from src.utils.images import open_rgb,save_referring_expression_overlay
from src.utils.geometry import valid_box

FILES=('objects/route_b.jsonl','unresolved_regions/route_b.jsonl','search_audit/route_b.jsonl')


def compact_audit(obj,phrase):
    blind=phrase.get('blind')
    decisions=phrase.get('decisions',[])
    return dict(version=REVIEW_VERSION,blind=blind,
        context_regions=(obj.get('context') or {}).get('regions',[]),
        locator_cue=phrase.get('locator_cue'),cue_type=phrase.get('cue_type'),
        peer_ids=sorted({c['object_id'] for d in decisions for c in d['comparisons']}),
        decisions=[{k:d[k] for k in ('outcome','target_kind_matches','locator_cue_valid',
                     'reference_scope_clear','blind_target_relation','issue_type','comparisons',
                     'describes_A','facts_visible','single_whole_target','grammatical',
                     'unique_in_full_image','also_matches_B','competing_object_ids','evidence','reason')}
                   for d in decisions])


def validate_phrase_audit(row):
    from src.routes.role_contract import phrase_verdict,BlindDecision
    audit=row.get('phrase_audit') or {}
    if audit.get('version')!=REVIEW_VERSION:raise ValueError('Missing independent phrase audit')
    blind=BlindDecision.model_validate(audit.get('blind')).model_dump()
    if blind_gate(blind):raise ValueError('Unresolved blind evidence cannot certify a phrase')
    decisions=audit.get('decisions',[])
    if not decisions or any(phrase_verdict(d)!='verified' for d in decisions):
        raise ValueError('Unresolved competition or referent mismatch')
    ids=[c['object_id'] for d in decisions for c in d['comparisons']]
    if len(ids)!=len(set(ids)) or sorted(ids)!=audit.get('peer_ids'):raise ValueError('Invalid peer audit coverage')
    cue=row.get('locator_cue')
    if (not cue or cue.casefold() not in row['final_referring_expression'].casefold()
            or row.get('cue_type') not in ('spatial','relation','attachment','action','appearance')
            or cue!=audit.get('locator_cue') or row['cue_type']!=audit.get('cue_type')):
        raise ValueError('Missing or inconsistent locating condition')


def delivery(states):
    objects=[];verified=[];unresolved=[];search=[];rejected=[]
    for state in states:
        for obj in state['objects']:
            phrase=obj['phrase']
            status='verified' if phrase and phrase['status']=='verified' else 'unresolved'
            row=dict(contract=CONTRACT,image_id=state['image_id'],region_id=obj['id'],
                     source_image=state['source_image'],image_width=state['image_width'],image_height=state['image_height'],
                     bbox_xyxy=obj['bbox_xyxy'],category=obj['query'],box_source=obj['source'],
                     box_version=obj['box_version'],object_verified=True,phrase_status=status,
                     phrase_review_version=state.get('phrase_review_version',1),referent=obj.get('referent'))
            objects.append(row)
            if status=='verified':
                verified.append({**row,'status':'verified_unique_referring_expression',
                    'final_referring_expression':phrase['text'],'pass_via':phrase['pass_via'],
                    'reground_passed':phrase['reground']['passed'],
                    'reground_iou':phrase['reground']['reground_iou'],
                    'locator_cue':phrase.get('locator_cue'),'cue_type':phrase.get('cue_type'),
                    'phrase_audit':compact_audit(obj,phrase)})
            else:
                unresolved.append({**row,'phrase':phrase['text'] if phrase else None,
                    'reason':(phrase['reason'] if phrase else 'no_phrase_before_search_drain_completed')[:1200],
                    'phrase_audit':compact_audit(obj,phrase) if phrase else None})
        for c in state['categories']:
            search.append(dict(image_id=state['image_id'],category=c['query'],full_image_searches=int(c['full_done']),
                local_searches=c['local_searches'],local_search_limit=state.get('local_search_limit',2),unprocessed_hints=sum(h['status']!='done' for h in c['hints']),
                stop_reason='budget_exhausted' if any(h['status']!='done' for h in c['hints']) else 'no_pending_evidence'))
        for c in state['candidates']:
            if c['status'] in ('rejected','unresolved_identity'):
                rejected.append(dict(image_id=state['image_id'],candidate_id=c['id'],bbox_xyxy=c['bbox_xyxy'],
                    category=c['query'],status=c['status'],reason=c.get('reason','')[:1200]))
    counts=Counter(r['image_id'] for r in verified)
    object_counts=Counter(r['image_id'] for r in objects)
    unresolved_counts=Counter(r['image_id'] for r in unresolved)
    return dict([('verified_regions/route_b.jsonl',verified),('objects/route_b.jsonl',objects),
                      ('unresolved_regions/route_b.jsonl',unresolved),('search_audit/route_b.jsonl',search),
                      ('rejected_regions/route_b.jsonl',rejected),
                      ('verified_regions/route_b_counts_by_image.jsonl',
                       [dict(image_id=s['image_id'],phrase_review_version=s.get('phrase_review_version',1),final_bbox_count=counts[s['image_id']],
                             object_count=object_counts[s['image_id']],unresolved_count=unresolved_counts[s['image_id']])
                        for s in states])])


def export(states,root):
    for path,rows in delivery(states).items():rewrite_jsonl_atomic(Path(root)/path,rows)


def review(root):
    from io import StringIO
    root=Path(root);directory=root/'human_review/route_b';directory.mkdir(parents=True,exist_ok=True)
    rows=[];expected=set()
    fields=('review_file','image_id','region_id','source_image','final_referring_expression',
            'bbox_x1','bbox_y1','bbox_x2','bbox_y2')
    for record in read_jsonl(root/'verified_regions/route_b.jsonl'):
        path=directory/(record['region_id']+'.jpg');expected.add(path.name)
        with open_rgb(record['source_image']) as image:
            save_referring_expression_overlay(image,tuple(record['bbox_xyxy']),record['final_referring_expression'],path)
        rows.append({k:record[k] for k in ('image_id','region_id','source_image','final_referring_expression')} |
            dict(review_file=str(path),**dict(zip(fields[-4:],record['bbox_xyxy']))))
    for old in directory.glob('*.jpg'):
        if old.name not in expected:old.unlink()
    buffer=StringIO();writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    atomic_write_text(directory/'index.csv',buffer.getvalue())


def validate(root,selected=None,require_review=True,expected_review_version=None):
    root=Path(root)
    required=(*FILES,'verified_regions/route_b.jsonl','verified_regions/route_b_counts_by_image.jsonl')
    for name in required:
        if not (root/name).is_file():raise FileNotFoundError('Missing role output: '+name)
    objects=list(read_jsonl(root/'objects/route_b.jsonl'));verified=list(read_jsonl(root/'verified_regions/route_b.jsonl'))
    unresolved=list(read_jsonl(root/'unresolved_regions/route_b.jsonl'));search=list(read_jsonl(root/'search_audit/route_b.jsonl'))
    index={o['region_id']:o for o in objects}
    if len(index)!=len(objects):raise ValueError('Duplicate object IDs')
    checkpoint=root/'route_b/role_state.jsonl'
    if checkpoint.is_file():
        states=list(read_jsonl(checkpoint))
        ids=[s['image_id'] for s in states]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate image checkpoint')
        if selected is not None and set(ids)!=set(selected):raise ValueError('Checkpoint image selection mismatch')
        confirmed={o['id']:(s['image_id'],o) for s in states for o in s['objects']}
        if len(confirmed)!=sum(len(s['objects']) for s in states):raise ValueError('Duplicate confirmed object')
        if set(confirmed)!=set(index):raise ValueError('Confirmed object lost during final export')
        for identity,row in index.items():
            image_id,obj=confirmed[identity]
            if row['image_id']!=image_id or row['bbox_xyxy']!=obj['bbox_xyxy'] or row['box_version']!=obj['box_version']:
                raise ValueError('Export differs from confirmed object state')
        # The checkpoint is the authoritative source before task cleanup. Check
        # phrases and audit rows too: a matching box alone is not a valid receipt.
        for name,expected_rows in delivery(states).items():
            canonical=lambda rows:Counter(json.dumps(r,sort_keys=True) for r in rows)
            if canonical(read_jsonl(root/name))!=canonical(expected_rows):
                raise ValueError('Export differs from confirmed state: '+name)

    for obj in objects:
        if obj.get('contract')!=CONTRACT or not obj.get('object_verified'):raise ValueError('Unverified object')
        if obj.get('phrase_review_version',1)==REVIEW_VERSION:
            referent=obj.get('referent') or {}
            if not referent.get('name') or referent.get('kind') not in ('physical_object','part','depiction','package'):
                raise ValueError('Missing exact referent identity')
        if not valid_box(tuple(obj['bbox_xyxy']),obj['image_width'],obj['image_height']):raise ValueError('Invalid object box')
        if selected is not None and obj['image_id'] not in selected:raise ValueError('Foreign image')
        if not Path(obj['source_image']).is_absolute() or not Path(obj['source_image']).is_file():raise ValueError('Missing original image')
    all_phrases=verified+unresolved
    if len(all_phrases)!=len(index) or {r['region_id'] for r in all_phrases}!=set(index):raise ValueError('Object/phrase partition mismatch')
    for row in all_phrases:
        obj=index[row['region_id']]
        if any(row.get(k)!=v for k,v in obj.items()):raise ValueError('Phrase switched object')
    for row in verified:
        if row.get('phrase_review_version')==REVIEW_VERSION:validate_phrase_audit(row)
        if row['phrase_status']!='verified' or not row['final_referring_expression'].lower().startswith('the '):raise ValueError('Invalid verified phrase')
        if row['pass_via'] not in ('reground','semantic_adjudication'):raise ValueError('Missing acceptance evidence role')
        if row['pass_via']=='reground' and not row['reground_passed']:raise ValueError('Failed reground falsely passed')
    for row in unresolved:
        if row['phrase_status']!='unresolved':raise ValueError('Invalid unresolved status')
    counts=list(read_jsonl(root/'verified_regions/route_b_counts_by_image.jsonl'))
    versions={r['image_id']:r.get('phrase_review_version',1) for r in counts}
    if expected_review_version is not None and any(v!=expected_review_version for v in versions.values()):
        raise ValueError('Expected phrase review version is missing or obsolete')
    if any(o.get('phrase_review_version',1)!=versions.get(o['image_id']) for o in objects):
        raise ValueError('Phrase review version mismatch')
    expected=Counter(r['image_id'] for r in verified)
    if selected is not None and {r['image_id'] for r in counts}!=set(selected):raise ValueError('Missing image results')
    if len({r['image_id'] for r in counts})!=len(counts):raise ValueError('Duplicate image count')
    if any(r['final_bbox_count']!=expected[r['image_id']] for r in counts):raise ValueError('Verified count mismatch')
    object_counts=Counter(r['image_id'] for r in objects);unresolved_counts=Counter(r['image_id'] for r in unresolved)
    if any(r.get('object_count')!=object_counts[r['image_id']] or
           r.get('unresolved_count')!=unresolved_counts[r['image_id']] for r in counts):
        raise ValueError('Object/unresolved count mismatch')
    if any(r['full_image_searches']!=1 or not 0<=r['local_searches']<=r['local_search_limit'] for r in search):
        raise ValueError('Invalid category search budget')
    if selected is not None and any(r['image_id'] not in selected for r in search):raise ValueError('Foreign search audit')
    if len({(r['image_id'],r['category'].casefold()) for r in search})!=len(search):raise ValueError('Duplicate category audit')
    if len({(r['image_id'],r['final_referring_expression'].casefold()) for r in verified})!=len(verified):
        raise ValueError('A unique phrase cannot identify two different objects')
    if require_review:
        with (root/'human_review/route_b/index.csv').open() as f:rows=list(csv.DictReader(f))
        if len(rows)!=len(verified) or {r['region_id'] for r in rows}!={r['region_id'] for r in verified}:raise ValueError('Review coverage mismatch')
        records={r['region_id']:r for r in verified}
        for row in rows:
            r=records[row['region_id']]
            if row['final_referring_expression']!=r['final_referring_expression'] or row['source_image']!=r['source_image']:raise ValueError('Review content mismatch')
            if [float(row['bbox_'+x]) for x in ('x1','y1','x2','y2')]!=r['bbox_xyxy']:raise ValueError('Review geometry mismatch')
            if not Path(row['review_file']).is_file():raise ValueError('Missing review image')
    policy=({'phrase_review_version':REVIEW_VERSION} if versions and set(versions.values())=={REVIEW_VERSION} else {})
    return dict(**policy,objects=len(objects),verified=len(verified),unresolved=len(unresolved),categories=len(search))
