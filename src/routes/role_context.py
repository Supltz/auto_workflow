"""Bounded context views; target-aware views never enter blind review."""
from __future__ import annotations
import math
from src.utils.geometry import iou, containment

REVIEW_VERSION = 2
DEFAULTS = dict(local_scale=3., min_image_fraction=.15, group_padding=.12,
                max_group_views=2, peer_page_size=12, nearest_peers=8)


def options(config):
    return {**DEFAULTS, **config.get('phrase_context', {})}


def expand(box, width, height, scale=1., minimum=0.):
    x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2
    w=min(width,max((x2-x1)*scale,width*minimum))
    h=min(height,max((y2-y1)*scale,height*minimum))
    left=max(0,min(width-w,cx-w/2));top=max(0,min(height-h,cy-h/2))
    return [math.floor(left),math.floor(top),math.ceil(left+w),math.ceil(top+h)]


def region_pixels(region, width, height):
    if (len(region)!=4 or any(not math.isfinite(v) for v in region)
            or not 0<=region[0]<region[2]<=1000 or not 0<=region[1]<region[3]<=1000):
        raise ValueError('Invalid normalized context region')
    return [math.floor(region[0]*width/1000),math.floor(region[1]*height/1000),
            math.ceil(region[2]*width/1000),math.ceil(region[3]*height/1000)]


def competitors(state, obj, config):
    """All known category peers plus nearby and overlapping cross-category targets.

    This is a retrieval aid, never an exhaustive scene inventory. Blind review
    searches the original scene independently of this list.
    """
    others=[o for o in state['objects'] if o['id']!=obj['id']]
    a=obj['bbox_xyxy'];ax=(a[0]+a[2])/2;ay=(a[1]+a[3])/2
    distance=lambda o: (((o['bbox_xyxy'][0]+o['bbox_xyxy'][2])/2-ax)/state['image_width'])**2+(((o['bbox_xyxy'][1]+o['bbox_xyxy'][3])/2-ay)/state['image_height'])**2
    nearest={o['id'] for o in sorted(others,key=distance)[:options(config)['nearest_peers']]}
    tokens=set(obj['query'].casefold().split())
    selected=[]
    for o in others:
        same=bool(set(o['category_ids']) & set(obj['category_ids']))
        lexical=bool(tokens & set(o['query'].casefold().split()))
        overlap=max(containment(tuple(a),tuple(o['bbox_xyxy'])),containment(tuple(o['bbox_xyxy']),tuple(a)))>=.6
        if same or lexical or overlap or o['id'] in nearest:
            selected.append(dict(id=o['id'],category=o['query'],bbox=o['bbox_xyxy'],
                                 referent=o.get('referent'),priority=0 if same or lexical else 1))
    return sorted(selected,key=lambda o:(o['priority'],distance({'bbox_xyxy':o['bbox']}),o['id']))


def context_views(state, card, boxes, config):
    """Return target-aware pixel windows, bounded without truncating peer review."""
    if not boxes:return []
    w,h=state['image_width'],state['image_height'];opt=options(config)
    views=[]
    for spec in card.get('context_regions',[])[:opt['max_group_views']]:
        region=region_pixels(spec['region'],w,h)
        views.append(expand(region,w,h,1+2*opt['group_padding']))
    peer_boxes=[p['bbox'] for p in card.get('peers',card.get('same_category_peers',[]))]
    if peer_boxes:
        group=[boxes[0],*peer_boxes]
        union=[min(b[0] for b in group),min(b[1] for b in group),max(b[2] for b in group),max(b[3] for b in group)]
        views.append(expand(union,w,h,1+2*opt['group_padding']))
    views.append(expand(boxes[0],w,h,opt['local_scale'],opt['min_image_fraction']))
    result=[]
    for view in views:
        # The unmodified full image is already supplied.
        if view==[0,0,w,h] or any(iou(tuple(view),tuple(b))>.97 for b in result):continue
        result.append(view)
    # Existing Qwen image limit stays 6: original, overlay, tight A, optional B.
    return result[:(2 if len(boxes)>1 else 3)]


def draft_valid(draft):
    cue=draft['locator_cue'].strip()
    return (draft['expression'].lower().startswith('the ') and bool(draft['visible_evidence'])
            and bool(cue) and cue.casefold() in draft['expression'].casefold())


def blind_gate(blind):
    """Multiple plausible referents cannot be overridden by target-aware approval."""
    if blind['outcome']=='multiple':return 'needs_rewrite'
    if blind['outcome']!='unique' or len(blind['matches'])!=1 or blind['requested_regions']:
        return 'unresolved'
    return None
