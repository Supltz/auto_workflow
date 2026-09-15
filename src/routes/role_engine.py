"""Per-image state machine; existing scheduler owns all leases and stage commits."""
from __future__ import annotations
import copy
from src.routes.role_contract import (CONTRACT, Scene, ObjectDecision, IdentityDecision, OCR,
    Draft, PhraseDecision, key, normalized_region, crop_to_original, geometry_audit,
    add_hint, reserve_search, phrase_verdict)
from src.regions.consensus import size_assessment
from src.utils.geometry import iou, containment, valid_box


def initial_state(row, signature):
    return dict(image_id=row['image_id'],source_image=row['image_path'],image_width=row['width'],
                image_height=row['height'],caption=row.get('caption',''),signature=signature,
                contract=CONTRACT,step=0,categories=[],candidates=[],objects=[],calls={},events=[])


class Engine:
    def __init__(self,state,config,backend,save):
        self.s,self.config,self.backend,self.save=state,config,backend,save
        self.s['local_search_limit']=int(config.get('local_search_limit',2))

    def call(self,name,card,schema,boxes=()):
        cache_key=key(['qwen',name,self.backend.prompt(name),card,boxes])
        if cache_key not in self.s['calls']:
            result,raw=self.backend.ask(name,self.s,card,schema,boxes)
            self.s['calls'][cache_key]={'result':result,'raw_answer':raw}
            self.save()
        return copy.deepcopy(self.s['calls'][cache_key]['result'])

    def ground(self,model,query,region=None):
        cache_key=key(['ground',model,query,region])
        if cache_key not in self.s['calls']:
            self.s['calls'][cache_key]=self.backend.ground(model,self.s,query,region)
            self.save()
        return copy.deepcopy(self.s['calls'][cache_key])

    def category(self,cid):
        return next(c for c in self.s['categories'] if c['id']==cid)

    def scene(self):
        # Bounded category pagination, not a cap on discovered object instances.
        for page in range(int(self.config.get('category_pages',3))):
            existing=[c['query'] for c in self.s['categories']]
            # Persist the page input so a restart cannot turn continuation into re-discovery.
            pages=self.s.setdefault('scene_pages',[])
            if page<len(pages):
                result=pages[page]
            else:
                result=self.call('scene',dict(already_seen=existing,caption_reference=self.s['caption'],
                     coordinates='original_image_normalized_1000',page=page+1),Scene)
                pages.append(result);self.save()
            for proposal in result['categories']:
                query=' '.join(proposal['query'].split())
                if not query or any(c['query'].casefold()==query.casefold() for c in self.s['categories']):continue
                category=dict(id='c_'+key([self.s['image_id'],query.casefold()]),query=query,
                    visible_evidence=proposal['visible_evidence'],locators=list(dict.fromkeys(proposal['locators'])),
                    hints=[],full_done=False,local_searches=0,peer_version=0)
                for hint in proposal['search_hints']:
                    region=normalized_region(hint['region'],self.s['image_width'],self.s['image_height'],
                                             self.config.get('local_padding',.2))
                    add_hint(category,region,hint['kind'],hint['evidence'])
                self.s['categories'].append(category);self.save()
            if not result['more_categories']:break
        self.s['category_search_budget_exhausted']=bool(result['more_categories'])
        self.save()

    def add_detections(self,category,result,model,region=None,query=None):
        added=[]
        for raw in result['predictions']:
            if raw.get('bbox_xyxy') is None:continue
            box=raw['bbox_xyxy']
            if region is not None:box=crop_to_original(box,region,result['view_size'])
            if not valid_box(tuple(box),self.s['image_width'],self.s['image_height']):continue
            identity='d_'+key([self.s['image_id'],category['id'],model,region,box])
            if any(c['id']==identity for c in self.s['candidates']):continue
            candidate=dict(id=identity,category_id=category['id'],query=category['query'],bbox_xyxy=box,
                           source=model,method=raw.get('box_method','detector_box'),view=region,
                           locator=query if model=='egm' else None,status='pending',evidence=raw)
            self.s['candidates'].append(candidate);added.append(candidate)
        self.save()
        return added

    def hint_box(self,category,box,source,reason):
        w,h=self.s['image_width'],self.s['image_height']
        region=normalized_region([box[0]*1000/w,box[1]*1000/h,box[2]*1000/w,box[3]*1000/h],w,h,
                                 self.config.get('local_padding',.2))
        add_hint(category,region,source,reason)

    def discover(self):
        # Every category gets its full-image opportunity before any EGM/local work.
        for category in self.s['categories']:
            if not category['full_done']:
                result=self.ground('sam31',category['query'])
                self.add_detections(category,result,'sam31');category['full_done']=True;self.save()
        for category in self.s['categories']:
            for locator in category['locators'][:int(self.config.get('category_locator_limit',2))]:
                if not locator.strip():continue
                result=self.ground('egm',locator)
                self.add_detections(category,result,'egm',query=locator)
                sam=[c for c in self.s['candidates'] if c['category_id']==category['id'] and c['source']=='sam31']
                for pred in result['predictions']:
                    box=pred.get('bbox_xyxy')
                    if box and not any(iou(tuple(box),tuple(c['bbox_xyxy']))>=.5 for c in sam):
                        self.hint_box(category,box,'egm_unmatched','EGM locator does not match full-image SAM instances')
                self.save()

    def identity(self,a,b):
        if a['query'].casefold()==b['query'].casefold() and iou(tuple(a['bbox_xyxy']),tuple(b['bbox_xyxy']))>=.98:
            return dict(same_object=True,certain=True,preferred=('B' if b['source']=='sam31' and a['source']!='sam31' else 'A'),
                        reason='same category and near-identical extent; prefer validated SAM when equivalent')
        return self.call('identity',dict(A={'category':a['query'],'bbox':a['bbox_xyxy'],'box_source':a['source']},
                    B={'category':b['query'],'bbox':b['bbox_xyxy'],'box_source':b['source']}),IdentityDecision,[a['bbox_xyxy'],b['bbox_xyxy']])

    def verify_objects(self):
        # Prefer SAM evidence before EGM-only candidates, not by confidence ranking.
        pending=sorted((c for c in self.s['candidates'] if c['status']=='pending'),key=lambda c:c['source']!='sam31')
        for candidate in pending:
            box=candidate['bbox_xyxy']
            size=size_assessment(tuple(box),self.s['image_width'],self.s['image_height'],self.config['aggregation'])
            candidate['size']=size
            if not size['size_pass']:
                candidate.update(status='rejected',reason=size['size_reject_reason']);self.save();continue
            decision=self.call('object',dict(category=candidate['query'],hypothesis=candidate.get('locator'),
                              source_view=candidate['view'],task='verify this exact box'),ObjectDecision,[box])
            candidate['verification']=decision
            if not ObjectDecision.model_validate(decision).accepted():
                candidate.update(status='rejected',reason=decision['reason']);self.save();continue
            duplicate=None;uncertain=False
            for obj in self.s['objects']:
                obox=obj['bbox_xyxy']
                if iou(tuple(box),tuple(obox))<.25 and max(containment(tuple(box),tuple(obox)),containment(tuple(obox),tuple(box)))<.8:continue
                identity=self.identity(obj,candidate)
                candidate.setdefault('identity_checks',[]).append({'object_id':obj['id'],**identity})
                if not identity['certain']:
                    uncertain=True;break
                if identity['same_object']:
                    duplicate=obj;break
            if candidate['source']=='egm' and not uncertain and not duplicate:
                # A SAM-unmatched object needs an additional object audit. Omit
                # the locator hypothesis so it cannot serve as its own evidence.
                extra=self.call('object',dict(category=candidate['query'],hypothesis=None,
                    task='additional full-image audit of a SAM-unmatched EGM candidate'),ObjectDecision,[box])
                candidate['extra_verification']=extra
                if not ObjectDecision.model_validate(extra).accepted():
                    candidate.update(status='rejected',reason='extra_object_audit: '+extra['reason']);self.save();continue
            if uncertain:
                candidate.update(status='unresolved_identity',reason='cannot establish whether this is a distinct instance')
            elif duplicate:
                candidate.update(status='duplicate',object_id=duplicate['id'])
                if candidate['id'] not in duplicate['detections']:duplicate['detections'].append(candidate['id'])
                if candidate['category_id'] not in duplicate['category_ids']:
                    duplicate['category_ids'].append(candidate['category_id']);self.category(candidate['category_id'])['peer_version']+=1
                # A verified SAM alternative replaces EGM geometry, keeping object identity.
                prefer_new=(identity['preferred']=='B')
                if prefer_new:
                    duplicate.update(bbox_xyxy=box,source=candidate['source'],method=candidate['method'],
                                     verification=decision,box_version=duplicate['box_version']+1)
                    for cid in duplicate['category_ids']:self.category(cid)['peer_version']+=1
            else:
                category=self.category(candidate['category_id'])
                obj=dict(id='o_'+key([self.s['image_id'],candidate['id']]),query=category['query'],
                     category_id=category['id'],category_ids=[category['id']],bbox_xyxy=box,box_version=1,
                     source=candidate['source'],method=candidate['method'],verification=decision,
                     detections=[candidate['id']],phrase=None,history=[],ocr=None)
                self.s['objects'].append(obj);category['peer_version']+=1
                candidate.update(status='verified',object_id=obj['id'])
            self.save()

    def recover(self):
        assert all(c['full_done'] for c in self.s['categories'])
        # One slot per category per sweep. Fixed limit shared by all categories.
        for category in self.s['categories']:
            hint=reserve_search(category,int(self.config.get('local_search_limit',2)))
            if hint is None:continue
            self.save()  # reservation survives retries; no quota is replenished.
            result=self.ground('sam31',category['query'],hint['region'])
            self.add_detections(category,result,'sam31',hint['region'])
            hint['status']='done';self.save()

    def peers(self,obj):
        return [dict(id=o['id'],category=o['query'],bbox=o['bbox_xyxy']) for o in self.s['objects']
                if o['id']!=obj['id'] and set(o['category_ids']) & set(obj['category_ids'])]

    def peer_key(self,obj):
        return key([obj['box_version'],[(cid,self.category(cid)['peer_version']) for cid in obj['category_ids']]])

    def ocr(self):
        for obj in self.s['objects']:
            if obj.get('ocr_version')!=obj['box_version']:
                obj['ocr']=self.call('ocr',dict(category=obj['query']),OCR,[obj['bbox_xyxy']])
                obj['ocr_version']=obj['box_version'];self.save()

    def describe(self,rewrite=False):
        for obj in self.s['objects']:
            previous=obj['phrase']
            if previous is not None:
                if not rewrite or previous['status']!='needs_rewrite':continue
                if previous['revision']>=int(self.config.get('max_refinement_rounds',2)):
                    previous.update(status='unresolved',reason='rewrite_budget_exhausted');self.save();continue
            card=dict(category=obj['query'],target_evidence=obj['verification'],verified_ocr=obj['ocr'],
                      same_category_peers=self.peers(obj)[:32],peers_are_not_exhaustive=True,
                      original_target_box=obj['bbox_xyxy'],previous=(dict(text=previous['text'],
                          revision=previous['revision'],status=previous['status'],reason=previous['reason'][:1200],
                          colliding_object_ids=previous.get('colliding_object_ids',[])) if previous else None))
            draft=self.call('phrase',card,Draft,[obj['bbox_xyxy']])
            text=' '.join(draft['expression'].split())
            if previous:obj['history'].append(previous)
            revision=0 if previous is None else previous['revision']+1
            obj['phrase']=dict(text=text,revision=revision,evidence=draft['visible_evidence'],
                status='draft' if text.lower().startswith('the ') and draft['visible_evidence'] else 'unresolved',
                reason=draft['reason'],reground=None,verified_peers=None,box_version=obj['box_version'])
            self.save()

    def reground(self,discover=True):
        for obj in self.s['objects']:
            phrase=obj['phrase']
            if phrase is None or not phrase['text'].lower().startswith('the ') or not phrase['evidence']:continue
            if phrase.get('reground') is None or phrase.get('box_version')!=obj['box_version']:
                result=self.ground('egm',phrase['text'])
                phrase['reground']=geometry_audit(obj['bbox_xyxy'],result['predictions'],self.config['aggregation'])
                phrase['box_version']=obj['box_version'];phrase['verified_peers']=None
                self.save()
            if not discover:continue
            category=self.category(obj['category_id'])
            for pred in phrase['reground']['predictions']:
                box=pred.get('bbox_xyxy')
                if box and not any(iou(tuple(box),tuple(o['bbox_xyxy']))>=.5 for o in self.s['objects']):
                    self.hint_box(category,box,'reground_unmatched','EGM reground returned an unrecorded candidate')
                    # May be a new object, not a license to switch the current phrase target.
                    self.add_detections(category,{'predictions':[pred]},'egm',query=phrase['text'])
            self.save()

    def verify_phrases(self):
        for obj in self.s['objects']:
            phrase=obj['phrase']
            if (not phrase or not phrase['text'].lower().startswith('the ')
                    or not phrase['evidence'] or not phrase.get('reground')):continue
            if phrase.get('box_version')!=obj['box_version']:
                phrase['reground']=geometry_audit(obj['bbox_xyxy'],phrase['reground']['predictions'],self.config['aggregation'])
                phrase['box_version']=obj['box_version'];phrase['verified_peers']=None
            peers_key=self.peer_key(obj)
            if phrase.get('verified_peers')==peers_key:continue
            normal=phrase['reground']['passed']
            name='verify' if normal else 'adjudicate'
            boxes=[obj['bbox_xyxy']]
            predictions=[p['bbox_xyxy'] for p in phrase['reground']['predictions'] if p.get('bbox_xyxy')]
            if not normal and predictions:boxes.append(predictions[0])
            peers=self.peers(obj)
            pages=[peers[i:i+32] for i in range(0,len(peers),32)] or [[]]
            decisions=[]
            for page,items in enumerate(pages):
                card=dict(target_A=obj['id'],category=obj['query'],phrase=phrase['text'],
                          target_evidence=obj['verification'],B_present=len(boxes)>1,
                          other_locator_boxes=predictions[1:] if not normal else [],
                          peers=items,peer_page=page+1,peer_pages=len(pages),
                          inspect_full_image_for_unlisted_alternatives=True)
                decisions.append(self.call(name,card,PhraseDecision,boxes))
            verdicts=[phrase_verdict(d) for d in decisions]
            status=('needs_rewrite' if 'needs_rewrite' in verdicts else
                    'unresolved' if 'unresolved' in verdicts else 'verified')
            phrase.update(status=status,verified_peers=peers_key,decisions=decisions,
                          pass_via=('reground' if normal else 'semantic_adjudication') if status=='verified' else None,
                          reason='; '.join(d['reason'] for d in decisions))
            self.save()
        # A deterministic collision is ambiguity evidence even if separate
        # model reviews both said unique. Surface it before rewrite rounds.
        self.mark_phrase_collisions()

    def mark_phrase_collisions(self):
        groups={}
        for obj in self.s['objects']:
            phrase=obj['phrase']
            if phrase and phrase['text']:
                groups.setdefault(phrase['text'].casefold(),[]).append(obj)
        for same in groups.values():
            if len(same)<2:continue
            ids=[o['id'] for o in same]
            for obj in same:
                phrase=obj['phrase']
                if phrase['status'] in ('verified','needs_rewrite'):
                    phrase.update(status='needs_rewrite',pass_via=None,
                        reason='same_phrase_for_distinct_objects; compare targets '+', '.join(ids),
                        colliding_object_ids=[i for i in ids if i!=obj['id']])
        self.save()

    def stage(self,action):
        if action in ('rewrite','finish_descriptions'):
            self.ocr()
            # New peers invalidate previous uniqueness; rewrite the same object.
            self.verify_phrases();self.describe(rewrite=True)
        elif action=='reground_final':
            # Close discovery after the last object audit. Retain A-vs-B evidence
            # for adjudication without starting another unbounded object chain.
            self.reground(discover=False)
        elif action=='finalize':
            self.mark_phrase_collisions()
            for obj in self.s['objects']:
                if obj['phrase'] and obj['phrase']['status']!='verified':
                    obj['phrase']['status']='unresolved'
            self.save()
        elif action=='review':pass
        else:getattr(self,action)()
