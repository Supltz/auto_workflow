"""One versioned stage definition shared by CLI, scheduler, progress and recovery."""
CONTRACT = 'role-grounding-v1'
STORAGE_CONTRACT = 'role-grounding-storage-v2'
LEGACY_STORAGE_CONTRACT = 'role-grounding-storage-v1'


def stage_plan(config=None):
    config = {"phrase_review_version": 2} if config is None else config
    blind = int(config.get("phrase_review_version",1)) >= 2
    stages = [
        ('qwen', 'category_inventory', 'scene', '类别清单'),
        ('other', 'instance_discovery', 'discover', 'SAM 实例发现与 EGM 定位'),
        ('qwen', 'object_verification', 'verify_objects', '对象与框审核'),
        ('other', 'local_search_initial', 'recover', '按类别局部补漏'),
        ('qwen', 'local_object_verification', 'verify_objects', '补漏对象审核'),
        ('qwen', 'target_ocr', 'ocr', '目标文字识别'),
        ('qwen', 'phrase_generation', 'describe', '生成对象描述'),
        ('other', 'phrase_reground', 'reground', 'EGM 描述定位'),
        ('qwen', 'phrase_adjudication', 'verify_phrases', '语义与歧义裁决'),
    ]
    rounds = int(config.get('max_refinement_rounds', 2))
    if not 0 <= rounds <= 10:
        raise ValueError('max_refinement_rounds must be between 0 and 10')
    for iteration in range(1, rounds + 1):
        for kind, name, action, label in [
            ('other', 'local_search', 'recover', '按类别补漏'),
            ('qwen', 'object_verification', 'verify_objects', '补漏对象审核'),
            ('qwen', 'phrase_rewrite', 'rewrite', '复核唯一性并改写'),
            ('other', 'phrase_reground', 'reground', 'EGM 描述定位'),
            ('qwen', 'phrase_adjudication', 'verify_phrases', '语义与歧义裁决'),
        ]:
            stages.append((kind, f'{name}_round_{iteration}', action, f'第{iteration}轮{label}'))
    stages.extend([
        ('qwen', 'final_object_verification', 'verify_objects', '末轮新增对象审核'),
        ('qwen', 'final_phrase_generation', 'finish_descriptions', '末轮对象描述与剩余改写'),
        ('other', 'final_phrase_reground', 'reground_final', '收尾描述定位（不再扩展候选）'),
        ('qwen', 'final_phrase_adjudication', 'verify_phrases', '最终唯一性复核'),
        ('other', 'finalize_objects', 'finalize', '导出对象与描述状态'),
        ('other', 'review_export', 'review', '生成审核图片'),
    ])
    if blind:
        expanded=[]
        for row in stages:
            if row[2]=="verify_phrases":
                expanded.append(("qwen",row[1].replace("adjudication","blind_review"),"blind_review","独立盲审指代集合"))
            expanded.append(row)
        stages=expanded
    return stages


def phases(config):
    groups = []
    for kind, name, _, _ in stage_plan(config):
        if not groups or groups[-1][0] != kind:
            groups.append((kind, []))
        groups[-1][1].append(name)
    return groups


if __name__ == '__main__':
    from src.utils.config import load_yaml
    for kind, name, _, _ in stage_plan(load_yaml('configs/route_b.yaml')):
        print(kind, name)
