# Phrase/context review v2

本次实现只生成 grounding phrase，不生成 QA。每条 phrase 至少有一个真实、有效的定位条件；有清晰场景关系时优先使用附着、相对位置或参照物，方便后续独立转换成问题。不恢复最少词数、两个线索、最小短边或 Top-30。

## 输入与证据的变化

| 环节 | 之前 | v2 |
|---|---|---|
| 对象审核 | 审核 category 与框 | 另外记录确切 referent_name/kind：实物、部件、图中印刷形象、包装；phrase 必须指同一粒度 |
| 局部视野 | 四周各扩 bbox 的 25%，即约 1.5 倍宽高 | 边界审核保留紧框；描述视野用 3 倍宽高与原图宽高各 15% 的较大者作兜底，边缘平移截限；这是 context 大小，不是目标尺寸门槛 |
| 对象群视野 | 目标中心小裁剪 | Qwen 先基于原图提出至多 2 个对象群、所属整体、参照物区域；另按当前竞争对象页构造 union。群边距默认 12%，每请求仍至多 6 张图 |
| 唯一性 | 展示 A 后判断 phrase 是否唯一 | 所有 phrase 先做独立调用，仅原图 + phrase；返回 0/1/多个/不确定和近似位置 |
| 放大 | 给审核者目标专属 crop | 盲审只可索取自己基于原图提出的至多 2 个无标框区域，最多再看一次；A 的坐标、专属 crop、生成理由不进入盲审 |
| 裁决 | 主要靠整体布尔判断 | 再揭示 A，与盲审匹配、EGM、候选竞争对象逐项比对；必须明确对象粒度、参照范围、定位条件及每个 competitor 的排除理由 |
| 改写 | 常规失败原因 | 传递 blind matches 与具体冲突对象；同一对象改写，预算不变 |

盲审调用是输入隔离的独立检查，不等于独立模型，也不保证找全。原图始终是比较范围；任何局部放大不重定义 leftmost/topmost/foreground。“right of”不能偷偷当成“immediately right of”。候选列表不是完整的场景清单。

生成和已知目标的复核优先保留模型规划的 group/anchor 视野，再使用当前 peer 页的 union 和局部兜底视野。受六图上限约束，有 B 时至多两个 context，无 B 时至多三个；原图始终提供。群体过大时允许两个重叠放大区域。具体倍率和视野配置可调，但修改策略会改变新 run 的输入签名。

## 验收规则

- EGM 成功也必须通过盲审和竞争对象审核。EGM 只输出一个框不证明唯一。
- 盲审 multiple：当前 phrase 必须改写，Qwen 的 supports_A 无法覆盖该证据。
- 盲审 none/uncertain，或放大后仍请求更多视野：unresolved。
- 盲审 unique：目标审核须明确唯一匹配就是 A；近似定位边界差不必视作换对象。
- EGM 失败仍可在盲审唯一且语义复核明确支持 A 时通过，保留原始失败和 semantic_adjudication 来源。
- 同类别竞争对象全部分页检查，不再截取前 32 个用于最终验收；另纳入相近类别、包含关系和邻近对象。每页默认 12 个，必须为每个 ID 返回一项对照。未排除的竞争对象、遗漏的对照或错误对象粒度均不得放行。
- 改写耗尽或无法判定，只把 phrase 标为 unresolved；已确认 object 留在对象索引。

## 阶段与恢复

默认两轮 refinement：29 stages，新增四个 `*_blind_review`。Context 规划在 phrase_generation/rewrite 中进行并缓存；不额外引入模型服务或 GPU 分配方式。

1. category_inventory
2. instance_discovery
3. object_verification
4. local_search_initial
5. local_object_verification
6. target_ocr
7. phrase_generation
8. phrase_reground
9. phrase_blind_review
10. phrase_adjudication
11. local_search_round_1
12. object_verification_round_1
13. phrase_rewrite_round_1
14. phrase_reground_round_1
15. phrase_blind_review_round_1
16. phrase_adjudication_round_1
17. local_search_round_2
18. object_verification_round_2
19. phrase_rewrite_round_2
20. phrase_reground_round_2
21. phrase_blind_review_round_2
22. phrase_adjudication_round_2
23. final_object_verification
24. final_phrase_generation
25. final_phrase_reground
26. final_phrase_blind_review
27. final_phrase_adjudication
28. finalize_objects
29. review_export

Family contract 保持 `role-grounding-v1`，新增 `phrase_review_version: 2`，storage contract 为 `role-grounding-storage-v2`。旧配置仍能解释旧 25-stage 进度，执行新版则要求 v2。旧 checkpoint 不迁移、不改写；旧审核结果不能伪装成通过新策略。新策略启动新 run，旧 run 如需续跑必须使用对应旧代码。

启动方式、1962/100 图片选择、GPU 归属、图片租约、任务分配和跨节点恢复工作模式不变。当前任务按用户要求仅在登录节点跑 CPU/mock 测试，不申请 GPU、不调用实际模型。因此这里不声明视觉质量和 GPU 性能已经验收。

## 归档和空间

最终保留对象索引、通过的 phrase 与原有审核图、unresolved phrase、简短类别搜索摘要。新增审计文本：确切对象粒度、定位条件、盲审匹配、所用 context 范围、逐项竞争判断和简短证据。归档汇总显式保留这些字段，完成校验检查策略版本和证据完整性。

不永久保存完整模型思考、历史回答、crop、mask 或 unresolved 额外图片。原有最终归档后任务目录清理继续运行。代码及 .local 的本地回滚备份保留；回滚不触碰任何 checkpoint。

## 样本对应

- 灯阵/穹顶：扩大到整组及参照物；盲审找出多个符合者，不能由红框引导后放行。
- 玩具宣传单/包装：先区分印刷人物、实物玩具和盒子，再检查黑白衣服等属性是否还适用于其他对象。
- 多个 Thor 盒子在 Black Widow 右侧：必须逐个排除，不能把一般方位解释成紧邻。
- 地面石板：bottom-left/upper-right/foreground 的宽泛区域本身不唯一；找不到真实区分条件就 unresolved，不编造编号。
- 树冠与整棵树：核对实际框对应的 referent 粒度，并检查拱门后方是否有多棵候选。
- 灯条、招牌、垃圾桶：优先补充“车顶上”“靠在餐车旁墙边”“橙色餐车左侧”等有可见依据的条件；不在本轮生成问题。
