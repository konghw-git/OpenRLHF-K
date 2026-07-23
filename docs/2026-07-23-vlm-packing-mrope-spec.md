# Spec:VLM 训练支持 sample packing(mRoPE × packing)

> 2026-07-23。前置阅读:`docs/2026-07-16-vlm-rollout-perturbation-grpo-design.md` §12.3.1(问题定性)、§13.5(非 packing 的 padding 代价)。
> 本 spec 基于对 **transformers 5.7.0 源码**(fork 钉死的版本,`requirements.txt:26`)、**OpenRLHF-K 打包全链路**、**verl 参考实现**(`docs/verl/`)的逐行核查,所有断言均带 `file:line` 证据。

## 0. TL;DR

OpenRLHF-K 的 VLM 通道禁 packing(`--data.max_images_per_prompt > 0` 断言,`openrlhf/cli/train_ppo_ray.py:618`),根因是 packing 分支只会算 1D position_ids,未重建 mRoPE 的 3D 位置。核查发现 **transformers 5.7.0 已原生内置了 verl 当年手工实现的全部机制**:4 行 position_ids 输入约定、从 position_ids 重启点自动推 cu_seqlens 的 varlen 路径、mask 式 deepstack 注入。因此方案收敛为:**在 Actor.forward 的 packing 分支预计算每样本 4 行 position_ids(text + t/h/w),随 token 一起打包,其余全部交给 HF 原生机制**。不需要 monkey-patch,不需要复刻 `get_rope_index`,预计新增/修改 ~150 行主代码 + ~200 行测试。验收标准:packing 与非 packing 前向的 per-token logprob 等价(bf16 容差内)。

> **适用边界(重要):** 本 spec 的正确性只对 **full-attention 的 mRoPE VLM(Qwen2-VL / 2.5-VL / 3-VL,C2 当前即 Qwen3-VL)** 成立。仓库同时支持训练的 **Qwen3.5 类模型(早期融合原生多模态,门控 DeltaNet 线性注意力 + MoE 的混合架构)+ packing 不在本 spec 覆盖内**——线性注意力层不吃 flash-attn 的 cu_seqlens 段切分,packing 下段间状态隔离未经核查(详见 §2 非目标 3、§5.7)。Qwen3.5 的**非 packing** 训练不受影响,仓库已支持。

## 1. 背景与动机

- C2(图像 MGT)阶段被迫关 packing,训练相按 `micro_batch_size` 逐组 padding 到组内最长序列。C1-4B 实测单步 4.5 min 中训练相占 3:13(128 次梯度累加);长尾样本会把整组 padding 撑大(实验记录 §13.5、§16)。
- **C2-4B 实测分解(2026-07-23,step≈148,单步 ~7.6 min)**:rollout+经验相 ~1:45(23%)、**训练相 ~4:10(55%,micro=2 → 每卡 128 次累加 micro-step)**、同步/切换开销 ~1:40(22%)。与 C1 相反,C2 瓶颈在训练相(MGT 生成短 ~230 tok + prefix caching 使 rollout 便宜;视觉塔解冻 + 非 packing padding 使训练重)——packing 正打在最大头上。
- packing 后训练相预期 1.5–2× 提速(去 padding + 减少累加次数),且消除"长尾 padding 使激活显存趋势性抬高"的问题(§13.5 的 8B 67.5/80G 风险)。
- verl 已证明 mRoPE × packing 可兼容;本 fork 钉的 transformers 5.7.0 让实现成本远低于 verl 当年(见 §3.1)。

## 2. 目标 / 非目标

**目标**
1. `--data.max_images_per_prompt > 0` 时允许 `--ds.packing_samples`,Qwen3-VL(mRoPE 家族)训练前向在 packing 下数值正确。
2. 覆盖两个 forward 消费点:训练步(`ppo_actor.py:294`)与经验采集 logprob(`ppo_actor.py:635`,含 reference model 走的 `launcher.py:134` 同构路径)——三者都汇聚到 `Actor.forward`,一处修改全覆盖。
3. 混合批(带图样本 + 纯文本样本混在同一 micro-batch)正确。
4. 等价性测试锁死正确性(§6)。

**非目标(显式排除)**
1. **ring attention / sequence parallel + VLM**:序列切分会把一张图的 token 劈到不同 rank,破坏 pixel_values ↔ image token 对齐。保留断言禁用。
2. **Gemma 类 VLM**(图像段内双向 attention):与 causal 的 flash_attn_varlen 原理不兼容,保留禁用。
3. **Qwen3.5 类混合线性注意力 VLM(门控 DeltaNet + MoE + 早期融合)+ packing**:本 spec 的段隔离全靠 full-attention 的 flash_attn_varlen(§3.1 T4 的 `_is_packed_sequence` → `_prepare_from_posids` → `flash_attn_varlen_func`,只作用于 full-attention 层)。而 Qwen3.5 的 decoder 是 `self_attn` + `linear_attn` **混合**(`openrlhf/models/utils.py:17-29` 有专为其 hybrid 布局写的 ZeRO-3 修复、`openrlhf/utils/vlm_utils.py:4` docstring 列其为支持目标,可证仓库确已训过该模型),线性注意力层是递推/扫描式,**不吃 cu_seqlens 段切分**;packing 拼接成 `(1,total)` 后若不在段边界重置递推状态,上一条序列的状态会泄漏到下一条。故本 spec **只覆盖 full-attention 的 mRoPE VLM(Qwen2/2.5/3-VL)**。澄清两点:(a) Qwen3.5 的**非 packing** 训练不受影响——每样本独立成行、状态天然隔离,仓库已支持;(b) MoE 与 packing 正交(逐 token 路由,§5.4),不是障碍,障碍只在线性注意力。"Qwen3.5 + packing" 的解禁路径见 §5.7,不在本 spec 范围。
4. **dynamic_batch + VLM**:v1 不开。机制上 `make_experience_batch` 对 list 字段(`mm_train_inputs`)按样本序 chain(`experience.py:251`),重排后仍对齐,大概率能直接工作,但留待 v2 单独验证。
5. critic + VLM:与 packing 无关,维持禁用(critic-free 估计器)。

## 3. 核查结论(事实基础)

### 3.1 transformers 5.7.0:verl 的三步解法已全部原生化

以下行号均指 wheel 内 `transformers/` 路径(`pip download transformers==5.7.0` 可复现):

| # | 机制 | 证据 |
|---|---|---|
| T1 | `Qwen3VLModel.get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask)` 暴露在模型类上,返回 `(3, bs, seqlen)` + deltas;支持整个 padded batch 一次调用(padding 位置初始化为 0,后续被 unpad 丢弃) | `models/qwen3_vl/modeling_qwen3_vl.py:1033` |
| T2 | **4 行 position_ids 输入约定**:`Qwen3VLTextModel.forward` 收到 `(4, bs, L)` 时拆出 `text_position_ids = position_ids[0]`(驱动 causal mask 与 FA packing 检测),`position_ids[1:]`(3 行 t/h/w)进 rotary;收到 3 行则 `text_position_ids = None`(FA 检测失效——这正是 HF 自算路径不管 packing 的原因,也说明 4 行入口就是留给打包调用方的) | `modeling_qwen3_vl.py:888-899` |
| T3 | `text_position_ids` 逐层传入 attention,两级调用:外层 TextModel→decoder_layer(`:919`,`position_ids=text_position_ids`),内层 decoder_layer→self_attn(`:548-553`)→ FA wrapper 的 `**kwargs` | `modeling_qwen3_vl.py:916-921, 548-553` |
| T4 | **FA varlen Case 1(自动推断)**:`_flash_attention_forward` 中 `_is_packed_sequence(position_ids, batch_size)`(batch==1 且位置非单调)→ `_prepare_from_posids` 用 `position_ids == 0` 切 `cu_seqlens` → `flash_attn_varlen_func`。与 verl `prepare_fa2_from_position_ids` 逐行同构 | `modeling_flash_attention_utils.py:437-472, 513-526, 745-798` |
| T5 | **FA varlen Case 2(显式 kwargs)**:`cu_seq_lens_q/k + max_length_q/k` 四者齐传即走 varlen,无需 position_ids 推断;`FlashAttentionKwargs` 从顶层 `forward(**kwargs)` 一路透传到 attention(仅进 language_model,不进 vision tower) | `modeling_flash_attention_utils.py:562-566, 744-747`;`modeling_qwen3_vl.py:1272, 867` |
| T6 | **deepstack 注入是 mask 式**:`hidden_states[visual_pos_masks] += embeds`,mask 来自 `input_ids == image_token_id`(`get_placeholder_mask`),与 batch 结构无关 → packing 安全 | `modeling_qwen3_vl.py:941-948, 1168-1190` |
| T7 | pixel_values 消费:`masked_scatter` 按 image token 出现顺序填充,packing 保序即对齐 | `modeling_qwen3_vl.py:1289-1300` |
| T8 | 显式传 position_ids 时完全绕过 `compute_3d_position_ids`(`if position_ids is None` 才自算),`mm_token_type_ids` 在 model forward 里再无其他用途(仅 `get_rope_index` 需要) | `modeling_qwen3_vl.py:1336-1345, 1209-1257` |
| T9 | FA2 下 `attention_mask=None` 时 `create_causal_mask` → `flash_attention_mask` 返回 None,不会构造显式 mask | `masking_utils.py:616-651, 733` |

### 3.2 OpenRLHF-K:现状与缺口

| # | 事实 | 证据 |
|---|---|---|
| K1 | 禁 packing 断言在 `max_images_per_prompt > 0` 分支,消息给了两个理由(pixel_values 对齐 + mRoPE position_ids)——由 T6/T7,前者在保序打包下自动成立;后者即本 spec 要解的 | `openrlhf/cli/train_ppo_ray.py:618-628` |
| K2 | packing 分支:`unpad_and_slice_tensor` 只算 1D `cumsum(attention_mask)-1`,`ring_attn_group=None` 时纯打包成 `(1, total)`、返回 `indices` 供 `gather_and_pad_tensor` 恢复 `(batch, seqlen)`;cu_seqlens 算了但只在 ring-attn 时使用,非 ring 场景靠 HF Case 1 自动推断(文本 packing 现状即如此) | `openrlhf/models/ring_attn_utils.py:103-148, 151-182` |
| K3 | 非 packing VLM 分支:`position_ids=None` 让模型自算;从全序列(prompt+response)重建 `mm_token_type_ids`(image=1, video=2)——该重建逻辑已被 C2 训练验证,packing 分支可直接复用 | `openrlhf/models/actor.py:246-260` |
| K4 | `rolled_sequences` 在 **打包前** 按行 roll 再 unpad,每段的 next-token 目标不跨段污染;段尾 token 的目标是 pad(被 action_mask 掩掉) | `ring_attn_utils.py:130-135` |
| K5 | `mm_train_inputs` 每样本一个 dict,仅含 `{pixel_values, image_grid_thw}`(token-type 字段显式排除、forward 时重建);`merge_mm_train_inputs` 按样本序 concat → 打包保序时对齐自动成立 | `openrlhf/utils/vlm_utils.py:113-117, 154-170` |
| K6 | Experience 的 list 字段在 split/make batch 时逐样本对齐(`value[i]` / chain),`balance_experiences` 重排后仍对齐 | `openrlhf/trainer/ppo_utils/experience.py:184-253` |
| K7 | 三个 forward 消费点(训练步 / actor logprob / ref logprob)全部汇聚到 `Actor.forward` | `ppo_actor.py:294, 635`;`launcher.py:134-143` |
| K8 | **既有 bug**:`dynamic_batch_enable`(`:647-650`)和 `ring_attn_size>1`(`:642-645`)会在 VLM 断言(`:618`)**之后**强制 `packing_samples=True`,静默绕过断言 → 本次一并修 | `train_ppo_ray.py:618-650` |
| K9 | `Actor.forward` 内 `self.model` 是裸 HF 模型(DS 在 Actor 外层包);`_vlm_config` 已在包装前缓存 | `actor.py:212-214` |

### 3.3 verl 交叉参照(`docs/verl/`)

| # | 事实 | 证据 |
|---|---|---|
| V1 | `get_rope_index` 是复刻版,docstring 明言"generated **before sharding** the sequence"、目标 transformers 4.57——5.7.0 下无需复刻(T1) | `verl/models/transformers/qwen3_vl.py:99-111` |
| V2 | 位置随数据流动的形状是 `(bs, 4, seq_len)`——四分量语义(text + t/h/w)与 T2 相同,但**轴序转置**(verl batch 在前,HF 约定 `(4, bs, L)` 分量在前),实现时不可直接照搬形状 | `verl/workers/utils/padding.py:47` |
| V3 | `prepare_fa2_from_position_ids`(`position_ids==0` 切 cu_seqlens → varlen)与 T4 的 HF 原生实现同构——证明该机制被 verl 生产验证过 | `verl/models/transformers/qwen2_vl.py:164-179, 228-240` |
| V4 | verl 的 deepstack/visual embeds 聚合逻辑与 HF 5.7.0 相同(mask 式) | `verl/models/transformers/qwen3_vl.py:205-297` |

## 4. 方案设计

### 4.1 核心思路

**打包前逐 micro-batch 预计算 4 行 position_ids,打包时随 token 同步 unpad,forward 显式传入;段边界识别交给 HF 原生 Case 1。**

```
非 packing(现状,保留):          packing(新增):
sequences (B, L) padded            sequences (B, L) padded
position_ids = None                ① mm_token_type_ids ← 从 sequences 重建(复用 K3 逻辑)
model 自算 3D 位置                  ② mrope (3,B,L) ← model.get_rope_index(...)   [T1]
                                   ③ text  (1,B,L) ← cumsum(attention_mask)-1, clamp(0)
                                   ④ pos4 = cat([text, mrope]) → (4,B,L)
                                   ⑤ unpad_and_slice_tensor 打包 token → (1,total), indices
                                   ⑥ pos4 用同一 indices 打包 → (4,1,total)
                                   ⑦ model(seq, attention_mask=None, position_ids=pos4,
                                            pixel_values, image_grid_thw)
                                      → HF 拆出 text 行 [T2] → attention 收到重启式 1D 位置
                                      → _is_packed_sequence 命中 → varlen [T4]
                                      → rotary 用 t/h/w 3 行;deepstack/pixel_values 走 mask [T6/T7]
```

关键性质:
- **text 行(row 0)每段从 0 重启**(right-padded 下 `cumsum(mask)-1` 首 token 恒为 0)→ Case 1 的 `position_ids==0` 切分点恰是段边界。单样本 micro-batch(只有一段)位置单调 → 走非 varlen 的整行 causal FA,同样正确。
- **纯文本样本混包**:该样本段内 mrope 3 行 = text 行广播(`get_rope_index` 对无图段的 fallback 行为,`modeling_qwen3_vl.py` 文本段处理;与非 packing 时模型自算结果一致)。整个 micro-batch 无图时跳过 ②,`pos4[1:] = text 行 expand(3)`。
- **logprob/entropy 恢复路径零改动**:`gather_and_pad_tensor` + `indices` 与位置无关(K2、K4)。

### 4.2 逐文件改动

#### (a) `openrlhf/models/actor.py` — 主改动(~70 行)

1. `__init__`:`is_vlm and packing_samples` 时解析并缓存 `get_rope_index`。**方法只定义在 `Qwen3VLModel` 上**(`modeling_qwen3_vl.py:1033`;`ForConditionalGeneration` 自身不定义,其 `:1604` 也是转调 `self.model.get_rope_index`),且 LoRA 时 PEFT 的 `__getattr__` 委托会让 `self.model.model` 解析到 `ForConditionalGeneration` 而非 `Qwen3VLModel` → 写死属性链会在 LoRA+VLM+packing 下 AttributeError(审稿 M1)。用包装无关的下钻循环:

   ```python
   inner = self.model
   while inner is not None and not hasattr(inner, "get_rope_index"):
       inner = getattr(inner, "model", None)   # PeftModel→LoraModel→ForCondGen→Qwen3VLModel 逐层下钻
   if inner is None:
       raise ValueError("packing_samples requires an mRoPE VLM exposing get_rope_index (e.g. Qwen2/3-VL); "
                        "Gemma-style VLMs are not supported")
   self._get_rope_index = inner.get_rope_index
   ```
   解析失败即启动时 raise,把不兼容(如 Gemma 类)暴露在启动时而非训练中。
2. 把 K3 的 `mm_token_type_ids` 重建逻辑提为私有方法 `_build_mm_token_type_ids(sequences)`(两分支共用)。
3. `forward` packing 分支,在 `unpad_and_slice_tensor` 调用前后插入:

```python
if self.packing_samples:
    vlm_pos4 = None
    if getattr(self, "is_vlm", False):
        assert ring_attn_group is None, "VLM packing does not support ring attention"
        token_type_ids = self._build_mm_token_type_ids(sequences)          # (B, L)
        text_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)    # (B, L)
        if mm_inputs.get("image_grid_thw") is not None or mm_inputs.get("video_grid_thw") is not None:
            mrope_pos, _ = self._get_rope_index(
                sequences, token_type_ids,
                image_grid_thw=mm_inputs.get("image_grid_thw"),
                video_grid_thw=mm_inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
            )                                                              # (3, B, L)
        else:
            mrope_pos = text_pos.unsqueeze(0).expand(3, -1, -1)
        vlm_pos4 = torch.cat([text_pos.unsqueeze(0), mrope_pos], dim=0)   # (4, B, L)

    sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
        sequences, attention_mask, ring_attn_group
    )
    if vlm_pos4 is not None:
        position_ids = pack_position_ids(vlm_pos4, indices)               # (4, 1, total)
    foward_attention_mask = None
```

注意:`get_rope_index` 是纯 tensor 运算(只读 config,不触参数),ZeRO-3 下安全(K9);原始 `sequences/attention_mask` 需在打包覆写前引用。

#### (b) `openrlhf/models/ring_attn_utils.py` — 新增 helper(~20 行)

```python
def pack_position_ids(pos4, indices):
    """(4, B, L) → (4, 1, total):对每行用与 token 相同的 indices 做 index_first_axis。"""
```
用 `rearrange(pos4, "r b l -> (b l) r")` + `index_first_axis` 一次完成——`unpad_input` 的 `indices` 来自 `attention_mask.flatten()`(行主序),与该 rearrange 的展平序严格一致(审稿已独立核对)。helper 末尾加两句防御断言:`assert attention_mask[:, 0].all()`(right-padding 不变量,§5.3)与 `assert packed_pos4[0, 0, 0] == 0`(打包后首段文本位置从 0 起,Case 1 切分的前提)。

#### (c) `openrlhf/cli/train_ppo_ray.py` — 断言精细化(~15 行)

1. `max_images_per_prompt > 0` 分支:删除"禁 packing"断言,替换为组合约束——
   - `packing_samples` 允许;
   - `packing_samples and ring_attn_size > 1` → 报错;
   - `packing_samples and dynamic_batch_enable` → v1 报错(消息注明 v2 计划);
2. **修 K8 顺序 bug**:把 VLM 组合校验移到 `ring_attn`/`dynamic_batch` 强制改写 `packing_samples` 之后(或在改写处复查),杜绝静默绕过。
3. 模型家族门禁(Gemma 类)不在 CLI 做(拿不到模型),由 (a).1 的启动时解析兜底。

#### (d) 文档与脚本

- `examples/scripts/train_vlm_math_hybrid_engine.sh` 加注释说明 VLM 可开 `--ds.packing_samples` 及其约束。
- 实验记录 §12.3.1 追加"已实现"注记(实施后)。

**不改**:`unpad_and_slice_tensor` 签名、`gather_and_pad_tensor`、`merge_mm_train_inputs`、Experience/replay buffer(K5/K6 已天然兼容)、experience_maker、vLLM rollout 侧(packing 只影响训练前向)。

### 4.3 设计决策与已否决的替代方案

| 决策 | 选择 | 否决项与理由 |
|---|---|---|
| 段边界传递 | **Case 1(HF 自动推断)** | Case 2(显式 `cu_seq_lens_*` kwargs,T5)更"显式",但要改 `unpad_and_slice_tensor` 返回值或重算 cu_seqlens,且 Case 1 是文本 packing 现状已依赖的同一机制(K2),行为一致性更好。若实测遇到 Case 1 判定问题,Case 2 是现成的 plan B(四个 kwargs 齐传即切换,T5 已核实透传通路)。 |
| position_ids 计算时机 | **Actor.forward 内、逐 micro-batch** | 在 experience 阶段预计算并存入 Experience:省重复计算(actor/ref/训练各算一次),但要给 Experience 加 `(4,L)` 字段、过 split/make/balance 全链路,改动面大;且 `get_rope_index` 开销相对前向可忽略。 |
| get_rope_index 来源 | **模型自带方法**(T1) | 复刻(verl 路线):版本无关但要维护副本;5.7.0 已内置且与模型行为定义一致(版本匹配优于版本无关)。 |
| monkey-patch attention | **不做** | T3/T4 证明原生通路完整;patch 是 4.x 时代的必需品(V1)。 |

## 5. 边界与约束

1. **`micro_batch=1`**:单段、位置单调 → 走整行 causal FA,正确(§4.1)。
2. **视频**:`video_token_id`/`video_grid_thw` 通路与图像对称(K3 已建 type=2;(a) 代码已带 `video_grid_thw`),但当前实验只有图像,视频等价性测试列为 TODO 不阻塞。
3. **左 padding**:全链路是 right-padding(`remove_padding_in_sequences` + `zero_pad_sequences(side="right")`),text 行"首 token 位置=0"依赖此;helper 内加一句 `assert attention_mask[:, 0].all()` 防御。
4. **Qwen3-VL MoE**:packing 不改变 expert 路由语义(逐 token);MoE 变体不需要额外处理。
5. **freeze_visual_encoder / LoRA / 梯度检查点 / EMA**:与位置计算正交;LoRA 仅影响 (a).1 的属性解析路径。
6. **transformers 版本漂移**:方案依赖 T1/T2/T4 三个 5.x 契约。fork 已钉 `==5.7.0`;若升级,等价性测试(§6)是回归门。
7. **模型家族边界(full attention vs hybrid 线性注意力)**:正确性证据(§3.1 T1–T9)全部取自 `modeling_qwen3_vl.py` 的 **full-attention** 路径,§6 等价性测试也用纯 full-attention 的 `Qwen3VLForConditionalGeneration`。因此本 spec 仅对 **full-attention mRoPE VLM** 成立(C2 用的 Qwen3-VL 属此类,直接适用)。**Qwen3.5 类混合线性注意力模型不在覆盖内**(理由见 §2 非目标 3):其 `linear_attn` 层在 packing 下的段间状态隔离未经核查,且 §6 用 full-attention 模型跑,测不到该泄漏。若将来把训练模型切到 Qwen3.5 并想开 packing,解禁前须:(a) 核对该模型 HF modeling 实现(非 `modeling_qwen3_vl.py`,应为带门控 DeltaNet 的 modeling 文件)的线性注意力 kernel 是否从与 full-attention 同一信号接收 `cu_seqlens`/段索引以在段边界重置递推状态;(b) 在真实 hybrid checkpoint(小配置)上新增"线性注意力跨段不泄漏"的白盒等价性断言(不能用 §6.1 的 full-attention 模型替代)。**安全性提醒**:§4.2(a).1 的启动门禁只按 `get_rope_index` 是否存在来拒 Gemma,而 Qwen3.5 是 mRoPE、**带 `get_rope_index`,会通过该门禁**——即当前设计**拦不住 Qwen3.5 + packing**,是静默出错(数值错但曲线未必立刻崩)而非启动报错。解禁前若暂不支持,应在门禁处显式加一条 hybrid/线性注意力检测(如 decoder 层是否含 `linear_attn` 子模块)并 raise,与 §2 非目标 3 一致。
8. **Multi-Token Prediction(MTP)与本 spec 正交,无需处理**(2026-07-23 核查)。Qwen3.5 血统(Qwen-Next / DeepSeek 系)自带 MTP 模块(HF config `num_nextn_predict_layers`),但它与 packing 无交互:
   - **RL 前向只走主 `lm_head`**:`Actor.forward` 只读 `output["logits"]`(`actor.py:267,285`),MTP 头不在梯度路径上;仓库全局无 MTP 处理(`grep -riE "mtp|nextn|multi_token|speculat"` 零命中)。
   - **verl 权威文档逐条印证**(`docs/verl/docs/advance/mtp.md`,美团,2026-02-15):§3 明列"base model 带 MTP 参数但**不训练** MTP"对训练结果**无显著影响**,唯一有影响的是"MTP loss 施加到全部参数 + `mtp_loss_scaling_factor=0.1`"。OpenRLHF-K 不开任何 MTP loss,等价于无影响场景。
   - **硬约束(记录防坑)**:verl §1 明确 **MTP *训练* 只支持 Megatron(mbridge/Megatron-Bridge + megatron),其它训练后端不兼容**。OpenRLHF-K 是 **DeepSpeed** 后端,即便将来想训 Qwen3.5 的 MTP 也需换栈,非增量改动。
   - **rollout 侧**:MTP 投机解码提升接受率 ~14%,但弱卡(H20)常致吞吐不升反降(verl §4:mimo-7B/H20/SGLang 降 ~50%),verl 当前建议不开;OpenRLHF-K 未配置 speculative decoding,现状无关。
   - **本机未能核实(需上训练机)**:transformers 5.7.0 加载 Qwen3.5 时是否**默认实例化** MTP 层(本机未装 transformers,`requirements.txt:26` 钉 `==5.7.0`)。若实例化且未被 `freeze_visual_encoder`(`actor.py:171-174`,冻结不含 `language_model`/`lm_head` 的参数)覆盖,MTP 参数会成为**零梯度可训练参数**,占 ZeRO 优化器状态并可能绊 ZeRO-3 未用参数 hook(与 hybrid Z3 leaf bug 同族,`models/utils.py:23-29`)。上机核查命令:加载后跑 `[n for n,_ in model.named_parameters() if 'mtp' in n.lower() or 'nextn' in n.lower()]`,空 = 默认不载入(最干净,连显存都不占),非空 = 需 freeze 掉或 `del` MTP 模块再包 DeepSpeed。

## 6. 验证方案(验收标准)

新增 `tests/test_vlm_packing_equivalence.py`,需在训练服务器(CUDA + flash-attn)跑:

1. **前向等价(核心)**:随机小配置 Qwen3-VL(2 层、小 hidden,`Qwen3VLForConditionalGeneration(config)` 随机权重,bf16)+ 构造 2–4 样本 micro-batch(含 1 个纯文本样本、不同尺寸假图);同一 `Actor` 分别以 `packing_samples=True/False` 前向,断言 per-token logprob `max|Δ| < 2e-3`(bf16 + varlen kernel 差异容差,为初始目标值,需实测标定——若 fp32 组通过而 bf16 超差,放宽容差并把实测差值范围记录进测试注释;fp32 下另跑一组 `< 1e-5` 排除实现性错误,**fp32 对照组不可省略**)。
2. **位置等价(白盒)**:packing 路径打包前的 `pos4[1:]`(逐样本)与非 packing 时模型自算的 `compute_3d_position_ids` 输出逐元素相等——把"位置对不对"与"kernel 数值差"分开归因。
3. **梯度等价**:同一 batch 两条路各做一次 backward,对比 language_model 若干层梯度范数(容差同上)。
4. **断言回归**:`ring_attn+VLM+packing`、`dynamic_batch+VLM+packing`、Gemma 类模型 + packing → 均应在启动期报错(修复 K8 后的顺序敏感用例)。
5. **冒烟训练**:C2 配方 + `--ds.packing_samples`,跑 ~10 step:reward/kl/grad-norm 曲线与非 packing 基线重合(小随机差),单步训练相耗时对比记录进实验文档。

1–4 过 = 功能正确;5 过 = 可上生产。

## 7. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| Case 1 判定在某些边角(如全 batch 单样本恰好非单调?)行为意外 | 低 | 测试 §6.1 覆盖 micro_batch=1;plan B 切 Case 2(§4.3) |
| bf16 下 varlen 与 padded kernel 数值差超容差,影响 PPO ratio | 低 | fp32 白盒测试归因;容差实测标定;PPO 本身对 old/new logprob 同路径计算,系统差抵消 |
| DS ZeRO-3 + `get_rope_index` 属性解析在 engine 包装下失效 | 低 | K9 已核实 Actor 内是裸模型;启动时解析、失败即 raise |
| deepstack 的 in-place `hidden_states[mask] += embeds` 与梯度检查点交互 | 极低 | 该路径非 packing C2 已在跑,packing 不改变此代码路径 |
| 收益不及预期(C2 瓶颈已移到 rollout) | ~~中~~ **已排除** | 2026-07-23 实测(§1):训练相占 55%,远超 30% 门槛。按 packing 训练相 1.5–2× 估算,单步 7.6 → ~5.5–6.2 min,端到端 ~1.2–1.4× |

回退:纯增量改动,关掉 `--ds.packing_samples` 即回到现状;不触碰非 packing 分支任何行为。

## 8. 工作量与实施顺序

| 步骤 | 内容 | 量级 |
|---|---|---|
| 0 | ~~从 C2 日志量出两相占比,确认收益~~ **✅ 已完成(2026-07-23):训练相 55%,闸门通过** | — |
| 1 | (b) helper + (a) actor.forward 改动 | ~90 行,0.5 天 |
| 2 | §6.1–6.3 等价性测试(先写测试,TDD) | ~200 行,0.5 天 |
| 3 | (c) CLI 断言 + §6.4 回归 | ~30 行,1h |
| 4 | §6.5 冒烟 + 吞吐记录 | 1 天(挂机) |

总计:~2 个工作日 + 1 天冒烟挂机。建议在 C2 当前 run 结束后实施,不中途换引擎(与实验记录 §12.3.1 的取舍一致)。

## 9. 审查记录

- **2026-07-23,模型家族边界澄清(补记)**:核对仓库 VLM 现状发现 `openrlhf/models/utils.py:17-29`(为 hybrid 布局写的 ZeRO-3 修复,含"hook 偷走 ~390/417 个内部参数梯度"的具体 bug 记录)与 `openrlhf/utils/vlm_utils.py:4`(docstring)把 **"Qwen3.5"**(早期融合原生多模态,门控 DeltaNet 线性注意力 + MoE 的 hybrid 架构)列为已支持的训练目标——即仓库确已在**非 packing** 路径训过该模型。据此明确本 spec 的适用边界:packing 机制仅对 **full-attention 的 mRoPE VLM(Qwen2/2.5/3-VL)** 成立;"Qwen3.5 + packing" 是 Gemma(段内双向)之外的**第二类 packing 未必安全的结构**(线性注意力段间状态隔离),已在 §0 边界、§2 非目标 3、§5.7 补充说明并给出解禁前置核查。当前实验(C2)用 Qwen3-VL,属 spec 直接适用范围,不受影响。
- **2026-07-23,MTP(Multi-Token Prediction)核查(补记)**:核对 Qwen3.5 血统的 MTP 是否影响本 spec。依据 `docs/verl/docs/advance/mtp.md`(美团权威指南)+ OpenRLHF-K 源码,结论:MTP 与 packing 正交、对本 spec 无影响(详见 §5.8)。关键事实:(a) RL 前向只走主 `lm_head`,仓库全局无 MTP 处理;(b) verl §3 证实"带 MTP 参数但不训练"对训练结果无显著影响;(c) 硬约束——verl §1 明确 MTP *训练* 仅支持 Megatron,DeepSpeed 系(含本仓库)不兼容;(d) 唯一未核实项(本机未装 transformers):Qwen3.5 加载时是否默认实例化 MTP 层,已在 §5.8 给出上机核查命令。C2(Qwen3-VL)无 MTP 模块,完全不受影响。
- **2026-07-23,Opus 4.8 对抗性审查**(独立读源码逐条核验,抽查事实表 8/18 条全部属实):1 条 MAJOR——LoRA 包装下 `get_rope_index` 属性链断裂(已修,§4.2(a).1 改为下钻循环);2 条 MINOR——V2 轴序表述、T3 行号引用(已修);2 条 QUESTION——bf16 容差需实测标定(已注入 §6.1)、packing 后位置断言加固(已注入 §4.2(b))。审查同时独立确认:打包展平序与 `unpad_input` indices 严格一致、pixel_values/image_grid_thw/get_rope_index 三处消费顺序一致、forward 消费点无遗漏(EMA 不做前向)、`get_rope_index` 无参数访问(ZeRO-3 安全)、deepstack `_deepstack_process` 内有 `hidden_states.clone()`(:946,非 in-place,梯度检查点安全)。结论:核心机制正确,修复 M1 后可进入实施。
