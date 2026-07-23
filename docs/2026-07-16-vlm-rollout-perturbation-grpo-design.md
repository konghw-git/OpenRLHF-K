# 设计文档:输入扰动作为小模型视觉推理 RL 的高效探索手段

> 诊断小模型 VLM RL 的"优势坍缩",并系统比较补救方案(以 NoisyRollout 的输入扰动为核心候选)

- 日期:2026-07-16
- 类型:RL 算法 / 训练系统 经验研究(empirical / systems paper)
- 落地框架:OpenRLHF-K(v0.10.4,原生 VLM GRPO)
- 状态:M0 通路冒烟已过;**C0 原始基座评测(Qwen3-VL-4B + 8B,全 8 基准)已完成**(2026-07-17);C1–C3 待训练。评测工具链已就绪(`kong_rl/eval/`)。进度与发现见 §10,目录/运行约定见 §11。

---

## 1. 背景与动机

小模型(3B–8B)VLM 的强化学习(GRPO 系)常常"训不动":大量 prompt 的一组 rollout 奖励**全相同**(全对或全错)。GRPO 的优势 = 组内奖励减均值除标准差,一旦组内奖励同质,**优势恒为 0,该样本不产生任何梯度**。在小模型 + 数据稀缺的设定下(正是 LMM-R1 的处境),这种"退化组"占比可能很高,导致翻倍的 rollout 算力被浪费。

两篇代表性工作各自触及了这个问题的一面,但都没把它当作核心问题系统研究:
- **LMM-R1**(arXiv 2503.07536):两阶段规则型 RL(文本 FRE → 多模态 MGT)提升 3B LMM 推理。贡献在**数据/课程**。
- **NoisyRollout**(NeurIPS 2025):每个 prompt 生成 `n` 条干净图 + `n` 条高斯加噪图的 rollout,放进**同一 GRPO 组**,用噪声制造 rollout 多样性提升探索;带 sigmoid 噪声退火。贡献在**采样/探索**。

**读码发现的关键留白(本项目的立足点)**:NoisyRollout 把额外一倍算力**几乎全用在"扰动组基线"上**——加噪 rollout 生成和更新都用噪图(纯 on-policy)、对噪声视图**无一致性约束、无重要性修正、无基于奖励的过滤**,连标记 clean/noisy 的 `image_status` 字段都写了从未被读。也就是说,"输入扰动"目前只是一个**未被论证是否最优**的探索手段。

## 2. 核心研究问题与假设

- **RQ1(诊断)**:小模型 VLM RL 中,优势坍缩(同质奖励组占比)到底多严重?它如何随模型规模、训练阶段、数据难度变化?
- **RQ2(比较)**:同样的额外 rollout 预算,用来"救活退化组、注入有效梯度",哪种手段最划算?
- **RQ3(机制)**:NoisyRollout 的收益,主要来自"扰动带来的奖励方差/组多样性",还是来自"视觉鲁棒性"本身?
- **假设 H**:输入扰动之所以有效,主要是因为它以极低成本给退化组注入奖励方差;因此**只对退化组做定向扰动**(而非全局盲扰),能以更低成本取得相当或更好的收益。

## 3. 相关工作与定位

- 与 **NoisyRollout** 的差异:我们不把它当"数据增强方法"复现,而是**把它放进一个更大的设计空间**,并回答"它是不是这笔预算的最优花法"。
- 与 **DAPO / dynamic sampling**(用重采样/过采样绕开零方差组)的差异:我们对比**输入扰动 vs 重采样**这两类补救,在**小模型 + 多模态**设定下谁更高效——这是现有工作没有系统比过的。
- 与 **LMM-R1** 的关系:借用其两阶段 recipe 与数据作为**测试床**,考察扰动式探索与"文本先训"是否可组合增益。
- **诚实定位**:算法新颖度中等,卖点是**问题诊断 + 严谨的比较研究 + 一个机制驱动的轻量变体**,以及在主流框架(OpenRLHF-K)上的开源实现。属于扎实的 empirical/systems 论文,不是"惊艳新算法"。

## 4. 方法

### 4.1 框架与基座
- 框架:**OpenRLHF-K** 原生 VLM 通道(`AutoModelForImageTextToText` + AutoProcessor,`position_ids=None` 让模型自算 mRoPE;VLM 禁用 critic/packing,用 critic-free 估计器)。**已核对源码**:VLM 自动经 HF config 的 `vision_config` 探测(`openrlhf/utils/utils.py:is_vlm_model`),`position_ids=None` 在 `openrlhf/models/actor.py:246`;禁 critic/packing 的断言在 `openrlhf/cli/train_ppo_ray.py:618`,**触发条件是 `--data.max_images_per_prompt > 0`**(不是模型探测),默认 0 = 纯文本。参考脚本 `examples/scripts/train_vlm_math_hybrid_engine.sh` 与本设计几乎逐行对得上,是第一手模板。
- **CLI 是嵌套点号命名空间**(不是扁平 flag,也不是 lmm-r1 的旧写法):`--algo.advantage.estimator`、`--data.prompt_dataset` / `--data.input_key` / `--data.label_key` / `--data.image_key` / `--data.apply_chat_template`、`--rollout.n_samples_per_prompt`、`--reward.remote_url`、`--reward.normalize_enable`、`--data.max_images_per_prompt`、`--actor.freeze_visual_encoder`。**直接抄 `docs/lmm-r1/examples/scripts/` 的命令会解析失败**;写命令前一律以本地 `--help` 为准。
- **分阶段 VLM 开关**:一阶段文本 FRE(deepscaler 无图)→ `--data.max_images_per_prompt 0`(纯文本,estimator 仍需 `n>1` 走 critic-free);二阶段多模态 MGT → `--data.max_images_per_prompt ≥1`(此时才触发 VLM 禁 critic/packing 断言,并设 vLLM `limit_mm_per_prompt`)。
- 基座模型:**Qwen3-VL-4B**(主),**Qwen3-VL-8B**(规模消融)。均已在本地 `models/`(全程只用 Qwen3 系列,不使用 Qwen2.5)。
- 算法:`group_norm`(GRPO)/ `reinforce_baseline`(`experience_maker.py:264-272`)。**规则奖励要自己写 `examples/python/*.py` 的 `reward_func`**——lmm-r1 的 HTTP `math_verifier` server 在本仓库**不存在**(无 `openrlhf/models/remote_rm/`),不可复用;OpenRLHF-K 自带的 `examples/python/math_reward_func.py` **只抽 `\boxed{}`**,而一阶段 deepscaler 的答案是 `$...$` 无 boxed(会导致每条 reward 恒 0、静默空训),故需写**统一奖励**:兼容 `$...$` 与 `\boxed{}`(或直接抽 `<answer>...</answer>` 内文,两阶段数据都带该标签),再复用 `openrlhf/utils/math_utils.py` 的 `grade_answer` 判分。

### 4.2 优势坍缩诊断(RQ1,先做)
在标准 GRPO 训练中记录每步:
- 退化组占比(std≈0 的**组**占比),按"全对/全错"细分。**注意实现细节**:OpenRLHF-K **没有 uid**,优势分组是**纯位置式**的——`experience_maker.py:252` `rewards.reshape(-1, args.rollout.n_samples_per_prompt)`,每连续 `n_samples_per_prompt` 条即一组。诊断按此 reshape 后逐组算 std 即可。
- 有效梯度样本占比、组内奖励方差分布;
- 随训练步、模型规模、数据难度的变化曲线。
这套诊断本身就是一份别人没细报的量化结果。

### 4.3 输入扰动 rollout(把 NoisyRollout 移植进 OpenRLHF-K)
**关键实现约束(已核对源码,比"数十行"重)**:
- **分组靠位置连续,不是共享 uid**:要把 `n` 干净 + `n` 扰动放进同一 GRPO 组,必须让它们在一个长度 `2n` 的**连续块**里,并把 `--rollout.n_samples_per_prompt` 设为 `2n`;`experience_maker.py:252` 的 reshape 自然成组,**分组侧无需改代码**(这一步反而是最省事的)。
- **`n` 是在 vLLM 引擎内部循环生成的,且一 prompt 只带一张图**:`samples_generator.py:_dispatch_prompts_to_vllm` 每 prompt 只传单个 `img` + `num_samples=n`,引擎内 `ray/vllm_engine.py:204` `for _ in range(num_samples)` 复制采样——所以一个 prompt 的 n 条共享**同一张图**。做 clean+noisy 需把同一 prompt **dispatch 两次**(干净图 `num_samples=n`、噪图 `num_samples=n`,保持两块连续),或扩 `generate_responses` 支持 per-sample 图。
- **噪声要在 tokenize 之前打进 PIL,并同时流入 rollout 与训练侧**:噪图 PIL 必须写进 `response["images"]` / `response["mm_train_inputs"]`(`samples_generator.py:306` 一带),让 actor/ref 前向也在噪图上算(NoisyRollout 对噪声视图是纯 on-policy)。
- **sigmoid 退火需要 global step,当前没往下传**:要把训练步下传到 `samples_generator`;退火/加噪逻辑可直接移植 `docs/NoisyRollout/verl/utils/image_aug.py`(扩散式 `q_x` + sigmoid 退火,自包含 torch/PIL,含 `gaussian_noise_step`/`decay_mode`/`decay_coef`/`decay_sig_mid_step` 等旋钮)。
- 扰动类型:高斯为主,裁剪/旋转/分辨率作消融。
- 这是 4.4/4.5 所有实验的公共底座;移植面集中在 `samples_generator.py`(dispatch 两次 + 加噪 + step 下传),分组与优势归一不动。

### 4.4 对照方案(设计空间,RQ2)
在**相同的额外 rollout 预算**下对比:
- (a) 单纯多采 `2n` 条干净 rollout;
- (b) 升 temperature 的干净 rollout;
- (c) DAPO 式 dynamic sampling / 过采样(丢弃零方差组、重采直到有方差);
- (d) NoisyRollout 式全局输入扰动(高斯);
- (e) 其他扰动类型(裁剪/旋转/分辨率)。
**这是 do-or-die 对照**:若扰动打不过 (a)/(b),则原故事不成立。

### 4.5 机制驱动的轻量变体(RQ3 + H)
- **定向扰动(Targeted Perturbation)**:只对"预测会坍缩/低方差"的 prompt 施加扰动,其余正常采样,省算力。
- **难度/方差自适应噪声**:用组内奖励方差或历史正确率决定每样本噪声强度(替代全局时间退火)。
- 消融:扰动的收益在多大程度上被"奖励方差注入"解释(把扰动 rollout 的答案强制按干净图重算 reward,隔离"鲁棒性"贡献)。

## 5. 数据与评测

### 5.1 训练数据(本地已有)+ 转换脚本(比"三列"复杂,已核对 loader 源码)
本地 `deepscaler_message.jsonl`(文本 FRE)、`mathv60k_message.jsonl`、`mathv_geo_message.jsonl`(多模态 MGT)是 **lmm-r1 格式**,与 OpenRLHF-K 的 loader(`openrlhf/datasets/prompts_dataset.py`)期望在**每个维度**都不一样,必须写转换脚本处理以下四点:
- **`message` 是 stringify 的 JSON 字符串,必须 `json.loads` 成 list**:loader `preprocess_data` 里 `chat = data[input_key]`,若 chat 是 str 会被整段当成**一句用户文本**包起来(`prompts_dataset.py:28-30`),对话结构全丢。转换后写进 `--data.input_key` 指定的键(如 `prompt`),值是消息 list。
- **图路径要从 content 内上提到顶层 `images` 键**:loader 从**顶层独立 `image_key`(默认 `images`)** 读图(`prompts_dataset.py:78,88`),而本地数据把路径嵌在 content 里(`{"type":"image","image":"/path"}`)。转换需把每个路径**上提**成顶层 `images: [...]` 列表,content 里留裸 `{"type":"image"}`。
- **绝对路径要重写到本地解压目录,且两个数据集规则不同**:mathv60k 路径形如 `/apdcephfs_gy2/share_302735770/.../mathv60k_img/...`(绝对集群路径),mathv_geo 是相对 `data/mathv_geo/mathv_geo_img/...`;tar(`mathv60k_img.tar.gz` 2.4G、`mathv_geo_img.tar.gz`)解压到顶层 `mathv60k_img/` 与 `mathv_geo_img/`。转换脚本需**按数据集**各写一条前缀重写规则,指向本地解压位置。解压前先 `df -h /ThetaAI`(共享盘 98%,CLAUDE.md 约束)。
- **答案格式两阶段不一致**:deepscaler 的 `answer` 是 `$...$`(无 boxed),mathv60k 是 `$\boxed{}$`,均在 `<answer>...</answer>` 内 —— 对应 §4.1 的统一奖励抽取方案。转换时把 `answer` 落进 `--data.label_key`(如 `label`)。
- 目标 schema(参考 `examples/scripts/train_vlm_math_hybrid_engine.sh`):`{"prompt": [<消息list>], "images": ["/abs/local/path.png"], "label": "..."}`;启动加 `--data.input_key prompt --data.label_key label --data.image_key images --data.apply_chat_template`。

### 5.2 评测框架:统一走 EvalScope
所有对外报告的基准评测**统一用 EvalScope**(魔搭 ModelScope 的开源评测框架,离线文档见 `OpenRLHF-K/docs/official-docs/evalscope.readthedocs.io/`),取代原先零散的 NoisyRollout eval 脚本,理由:
- **可复现、可写进简历**:一条 `evalscope eval` 命令 + 一份 config 就能锁定 model / datasets / generation-config / 采样次数,结果表格与 JSON 落盘,便于横向对比多个 checkpoint。
- **原生覆盖本项目需要的多模态数学/感知基准**(见 5.3),VLM 通道成熟;文本基准同框架一并跑,验证"文本推理不退化"。
- **一致的推理后端**:评测可用 vLLM 后端或 OpenAI-API service 模式起本地权重,和训练侧 rollout 引擎对齐,减少"评测口径与训练不一致"的偏差。
- 本地已有的 NoisyRollout eval 套件(`datasets/noisyrollout_evaluation_data/eval_data.zip`)保留作**交叉校验**:同一 checkpoint 两套口径都跑一遍,确认分数一致后再以 EvalScope 为准对外汇报。

> 写评测脚本前遵循 CLAUDE.md「官方文档与写训练脚本的流程」:先查 `evalscope.readthedocs.io` 的 `get_started/`(quick start、parameters)与 `get_started/supported_dataset/vlm.html`,再核对本地 EvalScope 版本的 `--help` 与数据集名,最后动手写。

### 5.3 评测基准分层(每个领域选两个代表性基准,共 8 个,克制而够用)
| 领域 | 选定的两个基准 | 为什么是这两个 |
|------|--------------|--------------|
| **专项 · 多模态数学/几何推理**(主线指标) | **MathVista**、**MathVision** | 直接对标 LMM-R1/NoisyRollout 与旧简历口径,社区最认;一个偏综合视觉数学、一个偏竞赛难度 |
| **专项 · 幻觉与视觉忠实度**(盯扰动会不会引入幻觉) | **HallusionBench**、**POPE** | 扰动式探索的最大风险是幻觉;HallusionBench 测视觉误导推理、POPE 测物体存在性幻觉,互补 |
| **通用 · 多模态综合能力**(看泛化,不只过拟合几何) | **MMMU**、**MMStar** | 覆盖广学科综合(MMMU)+ 去数据泄漏的精炼集(MMStar),证明提升可泛化 |
| **通用 · 纯文本推理**(证明文本能力不退化) | **GSM8K**、**MATH-500** | 呼应两阶段迁移叙事;GSM8K 对齐旧简历的 82.49%,MATH-500 加一档难度 |

以上 8 个基准全部由 EvalScope 原生支持(以本地 `evalscope eval --help` 的 `--datasets` 名为准);全部走同一份 config,四个 checkpoint 各跑一遍即得完整对比表。

### 5.4 评测检查点矩阵(核心:每个阶段各测一次,量化每一步的增量价值)
在**同一套基准 + 同一份 EvalScope config**下,对以下四类 checkpoint 各评测一次,逐列对比才能说明每个环节确实带来收益(而非把所有增益笼统归给"做了 RL"):

| # | Checkpoint | 说明 | 想证明的事 |
|---|-----------|------|-----------|
| C0 | **原始基座** Qwen3-VL-4B(未训) | 冷启动基线 | 起点分数,后续所有提升的分母 |
| C1 | **第一阶段(纯文本 RL)后** | 文本 FRE(deepscaler)训完 | 文本数学能力被激发;多模态基准此时可能持平/略降,记录迁移前状态 |
| C2 | **第二阶段(多模态图像 RL)后** | 在 C1 基础上做 MGT(mathv60k/geo) | 文本推理成功迁移到多模态,主线基准显著提升,且文本基准不退化 |
| C3 | **引入输入扰动(NoisyRollout / 定向扰动)后** | 在同预算下加扰动式探索 | 本项目的净贡献:相对 C2 的 baseline,扰动带来的额外提升,以及幻觉基准不恶化 |
| (可选) C3' | **对照方案** (a 多采 / b 升温 / c dynamic sampling) 各自的 checkpoint | 4.4 设计空间对照 | 扰动是不是这笔额外预算的最优花法(do-or-die) |

> 简历叙事对应:C0→C1→C2 复刻并强化原 MLLM-R1 的两阶段迁移(如原 Qwen2.5-VL-3B 在 MathVista 49.15%→57.36% 那类曲线),C2→C3 才是本次 spec 的新增贡献(扰动式探索)。四个点的分数表 + 训练侧诊断曲线共同构成"每一步都有意义"的证据链。

### 5.5 指标
- **基准侧**:各基准准确率(EvalScope 输出),按 5.4 的检查点矩阵逐列对比;主线基准报均值,并对高方差小基准用多次采样(`--repeats` / `mean_and_vote_at_k`)稳定估计。
- **训练侧**(RQ1 诊断,评测报告一并附上):退化组占比(reshape 成组后 std≈0 的组占比,按全对/全错细分;见 §4.2 无 uid、纯位置分组)、有效梯度样本占比、组内奖励方差分布、样本效率(达到同等分数所需 rollout 数)。
- **对外汇报口径固定**:同一 generation-config(temperature / max_tokens / few-shot 数)贯穿所有 checkpoint,避免解码设置差异污染跨阶段对比。

## 6. 实验计划与里程碑
- **M0 · Pilot(go/no-go,~1–2 天)**:
  - **Qwen3-VL 通路冒烟(硬门槛,先于一切)**:参考脚本与 `actor.py` 是为 **Qwen3.5/Gemma4** 写/测的,Qwen3-VL 未被官方点名;`actor.py:258` 靠 `"image_grid_thw" in mm_inputs` 选 mRoPE 分支(Qwen3-VL 应命中但**未验证**)。M0 先小规模确认:前向 + 生成 + 权重同步到 vLLM 跑通、`image_grid_thw` 分支正确、`attn_implementation` 选型(Qwen3-VL 用 flash-attn 还是 eager)无报错。此关不过不进训练。
  - 再跑 **C0 基线评测**(EvalScope,固定 config,拿到 Qwen3-VL-4B 起点分数);
  - 再量退化组占比(RQ1,按 §4.2 位置分组统计),并小规模对比 (a) vs (d) 能否降低坍缩、转化为有效梯度。结果决定是否继续。
- **M1**:移植 NoisyRollout 到 OpenRLHF-K,复现单阶段 GRPO baseline + NoisyRollout。
- **M1.5 · 两阶段复刻 + 分阶段评测**:跑通 FRE→MGT 两阶段,产出 **C1(纯文本 RL 后)** 与 **C2(图像 RL 后)** 两个 checkpoint,各用同一 EvalScope 套件评测,复刻并强化 MLLM-R1 的迁移曲线。
- **M2**:完成 4.4 全部对照(设计空间比较);对扰动方案与关键对照各产出 checkpoint,得到 **C3 / C3'** 评测,做设计空间横向对比。
- **M3**:定向/自适应变体(4.5)+ 机制消融;对最终变体补一次评测。
- **M4**:规模(3B/4B/8B)与两阶段(FRE→MGT)组合性实验;汇总 C0–C3 检查点矩阵成表,成文。

> 评测贯穿始终:每产出一个对外要汇报的 checkpoint(C0/C1/C2/C3)就立即用固定 EvalScope config 评测并落盘,避免训练完成后回头补测导致口径漂移。

## 7. 算力 / 磁盘约束
- 共享 8×80G 节点,`CUDA_VISIBLE_DEVICES` 只占空闲卡;VLM 不支持 packing,长序列吞吐受限,优先 3B/4B。
- 共享盘余量波动大:checkpoint 保留要克制,中间产物走 `/dev/shm`,写大文件前 `df -h /ThetaAI`。
- `2n` rollout 会翻倍 rollout 显存/时间,micro batch 与 `n` 需保守。

## 8. 风险与缓解
- **扰动打不过更笨的基线(a/b)** → M0/M2 优先证伪;若成立则转向"定向扰动省算力"这条更细的贡献。
- **与 DAPO dynamic sampling 撞车** → positioning 明确为"输入扰动 vs 重采样,在小模型/多模态上的效率对比",并把 DAPO 纳为对照而非对手。
- **VLM+critic/packing 受限** → 只用 critic-free 估计器(已确认可行,`--data.max_images_per_prompt>0` 时自动禁 critic/packing)。
- **Qwen3-VL 在 OpenRLHF-K 未被官方点名测过**(代码/脚本按 "Qwen3.5"/"Gemma4" 写) → M0 硬门槛冒烟(前向/生成/权重同步 + `image_grid_thw` 分支 + attn 选型),过关再训。
- **一阶段静默空训风险**(内置 boxed-only reward 对 deepscaler 的 `$...$` 答案恒判 0) → 用 §4.1 的统一奖励抽取,并在 M0 打印几条 pred/gold 对照确认非零 reward 后再放量。
- **数据转换踩坑**(message stringify、图路径内嵌+绝对路径、两数据集前缀不同) → 转换脚本按 §5.1 逐条处理,转完先抽样加载几条确认 loader 不报错、图能读到。
- **误用旧 CLI 写法** → 命令一律嵌套点号命名空间,以本地 `--help` 为准,不抄 lmm-r1 扁平 flag。

## 9. 简历卖点
分布式多模态 RL(Ray+vLLM+DeepSpeed)、GRPO/DAPO 机制理解、问题诊断与受控实验设计、跨框架移植(verl→OpenRLHF-K)、可复现开源实现,以及**基于 EvalScope 的系统化评测**——C0/C1/C2/C3 四检查点矩阵(原始基座 → 文本 RL → 图像 RL → 输入扰动)量化每一阶段的增量收益,每个领域取两个代表性基准共 8 个:多模态数学专项(MathVista/MathVision)、幻觉(HallusionBench/POPE)、通用多模态(MMMU/MMStar)、文本不退化(GSM8K/MATH-500),取代旧简历里"只报 MathVista/MathVision 两点"的粗糙口径。

---

## 10. 实验进度与关键发现(更新日志)

### 10.1 进度
- **2026-07-17 · C0 原始基座评测完成(4B + 8B,全 8 基准)**。同一份固定 config(贪心解码 temp 0 / max_tokens 2048,EvalScope 1.9.0 走 vLLM OpenAI 后端),两模型各 8 基准跑完并落盘。汇总表 `checkpoints/eval/RESULTS.md`;逐 run 结果在 `checkpoints/eval/C0-{4B,8B}/<ts>/shardN/reports/`。

  | 领域 | 基准 | C0-4B | C0-8B | N |
  |------|------|------:|------:|--:|
  | 多模态数学 | math_vista | 60.50 | 65.60 | 1000 |
  | 多模态数学 | math_vision | 26.15 | 28.16 | 3040 |
  | 幻觉 | hallusion_bench | 69.23 | 76.92 | 130 |
  | 幻觉 | pope | 87.00 | 86.87 | 9000 |
  | 通用多模态 | mmmu | 48.56 | 53.67 | 900 |
  | 通用多模态 | mm_star | 59.27 | 62.00 | 1500 |
  | 纯文本 | gsm8k | 93.40 | 94.54 | 1319 |
  | 纯文本 | math_500 | 78.00 | 80.60 | 500 |

  8B 在 7/8 基准上 ≥ 4B(POPE 已饱和,两者持平),曲线合理,可作为 C0→C1→C2→C3 矩阵的分母。
- **M0 通路冒烟**(先于本次评测,见 memory / `kong_rl/smoke_*.py`):Qwen3-VL-4B HF 前向/生成 + flash-attn + `image_grid_thw` mRoPE 分支;vLLM rollout 多模态生成均通过。
- **2026-07-17 · C1(纯文本 FRE)训练脚本就绪 + C1-4B 开训**。脚本 `kong_rl/train/train_c1.sh`(OpenRLHF-K 原生嵌套 CLI,非 lmm-r1 旧扁平 flag)。冒烟通过(奖励非零、无 packing/position_ids 报错、hybrid sleep/wake + NCCL 权重同步正常、梯度正常)。配方:`reinforce_baseline` / n=16 / lr 4e-7 / kl k2 1e-3 / normalize / 1 episode。**吞吐按 §12 激进调优**(gpu_mem 0.85 + max_len 3072 + max_new 2048 + CUDA graph + freeze_visual_encoder):rollout 生成 ~9×、端到端 ~4–5× 提速(~3 天 → ~16–20h),仍全程同步 on-policy。磁盘 `--ckpt.max_num 1`。
- **2026-07-18 · C1-4B 在 step160 OOM 崩、修复后断点续训;流水线加评测**。0.85 太激进(§15.12):hybrid 下 co-located DeepSpeed 常驻 ~12G,step160 存点尖峰 + 累积碎片把 razor-thin 余量压垮 → vLLM 唤醒 KV(`create_and_map`)OOM。**非超长样本**(全程 gen_len ~1000–1550,max tot_len ~2300 << max_len 5120)。修复:`gpu_mem` → 4B 0.70 / 8B 0.60;并发现 `expandable_segments:True` 破坏 colocate CUDA IPC 权重同步、已弃用(§15.13)。崩溃前 4B 在学(math_acc 0.51→0.59),故**从 step160 断点续训**(`--ckpt.load_enable`)续到 315,已验证无 OOM/IPC 错。编排流水线 `run_c1_pipeline.sh`(§17.2)现覆盖:等 4B → 记录 → 8B 冒烟定 micro_batch → 训 8B → **评测 C1-4B/C1-8B(`run_ckpt.sh`)→ 汇总 RESULTS.md → 停**。HF checkpoint 累积撑盘问题用 `prune_ckpts.sh` 兜住(§17.4)。
- **2026-07-22 · C1(纯文本 FRE)全部完成:4B+8B 训到 step314 + 8 基准评测**。汇总 `checkpoints/eval/RESULTS.md`。C0→C1 增量(4B / 8B):

  | 领域 | 基准 | C0-4B→C1-4B | C0-8B→C1-8B |
  |------|------|:--|:--|
  | 多模态数学 | math_vision | 26.15→**34.51** (+8.4) | 28.16→**37.66** (+9.5) |
  | 多模态数学 | math_vista | 60.50→**65.60** (+5.1) | 65.60→**67.90** (+2.3) |
  | 通用多模态 | mmmu | 48.56→**54.00** (+5.4) | 53.67→**60.56** (+6.9) |
  | 通用多模态 | mm_star | 59.27→61.00 (+1.7) | 62.00→62.73 (+0.7) |
  | 纯文本 | math_500 | 78.00→**84.40** (+6.4) | 80.60→**85.40** (+4.8) |
  | 纯文本 | gsm8k | 93.40→93.63 | 94.54→94.92 |
  | 幻觉 | pope | 87.00→87.18 | 86.87→86.95 |
  | 幻觉 | hallusion_bench | 69.23→**66.15 (−3.1)** ⚠ | 76.92→76.92 (0) |

  **核心结论**:纯文本数学 RL **强迁移到多模态推理**(math_vision +8~9.5、mmmu +5~7、math_vista +2~5)——FRE"文本→多模态"迁移假设成立;饱和项(gsm8k/pope)不动;8B 迁移幅度普遍 ≥ 4B。**唯一回退:4B hallusion_bench −3.1**(8B 持平,N=130 方差大)→ 纯数学 RL 或轻微牺牲视觉对齐鲁棒性,**正是 C2 图像 RL 要补的**。两个评测踩坑(均已修复+记录):① 训练 checkpoint 缺 `preprocessor_config.json` 等 → vLLM 起服务崩,修法 = 从基座 `cp -n` 补;② 新增 `kong_rl/eval/run_ckpt_par.sh`(8 卡并行预热,~2min vs 串行 ~10min)。

- **2026-07-22 · C2(图像 MGT)脚本就绪 + 开训**。脚本 `kong_rl/train/train_c2.sh`(同 train_c1.sh 骨架的原生嵌套 CLI)。**方案定案**(见 §C2 配方):pretrain=C1-<SIZE> step314;数据 `mathv60k`=**VerMulti-65K**(从 MathV360K 过滤+随机采 65k 的广谱多模态,图像路径零缺失)——**即复刻论文头号模型 MGT-PerceReason 的配方**(FRE-Text→全 VerMulti RL,MM Avg 全表最高);`mathv_geo`=VerMulti-Geo15K(同源 MathV360K 的几何向抽取,非 mathv60k 子集)对应 MGT-Geo,留作 C2-geo 变体。**论文机制印证 C2 目的**:FRE-Text 单独会掉"Vision Only"感知(−3.43%),MGT 阶段救回并 +11.68%——恰对应我们 C1 的 4B hallusion_bench −3.1,C2 应补回;**estimator=`group_norm`(GRPO)**(spec §325/§404,非 lmm-r1 的 reinforce_baseline/gae);超参取 lmm-r1 多模态经验(lr 1e-6 / kl 1e-3 / n=16 / temp1.0);**C2 相对 C1 五处差异**(详 §18.2):①初始化=C1 step314 ②去 `freeze_visual_encoder` ③`image_key=images`+`max_images_per_prompt=1`(触发 VLM 禁 critic/packing)④序列长度+分辨率 ⑤estimator=group_norm。**开训后 step11 撞长度硬崩**(`VLM prompt length exceeds max_prompt_length`),root-cause=图像分辨率不封顶、长尾单图 ~16k token(非 OOM);已把处理器 `longest_edge` 封到 2.0Mpx + `max_len→5120`、`save_steps→10`,重启后 ~7.8 min/步稳定推进(详 §18.5)。
- **待办**:C2-4B 冒烟→全量→评测 → (可选)C2-8B、C2-geo 变体 → C3(输入扰动);RQ1 优势坍缩诊断。

### 10.2 关键工程发现(评测侧,均已核对源码/实测)
1. **EvalScope 的 `llm_ckpt`(transformers)路径硬拒绝多模态输入**(`evalscope/models/modelscope.py`:`'Transformer model does not support multimodal content'`)——6 个 VLM 基准无法走它。**必须用 vLLM 起 OpenAI 兼容服务 + `--eval-type openai_api`**(即 §5.2 推荐路径)。旧 `kong_rl/run_eval.sh`(用 llm_ckpt)对 VLM 基准是坏的,已被 `kong_rl/eval/` 工具链取代。
2. **8 个基准全部规则判分**(boxed 抽取 / 选择题解析 / accuracy),**无需 judge 模型**,不用配 judge。
3. **ModelScope 下载走慢代理坑**:EvalScope 从 ModelScope 拉数据集,但 `modelscope.cn` / 其 OSS 后端 `aliyuncs.com` 原不在 `no_proxy` → 数据集下载被塞慢代理。已加进 `activate_rlhf.sh` 的 `no_proxy`,直连 ~7.4MB/s。`MODELSCOPE_CACHE` 指到大盘 `datasets/evalscope_cache/`。
4. **vLLM 引擎必须 TP=1**(本机 TP>1 崩,见 CLAUDE.md);4B/8B 单卡 80G 足够,`gpu_memory_utilization 0.85`。
5. **效率长尾**:`math_vision`(3040 样本、长 CoT)是长板,单卡 A800 ~50min,轻基准早完 → 空卡。静态分片(每副本 2 基准)会留下 straggler。**后续优化**:`math_vision` 有 `level 1`–`level 5` 子集,可用 `--dataset-args` 按子集切分做**基准内数据并行**,消除长尾。

## 11. 实验目录结构与运行约定(覆盖 C0–C3 及后续全部实验)

### 11.1 目录树(全部落 `/ThetaAI/kong` 大盘)
```
kong_rl/
  convert_data.py  reward_func.py            # 数据转换 + 统一奖励(训练用)
  smoke_qwen3vl.py  smoke_vllm_qwen3vl.py     # M0 冒烟
  eval/                                       # ★ 评测工具链(已就绪)
    serve_model.sh      # 起 1 个 TP=1 vLLM OpenAI 服务(1 GPU/1 副本),轮询 /health
    eval_bench.sh       # 对已起服务跑 evalscope(openai_api,固定贪心 config,--no-timestamp)
    run_c0.sh           # C0 编排:多副本 + 8 基准分片 + 跑完自动 teardown
    collect_results.py  # 跨分片/跨 tag 聚合成表,可 --out RESULTS.md
  train/                                      # (待建)C1/C2/C3 训练启动脚本
datasets/
  converted/                                  # 转成 OpenRLHF schema 的训练数据(§5.1)
  *_img/                                      # 解压的训练/基准图像
  evalscope_cache/                            # EvalScope/ModelScope 基准数据(MODELSCOPE_CACHE)
models/models/                                # 基座权重 Qwen3-VL-4B/8B
checkpoints/
  <run_name>/                                 # 各训练实验的 checkpoint(C1/C2/C3/对照变体)
  eval/
    C0-4B/<ts>/shardN/reports/...             # 逐 tag 评测结果(EvalScope 原生 reports/predictions/reviews)
    C0-8B/<ts>/...                            #   <ts> 目录 + latest 软链
    C1-4B/  C2-4B/  C3-4B/ ...                # 后续检查点评测(同结构)
    RESULTS.md                                # 跨检查点汇总表(collect_results.py 生成)
logs/
  eval/            C0-4B_shardN_<ts>.log      # 每个评测分片一份
    server/        C0-4B_gpuG_pP_<ts>.log     # 每个 vLLM 副本一份
  train/                                      # (待建)训练日志
```

### 11.2 检查点 / 评测 tag 命名约定
- **tag = `C{阶段}[变体]-{规模}`**:`C0-4B`、`C1-4B`、`C2-4B`、`C3-4B`,及 8B 对应 `C{n}-8B`;§4.4 对照方案用变体后缀,如 `C3a-4B`(多采)、`C3b-4B`(升温)、`C3c-4B`(dynamic sampling)。
- tag 同时用作 vLLM `--served-model-name` 与 EvalScope `--model-id`(报告里的模型名),保证结果目录、报告、汇总表三处 tag 一致。

### 11.3 运行约定
- **C0(原始基座)**:两模型并行占满 8 卡(4B→GPU 0-3,8B→GPU 4-7,各 4 副本、8 基准分片 2/副本):
  ```bash
  GPUS=0,1,2,3 nohup bash kong_rl/eval/run_c0.sh 4B > logs/eval/run_c0_4B.out 2>&1 &
  GPUS=4,5,6,7 nohup bash kong_rl/eval/run_c0.sh 8B > logs/eval/run_c0_8B.out 2>&1 &
  ```
- **C1–C3(训练后的检查点)**:同一评测口径,只是把服务指向 checkpoint 目录而非基座——`serve_model.sh` 已接受任意模型路径;计划把 `run_c0.sh` 泛化成接收 `<ckpt_path> <tag>` 的 `run_ckpt.sh`(小改,待建;**勿复用已弃用的 `run_eval.sh` 名字**),或手动 `serve_model.sh <ckpt> <gpu> <port> <tag> ...` + `eval_bench.sh <tag> <port> <workdir> <log> "<datasets>"`。
- **聚合**:`python kong_rl/eval/collect_results.py --tag C0-4B C0-8B --out checkpoints/eval/RESULTS.md`(可续加 C1/C2/C3 tag 做逐列对比)。
- **固定口径**(贯穿所有 tag,勿改):贪心解码 temp 0 / max_tokens 2048;8 基准 ID 见 §5.3;TP=1;`MODELSCOPE_CACHE=datasets/evalscope_cache`;跑前 `source activate_rlhf.sh`(设直连镜像 + LD_LIBRARY_PATH + conda)。
- **磁盘**:基准数据约 ~19G(POPE 的 COCO 图 + MMMU/MathVista/MathVision 图占大头),已在大盘 `datasets/evalscope_cache/`;写前先 `df -h /ThetaAI`。

---

## 12. 多卡训练提速经验(工程 trick + 本项目取舍 + 实测效果)

> 本节是「C1 起把训练跑快、跑稳」的工程沉淀,既是简历/面试可讲的调参八股,也记录**本项目在本服务器(8×A800-80G, 驱动 535/CUDA12.2, TP>1 崩, 共享盘 98% 满)下为什么这么选、效果如何**。训练脚本落地在 `kong_rl/train/train_c1.sh`(OpenRLHF-K 原生嵌套 CLI,吞吐旋钮全部 env 可覆盖)。第一手依据是官方 `performance` / `hybrid_engine` / `async_training` / `troubleshooting` 文档 + 源码,不是凭记忆。

### 12.1 并行架构:Ray + vLLM + DeepSpeed,Hybrid Engine 时分复用同一批卡
RL 训练一步分两相:**rollout 生成**(vLLM 推理)与 **训练**(DeepSpeed 前向/反向/优化)。朴素分布式让 vLLM 和 DeepSpeed 各占一组卡 → 生成时训练卡闲、训练时生成卡闲,浪费一半时钟。**Hybrid Engine**(`--train.colocate_all --vllm.enable_sleep --ds.enable_sleep`)把两者**放同一批卡上时分复用**:生成相 vLLM 醒/DeepSpeed 睡(offload),训练相反之;权重经 NCCL 广播同步(`--vllm.sync_backend nccl`)。**本项目选它而非 async 分布式**:8 卡单节点、小模型,colocate+sleep 能把 8 卡吃满且**全程 on-policy**(收敛稳,见 §12.4)。

### 12.1.1 四种并行维度:本实验用哪种、不用哪种、为什么(面试必答)
把"多卡"讲清楚,先分清四个正交的并行维度——它们回答的是不同问题:

| 维度 | 一句话定义 | 解决什么 | 本实验 | 为什么 |
|------|-----------|---------|--------|--------|
| **数据并行 DP**(Data Parallelism) | 每卡一份**完整模型**,各吃**不同数据分片**,梯度聚合 | 吞吐(单位时间过更多样本) | ✅ **主干** | 8 卡都能放下 4B/8B,DP 最简单、通信最少、扩展最好 |
| **张量并行 TP**(Tensor Parallelism) | 把**单层的矩阵**按行/列切到多卡,协同算**同一 batch 的同一层**,all-reduce 合 | 单卡放不下的大层 | ❌ **TP=1** | 4B/8B 单卡放得下,不需要;本机 **TP>1 直接崩**(CustomAllreduce,P2P/IPC 不可用);官方 perf 也主张"能不 TP 就不 TP"(通信贵) |
| **流水线并行 PP**(Pipeline Parallelism) | 把模型**按层切成 stage** 分到多卡,像流水线传激活 | 超大模型 + 多节点 | ❌ 不用 | 单节点小模型,PP 只会引入 bubble/复杂度,零收益 |
| **模型并行 MP**(Model Parallelism) | 广义:TP+PP 都算"把一个模型切开到多卡" | 模型本身放不下 | ❌ 不做模型切分 | 每卡都有完整模型副本;下面单独澄清 ZeRO-3 为何**不算** MP |

**最容易被追问的点:ZeRO-3 分片了参数,算不算模型并行?——不算,它是 DP 家族。**
- **ZeRO-3**(本实验训练侧)分片参数/梯度/优化器态,只是为了**省显存**;计算时**临时 all-gather 还原出完整的层**,再让每卡**各自独立算自己那份 batch**——数据流仍是数据并行,故称 "ZeRO-DP"。
- **TP** 则是把**一个矩阵乘本身**切开,多卡**协同算同一份 batch 的同一层**,每步都要 all-reduce 部分和——这才是模型(张量)并行。
- 一句话区分:**ZeRO-3 切"显存",TP 切"计算"**。两者可叠加(大模型上 ZeRO-DP × TP),本实验只用前者。

### 12.1.2 8 卡如何调度:Hybrid Engine 时分复用的时间线
本实验**不给 vLLM 和训练各分一半卡**(那样一半卡永远在闲),而是 `--train.colocate_all` 让**全部 8 卡在两个相位间时分复用**。一个 iteration:

```
     ┌───────────── 生成相 ─────────────┐ ┌──────── 训练相 ────────┐ ┌─ 同步 ─┐
卡0..7│ 8×vLLM 引擎醒 (各占1卡, TP=1)      │ │ DeepSpeed ZeRO-3 醒     │ │ 权重   │
     │ DeepSpeed offload 睡              │ │ vLLM sleep(释放KV)     │ │ 广播   │→ 下一轮
     │ ← engine-level 数据并行解码rollout│ │ ← 8卡DP前向/反向/优化   │ │ 到8引擎│
     └──────────────────────────────────┘ └────────────────────────┘ └────────┘
```

- **生成相**:8 个 vLLM 引擎醒(每卡一个引擎、**TP=1**、彼此独立),`128 prompt × 16 samples` 摊到 8 引擎并行解码——这是 **engine 级数据并行**。DeepSpeed 此时 offload/睡。
- **训练相**:vLLM `sleep`(cumem_allocator 释放 KV 显存),DeepSpeed 醒,8 卡 **ZeRO-3 数据并行**跑前向/反向/优化。
- **权重同步**:训练完 actor `all-gather` 出完整权重 → NCCL 广播到 8 个 vLLM 引擎(`--vllm.sync_backend nccl`),保证下轮生成用最新策略(**on-policy**)。
- 净效果:**同一批 8 卡,两个相位都吃满,无空闲卡**;代价是两相**串行**(不像 async 那样重叠),但换来严格 on-policy(§12.4)。

### 12.1.3 vLLM 与训练如何共处一张卡(colocate + sleep 的关键工程)
两个吃满显存的进程(vLLM KV + DeepSpeed 模型态)要轮流用同一张卡,靠三件事:
1. **双 sleep**:`--vllm.enable_sleep`(睡时 cumem_allocator 释放/解映射 KV,`wake_up` 时 `create_and_map` 重新映射回来)+ `--ds.enable_sleep`(DeepSpeed offload 优化器态/参数到 CPU)。谁醒谁占显存,另一方缩到最小常驻。
2. **给对方留额度**:`gpu_memory_utilization` **不能拉满**——即便对方 sleep,也有 ~10–20G 常驻(实测 vLLM sleep 后仍驻 ~10G;DeepSpeed 醒着 ~12–20G)。这正是 §15.12 的教训:0.85 → vLLM KV 吃 67G + DeepSpeed 常驻 12.5G = 撑爆 80G,故降到 **0.70(4B)/0.55–0.60(8B)**。
3. **权重同步走 CUDA IPC / NCCL**:`broadcast_to_vllm → _handle_cuda_ipc` 用 CUDA IPC 句柄零拷贝传权重;⚠ **不能设 `expandable_segments`**(cuMemMap 与 IPC 句柄不兼容,§15.13)。

> 这套方案的可讲点:**用"时分复用 + sleep/wake"把本该分两组卡的推理与训练压到同一组卡上,8 卡利用率拉满,同时靠留显存额度 + IPC 权重同步保证正确性**——是本实验在"单节点 8 卡、小模型、TP 不可用"约束下的最优解。

### 12.2 先定位瓶颈:小模型 RL 里 rollout 生成是大头
实测(C1-4B):一步里 **rollout 生成占绝对大头**(2048 条序列、每条上千 token 的 CoT),训练(2048 样本、非 packing)相对快。所以**提速资源全砸在提高 vLLM 解码吞吐**上,而不是训练侧。

### 12.3 提速旋钮与取舍(★=本项目实际采用,附原理)

| 旋钮 | 方向 | 原理 | 本项目 |
|------|------|------|--------|
| ★ **`vllm.gpu_memory_utilization`** | 0.6→**0.70(4B)/0.60(8B)** | KV 缓存越大 → vLLM 自动允许并发解码的序列越多(OpenRLHF-K 不暴露 `max_num_seqs`,并发由它 + max_model_len 决定)。**⚠ 曾试 0.85 结果 OOM**(见 §15.12):hybrid colocate 下 DeepSpeed 即便 sleep 也常驻 ~12–20G,必须给它留额度,不能像纯推理评测那样拉满 | 0.70 下 KV ~44G/卡,并发仍充裕、且留 ~13G 余量稳定跑 |
| ★ **`data.max_len`(= max_model_len)** | 5120→**3072** | 每序列预留 KV 与 max_model_len 成正比,调小 → 同显存能塞更多并发序列 | prompt 短 + 生成 2048 足够,砍掉冗余预留 |
| ★ **`rollout.max_new_tokens`** | 4096→**2048** | 生成上限;实测 deepscaler 答案 CoT ~900–1400 token,2048 已充裕且对齐评测口径 | 缩短长尾、提并发 |
| ★ **CUDA graphs**(去掉 `--vllm.enforce_eager`) | 开 | 消除解码 kernel launch 开销,小 batch 多步解码提速明显 | **实测本机 sleep/wake+权重同步循环下不卡**(冒烟验证 2 轮 wake-after-sleep 正常);官方把 enforce_eager 列为「同步卡住时的退路」,故 `EAGER=1` 保留为退路 |
| ★ **`vllm.enable_prefix_caching`** | 开 | 同 prompt 的 `n_samples` 条采样共享 prompt 前缀 KV,n=16 时省大量重复 prefill | 一直开 |
| ★ **`actor.freeze_visual_encoder`** | 开 | **纯文本 C1 视觉塔从不被激活**:冻结它 → 不给它建优化器态(省显存/checkpoint)+ 权重同步**跳过整个 visual tower**(实测同步日志里 `model.visual.blocks.*` 占大量传输) | 文本阶段纯赚;NoisyRollout/参考脚本图像阶段也冻结视觉塔 |
| ★ **`vllm.num_engines` = 卡数, `tensor_parallel_size 1`** | — | 官方 perf:**优先多引擎、最小化 TP**(TP 有通信开销);本机 TP>1 直接崩(CustomAllreduce) | 8 引擎 × TP1,每卡一引擎数据并行 rollout |
| **`ds.packing_samples`** | ✗ 不用 | 去 padding、训练大提速——**但本 fork 的 VLM 通道禁用**(`max_images>0` 断言关 packing),根因是 mRoPE 的 3D position_ids 只在非 packing 分支自算。这不是物理不可能,verl 已解(见 §12.3.1) | 文本阶段可开(C1 未开);图像阶段被迫**关** |
| **`train.dynamic_batch_enable`** | ✗ 不用 | 变长序列按 token 数动态组 batch,利用率更高——**但它强制开 packing** → 同上,VLM 用不了 | 关 |
| **`ds.overlap_comm`** | ★ 开 | 显存充裕时重叠反向与梯度 reduce | 80G 充裕,开 |
| **`ds.adam_offload`** | ✗ 不用 | 省显存但拖慢;显存充裕就别开 | 关 |

### 12.3.1 VLM 为什么不能 packing:mRoPE × packing 的真正矛盾,与 verl 的解法(面试深水区)
packing 把变长序列首尾相接成一条,用 `cu_seqlens` 划段、`flash_attn_varlen` 保证段间互不 attend——去掉 padding,是文本 RL 最大的单项提速。VLM 关掉它,矛盾**只**在 position_ids:

- **普通 1D RoPE**:位置 = `cumsum(attention_mask)-1`,packing 后每段从头计数即可,现算、不依赖模型。
- **Qwen-VL 的 mRoPE**:位置是 **3 维 (t,h,w)**,图像 token 的坐标由 `image_grid_thw` 按网格展开决定,必须**逐样本**算(HF 的 `get_rope_index`)。OpenRLHF-K 只在**非 packing 分支**实现了这套自算,故一旦 `max_images>0` 就**断言关 packing**——是**该 fork 未实现**,不是原理上做不到。

**verl 证明可兼容**(`docs/verl/verl/models/transformers/{qwen2_vl,qwen3_vl}.py`),一句话:**position_ids 在 packing 前按样本各自算好,跟着 token 一起打包,attention 时再从 position_ids 反推段边界**。三步:
1. **打包前逐样本预算**:`get_rope_index()` 在数据/rollout 管线(forward 之前)算出每条样本的 3D/4D mRoPE 位置。
2. **位置随 token 同步打包**:`input_ids` 与 `position_ids` 一起拉平(fsdp `transformer_impl.py`),段边界取自 nested-tensor 的 offsets。
3. **attention 反推 varlen**:`prepare_fa2_from_position_ids()` 用 `cu_seqlens = (position_ids==0).nonzero()`——每个样本首 token 位置归 0 即天然边界(图像 token 的 t 维被加了偏移、不为 0,故不会误判),交给 `flash_attn_varlen_func`;还能叠加 Ulysses 序列并行。

**本项目取舍**:C2 接受非 packing(靠资源分批 + 梯度累积兜底,§13.5),**不**中途移植——移植 = 复刻 `get_rope_index` + monkey-patch Qwen3-VL attention + 改 actor 打包逻辑,多日实验跑到一半不值当冒此风险。真遇吞吐瓶颈,更划算的是**用 token 预算动态分批(不碰 mRoPE)**,或**把 VLM 阶段整段换到 verl**(它对 Qwen3-VL 原生支持这套 + SP)。

### 12.4 同步 Hybrid vs 异步 Async:为什么 C1 **不**上 async
`--train.async_enable`(+ `--train.partial_rollout_enable` + 重要性修正 `is_correction`)让 rollout 与训练**并发重叠**,是官方「最高吞吐」路径。但官方 `async_training` 文档明确列出**不该用 async 的场景:小模型、短 rollout、稀疏奖励**——**C1 三条全中**(4B 小模型、~1k token 短 CoT、二值 0/1 稀疏奖励),async 的 off-policy 噪声会伤收敛。故 C1 走**同步 Hybrid**(先把收敛性坐实),把 async 留作后续「已验证收敛后再提吞吐」的可选项——这本身是个诚实、可讲的工程判断(资源换正确性,分场景选型)。

### 12.4.1 本实验的 on-policy 严格程度(以及唯一的隐式 off-policy 来源)
> 面试易被反问:「你说 on-policy,真的严格吗?是不是用了重采样/经验回放?」——先给结论:**C1 是"设计上严格 on-policy"的**,**没有**用经验回放、多轮复用、重要性修正等 off-policy/重采样技巧;唯一的"不严格"来自 **vLLM(行为策略)↔ HF(目标策略)的数值 mismatch**,由 PPO 裁剪比率兜底。逐条以本仓源码/实参核对:

**为什么说它严格 on-policy(源码级证据)**:
1. **一次 rollout = 一次策略更新**:本实验 `rollout 样本 = 128 prompt × 16 = 2048 = train.batch_size`,即**一个 rollout buffer 恰好凑一次 optimizer 更新**。`ppo_actor.py:189` 用 `iter_grad_accum_global_norm` 把 buffer 切成 64 个 micro-step,DeepSpeed **在累加边界只真正 step 一次**(§13.5)——所以这 2048 条样本全部用于**同一次更新**,且这次更新用的策略**就是生成它们的策略**。
2. **`--train.max_epochs 1`**:对每个 rollout buffer **只过一遍**(日志 `Train epoch [1/1]`)。若 >1,则第 2 遍起策略已被第 1 遍更新过、样本变成旧策略产物 → off-policy;我们设 1,规避了这点。
3. **同步 Hybrid,非 async**:`--train.async_enable` / `--train.partial_rollout_enable` / 重要性修正 `is_correction` **全未开**(§12.4)——这些才是 OpenRLHF 里真正引入 off-policy 的开关。
4. **无重采样/过滤**:`--algo.dynamic_filtering_enable` 默认 `False` 且未开(源码 `train_ppo_ray.py:562`);无经验回放跨步复用(`replay_buffer` 每步 `clear()`,`ppo_actor.py:602`)。

**那些"看着像重采样"的东西,其实不是 off-policy**:
- **`n_samples_per_prompt=16`**:对每个 prompt 采 16 条 completion,是 **GRPO/reinforce_baseline 的组内采样**(算组基线/优势用),**16 条全来自当前策略** → 是 on-policy 的"群体采样",不是 off-policy 的"重采样"。
- **`reward.normalize_enable` / 优势归一化**:是统计标准化,和采样分布无关,不影响 on/off-policy。

**唯一真实的隐式 off-policy 来源(这才是加分的诚实回答)**:
- **token 由 vLLM 采样,但训练用的 log-prob(新旧都)由 HF/DeepSpeed 前向重算**。二者共享同一份权重,但**内核/精度/采样实现不同 → 同一序列的概率不完全一致**。于是"生成这批 token 的行为策略"(vLLM)与"计算梯度的目标策略"(HF)存在**微小分布错配**,严格说构成一点点 off-policy。
- OpenRLHF 的 actor loss 是 **PPO 裁剪代理**(带重要性比率 `ratio=π_new/π_old` + clip),这个 mismatch 被 clip 吸收;又因**一次更新内权重不变**,`π_new≈π_old`、`ratio≈1`,clip 基本不触发 → 实际非常接近纯 on-policy REINFORCE 更新。
- 这就是社区常说的 **"rollout–training logprob mismatch"**;想彻底消除要么用 vLLM 返回的 logprob 做 `is_correction`(off-policy 修正),要么对齐两边精度——本实验不做,因误差小且一次更新 ratio≈1。

**一句话可讲**:C1 = 同步 Hybrid + 单 epoch + 一 rollout 一更新 + 无回放/无 async/无 IS 修正 = **设计上严格 on-policy**;唯一的不严格是 vLLM↔HF 数值 mismatch 带来的微小隐式 off-policy,由 PPO 裁剪比率兜底(且 ratio≈1)。**不要对面试官说"我们用了很多重采样策略"——那是误述,反而会被追着打。**

### 12.5 显存/磁盘账:zero3 checkpoint 里优化器态是大头
DeepSpeed ZeRO-3 断点态(`ckpt.path`)= fp32 Adam(m+v, 8 B/param)+ fp32 master(4 B/param)≈ **12 B/param**;4B 模型 ≈ ~48G,加 bf16 权重 ~8G ≈ **一份 ~56G**,而 `save_hf` 的 bf16 模型才 ~8G——**优化器态才是磁盘大头,不是模型本身**。共享盘 98% 满,故 **`--ckpt.max_num 1`**(只留最新一份,可断点续训)+ `freeze_visual_encoder`(视觉塔不进优化器态,进一步瘦身)。另:`/ThetaAI` 98% 满会让 **Ray 拒启**,需 `RAY_local_fs_capacity_threshold=0.99` + `ray start --temp-dir /dev/shm/...`(临时目录放内存盘)。

### 12.6 调参八股(通用 RL 训练 checklist,本项目取值)
- **batch 关系**:`train.batch_size = rollout.batch_size × n_samples_per_prompt`(OpenRLHF-K 约定);本项目 128×16=2048。
- **`n_samples_per_prompt`**:GRPO/reinforce_baseline 组内基线靠它,`n>1` 是硬性(否则优势恒 0 静默空训);本项目 **16**(组多样性好、利于 RQ1 优势坍缩诊断)。
- **advantage estimator**:C1 文本用 **`reinforce_baseline`**(REINFORCE++-baseline,复刻 LMM-R1 FRE);C2/C3 图像/扰动阶段转 **`group_norm`**(GRPO,对齐 NoisyRollout)。VLM 通道禁 critic,不能用 `gae`。
- **KL**:`--algo.kl.use_loss --algo.kl.estimator k2 --algo.kl.init_coef 1e-3`(小 KL 锚定不跑偏;GRPO 系用 KL-as-loss 配 k2/k3)。
- **lr**:actor `4e-7`(小模型 RL 保守,防训崩);`temperature 1.0`(rollout 要采样多样性,评测才用贪心 temp0)。
- **`reward.normalize_enable`**:奖励归一化,稳梯度。
- **梯度检查点** `actor.gradient_checkpointing_enable`:省激活显存,长序列必开。

### 12.7 实测效果(C1-4B, 8×A800, rollout_bs=128)

| 配置 | rollout 生成(128 prompt×16) | 一步 experience-making | 单步(全步) | 1 episode(315 步) |
|------|------|------|------|------|
| 保守 baseline(gpu_mem0.6 / max_new4096 / max_len5120 / enforce_eager)〔调优前推算〕 | ~9–10 min(~4.5 s/prompt) | ~12 min | ~15 min | **~3 天** |
| **激进实测(gpu_mem 0.70 / max_new2048 / max_len3072 / CUDA graph / freeze_vit)** | **~80 s(1:20,实测)** | **~80 s(exp-making 即生成主导)** | **~4.5 min(+训练相 3:13)** | **~24 h(实测步时×315)** |
| **提速** | **~7×** | ~9× | ~3.3× | ~3× |

> 结论(实测):rollout 生成提速 ~7×、端到端 ~3× 量级,且**保持同步 on-policy**(未牺牲收敛)。所有旋钮 env 可调,`EAGER=1` 为 CUDA-graph 出问题时的退路。⚠ **gpu_mem 从最初设想的 0.85 实测回退到 0.70**——0.85 在 4B 训到 step160 时 vLLM 唤醒 KV OOM(§15.12),8B 更紧(§17.1 降到 0.55);"激进 KV"要让位给"hybrid colocate 下 DeepSpeed 常驻 + 长序列尾部"的真实余量。

---

## 13. 训练 & 推理显存账(估算 + 实测验证)

> 面试高频:「4B 模型 8 卡训练,显存都花哪了?给我估一遍」。本节把 C1-4B 的显存**算清楚并与实测对齐**。本机 A800-80G/卡,bf16,Adam,DeepSpeed ZeRO-3,vLLM 与 DeepSpeed **colocate + sleep 时分复用同一批卡**,所以要**分相位**看。

### 13.1 训练相位(DeepSpeed 醒 / vLLM 睡)
模型状态用经典「混合精度 Adam ≈ 16 字节/参数」拆:
- fp32 优化器:master 权重 4 + Adam 动量 m 4 + 方差 v 4 = **12 B/param**
- bf16 参数 2 + bf16 梯度 2 = **4 B/param**
- 合计 **16 B/param**。4B 模型 → 全量 **64 GB 模型态**。

ZeRO-3 把参数/梯度/优化器**全部沿 8 卡分片** → 每卡 **64/8 ≈ 8 GB** 模型态。再加:
- 激活(开梯度检查点,只存层边界、反向重算)~ 数 GB/卡;
- vLLM sleep 后仍驻留 ~10 GB(实测日志 "10.09 GiB still in use");
- NCCL/通信 buffer、DeepSpeed 常驻等杂项。
- **`freeze_visual_encoder`** 让视觉塔不进优化器态(省它那份 12 B/param)。

→ **估算 ~25–45 GB/卡;实测训练相位 38–47 GB/卡 ✓ 对得上**(留了大量余量,见 §13.3)。

### 13.2 推理相位(vLLM 醒 / DeepSpeed 睡)
单引擎单卡(TP=1,每卡一个引擎,**权重不分片**):
- bf16 权重 4B×2 = **8 GB/卡**;
- `gpu_memory_utilization 0.85` × 80 = **68 GB 预算**,扣掉权重+CUDA graph+开销 → **~55–60 GB 全给 KV 缓存**(PagedAttention 分页管理);KV 越大 → 并发解码序列越多 → rollout 越快;
→ **估算 ~68–77 GB/卡;实测 rollout 相位 77 GB/卡 ✓**。

### 13.3 为什么留这么多余量 / 下一步还能怎么压榨
C1-4B **早期步**训练相只用了 ~42/80 GB,**说明当时 `train.micro_batch_size` 还能往上加**(非 packing 时训练是逐 micro-batch 串行,单步训练是 wall-clock 大头)。把 micro_batch 调大能减少梯度累加次数、提训练吞吐。**这是个能主动讲的点:先测出显存余量,再据此加 batch,而不是拍脑袋。**
> ⚠ **但"余量很多"是有条件的、会随训练收缩**:推理 RL 里 CoT 越训越长,非 packing 的 padding 会让训练相峰值**趋势性抬高**(C1-8B step≈25 实测已到 67.5/80、且跨卡不均)。所以"加 micro_batch 提速"只在**训练早期显存宽松时**成立;到中后期长尾变多,方向反而是**调小 micro_batch 保安全**。详见新增 §13.5。

### 13.4 权重同步的账
每步训练后,actor(ZeRO-3 分片)需 **all-gather 出完整权重 → NCCL 广播到 8 个 vLLM 引擎**。这步不小(4B 全量权重过一遍网络);`freeze_visual_encoder` 跳过视觉塔那部分传输(实测同步日志里 `model.visual.blocks.*` 占可观比例)。

### 13.5 训练相"长尾 padding"显存效应 + 梯度累积这个安全阀(2026-07-19 C1-8B 实测)
**观察**:C1-8B(step≈25, gpu_mem0.55)训练相显存**跨卡不均且偏高**:`卡0 57 / 卡1 62 / 卡2 55 / 卡3 59 / 卡4 64 / 卡5 63 / 卡6 64 / 卡7 67 GB`,且随训练推进抬升。逐层拆解:

1. **相位定位**:高显存在**训练相**(DeepSpeed 醒),**不是 rollout**(vLLM 此时睡、KV 已释放)。vLLM 的 KV 池启动即**预留死**(实测 0.55 → 每引擎 `Available KV 23.83 GiB / 173,520 tokens`),rollout 期恒定、不随样本长短涨落——所以"显存高"与"vLLM 分配激进"无关。
2. **非 packing 的 padding 放大**:VLM 不能 packing(mRoPE,§12.3),训练按 `train.micro_batch_size=4` 逐组前向/反向;**非 packing 下一组 4 条序列 padding 到组内最长的那条**。一条长尾样本(生成打满 `max_new=2048` → +prompt ≈ **3072**,日志见 `seq_len=3072`)就把整组撑到 3072,那一步激活显存成倍膨胀。
3. **跨卡不均的来源**:`2048 / 8 卡 / micro 4 = 64` 个梯度累加 micro-step;长尾样本随 shuffle 每步落在不同 DP rank → 各卡激活峰值不同 → 显存**又高、又不均、又忽上忽下**。ZeRO-3 模型态本身是均匀分片的,不均只可能来自激活 = 长尾指纹。
4. **随训练抬升**:推理 RL 里 reward↑ → CoT 越写越长 → 越多样本打满 2048 → 长尾变多 → 训练相峰值**趋势性抬高**(预期现象,非偶发)。

**危险性:高但有界**。`max_len=3072` 是硬顶 → 训练相最坏 = 一组 `4×3072`,**封顶在此、不会失控 OOM**。实测峰值 67.5/80,余量 ~12.5 GB,未崩。

**梯度累积 = 唯一便宜的安全阀(关键澄清,回答"本实验没用梯度累积吧")**:
- **本实验用了梯度累积,而且量很大**:`accum = train.batch_size / (train.micro_batch_size × world_size) = 2048 / (4 × 8) = **64**` —— 就是训练日志里 `Train epoch [1/1]: x/64` 那个进度条。所以**并非"没用累积",而是每次策略更新累积了 64 个 micro-step**。
- **`micro_train_batch_size` 与梯度累积在"固定全局 batch"下是同一个旋钮(反比)**:global `train.batch_size=2048` 不变时,micro 4→2 会自动把 accum 64→128。**"调小 micro batch" ≡ "调高梯度累积"**。
- 效果:**激活峰值近似砍半**(每组 padding 受害者从 4→2),**收敛完全不变**(全局 batch 一致 → 优化器更新逐比特相同),代价是训练相 wall-clock 变长(更多串行 micro-step)。这是经典的"显存↔速度"权衡,且**收敛中性**。

**给 step315 续训的建议(直接回答"调小 batch + 调高梯度累积是否更好")**:
- ✅ **只动 `micro_train_batch_size`(4→2),不要动全局 `train.batch_size`**:前者是纯显存↔速度、收敛中性的安全阀;后者会改 RL 动力学(梯度方差、`reinforce_baseline` 组基线质量、每次更新的 on-policy 样本量),**不是显存旋钮**,别为省显存去动它。
- ⚖️ **是否要调看约束**:续训中若训练相峰值逼近 **~75–78 GB**(response 变长会推高)→ micro 4→2 值得(换安全、几乎零收敛代价);若显存仍宽松 → 别动,徒增耗时。**先测峰值再决定**。
- ⚠ **只影响训练相**:调梯度累积对 rollout 相 / 卡死无帮助(那是 vLLM 侧的问题,与激活显存无关)。

---

## 14. 关键超参 / 架构选型的深入理由(逐条「为什么」)

| 选择 | 取值 | 为什么这么选(面试可展开) |
|------|------|--------------------------|
| **ZeRO stage** | **3** | 4B 本可 ZeRO-2 甚至不分片就放下,**但 colocate 的 vLLM 要抢显存做 KV**——ZeRO-3 连参数也分片,**给 KV 腾出最多显存 → rollout 并发最高**。代价是每次前向要 all-gather 参数(通信),小模型这点通信很便宜,划算。 |
| **TP=1 + 多引擎** | 8 引擎 | 官方 perf:优先加引擎、最小化 TP(TP 有 all-reduce 通信开销)。4B/8B 单卡放得下,不需要 TP。**且本机 TP>1 直接崩**(CustomAllreduce P2P/IPC 不可用)。 |
| **batch 结构** | rollout 128 × n16 = 2048 = train_bs;micro 2 | 全局训练 batch = 每步 rollout 产出的经验数(OpenRLHF-K 约定)。micro 2 → 128 次梯度累加 / 8 卡。batch 越大梯度越稳,但一步越慢;2048 是稳定性/步时的折衷。 |
| **n_samples_per_prompt** | **16** | GRPO/reinforce_baseline 的组内基线靠同一 prompt 的多条采样估计。**越大 → 优势方差估计越准、"全对/全错"退化组越少**(直接关系本项目 RQ1 优势坍缩)。`n>1` 是硬约束(否则优势恒 0 静默空训)。成本被 prefix caching 摊薄(16 条共享 prompt prefill)。 |
| **advantage estimator** | C1 `reinforce_baseline`;C2/C3 `group_norm` | C1 文本复刻 LMM-R1 FRE(REINFORCE++-baseline,只减组均值);C2/C3 对齐 NoisyRollout 用 GRPO(`group_norm`,减均值再除 std)。**VLM 通道禁 critic,不能用 `gae`**(源码断言,`max_images_per_prompt>0` 触发)。 |
| **KL** | use_loss + k2 + init 1e-3 | KL 作为 loss 项(GRPO 系惯例)而非 reward 惩罚;k2 估计器对正 KL 低方差;系数小(1e-3)只做轻锚定,允许模型探索。 |
| **lr / warmup / scheduler** | 4e-7 / 0.03 warmup / cosine_min_lr | 小模型 RL 用极小 lr 防训崩(策略梯度对 lr 敏感);3% warmup + cosine 退火。 |
| **temperature** | 训练 1.0 / 评测 0.0 | 训练 rollout 要采样多样性(探索、组内方差);评测固定贪心 temp0 保证可复现、跨检查点可比。 |
| **precision** | bf16 参数,fp32 logits | bf16 动态范围好(A800 原生);logits 转 fp32 算 log_prob/KL 数值稳(源码 `output.logits.to(float32)`)。 |
| **freeze_visual_encoder** | 文本阶段开 | 视觉塔在纯文本从不激活 → 不建优化器态(省显存/磁盘)+ 权重同步跳过它。 |
| **packing / dynamic_batch** | **关** | 本 fork 只在非 packing 分支自算 mRoPE 的 3D position_ids,故 `max_images>0` 断言关 packing → VLM 吞吐受非 packing 拖累。是 fork 实现约束,非物理下限(verl 已解,§12.3.1)。 |

---

## 15. 踩坑与教训(root-cause + 修复,时间线)

按遇到顺序,每条含**现象 → 根因 → 修复**,都是面试可讲的"我怎么 debug"素材:

1. **vLLM 一起引擎就 `CUDA driver version is insufficient`** → PyPI 的 vllm 0.22.1 是 CUDA13 构建,本机驱动 535/CUDA12.2 跑不了 → 换官方 **cu129 预编译轮子**(cu12 系,靠次版本兼容跑在 535 上)。见 [[vllm-cuda13-driver-blocker]]。
2. **TP>1 崩 `CustomAllreduce ... has no attribute '_ptr'`** → 本机 P2P/IPC 不可用 → **锁 TP=1**,靠多引擎并行。
3. **一阶段可能"静默空训"** → 内置 `math_reward_func` 只抽 `\boxed{}`,而 deepscaler 答案是 `$...$` 无 boxed → 每条 reward 恒 0 → 组同质 → 优势 0 → 无梯度但不报错。修复:写**统一奖励** `reward_func.py`(抽 `<answer>` 内文,兼容 `$...$`/`\boxed{}`),并在冒烟打印 pred/gold 确认 reward 非零。
4. **Qwen3-VL 纯文本能不能 packing** → 源码 `actor.py` mRoPE 自算 position_ids 分支只在**非 packing** 走;packing 路径对 VLM 未重建 3D 位置 → 判定**必须关 packing**,冒烟验证非 packing 前向正常。
5. **Ray 拒启** → `/ThetaAI` 98% 满,Ray 默认磁盘熔断阈值 95% → `RAY_local_fs_capacity_threshold=0.99` + `ray start --temp-dir /dev/shm/...`(临时目录挪内存盘)。
6. **旧复现脚本不能照搬** → lmm-r1 的 `train_fre_text.sh` 是**旧框架的扁平 flag** + 依赖不存在的 `remote_rm.math_verifier` server → OpenRLHF-K v0.10.4 全改成嵌套点号 CLI + 用 `--reward.remote_url <.py>`。只借它的**超参经验**,命令按新仓库源码/`--help` 重写。
7. **CUDA graph 会不会和 sleep/权重同步打架** → 官方把 `enforce_eager` 列为"权重同步卡住时的退路",有真实风险 → **专门冒烟验证** EAGER=0 跑通 2 轮 wake-after-sleep+同步不卡,才敢默认开;`EAGER=1` 保留退路。
8. **首版太保守(~3天)** → gpu_mem 0.6/enforce_eager/max_len5120 → 定位 rollout 为瓶颈,按 §12 激进调优 → ~9× rollout 提速(见 §12.7)。
9. **磁盘账认知修正** → checkpoint ~110G 不是模型大,是 **ZeRO-3 优化器态(~12 B/param)** → `--ckpt.max_num 1` + freeze 视觉塔。
10. **(注意)wandb key 被 config dump 打进训练日志** → 日志在私有大盘、未提交 git;后续可加日志脱敏。
11. **`save_hf` 的 HF checkpoint 不受 `--ckpt.max_num` 管、会累积撑盘** → `max_num` 只轮转 DeepSpeed 断点,`global_stepN_hf/` 每次另存不清理(4B~8G/8B~16G × 十几份)→ 写 `prune_ckpts.sh` 后台只留最新 1 个,`train_c1.sh` 自动起。详见 §17.4。
12. **`gpu_memory_utilization=0.85` 太激进,hybrid 下延迟 OOM** → colocate 模式下 DeepSpeed 即便 sleep 也常驻显存(实测 OOM 报告:vLLM 66.8G + DS 三个进程 ~12.5G = 占满 79.3G/卡)。4B 靠碎片累积撑到 **step160 才在 vLLM wake KV 时 OOM**(不是立刻崩,更隐蔽);8B(权重 16G)冒烟直接 OOM。**根因是给 vLLM 的 0.85 没给 co-located DeepSpeed 留位**。修复:`gpu_mem` 降到 **4B 0.70 / 8B 0.60**(留 ~13G/卡余量),其余提速旋钮不动。教训:**纯推理评测能用 0.85,但 hybrid 训练要给同卡的 DS 留额度**——两个场景显存账不一样。
   - **为什么偏偏 step160?(排查结论,非超长样本)**:查了全程序列长度——`gen_len` 稳定 ~1000–1550,**最大 `tot_len` 才 ~2300 token,远未触及 `max_len 5120`,step160 附近无任何飙升**,所以**排除"超长样本导致尖峰"**。真正触发点是:**step160 恰好是 checkpoint 保存步**(`save_steps=20`;日志可见此刻在 "Deleted checkpoint global_step140" 做 max_num 轮转)。存点时 `save_hf` 的 `_consolidated_16bit_state_dict()` 把整模型 all-gather 到 rank0 + 写 DS 分片,**制造瞬时显存尖峰**;紧接着 vLLM **两段式唤醒**(先 `['weights']` 再 `['kv_cache']`)在 `create_and_map` 重映射 sleep 丢弃的 ~37G KV 时,已无物理显存可映射 → OOM。早期 20/40/…/140 的存点也有尖峰,但那时碎片少、余量够,撑过去了;**160 是"累积碎片 + 存点尖峰"把 0.85 的 razor-thin 余量压垮的那一下(带随机性,可能是 140 也可能 180)**。降到 0.70 后 ~13G 余量足以吸收"存点尖峰 + 唤醒重映射",故稳定。
13. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 colocate 权重同步(CUDA IPC)不兼容** → 我一度想用它缓解碎片,结果 resume 首次 `broadcast_to_vllm→_handle_cuda_ipc` 直接 `CUDA driver error: invalid argument`。expandable_segments 用 cuMemMap 虚拟内存,**无法通过 CUDA IPC 句柄共享**(PyTorch 已知限制),而 colocate 下 actor→vLLM 权重就是走 IPC。**移除即恢复**;OOM 本就是"真占满"不是碎片,单靠降 `gpu_mem` 已解决,不需要 expandable_segments。

---

## 16. 面试高频拷打 Q&A(自测)

- **Q: 4B 模型 8 卡显存怎么分?** A: 见 §13。训练相位 ZeRO-3 模型态 64G/8≈8G/卡 + 激活 + vLLM 驻留 ~10G ≈ 实测 38–47G;推理相位 vLLM 权重 8G + KV ~60G ≈ 实测 77G。分相位是因为 hybrid 时分复用。
- **Q: 为什么 ZeRO-3 而不是 2?** A: 不是为了放下模型(2 就够),是为了给 colocate 的 vLLM KV 腾显存,换更高 rollout 并发;小模型 all-gather 通信便宜。
- **Q: 什么是优势坍缩?怎么浪费算力?** A: GRPO 组内奖励同质(全对/全错)→ 优势=0 → 该组零梯度,但 rollout 算力照花。这正是本项目 RQ1 要量化、RQ2/RQ3 要救的问题。
- **Q: 为什么不用 async 提吞吐?** A: async 引入 off-policy 噪声,官方明确不建议用于小模型/短 rollout/稀疏奖励——C1 三条全中,先保收敛;async 留作已验证后的可选项。
- **Q: rollout 怎么提速的?** A: 定位瓶颈是生成 → 提 KV(gpu_mem 0.85)、降 max_model_len(3072)、CUDA graph、prefix caching、freeze 视觉塔省同步;实测 rollout ~9×,端到端 ~4–5×,仍同步 on-policy。
- **Q: 权重怎么从训练同步到 vLLM?贵不贵?** A: 每步 all-gather 完整权重经 NCCL 广播到各引擎;freeze 视觉塔可省其传输。
- **Q: 为什么 VLM 不能 sample packing?吞吐受限是根本的吗?** A: 本 fork 的 mRoPE 3D position_ids 只在非 packing 分支自算,`max_images>0` 就断言关 packing——这是**实现约束不是原理不可能**。verl 已证可兼容:position_ids 打包前逐样本预算、随 token 一起打包、attention 从 `position_ids==0` 反推 `cu_seqlens` 走 flash_attn_varlen(详 §12.3.1)。本项目接受非 packing、不中途移植。
- **Q: KL 用 penalty 还是 loss?哪个估计器?** A: KL-as-loss + k2(正 KL 低方差),小系数轻锚定。
- **Q: batch 怎么定的?全局 batch 多大?** A: rollout 128 prompt × n16 = 2048 经验 = 全局训练 batch;micro 2 → 128 次梯度累加/8 卡。
- **Q: 训了多久?** A: **实测(非估):C1-4B 单步 ~4.5 min**(rollout 相 ~1:20 + 训练相 3:13〔128 次梯度累加,非 packing 训练是大头〕);**C1-8B 端到端 ~8.3 min/步**(实测 37.8h / 274 步,含存点/权重同步/sleep 转换开销;其中 rollout 相 ~1:55 + 训练相 3:50〔64 次累加〕,差额是相位切换与每 20 步存点 I/O)。两者均训到 **step314**(4B 从 step160 续、8B 从 step40 续),checkpoint 每 20 步存(max_num 1)。

---

## 17. C1-8B 训练设计 + 全流程无人值守编排

### 17.1 8B 相比 4B 的差异与调参(面试可讲的"规模化"经验)
8B 是 4B 的规模消融,同一套 recipe/脚本(`train_c1.sh 8B`),差异点:
- **显存翻倍的是优化器态**:ZeRO-3 模型态 8B ≈ 128 GB 全量 → /8 卡 = **16 GB/卡**(4B 是 8 GB/卡)。推理侧 vLLM 权重 8→16 GB/卡。
- **`train.micro_batch_size`(实测取 4)**:候选 `[4,2]` 冒烟从大到小试,8B 定 **MICRO=4**(2048/(4×8)=**64 次梯度累加**,与日志 `Train epoch 64/64` 吻合);4B 用 MICRO=2(128 次累加)。
- **`gpu_memory_utilization` 实测降到 0.55(不是 0.85)**:8B 起初 0.6 + CUDA graph 在 **~step26 卡死**(rollout 生成停滞,py-spy 抓栈确认;0.6 仍复现),降到 **0.55** + 保留 CUDA graph + 480s 停滞看门狗后**平稳训到 step314**。教训:8B 的 vLLM 权重(16G)+ colocate DeepSpeed 常驻,给 CUDA graph private pool 和长序列尾部留的头比 4B 更紧,0.85 那套"激进 KV"在 8B 上直接翻车。
- 其余(reinforce_baseline / n16 / **lr 4e-7** / kl k2 1e-3 / freeze_visual_encoder / CUDA graph 开(enforce_eager=False) / max_len 3072 / max_new 2048)与 4B 一致,保证 C1-4B / C1-8B **可比**(以上均从 8B run 的 config dump 实测确认)。

### 17.2 无人值守编排(`kong_rl/train/run_c1_pipeline.sh`)
因训练与评测都吃满 8 卡、不能并行,且单模型多小时,做成一条**串行流水线**,后台 nohup 跑,全程落状态文件,无需盯:
1. 等当前 **C1-4B** 训完(轮询其 PID)→ **记录 4B 结果** 到 `checkpoints/C1-4B/TRAIN_SUMMARY.md`;若无 checkpoint(异常退出)则**停下不盲目开 8B**。
2. **8B 冒烟自动定 `micro_batch`**(§17.1)→ 全量 **C1-8B** → 记录 8B 结果。
3. 两模型都训完 → **评测 C1-4B、C1-8B**(`run_ckpt.sh`,与 C0 同口径:8 基准每卡 1 个、贪心 temp0/max_tokens2048、vLLM openai_api、TP=1)。
4. **汇总** `collect_results.py --tag C0-4B C1-4B C0-8B C1-8B --out checkpoints/eval/RESULTS.md`(C0→C1 逐列对比,直接看文本 RL 的增量)→ **停**,GPU 释放,等下一步 C2。

监控:`cat logs/train/c1_pipeline_STATUS.txt`(一句话当前阶段)、`tail -f logs/train/c1_pipeline.log`。

### 17.3 评测入口:`run_ckpt.sh`(泛化自 `run_c0.sh`)
`run_c0.sh` 把模型路径写死为基座;新增 `kong_rl/eval/run_ckpt.sh` 接收 `<model_path> <tag>`,指向训练产出的 HF checkpoint 即可,其余分片/服务/判分逻辑与 C0 完全一致——保证 C0/C1 口径不漂。

### 17.4 关键坑:`save_hf` 的 HF checkpoint 不受 `--ckpt.max_num` 管,会累积撑盘
**现象/根因**:`--ckpt.max_num` 只轮转 DeepSpeed 断点(`ckpt.path/_actor/`);而 `save_hf` 每 `save_steps` 另存一个 `ckpt.path/global_step{N}_hf/`(4B ~8G、8B ~16G),**无任何清理**(源码 `ppo_actor.py:save_checkpoint` → `save_model`,tag 是 `global_step{N}` 递增)。315 步 / 每 20 步 ≈ 15 份 → 4B ~120G、8B ~240G,**在 98% 满的共享盘上必爆**,也违反"最多留 1 个 checkpoint"。
**修复**:写 `kong_rl/train/prune_ckpts.sh`——后台循环,只保留最新 1 个 `*_hf`(正在写的那个天然最新、不会误删),删掉旧的。`train_c1.sh` 全量训练时自动起它(trap 清理);对早于该改动就已在跑的 4B,单独起一个 pruner 兜底。**这条是很典型的"框架默认行为 × 真实磁盘约束"踩坑,面试值得讲。**

### 17.5 显存实测汇总(供快速回答)
| 相位 | C1-4B 实测 | C1-8B 实测 |
|------|-----------|-----------|
| 训练(DS 醒/vLLM 睡) | 38–47 GB/卡 | **44–75 GB/卡(跨卡不均,长尾 padding 指纹),峰值 74.8G** |
| 推理 rollout(vLLM 醒/DS 睡) | 77 GB/卡(gpu_mem 0.85,评测口径;训练侧 4B 用 0.70) | gpu_mem **0.55**(权重 16G + KV;0.6 会卡死,见 §17.1) |
| ckpt 磁盘(留 1 份) | DS 断点 **46G** + HF **8.3G** | DS 断点 **93G** + HF **17G** |

> 8B 训练相显存"又高、又不均、又忽上忽下"是非 packing 长尾 padding 的指纹(§13.5);峰值 74.8G 逼近 80G,靠 gpu_mem 0.55 留头 + micro 4 未爆。磁盘均为实测 `du -sh`:8B 一份 DS 断点(fp32 Adam 态)就 93G,是"留 1 份"铁律的由来。

---

## 18. C2 图像 RL(MGT)全链路:数据构造 → 训练细节 → 决策/调优 → 踩坑 → 机制预期

> 本章是 C2 阶段的"教程级"沉淀,对齐 §17 的 C1-8B 专章。C2 = 在 C1(FRE-Text)之上做**多模态泛化训练 MGT**,复刻 LMM-R1(arXiv 2503.07536)的**头号开源模型 MGT-PerceReason**。脚本 `kong_rl/train/train_c2.sh`;论文/数据认知的一手依据见 `docs/lmm-r1/README_zh.md` 与 `OpenRLHF-K/docs/khw/MinerU_markdown_2503.07536v2_*.md`。

### 18.1 数据构造全链路(面试必答:多模态可验证 RL 数据怎么来的)
**一句话来源链**:`MathV360K(360k 多模态题)` → 过滤出"答案可规则验证"(数值 123/4.11 或选项 A/B/C/D)得 **130k** → 两种抽法得两份训练集:

| 数据集 | 本地文件 | 抽法 | 规模 | 构成(实测 = 论文 Table A4/A5 逐条吻合) | 训练目标 |
|--------|---------|------|------|------|---------|
| **VerMulti-65K** | `converted/mathv60k.jsonl` | 从 130k **随机采 65k**(广谱) | 65,118 | 24 源:IconQA 7166 / PMC-VQA 6760 / TabMWP 6732 / A-OKVQA / FigureQA / ScienceQA / 图表 / 文档 / 科学…几何仅 GeoQA+4062+Geo3K2845+UniGeo2767 | FRE-Multi(对照)+ **MGT-PerceReason(最优)** |
| **VerMulti-Geo15K** | `converted/mathv_geo.jsonl` | 从 MathV360K **抽几何** 15k | 19,810 | 纯几何:GeoQA+ **8155** / Geometry3K 2776 / UniGeo 5583 / GEOS 271 / TQA 25 | MGT-Geo |

- **两份是同源(MathV360K)的两种抽法,谁都不是谁的文件子集**:Geo 特意多抽几何(GeoQA+ 8155 ≫ 65k 里的 4062),题干交集仅 ~52.6%(占 Geo),因为两次过滤都保留了进入各自集合的那部分标准几何基准题。**别在面试里说"geo 是 mathv60k 的子集"**——不准确。
- **本项目 C2 选 `mathv60k`(全 VerMulti)= 复刻 MGT-PerceReason**:它是论文 MM Avg 最高(40.95)的头号开源模型,广谱、平衡感知与推理、且**提升视觉不牺牲推理**;`mathv_geo` 对应 MGT-Geo(几何专项),留作 C2-geo 变体做领域对照。
- **转换四要点**(与 §5.1 同,C2 侧再确认):① `message` 是 stringify 的 JSON → `json.loads` 成消息 list 落到 `prompt`;② 图路径从 content 内**上提**到顶层 `images` 键(loader 从独立 `image_key` 读图);③ 绝对/相对路径按**数据集各写一条前缀重写规则**指向本地解压目录(`mathv60k_img/`、`mathv_geo_img/`);④ 答案 `$\boxed{}$` 落 `label`,与 deepscaler 的 `$...$` 由统一奖励抽取(§4.1)兼容。
- **上线前校验(已做)**:两份行数 65118 / 19810;抽查各 500 条**图像路径零缺失**;schema `{prompt:list, images:[abs_path], label:"$\boxed{}$"}` 正确;奖励函数 `reward_func.py` 对 boxed 与 `$...$` 双格式均可判分(设计时即为两阶段兼容)。

### 18.2 C2 训练配方:相对 C1 的**五处改动**(逐条为什么)
同一套 OpenRLHF-K 嵌套 CLI 骨架(`train_c2.sh` fork 自 `train_c1.sh`),保证阶段间可比;差异仅五处,每处都有明确动机:

| # | 改动 | C1 | C2 | 为什么 |
|---|------|----|----|--------|
| 1 | **初始化** | 原始基座 | `checkpoints/C1-<SIZE>`(step314 最终模型) | MGT 定义就是"在 FRE 之上继续 RL";用最新 C1 检查点延续已激发的推理能力 |
| 2 | **视觉塔** | `--actor.freeze_visual_encoder`(文本阶段视觉塔不用,冻结省同步+优化器态) | **去掉冻结**(视觉塔参与训练) | 图像 RL 要学"从图里抽视觉信息";冻结=只让 LLM 适配,违背 MGT 目的 |
| 3 | **图像输入** | `--data.max_images_per_prompt 0`(纯文本) | `--data.image_key images --data.max_images_per_prompt 1` | 打开多模态通道;**`>0` 触发 VLM 禁 critic/禁 packing 断言**,并让 vLLM 内部设 `limit_mm_per_prompt` |
| 4 | **序列长度 + 图像分辨率** | `--data.max_len 3072` | **5120 + 处理器 `longest_edge` 封顶 2.0Mpx** | 压力在 prompt 侧的图像 token(非生成;论文 Fig4:MGT 生成仅 ~200–250 tok)。**但光调 max_len 不够**:分辨率不封顶时长尾单图可达 ~16k token 撑破预算并硬崩(§18.5),须同时把单图封到 ≤1976 token |
| 5 | **算法** | `reinforce_baseline`(REINFORCE++-baseline,复刻 FRE-Text) | **`group_norm`(GRPO,减均值再除 std)** | spec §325/§404:C2/C3 对齐 NoisyRollout 用 GRPO;**VLM 通道禁 critic → 不能用 gae**(lmm-r1 的 mgt_geo 用 gae 是旧框架带 critic,我们这条路走不通) |

其余超参取 **lmm-r1 多模态经验**:`lr 1e-6`(C1 文本是 4e-7,多模态阶段略高)、`kl k2 1e-3`、`n_samples 16`、`temp 1.0`、`rollout_bs 128`、`train_bs 2048`、`1 episode`、`max_new_tokens 1024`(MGT 生成本就短 ~200–250 tok,1024 已充裕、比 2048 省长尾 padding)。显存:`gpu_mem 0.50(4B)/0.45(8B)`、`micro 2(4B)/1(8B)`;`save_steps 10`(崩溃保险,`max_num 1` 磁盘不涨)。

### 18.3 决策思路(把"选择"讲成"权衡")
- **为什么 mathv60k 而非 geo?** geo 太窄(纯几何)、只对应中间模型 MGT-Geo;mathv60k=全 VerMulti=论文最优 MGT-PerceReason,广谱覆盖我们四个评测域(math_vista/math_vision/mmmu/mm_star),泛化证据更强。
- **为什么 group_norm 而非照抄 lmm-r1?** lmm-r1 fre_multi 用 reinforce_baseline、mgt_geo 用 gae(PPO+critic)。本框架 VLM 通道**源码断言禁 critic**(`max_images>0` 触发),gae 直接出局;group_norm 是 critic-free、且与后续 C3(NoisyRollout GRPO)口径统一——一个选择同时满足"框架约束 + 阶段一致性"。
- **为什么解冻视觉塔却又怕显存?** 解冻是 MGT 的必需(否则学不到视觉);代价是视觉塔多一份优化器态 + 每步权重同步多传视觉塔。这就是 C2 比 C1 更吃显存、要调低 gpu_mem/micro 的根因——**"能力换显存"的显式权衡**。
- **为什么先只跑 4B?** 每个规模都是多日活(共享节点);4B 是主线,先拿 4B 的 C0→C1→C2 完整迁移曲线,再决定 8B。

### 18.4 调优与冒烟(smoke-first 工作流)
- **先冒烟再放量**(已做):`SMOKE=1` 小 batch(rollout 16 / max_samples 32)沿用真实 `gpu_mem/max_len/micro`,几分钟内验证**四件事**:图像能加载进 rollout、多模态生成不报错、`[Reward]` 非零(实测 42×r=1/84×r=0,几何题 gold='C'、CLEVR gold='6' 正常判分)、`Experience→Train` 走通(group_norm 的 act_loss/reward 正常)。**冒烟峰值 48G/80G**,gpu_mem 0.5/micro 1 安全、留足余量。
- **显存若宽松可上调**:冒烟只跑短序列,真实长尾要留神(见下)。若全量早期显存稳,可把 `MICRO 1→2` 提训练吞吐;若中后期逼近 OOM,反向降 micro 或 gpu_mem。
- **长尾风险比 C1 更重(但已被分辨率封顶驯服)**:C1 的教训(§13.5)是非 packing 下一组按最长样本 padding、长尾样本撑爆激活;C2 图像 token 让序列更长更不均。封顶 `longest_edge 2.0Mpx` 后单图 ≤1976 token、最坏 prompt 2002 token(§18.5),长尾被**硬约束住**,故 4B 敢用 micro 2;gpu_mem 压到 0.5 给激活尖峰留头。
- **训练预算**:mathv60k 65k / 128 ≈ **508 步/episode**,C2 每步比 C1 慢(图像编码 + 长序列 + 视觉塔训练),实测 ~7.8 min/步(4B);全量 ~3 天(4B)/ ~5 天(8B)。每 10 步存最新 checkpoint(`max_num 1`),随时可停下评测、崩溃至多丢 10 步。可用 `MAXSAMP` 截断到 ~40k(~315 步,与 C1 预算持平便于对比)或 ~20k(快出方向验证)。

### 18.5 踩坑指南(C2 专属,均已核对)
- **C1 检查点作 pretrain 缺处理器文件**:OpenRLHF `save_hf` 存的 C1 模型缺 `preprocessor_config.json` / `video_preprocessor_config.json` 等,直接拿去起 vLLM 评测会 `Can't load image processor` 崩(与 §17.4 同类"框架默认行为"坑)。修法:从基座 snapshot `cp -n` 补这些静态预处理配置(视觉塔冻结时预处理定义不变;C2 解冻后仍是同一套图像预处理参数,补齐即可)。C2 训练侧用 transformers 加载或许不炸,但**评测 C2 时会复现**——已在 C1 收官时给 `checkpoints/C1-*` 根目录与 `global_step300_hf` 都补齐处理器文件。
- **`freeze_visual_encoder` 反向坑**:C1 为省显存冻结了视觉塔;若 C2 忘记去掉,会**静默只训 LLM**、视觉能力学不动,分数上不去还难查。C2 脚本已显式移除该 flag。
- **图像分辨率不封顶 → 单条超长 prompt 硬崩整轮(root-cause,已修)**:C2-4B 全量在 step 11 崩于 `ValueError: VLM prompt length (8734) exceeds max_prompt_length (7168)`。**不是生成、不是 OOM**:Qwen3-VL 处理器默认 `size.longest_edge=16.7Mpx` ≈ 不限分辨率,mathv60k 中位图仅 150 token,但高分辨率长尾(图表/文档,最大一张 7123×9723)膨胀到 ~16k 图像 token;`max_prompt_length = max_len − max_new` 被单条撑破时,OpenRLHF-K 的 VLM 通道**拒绝截断图像 prompt**(截断会错位 image-token↔pixel_values)→ 直接 kill 整个多日任务。**一条坏样本就够**。修法两处:① pretrain 模型目录 `preprocessor_config.json` 把 `size.longest_edge` 封到 **2.0Mpx**(单图 ≤1976 token,实测全库最坏 prompt 2002 token);② `max_len → 5120`(预算 4096 ≫ 2002)。非 packing 按批内最长样本 padding(不是 pad 到 max_len),故宽松 max_len 只是安全余量、**不增显存**。评测侧不封顶(用基座配置)以保 C0/C1/C2 口径一致。**降 gpu_mem 对此类报错无效**——那是 KV 额度,与 prompt 长度无关(易被误当 OOM 处理)。本 fork 无 lmm-r1 的 `--processor_kwargs` 像素封顶入口,故只能改模型目录配置。
- **`max_images_per_prompt>0` 的连锁断言**:一旦 >0,框架自动**禁 critic、禁 packing**——gae 用不了(§18.3)、吞吐受非 packing 拖累。禁 packing 的机制与 verl 的破解见 §12.3.1;这是 fork 实现约束,非 VLM RL 的物理下限。
- **数据规模翻倍时间**:mathv60k 是 deepscaler 的 ~1.6 倍且每步更慢,别按 C1 的 ~26h 直觉估 C2,要按 ~4–5 天规划共享节点占用。

### 18.6 机制预期(论文已验证,不是猜)
- **视觉感知回收**:论文 §5.2.1 明确——**FRE-Text 单独会掉"Vision Only"感知**(MathVerse VO 比基座 −3.43%),而 **MGT 阶段救回并大涨(比 FRE-Text +11.68%、比基座 +8.25%)**。这与我们 **C1 的 4B hallusion_bench −3.1** 是同一机制的两个观测点 → **C2 应把这个回退补回甚至反超**,是 C2 最直接的成功判据之一。
- **响应长度轨迹**:FRE-Multi 越训越短(150→<80 tok,退化成"直接报视觉答案");FRE-Text 越训越长(600→800 tok,推理更充分);**MGT-PerceReason 稳定在 200–250 tok**(推理深度与视觉识别效率的平衡)。看 C2 的 `gen_len` 曲线若从 C1 的 ~1200 降到几百,是"学会了在视觉任务上适度推理"的健康信号,不是退化。
- **两阶段协同**:直接多模态 RL(FRE-Multi / Direct-RL-*)会伤纯推理;先文本 FRE 再图像 MGT 才能"视觉↑且推理不掉",这正是 C0→C1→C2 三点曲线要讲的故事。

### 18.7 C2 面试拷打 Q&A(自测)
- **Q: C1 和 C2 数据什么关系?** A: C1 用纯文本 DeepScaleR-40K;C2 用 mathv60k=VerMulti-65K(从 MathV360K 过滤+随机采 65k 的广谱多模态)。geo 是同源另一抽法(几何专项),非子集。
- **Q: C2 为什么不能用 PPO(gae)?** A: 本框架 VLM 通道源码断言禁 critic(`max_images>0` 触发),gae 需 critic 故不可用;改用 critic-free 的 group_norm(GRPO),兼与 C3 口径统一。
- **Q: 从 C1 继续训,视觉塔冻不冻?** A: 必须解冻——MGT 目的就是学视觉推理,冻了只训 LLM。代价是显存(多一份优化器态 + 视觉塔权重同步),故调低 gpu_mem/micro。
- **Q: max_len 怎么定的?生成不是才 1024?** A: 长度压力在 prompt 侧的图像 token,不是生成(MGT 生成仅 ~200-250 tok)。关键是**光调 max_len 不够**——不封图像分辨率时长尾单图可达 ~16k token,任一超长样本会让 VLM 通道硬崩(§18.5);正解是**先把处理器 `longest_edge` 封到 2.0Mpx(单图≤1976),再把 max_len 定到 5120**(预算 4096 ≫ 最坏 2002)。(VLM 为何不能 packing 见 §12.3.1 / §16。)
- **Q: 怎么判断 C2 成功?** A: ① 多模态主线(math_vista/vision/mmmu/mm_star)在 C1 基础上再涨;② 文本基准(gsm8k/math_500)不退化;③ **幻觉/视觉忠实度(hallusion_bench)把 C1 的 −3.1 回退补回**(论文验证的 vision-only 回收机制);④ gen_len 收敛到数百的健康区间。
