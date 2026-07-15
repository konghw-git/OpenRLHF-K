# OpenRLHF 七天精通指南

> 面向：有 ML/LLM 基础、想吃透分布式 RLHF 训练全部细节与调参技巧的工程师。
> 方法论：**以一条数据的生命周期为主线读代码**——一个 prompt 如何变成 rollout、变成 experience、变成梯度、再变成 vLLM 里的新权重。架构和算法都挂在这条主线上，而不是按目录顺序啃文件。

配套资料（本仓库内，均可离线阅读）：

- 官方文档镜像（中英双语）：`docs/official-docs/openrlhf.readthedocs.io/{en,zh}/latest/`，纯文本版在 `_sources/*.rst.txt`（比 HTML 好读）
- 仓库自带导航：`.claude/docs/architecture.md`、`.claude/docs/official-docs-map.md`
- 可运行参考配置：`examples/scripts/*.sh`（**flag 组合的事实标准**）

---

## 第 0 章：一张图看懂整个系统（开始前必读）

一次 RL 训练迭代（sync 模式）的完整数据流：

```
 prompts_dataloader（rollout.batch_size 个 prompt）
        │
        ▼
 SamplesGenerator ──► vLLM engines（每 prompt 生成 n_samples_per_prompt 条）
   samples_generator.py      │ token-in-token-out：只传 token id + logprob，从不回到文本
        │                    │ （可选）动态过滤：整组 reward 全对/全错的 prompt 丢弃重采
        ▼
 RemoteExperienceMaker（experience_maker.py）
   ├─► ReferenceModelActor：算 log π_ref → KL
   ├─► RewardModelActor / remote RM URL / reward_func.py：算 reward
   ├─► CriticModelActor：算 V(s)（仅 GAE）
   ├─ length penalty（DAPO overlong / ProRL 截断惩罚）
   ├─ reward baseline 减法（rloo/grpo/reinforce_baseline…在这里发生）
   └─ compute_advantages_and_returns（6 种 estimator）
        │
        ▼
 ReplayBuffer（replay_buffer.py）──► seqlen balancing / dynamic batch
        │
        ▼
 PolicyModelActor.ppo_train（ppo_actor.py + models/loss.py）
   PolicyLoss（PPO clip / dual-clip / GSPO）+ 可选 KL loss + 可选 entropy
   CriticModelActor 同步更新 ValueLoss（仅 GAE）
        │
        ▼
 broadcast_to_vllm（NCCL 或 CUDA IPC）──► vLLM 拿到新权重，进入下一轮
```

四类 GPU actor + vLLM engine，全部由 `openrlhf/cli/train_ppo_ray.py` 编排：

| 角色 | 文件 | 何时可以省掉 |
|---|---|---|
| PolicyModelActor（被训练的 actor） | `trainer/ray/ppo_actor.py` | 永远需要 |
| CriticModelActor | `trainer/ray/ppo_critic.py` | `advantage.estimator != gae` 时自动不创建 |
| ReferenceModelActor | `trainer/ray/launcher.py` | `algo.kl.init_coef = 0` 时不创建 |
| RewardModelActor | `trainer/ray/launcher.py` | 用 remote RM URL 或 reward_func.py 时不部署 |
| vLLM engines | `trainer/ray/vllm_engine.py` | 负责全部生成（训练 80% 时间在这） |

**读代码前的两个关键约定：**

1. CLI flag 是点分层级的（`--algo.kl.init_coef`），`openrlhf/utils/config.py` 的 `hierarchize()` 把它们变成嵌套命名空间 `args.algo.kl.init_coef`。**grep 时搜嵌套名**（如 `kl.init_coef`），别搜 CLI 拼写。
2. 全部 flag 的权威语义在文档 `common_options.rst.txt`；全部默认值在 `cli/train_ppo_ray.py` 的 argparse 定义里。

---

## Day 1 — 全局架构与启动流程

**目标**：能白板画出 Ray actor 拓扑；理解 colocation 三种模式的 GPU 布局；跑通测试。

### 阅读清单（按顺序）

1. 文档 `index.rst.txt` + `architecture.rst.txt`（20 分钟，建立官方视角）
2. `openrlhf/utils/config.py`（很短，先懂参数系统）
3. `openrlhf/cli/train_ppo_ray.py` 通读：
   - `train_ppo_ray.py:40-64` placement group 构建与 colocate 约束
   - `train_ppo_ray.py:84-141` 四类 actor + vLLM 的 GPU 分配（PG 下每 actor 只申请 0.2 GPU 以实现共卡）
   - `train_ppo_ray.py:170-188` 初始化顺序（critic 必须等 actor，因为 scheduler 依赖 max_steps）
   - 文件末尾的 argparse 全量 flag（当字典翻，不用背）
4. `openrlhf/trainer/ray/launcher.py:202-373`：`RayActorGroup` 如何拉起 world_size 个 actor、`async_run_method_batch` 如何把 batch round-robin 切给各 rank（ring/TP 组内的 rank 拿同一份数据，`duplicate_actors = ring_attn_size * tp_size`）

### 核心概念

**三种部署形态**（性能文档的第一决策点）：

| 模式 | flag | GPU 布局 | 适用 |
|---|---|---|---|
| Hybrid Engine（推荐起步） | `--train.colocate_all --vllm.enable_sleep --ds.enable_sleep` | 所有角色 + vLLM 时分复用同一组卡 | 单机 4-8 卡、1.5B-13B |
| 部分共卡 | `--train.colocate_actor_ref` + `--train.colocate_critic_reward` | actor/ref 一组、critic/reward 一组、vLLM 独立 | 多机中等模型 |
| 全分离（Distributed） | 不加 colocate | 每角色独占 GPU 组 | 70B+ 或异构硬件 |

约束记牢：`colocate_all`（非 async）要求 `actor 卡数 == vllm.num_engines × vllm.tensor_parallel_size`（`train_ppo_ray.py:57-64`）；`vllm.enable_sleep` 与 `train.async_enable` **互斥**。

### 动手

- `pytest tests/` 在 Mac 上能直接跑（纯逻辑测试），确认环境 OK。
- 打开 `examples/scripts/train_ppo_ray_hybrid_engine.sh`，逐行对照今天读过的 flag，画出它的 GPU 分配图（8 卡：actor/ref/critic/reward 各 0.2 卡共享 + 4 个 TP=2 的 vLLM engine，全部落在同 8 张卡上）。

### 自测

1. 为什么 PG 模式下每个 actor 只申请 0.2 GPU？（答：让 actor/ref/critic/reward/vLLM 五方共享同一张物理卡，靠 sleep/offload 时分复用）
2. `algo.kl.init_coef=0` 时系统少了哪个 actor？为什么可以少？
3. GRPO 训练时哪两个 actor 不存在？

---

## Day 2 — 采样：从 prompt 到 token 轨迹

**目标**：吃透 rollout 生成、agent 抽象、Experience 数据结构——理解"token-in-token-out"为什么是这个框架的设计基石。

### 阅读清单

1. 文档 `agent_paradigm.rst.txt`（execution mode × RL algorithm × pipeline 三轴正交的设计哲学）
2. `openrlhf/trainer/ppo_utils/samples_generator.py` 全文：
   - `generate_samples`（:88-138）：缓冲区 + chunk 机制，`chunk_size = rollout.batch_size × n_samples_per_prompt`
   - `_dispatch_prompts_to_vllm`（:199-245）：用最小堆做 vLLM 引擎负载均衡
   - 动态过滤（:169-195）：同 prompt 的 n 条样本平均 reward 落在 `dynamic_filtering_range` 外则**整组丢弃并补新 prompt**（DAPO 的动态采样）
3. `openrlhf/utils/agent.py` 全文：
   - `SingleTurnAgentExecutor`（:184-356）：单轮生成 + reward 三种来源（HTTP remote RM :322-356 / 本地 reward_func.py :303-320 / RM actor 走 experience_maker 不经过这里）
   - `MultiTurnAgentExecutor`（:31-181）：`while not done: generate → env.step → 拼接 feedback token`，`action_ranges` 记录哪些 token 是模型生成的（只有它们算 loss），env feedback token 的 rollout_log_prob 补 0
4. `openrlhf/trainer/ppo_utils/experience.py:28-69`：Experience 的每个字段。重点分清三个 log_probs：
   - `rollout_log_probs`：vLLM 生成时的策略（π_old^rollout，可能已过时）
   - `action_log_probs`：训练时 actor 重算的当前策略（π_θ）
   - `base_action_log_probs`：reference model（π_ref，用于 KL）
5. `openrlhf/trainer/ray/vllm_engine.py:183-207`：engine 侧怎么调 executor。

### 核心概念

**Token-in-token-out**：轨迹永远以 token id + logprob 传递，训练与采样之间从不重新 detokenize。这消灭了 chat template 漂移、BOS/EOS 不一致等一整类 silent bug——很多自研 RLHF 框架的收敛玄学问题根源就在这。

**action_mask 的偏移**：`samples_generator.py:267` 中 `action_mask = action_mask[1:truncate_length]`——因为 log_prob 对应"预测下一个 token"，掩码相对 sequences 右移一位。读 loss 代码时脑中要有这个对齐关系。

**自定义 reward 函数签名**（RLVR 最常用）：

```python
def reward_func(queries, prompts, labels):
    return {"rewards": tensor,   # 进 advantage 计算
            "scores": tensor,    # 进 dynamic_filtering 判断
            "extra_logs": dict}
```

### 自测

1. `n_samples_per_prompt=8` 且 `rollout.batch_size=128`，一次 generate_samples 产出多少条序列？哪些算法**必须** n>1？
2. 多轮 agent 里环境反馈的 token 会参与 policy loss 吗？靠什么机制排除？
3. 动态过滤为什么要求 reward 在生成阶段就可得？（答：过滤发生在 samples_generator，早于 experience_maker，所以只支持 remote RM / reward_func / agent 路径）

---

## Day 3 — 算法核心 I：reward 管线与六种 advantage estimator

**目标**：对每种算法（PPO/REINFORCE++/RLOO/GRPO/Dr.GRPO）能手写 advantage 公式，并说出它们在代码里差在哪几行。

### 阅读清单

1. `openrlhf/trainer/ppo_utils/experience_maker.py` 全文，重点 `compute_advantages_and_returns`（:238-329）
2. `openrlhf/models/utils.py:64-126`（KL 三种估计器 + `compute_reward`）
3. `openrlhf/trainer/ppo_utils/kl_controller.py`（全文 30 行）
4. `openrlhf/trainer/ppo_utils/length_penalty.py`（全文）
5. 文档 `agent_training.rst.txt` 的算法表格部分

### 核心概念：reward 的加工流水线（顺序很重要）

对每条样本，reward 依次经过：

```
原始 reward（RM / reward_func / env）
  → length penalty（DAPO overlong、ProRL 截断惩罚）      length_penalty.py
  → 按 estimator 做组内 baseline 减法                     experience_maker.py:246-272
  → reward clip 到 (-10,10)                              models/utils.py compute_reward
  → KL 惩罚：-kl_coef·KL 加到每个 token，原始 r 只加在 EOS  （仅 use_kl_loss=False 时）
  → 折扣累积 / GAE → advantage
  → （部分 estimator）全局 (mean,std) 归一化               experience_maker.py:315-328
```

### 六种 advantage estimator（`--algo.advantage.estimator`）

设同一 prompt 的 n 条样本 reward 为 r_1..r_n：

| estimator | 论文名 | baseline 减法 | 全局归一化 | 需要 critic | 公式 |
|---|---|---|---|---|---|
| `gae` | PPO | 无（用 V(s)） | ✅ (mean,std) | ✅ | δ_t = r_t + γV_{t+1} − V_t；A_t = δ_t + γλA_{t+1} |
| `reinforce` | REINFORCE++ | 无 | ✅ | ❌ | A_t = G_t = Σ γ^k r_{t+k} |
| `rloo` | RLOO | 留一均值 (Σr−r_i)/(n−1) | ❌ | ❌ | A = r_i − b_i |
| `reinforce_baseline` | REINFORCE++-baseline（**RLVR 官方推荐**） | 组均值 mean(r) | ✅ | ❌ | A = r_i − r̄，再全局 /std |
| `group_norm` | GRPO | 组均值 | ❌（组内已除 std） | ❌ | A = (r_i − r̄)/(std(r)+1e-9) |
| `dr_grpo` | Dr. GRPO | 组均值 | ❌ | ❌ | A = r_i − r̄（去掉 GRPO 的 /std，修正梯度偏差） |

记忆锚点：**gae 之外全部 critic-free 且 γ 强制 1.0**；`train_ppo_ray.py:600-601` 非 gae 时直接把 critic 路径设 None。`--algo.advantage.no_std_norm` 可保留减均值但关掉除 std。

### KL 的两种用法（最容易混淆的设计分叉）

| | `algo.kl.use_loss=False`（默认，PPO 传统） | `algo.kl.use_loss=True`（GRPO 式） |
|---|---|---|
| KL 去哪 | 变成 per-token reward 惩罚：r_t += −β·KL_t | 作为独立 loss 项加在 policy loss 旁 |
| 推荐 estimator | 只用 `k1` | `k2` 或 `k3`（k1 无效） |
| ref logprobs | 用完即弃 | 保留传给 actor 训练 |
| 典型系数 | 0.01（RLHF 对话） | 1e-5（RLVR）；GRPO 常配 k3 |

三种 KL 近似（`models/utils.py:64-95`，源自 Schulman 博客）：Δ = logπ − logπ_ref

- **k1** = Δ（无偏但可为负）
- **k2** = Δ²/2（非负，低方差）
- **k3** = e^{−Δ} − 1 + Δ（非负、无偏，GRPO 论文用的就是它）

自适应 KL（`kl_controller.py:4-18`）：设了 `--algo.kl.target` 才启用，按 `value *= 1 + clip(kl/target − 1, ±0.2) · n_steps/horizon` 比例调节。

### 长度惩罚（reasoning 训练必调）

- **DAPO overlong**（`length_penalty.py:16-58`）：设软区间 `overlong_buffer_len`，超出 `max_new_tokens − buffer` 的部分按比例罚，最大罚 `−penalty_factor`。
- **ProRL stop-properly**（:61-106）：对 `finish_reason=="length"` 的截断样本，系数 ≥0 时 reward 乘系数（0=清零），<0 时直接覆盖为该负值。两者可叠加。

### 自测

1. Dr. GRPO 相对 GRPO 只改了一件事，是什么？为什么这个改动能修正"简短正确答案被过度奖励"的偏差？
2. `use_kl_loss=True` 时 `experience_maker.py` 为什么跳过 KL 计算但保留 `base_action_log_probs`？
3. 为什么 rloo/group_norm/dr_grpo 不做全局 advantage 归一化而 reinforce_baseline 做？（对照 `experience_maker.py:315` 的列表）

---

## Day 4 — 算法核心 II：loss、聚合与训练步

**目标**：吃透 `models/loss.py` 每一个分支；理解 token-level vs sequence-level 聚合的数学差异；理解 off-policy 校正（这是 async 训练能 work 的前提）。

### 阅读清单

1. `openrlhf/models/loss.py` 全文（重中之重）
2. `openrlhf/utils/loss_utils.py`（梯度累积下的全局归一化，被 `tests/test_loss_aggregation.py` 覆盖——**先跑测试再读实现**）
3. `openrlhf/trainer/ppo_utils/replay_buffer.py`（dynamic batch + Karmarkar-Karp 序列长度均衡）
4. `openrlhf/trainer/ppo_trainer.py:213-298`（train_step / ppo_train 编排）+ `trainer/ray/ppo_actor.py:155-259`（worker 内真正的优化循环）

### 核心概念

**PolicyLoss 主干**（`loss.py:116-231`）：

```
ratio = exp(clamp(logπ_θ − logπ_old, ±20))
surr1 = ratio · A
surr2 = clamp(ratio, 1−ε_low, 1+ε_high) · A
loss  = −min(surr1, surr2)                          # 标准 PPO
# dual-clip（--actor.dual_clip c>1，A<0 时加下界防过度惩罚）：
loss  = −where(A<0, max(min(surr1,surr2), c·A), min(surr1,surr2))
```

- ε 默认 0.2，可用 `--actor.eps_clip_low_high 0.2 0.27` 设非对称 clip（DAPO 的 clip-higher 技巧：抬高上界鼓励探索）。
- **GSPO**（`--actor.policy_loss_type gspo`，`loss.py:170-178`）：ratio 换成**序列级几何平均** `exp(mean_t Δlogp_t)`，且强制 sequence-level 聚合。适合 MoE 等 token ratio 噪声大的场景。

**聚合语义**（`aggregate_loss`，`loss.py:11-39`）——这是本仓库最精细的工程点：

- token-level（默认）：`Σ(loss·mask) / 全局token数 × dp_size`——长回答权重大
- sequence-level：先序列内平均、再样本间平均——每条样本等权（GRPO 论文原味）
- `× dp_size` 是为了抵消 DeepSpeed 跨 DP rank 的梯度平均，保证**梯度与单卡全量 batch 严格等价**
- `loss_utils.py:80-105`：梯度累积时把同一 optimizer step 的所有 micro-batch 的 token 数**先汇总再分摊**，避免"均值的均值"偏差——`tests/test_loss_aggregation.py` 专门验证 token 数不均时两种算法结果不同

**vLLM off-policy 校正**（`--algo.advantage.is_correction_enable`，`loss.py:196-219`）：rollout 用的 π_rollout 与训练时的 π_old 可能不同（async/partial rollout 下必然不同），用 `w = exp(logπ_old − logπ_rollout)` 做重要性采样修正：

| type | 行为 | 场景 |
|---|---|---|
| `tis` | w 截断到 [0.5, 5.0] 后乘 loss | 默认 |
| `icepop` | 阈值外的 token 直接置 0（mask 掉） | partial rollout 官方推荐 |
| `seq-mask-tis` | 序列级几何均值超阈值→整句 mask，句内再乘 w | 更保守 |

**Critic 侧**（`loss.py:234-270`）：value clip（默认 0.5）+ MSE 取 max；`--critic.freezing_steps N` 让前 N 步只训 critic 不动 actor（PPO 冷启动稳定性技巧）。

**ReplayBuffer 的 dynamic batch**（`replay_buffer.py:91-177`）：开 `--train.dynamic_batch_enable` 后不再按固定 micro_batch_size，而是按 `max_tokens_per_gpu` 的 token 预算装箱，用 Karmarkar-Karp 算法均衡各 micro-batch 的总长，跨 DP rank 对齐 step 数。变长 reasoning 输出下利用率远高于定长 batch。

### 自测

1. token-level 和 sequence-level 聚合，哪个会让"一条 8000 token 的长回答"对梯度影响更大？GRPO 论文的原始设定是哪个？
2. dual_clip 只在 advantage 为负时起作用，为什么？
3. async 训练不开 is_correction 会发生什么？（提示：π_rollout 落后 π_old 一到多个 batch，ratio 分布整体偏移）

---

## Day 5 — 分布式基础设施：hybrid engine、权重同步、async

**目标**：理解"训练 80% 时间在生成"这一根本矛盾的三种工程解法；能解释 NCCL vs CUDA IPC 权重同步的选择逻辑。

### 阅读清单

1. 文档 `hybrid_engine.rst.txt` + `async_training.rst.txt` + `performance.rst.txt`
2. `openrlhf/trainer/ppo_trainer.py:300-319, 499-570`（broadcast_to_vllm 入口 + fit 主循环）
3. `openrlhf/trainer/ray/ppo_actor.py:102-153, 438-478`（同步后端选择 + IPC/NCCL 实现 + ZeRO-3 GatheredParameters）
4. `openrlhf/trainer/ray/vllm_worker_wrap.py`（vLLM worker 侧如何收权重：`update_weight` / `update_weight_cuda_ipc` → `load_weights`）
5. `openrlhf/utils/deepspeed/deepspeed.py` + `deepspeed_utils.py`（DeepspeedStrategy：ZeRO 配置、Muon、ring attention mesh、offload/reload states）
6. `openrlhf/trainer/ppo_trainer_async.py`（GenerateSamplesActor / TrainingActor / rollout_queue / VLLMLock / partial rollout）

### 核心概念

**Hybrid engine 一个 step 的时序**（colocate_all + 双 sleep）：

```
vLLM wake(全部) → 生成 rollout → vLLM sleep
DS reload states → critic 训练 → DS offload
DS reload states → actor 训练 → DS offload
vLLM wake(仅 weights) → broadcast 权重 → （KV cache 下次生成前才 wake）
```

sleep 模式下 critic 和 actor **必须串行**（共享 GPU）；非 sleep 模式二者并行 fit（`ppo_trainer.py:283-296`）。

**权重同步三条路**（`ppo_actor.py:102-103`）：

- **CUDA IPC**：`nccl + colocate_all + 非async` 时启用——actor 和 vLLM 在同一张物理卡上，直接传 IPC handle 零拷贝共享显存。
- **NCCL broadcast**（默认）：DS rank0 与所有 vLLM worker 组一个独立 process group（world = engines×tp + 1）逐参数广播。
- ZeRO-3/TP 下参数是分片的，广播前须 `GatheredParameters` 聚合。

**Async 的本质**：生成和训练变成生产者/消费者，`--train.async_queue_size`（默认 1）就是**最大允许的 off-policy staleness（按 batch 计）**。再加 `--train.partial_rollout_enable` 则权重切换时 vLLM 只 pause/resume，in-flight 序列前半旧权重后半新权重——所以官方要求配 `is_correction`（icepop）。

**DeepspeedStrategy 要点**：

- 梯度累积自动算：`gas = train.batch_size × ring×tp / micro_batch_size / world_size`
- Muon 优化器（`--actor.optim muon`）：2D 权重走 Muon（lr 0.02）、embedding/head/1D 参数走 aux-Adam；与 `adam_offload` 不兼容；DS≥0.18.9；用 Muon 时 grad clip 建议设 0（DS 的 clip 发生在 Newton-Schulz 之后会把更新缩小约 700 倍——典型的"读了源码才知道"的坑）
- Ring attention：3D mesh (dp, sp, tp)，`ring_attn_size>1` 强制要求 `packing_samples`
- EMA（`--train.enable_ema`，β=0.992）：每 step 后在 CPU 上做滑动平均，保存时存 EMA 权重——RLHF 出模型更稳的老技巧

### 自测

1. 为什么 `vllm.enable_sleep` 和 `async_enable` 互斥？（sleep 靠训练/生成严格串行时分复用；async 恰恰要二者并行）
2. partial rollout 相比普通 async 多引入了什么噪声？框架用什么补救？
3. 什么条件下权重同步走 CUDA IPC 而不是 NCCL？为什么该条件下 IPC 更优？

---

## Day 6 — 调参艺术：性能与算法双维度

**目标**：形成自己的调参决策树。今天以官方 performance 文档 + 示例脚本对照为主。

### 性能调参决策树（官方 performance.rst 提炼）

**第一步：选部署模式**

- 单机 4-8 卡、≤13B → 模板 A：`colocate_all + vllm.enable_sleep + ds.enable_sleep + ZeRO-3 + packing_samples + dynamic_batch + sync_backend nccl`
- 多机、模型放得下但卡不够 → 模板 B：`colocate_actor_ref + colocate_critic_reward + 独立 vLLM + adam_offload`
- 追极限吞吐、已在 sync 验证收敛 → 模板 C：`async_enable (+ partial_rollout_enable + icepop)`
- 70B+ → 全分离 distributed 模式

**始终开启**：`--ds.packing_samples`（去 padding，大幅提速）、`--vllm.sync_backend nccl`、`--train.dynamic_batch_enable + max_tokens_per_gpu`。

**vllm.gpu_memory_utilization 经验值**：8B→0.6，13B→0.5，34B→0.4。

**OOM 处理优先级**（顺序执行，别乱跳）：

1. `packing_samples` + `gradient_checkpointing_enable`
2. 降 `train.micro_batch_size` / `rollout.micro_batch_size`
3. 降 `vllm.gpu_memory_utilization`
4. `ds.adam_offload` + 提 `ds.zero_stage`（2→3）
5. 最后手段：去掉 colocation

**长上下文（>8K）**：`ring_attn_size 2 + ring_attn_head_stride 2` 起步，配 ZeRO-3 + packing。

**batch 关系式**：`train.batch_size = rollout.batch_size × n_samples_per_prompt`（examples 脚本基本都守这个）；生成侧偏好**多 engine 小 TP** 而非少 engine 大 TP。

### 算法调参速查（结合示例脚本实证）

| 目标 | 推荐配置 | 出处 |
|---|---|---|
| RLVR / 数学推理（首选） | `estimator=reinforce_baseline, n_samples=8-16, kl.use_loss + k2, init_coef=1e-5, dynamic_filtering 0 1` | `train_prorlv2_math_hybrid_engine.sh`（ProRL V2 实战配置） |
| 经典 RLHF（有 RM） | `estimator=gae, kl k1, init_coef=0.01, actor lr 1e-6 / critic lr 9e-6, critic.freezing_steps 预热` | `train_ppo_ray_hybrid_engine.sh` |
| DAPO 复现 | `estimator=group_norm, eps_clip_low_high 0.2 0.27, kl.use_loss + k3, n_samples=8, dynamic_filtering, overlong_buffer_len` | `train_dapo_ray_hybrid_engine.sh` |
| 长 CoT 防刷长度 | `reward.overlong_buffer_len + overlong_penalty_factor`；截断样本 `stop_properly_penalty_coef 0`（清零）或负值 | ProRL 脚本 |
| async/partial rollout | 必配 `is_correction_enable + is_correction_type icepop`，阈值默认 0.5 5.0 | `train_reinforce_baseline_ray_agent_async.sh` |
| MoE / token ratio 噪声大 | `policy_loss_type gspo` | GSPO 论文场景 |
| 训练不稳、被负样本拖崩 | 加 `dual_clip 3` | dual-clip PPO 论文 |

**几个关键默认值**（背下来）：actor lr `1e-6`、critic lr `9e-6`、eps_clip `0.2`、value_clip `0.5`、kl init_coef `0.01`、γ=λ=`1.0`、reward clip `(-10,10)`、scheduler `cosine_with_min_lr`（warmup 3%、min_lr_ratio 0.1）、EMA β `0.992`。

### Troubleshooting 高频坑（troubleshooting.rst + 源码验证）

- 旧版平铺 flag（`--actor_num_nodes`）全部失效，报 unrecognized arguments → 查 common_options 的迁移表
- Ray 下 GPU device index 错乱 → `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
- vLLM 挂起 → 试 `--vllm.enforce_eager`（禁 CUDA graphs）
- Muon：需 DS≥0.18.9、与 adam_offload 互斥、ns_steps/nesterov 是占位符改了无效、grad clip 设 0
- 卡死排查：`py-spy top --pid`（容器要 `--cap-add=SYS_PTRACE`）
- LoRA 只支持 SFT/RM/DPO，**Ray+vLLM PPO 不支持**
- VLM：不支持 packing_samples、不支持 critic（必须 critic-free estimator）、attn 用 eager

### 动手

把 `train_prorlv2_math_hybrid_engine.sh`、`train_dapo_ray_hybrid_engine.sh`、`train_reinforce_baseline_ray_agent_async.sh` 三个脚本做一张 diff 表：同一列是 flag，每行一个脚本——你会直观看到"算法差异只是十来个 flag 的差异"，这正是 agent 范式解耦的价值。

---

## Day 7 — 收官：非 RL 路径、多轮 Agent 与综合实战

**目标**：补齐 SFT/RM/DPO 三条支线；能独立写多轮 agent；用三个实战练习检验一周成果。

### 阅读清单

1. 文档 `non_rl.rst.txt` + `openrlhf/trainer/{sft,rm,dpo}_trainer.py` + `openrlhf/datasets/`（packing、chat template 处理）
   - RM 训练：PairWiseLoss `−logσ(r_c − r_r − margin)` 或 LogExpLoss；value_head_prefix 默认 `score`
   - DPO：β 典型 0.1-0.5、`ipo_enable`、`label_smoothing`(cDPO)、`nll_loss_coef`（Llama 3.1 报告的 NLL 正则）
2. 文档 `agent_training.rst.txt` 多轮部分 + `examples/python/` 下的 agent 示例（含 OpenAI 兼容 server executor：把本地 vLLM 包成 `/v1/chat/completions` 同时截获 token trace 训练——工具调用 RL 的标准姿势）
3. 文档 `checkpoint.rst.txt`：DS ckpt（可恢复，含 optimizer/dataloader 状态）vs HF 格式（`--ckpt.save_hf`）；best checkpoint 按 eval metric 轮转；`--ds.use_universal_ckpt` 跨并行度恢复

### 实战练习（检验一周成果）

**练习 1（算法）**：不看笔记，手推 REINFORCE++-baseline 和 GRPO 的完整 advantage 计算（从 raw reward 到进 loss 的张量），然后在 `experience_maker.py` 里逐行验证。能指出二者在"全局归一化"上的差异算过关。

**练习 2（代码）**：给 `PolicyLoss` 加一个假想的新 `policy_loss_type`（比如把 clip 换成 KL 罚项的 PPO-penalty 变体），在 `tests/` 下仿照 `test_loss_aggregation.py` 写单测并跑通（Mac 上可跑，纯 CPU 逻辑）。这会强迫你吃透 ratio/mask/聚合的全部张量形状。

**练习 3（系统设计）**：给定"2 节点 × 8×A100-80G，训 32B 模型做数学 RLVR"，写出完整启动脚本：选部署模式、estimator、KL 方案、长度惩罚、batch 关系、vLLM 参数，每个 flag 写一行理由。写完对照 examples 里最接近的脚本互评。

**练习 4（多轮 agent，可选）**：实现一个 `AgentInstanceBase` 子类做简单环境（如 20 questions 或计算器工具调用），接到 `--train.agent_func_path`，理解 reset/step 协议和 environment_feedback 的 token 拼接。

### 延伸阅读（源码对应的论文）

| 代码 | 论文 |
|---|---|
| `reinforce`/`reinforce_baseline` | REINFORCE++ (arXiv:2501.03262)；ScaleRL、ProRL V2 为其大规模实证 |
| `group_norm` | GRPO（DeepSeekMath, arXiv:2402.03300） |
| `dr_grpo` | Dr. GRPO (arXiv:2503.20783) |
| `dynamic_filtering` + `eps_clip_low_high` + overlong penalty | DAPO (arXiv:2503.14476) |
| `policy_loss_type=gspo` | GSPO (arXiv:2507.18071) |
| `dual_clip` | Dual-clip PPO (arXiv:1912.09729) |
| k1/k2/k3 | Schulman, *Approximating KL Divergence*（joschu.net/blog/kl-approx） |
| TIS/ICEPOP | fengyao.notion.site/off-policy-rl |
| Adaptive KL | Ziegler et al. (arXiv:1909.08593) |

---

## 附录 A：一页公式速查

```
PPO:        L = −E[min(r·A, clip(r, 1−εl, 1+εh)·A)],  r = π_θ/π_old
dual-clip:  A<0 时 L = −max(min(surr1,surr2), c·A)
GSPO:       r_seq = exp(mean_t log r_t)，seq-level 聚合
GAE:        δ_t = r_t + γV_{t+1} − V_t;  A_t = δ_t + γλ·A_{t+1};  ret = A + V
REINFORCE++:A_t = Σ_k γ^k r_{t+k}，全局 (μ,σ) 归一化
RLOO:       A_i = r_i − (Σ_j r_j − r_i)/(n−1)
R++-baseline: A_i = r_i − r̄_group，再全局 /σ
GRPO:       A_i = (r_i − r̄)/(σ_group + 1e-9)
Dr.GRPO:    A_i = r_i − r̄（无任何 σ 归一化）
KL:  k1 = Δ;  k2 = Δ²/2;  k3 = e^{−Δ} − 1 + Δ   （Δ = logπ − logπ_ref, clamp ±10）
reward 合成（use_kl_loss=False）: r_t = −β·KL_t + 𝟙[t=EOS]·clip(r, −10, 10)
DAPO overlong: penalty = −min(len − (max_new − buf), buf)/buf × factor
token-level 聚合: Σ(l·m)/N_tokens_global × dp    seq-level: mean_i(mean_t l_it) × dp
IS 校正: w = exp(logπ_old − logπ_rollout)；tis=clamp(w)，icepop=mask(w∉[lo,hi])
```

## 附录 B：学习习惯建议

1. **每读一个模块先跑/写测试**：`tests/` 在 Mac 可跑，是唯一的本地验证手段（训练路径需要多卡 CUDA，别在本机试图启动训练）。
2. **grep 用嵌套名**：`rg "kl\.init_coef" openrlhf/` 而不是搜 `--algo.kl.init_coef`。
3. **文档→源码→脚本三角验证**：文档说语义、源码定真相、examples 脚本给实证组合。有出入以源码为准（本仓库文档与代码版本同步得很好，但读代码的习惯要保持）。
4. **中文文档**在 `zh/latest/`，与英文同名对照读，术语翻译由管线生成、以英文为准。
5. 学完后真正的毕业考：**去 upstream 的 issue 列表挑一个训练稳定性相关的 issue，用这一周的知识写出诊断分析。**
