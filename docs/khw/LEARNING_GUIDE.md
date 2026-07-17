# OpenRLHF 七天吃透指南（案例驱动版）

> 面向：有 ML/LLM 基础、想吃透分布式 RLHF 训练全部细节与调参技巧的工程师。
> 方法论：**费曼学习法 × 一条真实数据的生命周期**。每一天你都跟着同一条数学题样本在系统里走几站，走到哪读到哪，最后能不看笔记把整条链路讲给别人听。每天末尾的"费曼自测"就是检验——答不上来就回去重读那一站的源码。

配套资料（本仓库内，均可离线阅读）：

- 官方文档镜像（中英双语）：`docs/official-docs/openrlhf.readthedocs.io/{en,zh}/latest/`，纯文本源码在 `_sources/*.rst.txt`（比 HTML 好读，本指南按此引用）
- 仓库自带导航：`.claude/docs/architecture.md`、`.claude/docs/official-docs-map.md`
- 可运行参考配置：`examples/scripts/*.sh`（**flag 组合的事实标准**）

**三角验证习惯**：文档说语义、源码定真相、examples 脚本给实证组合。本指南所有行号都对着当前仓库源码核对过，但代码会演化——读的时候保持"行号是路标不是真理"的心态。

---

## 序章：贯穿全书的主线案例

费曼说"凡我不能创造的，我就不理解"。对一个训练框架来说，"创造"的最小形式是：**给定一条具体输入，手算出它在每个环节的具体输出**。所以我们先固定一个真实实验和一条具体数据，后面七天全部围绕它展开。

### 实验配置：官方文档的默认 RL 示例

这是官方文档 `hybrid_engine.rst.txt` 给出的标准 launch recipe——**Qwen3-4B-Thinking 在数学数据集上做 RLVR**（Reinforcement Learning with Verifiable Rewards），奖励来自一个 Python 验证函数而非 reward model。仓库里最接近的脚本是 `examples/scripts/train_reinforce_baseline_hybrid_engine.sh` 与 `train_prorlv2_math_hybrid_engine.sh`。

| 维度 | 取值 | 对应 flag |
|---|---|---|
| 被训模型 | Qwen3-4B-Thinking-2507 | `--actor.model_name_or_path` |
| 数据集 | dapo-math-17k（约 17k 条 `{prompt, label}`） | `--data.prompt_dataset`，`--data.input_key prompt --data.label_key label` |
| 奖励 | `examples/python/math_reward_func.py`（验证 `\boxed{}` 答案，对=1 错=0） | `--reward.remote_url <py文件>` |
| 硬件 | 1 节点 × 4 GPU，全员共卡 | `--actor.num_gpus_per_node 4 --ref.num_gpus_per_node 4 --train.colocate_all` |
| vLLM | 2 个 engine × TP=2，睡眠模式 | `--vllm.num_engines 2 --vllm.tensor_parallel_size 2 --vllm.enable_sleep` |
| 算法 | REINFORCE++-baseline（critic-free） | `--algo.advantage.estimator reinforce_baseline` |
| KL | 作为 loss 项，k2 估计器，系数 1e-5 | `--algo.kl.use_loss --algo.kl.estimator k2 --algo.kl.init_coef 1e-5` |
| 采样规模 | 128 prompt/轮 × 每题 8 条 = 1024 条序列 | `--rollout.batch_size 128 --rollout.n_samples_per_prompt 8` |
| 训练批 | 1024（= 一轮 rollout 恰好一次参数更新） | `--train.batch_size 1024` |
| 动态过滤 | 组平均 reward ∉ (0,1) 的 prompt 整组丢弃 | `--algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0 1` |
| off-policy 校正 | ICEPOP，阈值 [0.5, 5.0] | `--algo.advantage.is_correction_enable --algo.advantage.is_correction_type icepop` |
| 序列长度 | max_len 74240 / max_new_tokens 64000（长 CoT） | `--data.max_len --rollout.max_new_tokens` |
| 并行 | ZeRO-3 + bf16 + ring attention 2 + 动态 batch | `--ds.zero_stage 3 --ds.ring_attn_size 2 --train.dynamic_batch_enable` |

几个立刻能推出来的派生量（Day 1 会解释每一个）：

- actor world_size = 4；ring_attn_size=2、tp=1 ⇒ **DP 组数 = 4/2/1 = 2**，`duplicate_actors = 2`
- 一轮 rollout 产出 128×8 = 1024 条序列 = 恰好 1 个 `train.batch_size` ⇒ **每轮 rollout 只做 1 次 optimizer step**
- `colocate_all` 断言：actor 卡数 4 == vllm 2×2 ✓（`train_ppo_ray.py:56-64`）
- estimator ≠ gae ⇒ critic 不创建（`train_ppo_ray.py:600-601`）；reward 走 py 文件 ⇒ RewardModelActor 不部署

### 主线数据：样本 S₁

从 dapo-math-17k 里取一条（题目为教学而选的简短示例，字段格式与真实数据一致）：

```json
{
  "prompt": "Find the sum of all positive divisors of 36. Please reason step by step, and put your final answer within \\boxed{}.",
  "label": "91"
}
```

（36 的正因数：1+2+3+4+6+9+12+18+36 = 91。）

约定几个贯穿全书的记号（token 数为教学假设值，标 ≈ 的都是示意）：

- **P** = 这条 prompt；套完 chat template 后记 prompt 长度 **L_p = 48 token**
- 该 prompt 采样 8 条回答，记 **S₁…S₈**；奖励向量 **r = [1, 1, 0, 0, 0, 0, 0, 0]**（2 对 6 错，组均值 0.25）
- 重点跟踪 **S₁**（答对的那条）：回答 **L_a = 210 token**，以 `...\boxed{91}<|im_end|>` 结尾，`finish_reason="stop"`（未截断）
- 完整序列长度 **T = 48 + 210 = 258**

### S₁ 的十二站旅程（全书地图）

```
┌── Day 2 ──────────────────────────────────────────────────────────┐
│ ① PromptDataset：套 chat template，P 变成带 <|im_start|> 的字符串    │
│ ② SamplesGenerator：最小堆负载均衡，把 P 派给 2 个 vLLM engine 之一  │
│ ③ SingleTurnAgentExecutor：tokenize → vLLM 生成 210 token           │
│    → decode 全文 → math_reward_func 判卷 → reward = 1.0            │
│ ④ 组装 Experience：sequences(1,258) / action_mask(1,257)（右移1位） │
│    动态过滤：组均值 0.25 ∈ (0,1) ⇒ 保留（全对/全错的组被丢弃重采）    │
├── Day 3 ──────────────────────────────────────────────────────────┤
│ ⑤ RemoteExperienceMaker：actor/ref 前向 → 三种 log_probs 齐了       │
│ ⑥ reward 流水线：长度惩罚 → 组内减 baseline（1→0.75）→ clip →       │
│    散播到 EOS → 逐 token return=0.75 → 全局归一化（≈1.50，示意）     │
├── Day 4 ──────────────────────────────────────────────────────────┤
│ ⑦ balance_experiences + ReplayBuffer：按长度均衡切给 2 个 DP rank    │
│ ⑧ 动态 batch：Karmarkar-Karp 按 16192 token/GPU 预算装箱            │
│ ⑨ training_step：PolicyLoss（ratio≡1 首步）+ k2 KL loss + icepop    │
│ ⑩ 梯度累积 → 唯一一次 optimizer step，权重 θ₀ → θ₁                  │
├── Day 5 ──────────────────────────────────────────────────────────┤
│ ⑪ broadcast_to_vllm：本配置命中 CUDA IPC 零拷贝路径                 │
│ ⑫ vLLM 拿到 θ₁ → 下一轮采样；周期性 eval（pass@k）与 checkpoint     │
└───────────────────────────────────────────────────────────────────┘
```

### 读代码前的两个关键约定

1. **CLI flag 是点分层级的**（`--algo.kl.init_coef`），`openrlhf/utils/config.py` 的 `hierarchize()`（全文 26 行）把平铺 argparse 命名空间按 `.` 重组为嵌套 SimpleNamespace：`args.algo.kl.init_coef`。**grep 时搜嵌套名**（如 `rg "kl\.init_coef"`），别搜 CLI 拼写。全部 flag 的权威语义在文档 `common_options.rst.txt`；全部默认值在 `cli/train_ppo_ray.py:198-587` 的 argparse 定义里。
2. **argparse 之后还有一段"参数改写区"**（`train_ppo_ray.py:594-717`）：`agent_func_path` 会把 `reward.remote_url` 改成 `"agent"`；非 gae 直接抹掉 critic 路径；`ring_attn_size>1` 强制打开 `packing_samples`；`n_samples_per_prompt=1` 配组内 baseline 类 estimator 会直接 assert 掉（否则 advantage 全 0，训练变成静默空转）。**很多"为什么我设了 X 没生效"的答案都在这一段。**

---

## Day 1 — 全局架构与启动流程

> **目标**：能白板画出本案例的 Ray actor 拓扑与 4 张 GPU 的分时复用布局；理解 placement group 与三种 colocation 模式；跑通本地测试。

### 阅读清单（按顺序）

1. 文档 `index.rst.txt` + `architecture.rst.txt`（20 分钟，建立官方视角：Ray 编排 + vLLM 生成 + DeepSpeed 训练三位一体）
2. `openrlhf/utils/config.py`（26 行，先懂参数系统）
3. `openrlhf/cli/train_ppo_ray.py` 的 `train()`（:19-195）逐行通读——这就是整个系统的"main 函数"
4. `openrlhf/trainer/ray/launcher.py`（:202-373 `RayActorGroup`；:17-101 `BaseDistributedActor`/`BaseModelActor`）
5. `openrlhf/trainer/ray/vllm_engine.py` 的 `create_vllm_engines`（:209-311）

### 案例推演：这条命令跑起来的前 60 秒

对照 `train()` 源码，本案例的启动时序是：

1. **`ray.init`**（:21-32）：设置 `NCCL_DEBUG=WARN`、零拷贝 tensor 等运行时环境变量。
2. **建 placement group**（:41-50）：`colocate_all` ⇒ 为 actor 建 4 个 `{GPU:1, CPU:1}` bundle 的 PG（策略 PACK）。`kl.init_coef=1e-5 > 0` ⇒ 断言 ref 与 actor 的卡数配置一致。
3. **创建 vLLM engines**（:52-82 → `vllm_engine.py:209-311`）：2 个 `RolloutRayActor`，TP=2 ⇒ `distributed_executor_backend="ray"`，每个 engine 通过 `VLLM_RAY_BUNDLE_INDICES` 钉在 PG 的指定 bundle 上。**engine 内部持有 executor**：本案例 `remote_rm_url` 是 py 文件 ⇒ `SingleTurnAgentExecutor`（Day 2 主角）。`is_correction_enable` ⇒ `logprobs_mode="processed_logprobs"`（要求 vLLM > 0.10.0），生成时返回逐 token logprob。
4. **创建 4 个 `RayActorGroup`**（:84-142）：
   - PolicyModelActor 组：4 个 actor，PG 下每个只申请 **0.2 GPU**（:89）——这是共卡的实现手法：actor/ref/vLLM 三方在同一张物理卡上各占 0.2 的"记账额度"，实际显存靠 sleep/offload 分时腾挪；
   - ReferenceModelActor 组：同上（`kl.init_coef>0` 才建）；
   - CriticModelActor：**本案例不存在**（estimator=reinforce_baseline）；
   - RewardModelActor：**本案例不存在**（reward 是 py 函数，在 vLLM engine 侧算）。
5. **选 trainer**（:144-148）：`async_enable` 未开 ⇒ `ppo_trainer.PPOTrainer`（一个 CPU 上的 Ray actor，"single controller"）。
6. **初始化顺序**（:170-185）：先 actor/ref/reward 并行 `init_model_from_pretrained`，**critic 必须最后**——它的 lr scheduler 需要 `max_steps`，而 `max_steps` 由 trainer 里的 dataset 长度算出（`ppo_trainer.py:72-79`：`len(dataset) × n_samples // train_bs × episodes × epochs`，本案例 ≈ 17000×8/1024 ≈ 132 步/episode）。
7. **`ppo_trainer.fit.remote()`**（:188）进入训练主循环（Day 2 起逐站拆）。

最终 GPU 布局（把它画在白板上，这是 Day 1 的验收标准）：

```
GPU0        GPU1        GPU2        GPU3
─────────── ─────────── ─────────── ───────────
actor r0    actor r1    actor r2    actor r3      ← DeepSpeed ZeRO-3, 各占0.2
ref   r0    ref   r1    ref   r2    ref   r3      ← 各占0.2
vLLM eng0-w0 eng0-w1    eng1-w0     eng1-w1       ← 2 engine × TP2, 各占0.2
─────────── ─────────── ─────────── ───────────
ring attention 分组: (r0,r1)=sp组0, (r2,r3)=sp组1 ⇒ DP=2
同一时刻只有一方真正"醒着"占大头显存（Day 5 讲 sleep/wake 时序）
```

### RayActorGroup：控制面的核心抽象

`launcher.py:202-373`。一组同类 GPU actor 的封装，三件事必须吃透：

- **拉起 world_size 个 rank**（:240-290）：rank0 先起，把自己的 IP:port 通过 `get_master_addr_port` 告诉其余 rank；每个 actor 是 `@ray.remote(num_gpus=1)` 类，自己在 `BaseDistributedActor.__init__` 里设 `MASTER_ADDR/RANK/WORLD_SIZE` 环境变量，再由 DeepspeedStrategy 建 torch.distributed 进程组。
- **`async_run_method(method_name, ...)`**：同一调用广播到全部 rank，返回 ref 列表（用于 `fit`、`offload_states` 这类全员动作）。
- **`async_run_method_batch(...)`**（:319-373）：把一个 batch 切成 `effective_actors = world_size // duplicate_actors` 份，**同一 ring/TP 组内的 rank 拿同一份数据**（`duplicate_actors = ring_attn_size × tensor_parallel_size`，本案例 = 2）。数据先切片再 `ray.put`，避免整批数据广播到所有节点。Day 3 的 forward fan-out、Day 4 的 experience 下发都走它。

配套的 `BaseModelActor.execute_batch`（:67-101）在 worker 侧逐条调用目标方法——所以 `forward` RPC 的粒度是"每 rank 一串 micro-batch"。

### 三种部署形态（性能文档的第一决策点）

| 模式 | flag | GPU 布局 | 适用 |
|---|---|---|---|
| **Hybrid Engine**（本案例，官方推荐起步） | `--train.colocate_all --vllm.enable_sleep --ds.enable_sleep` | 所有角色 + vLLM 分时复用同一组卡 | 单机 4-8 卡、~1.5B-13B；官方定位"max stability" |
| 部分共卡 | `--train.colocate_actor_ref` + `--train.colocate_critic_reward` | actor/ref 一组、critic/reward 一组、vLLM 独立 | 多机中等模型 |
| 全分离（Distributed） | 不加任何 colocate | 每角色独占 GPU 组 | 70B+ 或异构硬件 |

三条硬约束（源码里都有 assert/改写，`train_ppo_ray.py:56-64, 663-674`）：

1. `colocate_all`（非 async）要求 `actor 卡数 == vllm.num_engines × vllm.tensor_parallel_size`；
2. `vllm.enable_sleep` 与 `train.async_enable` **互斥**（sleep 靠严格串行分时，async 恰恰要并行）；
3. async 模式下 `colocate_all` 只共卡 DeepSpeed 各模型，vLLM 独立占卡（要一直生成）。

### 费曼自测（讲不出来就回去重读）

1. 为什么 PG 模式下每个 actor 只申请 0.2 GPU？"0.2"是显存配额吗？
   <details><summary>答案</summary>不是显存配额，只是 Ray 的调度记账值，目的是让 actor/ref/critic/reward/vLLM 五方能被调度到同一张物理卡。真实显存靠 vLLM sleep + DeepSpeed offload 分时腾挪。</details>
2. 本案例里为什么 critic 和 RewardModelActor 都不存在？各自省在哪一行？
   <details><summary>答案</summary>estimator=reinforce_baseline ⇒ `train_ppo_ray.py:600-601` 把 `critic.model_name_or_path` 置 None；reward 走 py 文件 ⇒ `:132-142` 不建 RewardModelActor，判卷逻辑在 vLLM engine 侧的 SingleTurnAgentExecutor 里执行。</details>
3. 为什么 critic 的初始化必须等 actor 之后？
   <details><summary>答案</summary>critic 的 lr scheduler 需要 max_steps；max_steps 依赖 trainer 构建 prompts dataset 之后才能算出（`train_ppo_ray.py:181-185` 的注释）。</details>
4. `duplicate_actors` 是什么？本案例等于几？如果改成 `--ds.tensor_parallel_size 2` 会变成几？
   <details><summary>答案</summary>ring_attn_size × ds tensor_parallel_size，即"共享同一份数据的 rank 数"。本案例 2×1=2；再开 TP=2 就是 4——此时 4 张卡只剩 1 个 DP 组。</details>

### 动手

- `pytest tests/` 在 Mac 上能直接跑（纯逻辑测试），确认环境 OK。
- 打开 `examples/scripts/train_ppo_ray_hybrid_engine.sh`（经典 PPO+RM 版），对照今天内容画出它的 8 卡布局图：actor/ref/critic/reward 各 0.2 共享 + 4 个 TP=2 的 vLLM engine，全部落在同 8 张卡上。和本案例的差异恰好是"多了 critic 和 reward 两类 actor"。
---

## Day 2 — 采样：S₁ 的诞生（第①-④站）

> **目标**：吃透从 prompt 字符串到 `Experience` 对象的每一步；理解"token-in-token-out"为什么是这个框架的设计基石；能手画 action_mask 的右移对齐图。

### 阅读清单

1. 文档 `agent_paradigm.rst.txt`（设计哲学：**执行模式 × RL 算法 × pipeline 三轴正交**，所有组合都合法）
2. `openrlhf/datasets/prompts_dataset.py`（:26-46 `preprocess_data`）
3. `openrlhf/trainer/ppo_utils/samples_generator.py` 全文（314 行）
4. `openrlhf/utils/agent.py`：`SingleTurnAgentExecutor`（:184-301）精读，`MultiTurnAgentExecutor`（:31-181）通读
5. `openrlhf/trainer/ray/vllm_engine.py`：`RolloutRayActor.generate` / `generate_responses`（:153-206）
6. `openrlhf/trainer/ppo_utils/experience.py`：`Experience` 字段（:28-69）+ `_process_response_into_experience`（`samples_generator.py:247-314`）
7. `examples/python/math_reward_func.py`（52 行，判卷函数真身）

### 第①站：PromptDataset —— 字符串阶段的唯一一次加工

`prompts_dataset.py:26-46`：开了 `--data.apply_chat_template`，P 被包成消息再由 tokenizer 的 chat template 展开（`add_generation_prompt=True`）：

```
<|im_start|>user
Find the sum of all positive divisors of 36. Please reason step by step,
and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
```

此后 P 以**字符串**形式存进 `PromptDataset.prompts`，label `"91"` 存进 `labels`。注意：**tokenize 不在这里发生**——发生在 vLLM engine 侧的 executor 里（第③站）。dataloader 的 batch_size 是 1（`ppo_trainer.py:42-49`），由 SamplesGenerator 自己攒批。

### 第②站：SamplesGenerator —— 攒批、派发、过滤的调度中枢

入口是训练主循环 `PPOTrainer.fit`（`ppo_trainer.py:516-564`）每轮调用的 `generate_samples`（`samples_generator.py:89-138`）：

- **chunk 机制**：一个训练 chunk = `rollout.batch_size × n_samples_per_prompt` = 128×8 = **1024 条 Experience**。`_sample_buffer` 不够就唤醒 vLLM 生成一批（`vllm_generate_batch_size` 默认 = rollout.batch_size；设得更大做超采样需要 async 模式）。
- **sleep 协同**（:106-120）：生成前 `wake_up`，生成后 `sleep`——hybrid engine 的分时复用在这里落地。
- **最小堆负载均衡**（`_dispatch_prompts_to_vllm`，:199-245）：先查询每个 engine 的未完成请求数入堆，每派一个 prompt 弹出最闲的 engine、计数 +n_samples 后压回。本案例 128 个 prompt 在 2 个 engine 上交替派发，每次 `generate_responses.remote(prompt, label, sampling_params, ...)` 让 engine 一次生成该 prompt 的全部 8 条。
- **SamplingParams**（:203-211）：temperature 1.0、`max_tokens=64000`（来自 `max_new_tokens`；不设时为 None=按 prompt 长度动态分配）、`logprobs=1`——**只因为开了 is_correction 才要 vLLM 返回逐 token logprob**，不开就不传输。
- **流式回收**（:161-196）：`ray.wait` 每完成一个 prompt 的 8 条就立即处理，不等全批。

### 第③站：SingleTurnAgentExecutor —— 生成 + 判卷都在 engine 侧

`agent.py:184-301`。engine 收到 `generate_responses` 后并发跑 8 个 `executor.execute`（`vllm_engine.py:194-206`，asyncio.gather）。对 S₁ 逐行推演：

1. **tokenize**（:211-213）：`add_special_tokens=False`（template 已含特殊 token）→ 48 个 prompt token id。
2. **生成预算**（:216-234）：本案例显式设了 `--rollout.max_new_tokens 64000` ⇒ `max_tokens=64000`，prompt 预算只剩 74240−64000=10240，超长 prompt 会被**左截断**保住生成空间。若不设 max_new_tokens 则走动态分支：`max_tokens = max(1, max_len − L_p) = 74192`。
3. **生成**（:240-243）：`llm_engine.generate(prompt_token_ids, ...)` **token 进 token 出**——engine 的 `generate`（`vllm_engine.py:153-177`）收发的都是 token id。S₁ 得到 210 个 action token，`finish_reason="stop"` ⇒ `is_truncated=False`。
4. **拼接与标注**（:249-251）：`observation_tokens = prompt(48) + action(210)`，`action_ranges = [(48, 258)]`。
5. **rollout log probs**（:253-259）：prompt 段补 48 个 0.0 占位，action 段逐 token 取 vLLM 返回的 logprob → 长度 258 的列表。
6. **判卷**（:281-299）：把 258 个 token **decode 回文本**（这是唯一一次 detokenize，且只用于算 reward，不回流训练），调 `reward_func([query], [prompt], ["91"])`：

```python
# examples/python/math_reward_func.py（节选）
pred_answer = extract_boxed_answer(response)   # → "91"
is_correct = grade_answer(pred_answer, label)  # → True
rewards.append(1.0 if is_correct else 0.0)
return {"rewards": tensor([1.0]), "scores": tensor([1.0]),
        "extra_logs": {"math_accuracy": ...}}
```

自定义 reward 函数的签名（RLVR 最常用扩展点）就是这个：`reward_func(queries, prompts, labels) → {"rewards", "scores", "extra_logs"}`。`rewards` 进 advantage 计算，`scores` 进动态过滤判断，`extra_logs` 进 wandb。HTTP remote RM 是同一接口的网络版（:322-356，5 次重试）。

> **Token-in-token-out（全框架第一设计基石）**：轨迹从 vLLM 到训练全程以 token id + logprob 传递，**从不重新 tokenize 文本**。这消灭了 chat template 漂移、BOS/EOS 不一致、特殊 token 处理差异等一整类 silent bug——官方文档明说这是"最常见的 RLHF 隐性 bug 来源"。很多自研框架的"收敛玄学"根源就在这。

### 第④站：组装 Experience —— 那个必须画在纸上的右移

`_process_response_into_experience`（`samples_generator.py:247-314`）把 executor 的返回打包成张量。S₁ 的具体形状：

```
索引:        0 ... 46  47 | 48 ... 256  257
token:      <prompt 48 个>  <action 210 个, 末尾 <|im_end|>>

sequences      (1, 258)  全部 258 个 token id
attention_mask (1, 258)  全 1（右 padding 之前）
action_mask 原始: 长 258，位置 48..257 为 1
action_mask 切片: action_mask[1:truncate] → (1, 257)，位置 47..256 为 1   ← 左移了一位!
rollout_log_probs[1:] → (1, 257)，与 action_mask 对齐
```

**为什么切 `[1:]`**（:267）：causal LM 里 `log_prob[i]` 是"看到第 0..i 个 token 后预测第 i+1 个"的对数概率。第一个 action token（索引 48）的 logprob 来自模型在索引 47 处的输出。所以逐 token 量（logprob/KL/advantage/value）的坐标系整体比 `sequences` **左移一位**，长度 T−1。后面读 loss 代码时脑中必须有这幅对齐图——`action_mask` 出现的每一处，形状都是 `(B, T-1)` 坐标系（源码注释里的 `(B, A)`）。

`Experience`（`experience.py:28-69`）字段按 RL 语义分组，用 `tensor_field("step"/"episode")` 标记：step 张量形如 `(B, T)` 或 `(B, T-1)` 参与 padding 合并；episode 张量形如 `(B,)` 直接 stack。**三个最容易混淆的 log_probs**：

| 字段 | 含义 | 谁算的 | 何时用 |
|---|---|---|---|
| `rollout_log_probs` | log π_rollout：**生成这条样本时** vLLM 的策略 | vLLM（采样时） | off-policy 校正（icepop/tis） |
| `action_log_probs` | log π_old：训练前 actor 用当前权重重算 | actor forward（Day 3） | PPO ratio 的分母 |
| `base_action_log_probs` | log π_ref：冻结参考模型 | ref forward（Day 3） | KL 正则 |

sync 模式下 π_rollout 和 π_old 是**同一套权重**，但 vLLM 与 DeepSpeed 的 bf16 kernel 数值不同，logprob 仍有微小差异；async/partial rollout 下二者是**不同版本的权重**，差异是系统性的——这就是 is_correction 存在的原因（Day 4 展开）。

### 动态过滤（DAPO 的 Dynamic Sampling）：整组生死

回到 `_generate_vllm`（:169-196）。每个 prompt 的 8 条全部到齐后：

- **S₁ 所在组**：scores = [1,1,0,0,0,0,0,0]，均值 0.25，落在开区间 (0,1) 内 ⇒ **保留**。
- **反例组 P′**（比如"1+1=?"）：8 条全对，均值 1.0，不满足 `0 < 1.0 < 1` ⇒ **整组丢弃**，并立刻从 dataloader 补取 1 个新 prompt 派发（:184-195）。全错组（均值 0）同理。

直觉：全对/全错的组减掉组均值后 advantage 恒为 0，白白浪费一次前向反向；过滤掉它们等于把算力集中在"有梯度信号"的题目上。`filter_pass_rate` 会被记进日志——它也是**课程难度指示器**（pass rate 越来越高说明题变简单了）。

工程约束（`train_ppo_ray.py:700-709` 的 assert）：过滤发生在 samples_generator，**早于** experience_maker，所以 reward 必须在生成阶段就能拿到 ⇒ 只支持 remote RM / reward_func / agent 路径（RM actor 的 reward 要到 Day 3 才算出来，赶不上）。

### 多轮 Agent 预览（Day 7 深入）

`MultiTurnAgentExecutor`（`agent.py:31-181`）是同一接口的多轮版：`while not done: 生成 → env.step() → 把环境反馈 tokenize 后拼进上下文`。两个关键机制现在先记住：

- `action_ranges` 记录**多段** (start, end)——只有模型生成的 token 算 loss，环境反馈 token 的 `action_mask=0`；
- 环境反馈 token 的 `rollout_log_probs` 补 0.0 占位（:159），反正 mask 会把它们排除在 loss 外。

### 费曼自测

1. `n_samples_per_prompt=8` 且 `rollout.batch_size=128`，一次 `generate_samples` 至少消耗多少个 prompt？开了动态过滤后可能更多还是更少？
   <details><summary>答案</summary>产出 1024 条序列、至少消耗 128 个 prompt；动态过滤会**更多**——每个被过滤的组都会补发新 prompt（prompts_consumed 单调增），pass rate 60% 时约消耗 213 个。</details>
2. 整条链路里唯一一次 detokenize 发生在哪、为什么它不违反 token-in-token-out 原则？
   <details><summary>答案</summary>`SingleTurnAgentExecutor.execute` 里 decode 全文给 reward_func 判卷（agent.py:283）。它只用于计算 reward 标量，文本不回流到训练数据——训练消费的仍是原始 token id。</details>
3. 为什么 `action_mask` 要做 `[1:truncate_length]` 切片？切完之后第一个 action token 的 logprob 对应 mask 的哪个下标？
   <details><summary>答案</summary>logprob 坐标系比 token 坐标系左移一位（预测下一个 token）。S₁ 的第一个 action token 在 sequences 索引 48，其 logprob/mask 在切片后坐标系的索引 47。</details>
4. 动态过滤为什么设成开区间 (0,1) 而不是闭区间？
   <details><summary>答案</summary>目的恰恰是剔除均值恰好等于 0（全错）或 1（全对）的组——这些组组内无方差、baseline 减完 advantage 全 0，无学习信号。</details>

---

## Day 3 — Reward 管线与六种 Advantage（第⑤-⑥站）

> **目标**：对每种算法（PPO/REINFORCE++/RLOO/GRPO/Dr.GRPO）能**手写** advantage 公式并代入 S₁ 的数字算出来；说清 KL 的两种用法在代码里的分叉点。

### 阅读清单

1. `openrlhf/trainer/ppo_utils/experience_maker.py` 全文（411 行，今天的主战场）
2. `openrlhf/models/utils.py`：`compute_approx_kl`（:64-95）+ `compute_reward`（:98-126）
3. `openrlhf/trainer/ppo_utils/kl_controller.py`（全文 29 行）
4. `openrlhf/trainer/ppo_utils/length_penalty.py`（全文 153 行）
5. 文档 `agent_training.rst.txt` 的算法表格部分

### 第⑤站：make_experience —— 一次前向 fan-out

入口：`PPOTrainer.train_step`（`ppo_trainer.py:213-264`）拿到 1024 条 rollout 后调 `experience_maker.make_experience_batch`（`experience_maker.py:80-96`），分三步。

**(a) `split_rollout_samples`**（:44-77）：开了 dynamic_batch ⇒ 按 `rollout.max_tokens_per_gpu=32768` 的 token 预算，用 Karmarkar-Karp 装箱把 1024 条切成若干前向 micro-batch，且 batch 数对齐到 effective_actor_num（本案例 2）的整数倍；每个 micro-batch 内右 padding 对齐（`Experience.concat_experiences`）。

**(b) `make_experience`**（:113-233）：把 micro-batch 列表 fan-out 给各 actor group 做 forward——

- **reward model**：`samples_list[0].rewards is None` 才需要。本案例 reward 已在生成时算好 ⇒ 跳过；
- **actor forward** → `action_log_probs`（log π_old）；
- **critic forward**：本案例无 critic ⇒ 用 `dummy_ref` 占位（值全 None）；
- **ref forward** → `base_action_log_probs`。

**colocation 决定同步纪律**（`_dispatch_forward`，:100-106）：`colocate_all` 下 actor 和 ref 共卡，必须**串行**——每个 group 前向完 `ray.get` + `empty_cache` 再放下一个；全分离模式则四路并行。结果回收时 `[::duplicate_factor]` 去重（:108-110）——ring/TP 组内每个 rank 都算了同一份数据，只取一份。

**(c) KL 分叉**（:205-219，**最容易混淆的设计点**）：

```python
if (有 ref) and (not args.algo.kl.use_loss):     # PPO 传统路径
    kl = compute_approx_kl(action_log_probs, base_action_log_probs, ...)
else:                                            # 本案例（GRPO 式）
    kl = zeros_like(...)                         # 不在 reward 里罚 KL
if not args.algo.kl.use_loss:
    base_action_log_probs = None                 # 用完即弃
# use_loss=True 时保留 base_action_log_probs，Day 4 在 actor 训练里算 KL loss
```

### 第⑥站：reward 加工流水线（顺序就是语义）

`compute_advantages_and_returns`（:238-330）。对 S₁ 逐步代入：

```
原始 reward r=1.0
 ① 长度惩罚 apply_length_penalties        本案例未配置 ⇒ 不变      length_penalty.py
 ② 按原始 prompt 顺序重排 + reshape 成 (128 组, 8)                  :246-252
 ③ 组内 baseline 减法（estimator 决定）    1.0 − 0.25 = 0.75        :264-271
 ④ compute_reward: clip 到 (−10,10) → 散播到 EOS 位置               models/utils.py:98-126
    S₁: 长度 257 的逐 token reward 向量，只有索引 256（EOS）= 0.75，其余全 0
    （use_kl_loss=False 时这里还会加逐 token 的 −β·KL；本案例 kl 是零张量）
 ⑤ get_cumulative_returns(γ=1): 从后往前累加 ⇒ 210 个 action token 的
    return 全部 = 0.75；advantages = returns.clone()                :379-411
 ⑥ 全局归一化（仅 gae/reinforce/reinforce_baseline）：
    对全部 1024 条的 action token 求 (μ, σ)，A ← (A−μ)/σ            :315-328
    设 μ≈−0.03、σ≈0.52（示意值）⇒ S₁ 每个 token 的最终 A ≈ 1.50
```

第④步的 EOS 散播值得单看一眼（`models/utils.py:121-122`）：用 `fliplr().argmax()` 找每行最后一个 action 位置，`scatter_` 把标量 reward 放上去——一行向量化代码替代了双重循环（注释里保留了等价循环版，对照读）。

### 六种 advantage estimator：同一组 r，六种算法的手算对照

`--algo.advantage.estimator` 的分支就在 :264-271（baseline 减法）和 :284-309（return 计算）。设同组 n=8 条样本 reward 为 r₁..r₈ = [1,1,0,0,0,0,0,0]（均值 r̄=0.25，样本标准差 σ=0.463）：

| estimator | 论文 | baseline 减法 | 全局归一化 | 需要 critic | S₁（r=1）算出的 A | 错误样本（r=0）的 A |
|---|---|---|---|---|---|---|
| `gae` | PPO | 无（用 V(s)，逐 token） | ✅ | ✅ | 依赖 V | 依赖 V |
| `reinforce` | REINFORCE++ | 无 | ✅ | ❌ | 1.0 → 全局归一 | 0 → 全局归一 |
| `rloo` | RLOO | 留一均值 (Σr−rᵢ)/(n−1) | ❌ | ❌ | 1 − 1/7 = **0.857** | 0 − 2/7 = **−0.286** |
| `reinforce_baseline` | REINFORCE++-baseline（**本案例，官方 RLVR 推荐**） | 组均值 | ✅ | ❌ | 0.75 → 全局归一 ≈ **1.50** | −0.25 → ≈ **−0.42** |
| `group_norm` | GRPO | 组均值再除组内 std | ❌ | ❌ | 0.75/0.463 = **1.620** | −0.25/0.463 = **−0.540** |
| `dr_grpo` | Dr. GRPO | 组均值 | ❌ | ❌ | **0.75** | **−0.25** |

记忆锚点：

- **gae 之外全部 critic-free 且 γ 强制 1.0**（:293-300 会 warning 并改写）；此时"逐 token return"退化为"整句共享一个标量 advantage"。
- Dr. GRPO 相对 GRPO 只删了一件事：**组内 /σ**。GRPO 的 /σ 会放大低方差组（比如 7 对 1 错）的梯度，造成"简短、模型已会的题被过度奖励"的偏差；Dr. GRPO 论文证明去掉后梯度无偏。
- `--algo.advantage.no_std_norm` 可让第⑥步只减 μ 不除 σ（reinforce_baseline 的变体）。
- 为什么 rloo/group_norm/dr_grpo **不做**第⑥步全局归一化？对照 :315 的列表——它们的组内操作已经完成了尺度控制（rloo 的留一均值理论无偏、group_norm 已除过 σ、dr_grpo 刻意保持原始尺度），再做全局归一会破坏各自论文的定义。

### KL 的两种用法（一定要能画出这张分叉表）

| | `algo.kl.use_loss=False`（默认，PPO 传统） | `algo.kl.use_loss=True`（GRPO 式，**本案例**） |
|---|---|---|
| KL 去哪 | 变成逐 token reward 惩罚：r_t += −β·KL_t（experience_maker） | 独立 loss 项：`loss = policy_loss + β·kl_loss`（Day 4，ppo_actor.py:319-337） |
| 计算时机 | 采样后一次性（用 π_old） | 每个训练 micro-batch 用**当前** π_θ 重算 |
| 推荐 estimator | `k1` | `k2` 或 `k3`（k1 均值为 0 无意义，argparse help 里明说） |
| ref logprobs | 用完即弃 | 保留传给 actor 训练 |
| 典型系数 | 0.01（RLHF 对话） | 1e-5（RLVR，本案例）；GRPO 常配 k3 |

三种 KL 近似（`models/utils.py:64-95`，源自 Schulman 博客 joschu.net/blog/kl-approx，Δ = logπ − logπ_ref，结果 clamp ±10）：

- **k1** = Δ（无偏、可为负、方差大）
- **k2** = Δ²/2（非负、低方差；有偏但实践近似好，本案例用它）
- **k3** = e^{−Δ} − 1 + Δ（非负、无偏，GRPO 论文用它）

自适应 KL 控制器（`kl_controller.py:4-18`）：设了 `--algo.kl.target` 才启用，`value *= 1 + clip(kl/target − 1, ±0.2) × n_steps/horizon`（Ziegler et al. 2019）。默认是 `FixedKLController`（本案例，β 恒 1e-5）。

### 长度惩罚（reasoning 训练必调，`length_penalty.py`）

发生在流水线第①步，直接改 `experience.rewards`：

- **DAPO overlong**（:16-58）：软区间惩罚。设 `expected = max_new_tokens − overlong_buffer_len`，超出部分按 `−min(exceed, buffer)/buffer × factor` 线性罚。例：max_new=8192、buffer=6144、factor=1 时，一条 4096 token 的回答罚 −(4096−2048)/6144 = −0.33。
- **ProRL stop-properly**（:61-106）：只看 `finish_reason=="length"`（truncated 标志）。系数 ≥0 时 reward **乘**系数（`0.0` = 截断样本 reward 清零，prorlv2 脚本的选择）；<0 时 reward **直接覆盖**为该负值（如 −0.5）。
- 两者可叠加；改完会同步 `info["reward"]` 保证日志一致（:150-153）。

假如 S₃（答错且被截断）在 prorlv2 配置下：reward 0 × 0.0 = 0（本来就 0）；假如某条**答对但被截断**：1.0 × 0.0 = 0——"没写完的对答案不算对"，逼模型学会在预算内收尾。

### 费曼自测

1. 把 r=[1,1,0,0,0,0,0,0] 换成 [1,1,1,1,1,1,1,0]，手算 group_norm 和 dr_grpo 下答对样本的 A，解释为什么 GRPO 会"过度奖励简单题"。
   <details><summary>答案</summary>r̄=0.875，σ=0.354。group_norm：(1−0.875)/0.354≈**+0.354**；dr_grpo：+0.125。同样是"答对"，GRPO 在低方差组里把 0.125 放大近 3 倍——模型越会做的题梯度越被放大，Dr. GRPO 删掉 /σ 就是修这个。</details>
2. `use_kl_loss=True` 时 experience_maker 为什么把 kl 置零但保留 `base_action_log_probs`？
   <details><summary>答案</summary>KL 不进 reward（置零防止 compute_reward 加惩罚），但 Day 4 训练时要用当前 π_θ 对 π_ref 重算 KL loss，所以 ref 的 logprobs 必须随 Experience 传给 actor。</details>
3. compute_reward 后 S₁ 的逐 token reward 向量长什么样？γ=1 的累计 return 为什么让每个 action token 拿到同一个值？
   <details><summary>答案</summary>长 257、只有 EOS 位（索引 256）=0.75 其余全 0。从后往前累加时，每个位置的后缀和都恰好包含那唯一的 0.75。</details>
4. 本案例的 KL 系数 1e-5 起什么作用？训练早期它几乎为 0，为什么还要配？
   <details><summary>答案</summary>k2 = Δ²/2，早期 π_θ≈π_ref ⇒ KL loss≈0；它是**长跑的信任域**——ProRL 式长期训练中防止策略漂离基座太远导致熵坍缩/退化，是随偏离量二次增长的"弹簧"而非常数拖拽。</details>
---

## Day 4 — Loss、聚合与训练步（第⑦-⑩站）

> **目标**：吃透 `models/loss.py` 的每一个分支；理解 token-level vs sequence-level 聚合的数学差异（能用数字例子说明）；理解 off-policy 校正——这是 async 训练能 work 的前提。

### 阅读清单

1. `openrlhf/models/loss.py` 全文（336 行，重中之重）
2. `openrlhf/utils/loss_utils.py`（105 行，梯度累积下的全局归一化——**先跑 `pytest tests/test_loss_aggregation.py -v` 再读实现**）
3. `openrlhf/trainer/ppo_utils/replay_buffer.py`（177 行）+ `experience.py:270-303`（`balance_experiences`）
4. `openrlhf/trainer/ppo_trainer.py:213-298`（train_step / ppo_train 编排）
5. `openrlhf/trainer/ray/ppo_actor.py:155-406`（worker 内真正的优化循环 `ppo_train` + `training_step`）

### 第⑦站：experience 下发与长度均衡

`train_step`（`ppo_trainer.py:213-264`）拿到算好 advantage 的 experiences 后：

1. **`balance_experiences`**（`experience.py:270-303`，dynamic_batch 时）：把 1024 条按 `total_length` 降序排序，再用"首尾交错"法分给 2 个 DP rank——docstring 里的例子：长度 [8,7,6,5,4,3,2,1] 分 2 组 ⇒ [8,1,6,3] 和 [7,2,5,4]，两组总长 18 vs 18。**目的**：防止某个 rank 全拿长序列拖慢整个 step（DP 同步在 optimizer step 处等齐）。
2. **`async_run_method_batch(method_name="append", ...)`**：切好的 experiences 发给各 actor rank，进各自的 `NaiveReplayBuffer`（`replay_buffer.py:49-59`）——注意 append 时先 `split_experience_batch` 拆回单条并 `remove_padding_in_sequences` 去掉右 padding（存"裸"样本，用时再拼）。

### 第⑧站：动态 batch —— Karmarkar-Karp 装箱

`setup_dynamic_batch`（`replay_buffer.py:91-177`），每次 `fit` 开头重建。本案例每个 DP rank 持有 512 条：

1. `local_train_batch_size = train.batch_size / dp_size = 1024/2 = 512` ⇒ `expected_num_steps = 128×8/1024 = 1` 个 optimizer step；
2. 对这 512 条的长度列表调 `get_minimum_num_micro_batch_size`：按 `train.max_tokens_per_gpu = 16192` 的预算算最少要几个 micro-batch，跨 DP rank `all_reduce(max)` 对齐 step 数；
3. `get_seqlen_balanced_partitions`（`seqlen_balancing.py`，从 verl 移植的 **Karmarkar-Karp 最大差分法**）把 512 条分进 N 个 micro-batch，使各箱 token 总数尽量均衡；
4. 预计算四组标量供训练用：`dynamic_batch_num_tokens`（整个 optimizer step 的**全局** action token 数）、`dynamic_global_batch_size`（全局有效样本数）、`dynamic_sample_loss_scale`、`dynamic_optimizer_step`（哪个 micro-batch 之后才真正 step，形如 [0,0,...,1]）。

对比不开 dynamic_batch 的固定路径：按 `micro_train_batch_size` 均匀切，梯度累积步数 gas 由公式算（`deepspeed.py:110-116`）：`gas = train_bs × ring × tp / micro_bs / world_size`。变长 reasoning 输出下固定 batch 的利用率远低于 token 预算装箱——这就是官方 performance 文档把 dynamic batch 列为"始终建议开启"的原因。

### 第⑨站：training_step —— 一个 micro-batch 的完整解剖

`ppo_actor.py:261-406`。输入：experience（含 sequences/action_mask/advantages/action_log_probs=π_old/base_action_log_probs=π_ref/rollout_log_probs=π_rollout）。流程：

```
actor 前向（当前 π_θ）→ action_log_probs                     :294-303
→ PolicyLoss(π_θ, π_old, A, mask, π_rollout, 归一化参数)      :306-313
→ use_kl_loss ⇒ 用 π_θ 与 π_ref 重算 KL → aggregate_loss     :319-333
→ loss = actor_loss + kl_loss × kl_ctl                       :337
→ (可选) MoE aux_loss、entropy 正则                           :339-352
→ strategy.backward → optimizer_step（dynamic 时看 flag）     :354-359
→ (可选) EMA 滑动平均                                         :361-366
```

### PolicyLoss 主干（`loss.py:116-231`）——代入 S₁

```
ratio = exp(clamp(logπ_θ − logπ_old, ±20))
surr1 = ratio · A
surr2 = clamp(ratio, 1−ε_low, 1+ε_high) · A
loss  = −min(surr1, surr2)                       # 标准 PPO（本案例 ε=0.2 对称）
```

**本案例的一个反直觉事实**：每轮 rollout 恰好 1 次 optimizer step、max_epochs=1 ⇒ 训练时 π_θ 与 π_old 是**同一套权重**（梯度累积期间参数不动）⇒ ratio ≡ 1、clip 永不触发，loss 退化为 −A·(权重梯度方向)。clip 在这里是"保险丝"。什么时候真正起作用？`train.batch_size < rollout.batch_size × n` 时——比如 prorlv2 脚本 rollout 512×16=8192、train_bs 1024 ⇒ 每轮 8 次 optimizer step，从第 2 步起 π_θ 已更新而 π_old 还是旧值，ratio 开始漂移，clip 生效。

其余分支逐个看：

- **非对称 clip**（DAPO clip-higher）：`--actor.eps_clip_low_high 0.2 0.27` 抬高上界——正 advantage 的低概率 token 允许涨得更多，鼓励探索。数字例：A=+1.5、ratio=1.35 时，对称 clip 卡在 1.2×1.5=1.8，clip-higher 卡在 1.27×1.5=1.905。
- **dual-clip**（:185-194，arXiv:1912.09729）：`--actor.dual_clip 3`。只在 **A<0** 时加下界 `max(min(surr1,surr2), c·A)`。数字例：A=−1、ratio=5 ⇒ 标准 PPO loss=−min(−5, −0.8×?)…实际取 −(−5)=5，梯度巨大；dual-clip 封在 −3·(−1)=3。防止"策略已大幅偏离的负样本"贡献爆炸梯度。为什么只管 A<0？A>0 时 min(surr1,surr2) 已经被 surr2 的上界封住了。
- **GSPO**（:170-178，`--actor.policy_loss_type gspo`）：ratio 换成**序列级几何平均** `exp(mean_t Δlogp_t)` 并广播回每个 token，同时强制 sequence-level 聚合（:143-144）。MoE 等逐 token ratio 噪声大的场景用。

### off-policy 校正（`--algo.advantage.is_correction_enable`，:196-219）

背景（fengyao.notion.site/off-policy-rl）：vLLM 的 π_rollout 与训练引擎的 π_old **从来不严格相等**（kernel 数值差异），async/partial rollout 下更是差整整一到多个版本。校正系数 `w = exp(logπ_old − logπ_rollout)`，三种用法：

| type | 行为 | 场景 |
|---|---|---|
| `tis` | w 截到 [0.5, 5.0] 后乘 loss | 默认 |
| `icepop`（**本案例**） | 阈值外的 token 系数直接置 0（mask 掉） | 官方 partial rollout 推荐 |
| `seq-mask-tis` | 序列级几何均值出界 ⇒ 整句 mask；句内仍乘 token 级 w | 更保守 |

代入 S₁ 的某个 token：DS 前向 logπ_old=−1.20，vLLM 报 logπ_rollout=−1.15 ⇒ w=e^{−0.05}=0.951 ∈ [0.5,5] ⇒ 该 token 的 loss ×0.951。另一个数值分歧大的 token：w=e^{−1.2}=0.30 < 0.5 ⇒ icepop 直接把它从 loss 里抹掉（tis 则会夹到 0.5 继续参与）。副产品 `vllm_kl`（:219）被记录进日志——**监控两引擎分歧度的免费探针**。

### aggregate_loss：本仓库最精细的工程点（:11-39）

两种归约语义，用一个数字例子焊死记忆：样本 A = 10 个 token、每个 loss 1.0；样本 B = 90 个 token、每个 loss 0.1。

- **token-level**（默认）：Σ(loss·mask)/全局token数 = (10×1.0+90×0.1)/100 = **0.19**——长回答权重大；
- **sequence-level**：先句内平均再句间平均 = (1.0+0.1)/2 = **0.55**——每条样本等权（GRPO 论文原味）。

两个"全局"修正因子是精髓：

- `× dp_size`：DeepSpeed/DDP 会把梯度跨 DP rank **平均**，乘回 dp_size 抵消它，保证梯度与"单卡跑全量 batch"严格等价；
- `batch_num_tokens / global_batch_size` 传**全局**总数（跨 rank all_reduce 或由 replay buffer 预计算），使 loss 对"数据怎么切片"不变。

梯度累积再叠一层（`loss_utils.py:54-105`）：DeepSpeed 对每个 backward 的 loss 自动 ×1/gas，所以 `iter_grad_accum_global_norm` 把同一 optimizer step 窗口内所有 micro-batch 的 token 数**先汇总再 ÷gas** 作为分母——gas 个 micro-batch 的贡献最终加成"整个 step 的一个 token 均值"，而不是"均值的均值"（micro-batch token 数不均时二者不同）。`tests/test_loss_aggregation.py:151-196` 专门验证这两种算法的差异——**这是全仓库唯一能在 Mac 上跑的算法级测试，先跑再读**。

### KL loss、entropy 与 critic 侧

- **KL loss**（`ppo_actor.py:319-333`）：k2 = (logπ_θ − logπ_ref)²/2，同一套 aggregate_loss 归约，×β=1e-5 加进总 loss。
- **entropy**（:345-352）：`--actor.entropy_coef` 设 0 表示**只记日志不进 loss**（本案例；熵是崩溃监控指标），设正值则 `loss −= coef × entropy`鼓励探索。
- **ValueLoss**（`loss.py:234-270`，仅 gae）：value clip（默认 0.5）+ 双 MSE 取 max，×0.5。`--critic.freezing_steps N`：前 N 步只训 critic 不动 actor（`ppo_trainer.py:272` 的 `run_actor` 条件）——PPO 冷启动时 critic 还是随机的，先冻 actor 让 V(s) 追上再放开。
- **指标归约**（`ppo_actor.py:196-259`）：每个指标带 "token"/"sample" 权重标签，跨 rank all_reduce 时按 token 数或样本数加权——保证日志里的 `policy_loss`、`kl` 等与单卡全量计算一致。

### 费曼自测

1. token-level 和 sequence-level 聚合，哪个让"一条 8000 token 的长回答"对梯度影响更大？GRPO 论文原始设定是哪个？OpenRLHF 默认是哪个？
   <details><summary>答案</summary>token-level 影响大（8000 个 token 每个都是一票）。GRPO 原味是 sequence-level（每句等权）。OpenRLHF 默认 token-level（`token_level_loss=True`），GSPO 强制 seq-level。</details>
2. 为什么 aggregate_loss 要 `× dp_size`？删掉它梯度会差多少？
   <details><summary>答案</summary>DeepSpeed 跨 DP rank 平均梯度，等效把 loss ÷dp_size；不乘回去，梯度会比"单卡全量 batch"小 dp_size 倍——lr 的有效值随卡数变化，跨规模复现实验时是灾难。</details>
3. 本案例训练时 ratio≡1，那 icepop 还有作用吗？
   <details><summary>答案</summary>有。icepop 校正的是 π_old 与 π_rollout（vLLM）的差异，与 ratio=π_θ/π_old 无关。就算完全 on-policy，vLLM/DeepSpeed 的 bf16 数值分歧也存在，个别 token 的 w 可能出界。</details>
4. async 训练不开 is_correction 会发生什么？
   <details><summary>答案</summary>π_rollout 系统性落后 π_old 一到多个 batch，w 分布整体偏移；直接拿 π_old 当采样分布做 PPO 等于用错误的重要性权重，实践表现为熵异常、reward 虚高后崩掉。所以官方 async/partial rollout 配方必带 icepop。</details>
5. dual_clip 为什么只在 advantage<0 时起作用？
   <details><summary>答案</summary>A>0 时 surr2=clamp(ratio)·A 已提供上界；A<0 时 min(surr1,surr2) 会选 ratio 大的那支（更负），ratio 极大时 loss 无界——dual_clip 用 c·A 兜底。</details>

---

## Day 5 — 分布式基础设施：hybrid engine、权重同步、async（第⑪-⑫站）

> **目标**：理解"训练 80% 时间在生成"这一根本矛盾的三种工程解法；能解释本案例为什么走 CUDA IPC 而不是 NCCL broadcast；能画出 async 模式的生产者-消费者拓扑。

### 阅读清单

1. 文档 `hybrid_engine.rst.txt` + `async_training.rst.txt` + `performance.rst.txt`（三份都不长，先读文档再读码）
2. `openrlhf/trainer/ppo_trainer.py:266-319`（ppo_train 的 sleep 编排 + broadcast_to_vllm 入口）
3. `openrlhf/trainer/ray/ppo_actor.py:102-153, 408-489`（同步后端选择 + NCCL/IPC 两条实现 + ZeRO-3 GatheredParameters）
4. `openrlhf/trainer/ray/vllm_worker_wrap.py`（73 行，vLLM worker 侧怎么收权重）
5. `openrlhf/utils/deepspeed/deepspeed.py` + `deepspeed_utils.py:151-211`（策略封装、offload/reload 的真实语义）
6. `openrlhf/trainer/ppo_trainer_async.py` 全文（353 行）

### Hybrid engine：一个训练 step 的完整时序（本案例）

把 Day 2-4 的站点按 GPU 时间轴重排（`colocate_all` + 双 sleep）：

```
t0  vLLM wake_up(全部)          ← samples_generator.py:106-107
t1  生成 1024 条 rollout（S₁ 在此诞生）        ~全 step 的 60-80% 时间
t2  vLLM sleep                  ← :119-120，释放权重+KV cache 显存
t3  make_experience: actor/ref 前向（串行 + empty_cache）
      注意: DS 的 offload_deepspeed_states 只搬走 optimizer states/
      梯度缓冲/hp_params，bf16 模型参数留在 GPU（deepspeed_utils.py:167-178）
      ⇒ 前向不需要 reload，"睡着"也能推理
t4  actor 训练: reload_states → fit(ppo_train) → offload_states
      ← ppo_trainer.py:274-286 _run_sleep；有 critic 时 critic 先训
t5  vLLM wake_up(tags=["weights"]) ← 只醒权重不醒 KV cache，省峰值显存
t6  broadcast_to_vllm: θ₁ 进入 vLLM               ← ppo_trainer.py:300-319
t7  （下轮 t0 时 KV cache 才被唤醒）
```

sleep 模式下 critic 和 actor **必须串行**（共卡）；非 sleep 模式二者并行 `fit`（:288-296）。

### 第⑪站：权重同步的三条路

选择逻辑一行（`ppo_actor.py:103`）：`use_cuda_ipc = (backend=="nccl") and colocate_all and (not async_enable)`。

- **CUDA IPC（本案例命中）**：actor 与 vLLM worker 在同一张物理卡 ⇒ 不走网络。每个参数：ZeRO-3 先 `GatheredParameters` 聚合分片 → `reduce_tensor` 拿 IPC handle → `all_gather_object` 按物理 GPU id 汇总 → vLLM worker `update_weight_cuda_ipc` 直接映射同卡显存 `load_weights`（`vllm_worker_wrap.py`）——**零拷贝**。
- **NCCL broadcast（默认路径）**：DS rank0 与所有 vLLM worker 组一个独立进程组，world = engines×tp + 1（`_init_vllm_sync_group` :110-153 的布局注释值得抄进笔记）。逐参数：rank0 发 `update_weight.remote(name, dtype, shape)` 元数据 → NCCL broadcast 张量 → worker `load_weights`。
- **Ray collective**（`--vllm.sync_with_ray`）：NCCL 建组失败环境的兜底。

公共细节：ZeRO-3/TP 下参数是分片的，广播前须逐参数 gather（:467-471）；开 prefix cache 时同步前要 `reset_prefix_cache`（旧 KV 对应旧权重！:409-414）；最后一个参数带 `empty_cache=True`。

### Async 模式：生产者-消费者 + 三个同步原语

`ppo_trainer_async.py`。sync 模式"生成→训练"轮流空转 GPU，async 把二者拆成两个常驻 Ray actor 并行跑：

```
GenerateSamplesActor ──(rollout_queue, maxsize=async_queue_size)──▶ TrainingActor
        ▲                                                              │
        └────────────(rollout_slots: 令牌信号量, 携带 global_step)──────┘
                VLLMLock: eval/broadcast 与生成互斥
```

- **`rollout_queue`**：产出的 rollout batch 排队；`--train.async_queue_size`（默认 1）就是**最大 off-policy staleness（按 batch 计）**。
- **`rollout_slots`**（:295-297）：令牌池实现背压——trainer 消费一个 batch 才放回一个令牌，generator 拿到令牌才能生产下一批。令牌上还捎带 trainer 的最新 global_step，generator 用它判断"该做 eval 了"（:109-127，`_eval_just_done` 防连评）。
- **`VLLMLock`**（:19-34）：普通 async 下，生成与权重广播用锁互斥（保证一个 batch 内权重一致）；**partial rollout**（`--train.partial_rollout_enable`）改用 vLLM 的 `pause_generation`/`resume_generation`（:255-265）——广播时只暂停不清空，in-flight 序列**前半旧权重后半新权重**。这是最激进的重叠，官方明确要求配 `is_correction`（icepop）。
- 超采样：`--rollout.vllm_generate_batch_size > rollout.batch_size` 仅 async 可用（多出的存 `_sample_buffer` 下轮直接用）。

官方 performance 文档的定位一句话：**async = max throughput，hybrid engine = max stability**；async 上线前必须先在 sync 模式验证收敛。

### DeepspeedStrategy 要点（`utils/deepspeed/deepspeed.py`）

- **3D mesh**：`(dp, sp, tp)`（:104-108），sp=ring attention 组。`ring_attn_size>1` 强制 `packing_samples`。
- **梯度累积**：`gas = train_bs × ring × tp / micro_bs / world_size`（:110-116）。
- **Muon 优化器**（`--actor.optim muon`）：2D 权重走 Muon（lr 0.02）、embedding/head/1D 参数走 aux-Adam；要求 DS ≥ 0.18.9；与 `adam_offload` **不兼容**；`ns_steps`/`nesterov` 是占位符（DS 硬编码 5/True，改了无效，源码 :316-326 有 warning）；**用 Muon 时 grad clip 建议设 0**——DS 的 clip 发生在 Newton-Schulz 之后，会把更新缩小约 700 倍（典型的"读了源码才知道"的坑，:328 附近注释）。
- **EMA**（`--train.enable_ema`，β=0.992）：每个 optimizer step 后在 CPU 上滑动平均（:507-521），保存时存 EMA 权重——RLHF 出模型更稳的老技巧。

### 第⑫站：checkpoint、恢复与评估

- **两种格式**：DS ckpt（含 optimizer/scheduler/dataloader 状态，可断点续训）+ HF 格式（`--ckpt.save_hf`，直接部署）。`client_states` 里存 `episode / global_step / total_consumed_prompts / data_loader_state_dict`（`ppo_trainer.py:548-553`）。
- **恢复时序**（`fit` :499-514）：读 ckpt → **先 broadcast_to_vllm**（旧权重同步给 vLLM，否则第一轮采样用的是基座！）→ 恢复 dataloader 状态。
- **best checkpoint**（:337-381）：eval 指标（默认自动检测第一个 `*_pass1`）创新高时保存 `best_global_stepN`。
- **eval 指标**（`compute_eval_metrics` :82-145）：按 datasource 分组算 pass@1（n 条平均）与 pass@k（n 条取 max），外加长度与截断率。案例若配 `--eval.dataset OpenRLHF/aime-2024 --eval.n_samples_per_prompt 4` 就得到 `eval_aime_pass1 / pass4`。
- 跨并行度恢复：`--ds.use_universal_ckpt`。

### 费曼自测

1. 为什么 `vllm.enable_sleep` 和 `async_enable` 互斥？
   <details><summary>答案</summary>sleep 的前提是训练/生成在同一组卡上严格串行分时；async 的意义恰恰是二者并行。async 下 vLLM 必须常驻自己那组卡（colocate_all 在 async 里只共卡 DS 模型）。</details>
2. 本案例满足哪三个条件走 CUDA IPC？换成 2 节点部署后走哪条路、为什么？
   <details><summary>答案</summary>backend=nccl + colocate_all + 非 async。2 节点时 actor 与 vLLM 不再同卡，IPC handle 跨不了机器，回落 NCCL broadcast（rank0 gather 后逐参数广播）。</details>
3. DS "睡着"时为什么还能做 make_experience 的前向？
   <details><summary>答案</summary>offload_deepspeed_states 只搬 optimizer states/梯度缓冲/hp_params，bf16 参数（lp_params）留在 GPU——前向只要参数在就行；训练前才 reload。</details>
4. partial rollout 相比普通 async 多引入了什么噪声？框架用什么补救？
   <details><summary>答案</summary>单条序列内部混合新旧权重生成的 token（π_rollout 不再是单一分布）。补救：token 级 IS 校正，官方推荐 icepop（出界 token 直接 mask，不放大方差）。</details>
5. 断点续训时为什么必须先 broadcast_to_vllm 再开始采样？
   <details><summary>答案</summary>vLLM engines 是从 HF 基座权重初始化的；不广播 ckpt 权重，恢复后第一轮 rollout 就是"基座采样 + ckpt 训练"的严重 off-policy 错配。</details>
---

## Day 6 — 调参艺术：性能与算法双维度

> **目标**：形成自己的调参决策树。今天以官方 `performance.rst.txt` + 三个示例脚本的对照为主，案例 recipe 作为锚点。

### 性能调参决策树（官方 performance 文档提炼）

**第一步：选部署模式**（官方原话："Max throughput → async；Max stability → Hybrid Engine；Distributed 是大模型兜底"）

- 单机 4-8 卡、≤13B、追稳定 → **模板 A（本案例）**：`colocate_all + vllm.enable_sleep + ds.enable_sleep + ZeRO-3 + packing_samples + dynamic_batch + sync_backend nccl`
- 多机、模型放得下但卡不够 → 模板 B：`colocate_actor_ref + colocate_critic_reward + 独立 vLLM + adam_offload`
- 追极限吞吐、已在 sync 验证收敛 → 模板 C：`async_enable (+ partial_rollout_enable + icepop)`，`async_queue_size` 从 1 起步
- 70B+ → 全分离 distributed 模式

**始终开启**：`--ds.packing_samples`（去 padding，官方标注 "Always on"）、`--vllm.sync_backend nccl`、`--train.dynamic_batch_enable`（注意有**两个**预算旋钮：`train.max_tokens_per_gpu` 管训练、`rollout.max_tokens_per_gpu` 管前向，本案例 16192/32768）。

**vllm.gpu_memory_utilization 经验值**（官方表，8×A100-80G）：8B→0.6，13B→0.5，34B→0.4，70B+ 改分离模式。本案例 4B 用 0.7。

**OOM 处理优先级**（官方给的顺序，别乱跳）：

1. `packing_samples` + `gradient_checkpointing_enable`
2. 降 `train.micro_batch_size` / `rollout.micro_batch_size`
3. 降 `vllm.gpu_memory_utilization`（0.6→0.5→0.4）
4. `ds.adam_offload` + 提 `ds.zero_stage`（2→3）
5. 最后手段：去掉 colocation 转分离模式

**长上下文（>8K）**：`ring_attn_size 2 + ring_attn_head_stride 2` 起步（本案例即是），配 ZeRO-3 + packing。

**batch 关系式**：`train.batch_size = rollout.batch_size × n_samples_per_prompt` 是常见起点（本案例 1024=128×8，每轮 1 次更新、严格 on-policy）；把 train_bs 调小（如 prorlv2 的 8192/1024）= 每轮 8 次更新，样本效率高但 off-policy 度升高、PPO clip 开始真正工作。生成侧偏好**多 engine 小 TP**而非少 engine 大 TP。

### 算法调参速查（结合示例脚本实证）

| 目标 | 推荐配置 | 出处 |
|---|---|---|
| RLVR / 数学推理（首选） | `estimator=reinforce_baseline, n_samples=8-16, kl.use_loss + k2, init_coef 1e-5~1e-4, dynamic_filtering 0 1, is_correction icepop` | 本案例 recipe（`hybrid_engine.rst`）+ `train_prorlv2_math_hybrid_engine.sh` |
| 经典 RLHF（有 RM） | `estimator=gae, kl k1 (不开 use_loss), init_coef 0.01, actor lr 1e-6 / critic lr 9e-6, reward.normalize_enable, critic.freezing_steps 预热` | `train_ppo_ray_hybrid_engine.sh` |
| DAPO 复现 | `estimator=group_norm, eps_clip_low_high 0.2 0.27, kl.use_loss + k3, n_samples=8, dynamic_filtering, overlong_buffer_len` | `train_dapo_ray_hybrid_engine.sh` |
| 长 CoT 防刷长度 | `reward.overlong_buffer_len + overlong_penalty_factor`；或截断样本 `stop_properly_penalty_coef 0`（清零）/负值（覆盖） | ProRL 脚本注释 |
| async / partial rollout | 必配 `is_correction_enable + is_correction_type icepop`（阈值默认 0.5 5.0） | `train_reinforce_baseline_ray_agent_async.sh` + `async_training.rst` |
| MoE / token ratio 噪声大 | `policy_loss_type gspo` | GSPO 论文场景 |
| 训练不稳、被负样本拖崩 | 加 `dual_clip 3` | dual-clip PPO 论文 |
| 探索不足、熵掉太快 | `eps_clip_low_high 0.2 0.27`（clip-higher）；或 `entropy_coef` 小正值 | DAPO / prorlv2 |

**几个关键默认值**（背下来，全在 `train_ppo_ray.py` argparse）：actor lr `1e-6`、critic lr `9e-6`、eps_clip `0.2`、value_clip `0.5`、kl init_coef `0.01`、γ=λ=`1.0`、reward clip `(-10,10)`、scheduler `cosine_with_min_lr`（warmup 3%、min_lr_ratio 0.1）、EMA β `0.992`、is_correction 阈值 `[0.5, 5.0]`、vllm.gpu_memory_utilization `0.95`（分离模式默认，共卡必须调低）。

### 监控什么（案例视角）

- `rollout/reward_mean`：题目正确率的直接体现；配合 `dynamic_filtering_pass_rate` 看课程难度。
- `kl` / `logprobs_diff`：偏离基座的程度；`vllm_kl`：vLLM 与训练引擎的分歧（异常增大 = 数值问题或 off-policy 过头）。
- `ppo_clip_ratio`：被 clip 的 token 占比；每轮 1 次更新的配置下应≈0，多次更新时持续 >0.2 说明 lr 过大或 staleness 过高。
- `rollout/response_length_mean` + `rollout/truncated_rate`：长度失控/截断率上升要么加长度惩罚要么放宽 max_new_tokens。
- `entropy_loss`（coef=0 时纯监控）：熵坍缩预警。

### Troubleshooting 高频坑（troubleshooting.rst + 源码验证）

- 旧版平铺 flag（`--actor_num_nodes`）全部失效，报 unrecognized arguments → 查 `common_options.rst` 的迁移表
- Ray 下 GPU device index 错乱 → `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
- vLLM 挂起 → 试 `--vllm.enforce_eager`（禁 CUDA graphs；官方 recipe 默认带上）
- Muon：DS≥0.18.9、与 adam_offload 互斥、ns_steps/nesterov 改了无效、grad clip 设 0
- 卡死排查：`py-spy top --pid`（容器要 `--cap-add=SYS_PTRACE`）
- LoRA 只支持 SFT/RM/DPO，**Ray+vLLM PPO 不支持**
- VLM：不支持 packing_samples、不支持 critic（必须 critic-free estimator）、attn 用 eager（`train_ppo_ray.py:617-627` 的 assert）

### 动手

把 `train_prorlv2_math_hybrid_engine.sh`、`train_dapo_ray_hybrid_engine.sh`、`train_reinforce_baseline_ray_agent_async.sh` 三个脚本做一张 diff 表：同一列是 flag，每行一个脚本。你会直观看到"算法差异只是十来个 flag 的差异"——这正是 agent 范式三轴解耦（算法 × 执行模式 × pipeline）的价值。

---

## Day 7 — 收官：非 RL 路径、多轮 Agent 与综合实战

> **目标**：补齐 SFT/RM/DPO 三条支线；能独立写多轮 agent；用四个实战练习检验一周成果。

### 阅读清单

1. 文档 `non_rl.rst.txt` + `openrlhf/trainer/{sft,rm,dpo}_trainer.py` + `openrlhf/datasets/`（packing、chat template 处理）
2. 文档 `agent_training.rst.txt` 多轮部分 + `examples/python/` 全部 agent 示例
3. 文档 `checkpoint.rst.txt`（Day 5 已覆盖大半，补 DS↔HF 转换与 `ckpt_ds_zero_to_universal.sh`）

### 非 RL 三条支线（与 RL 共享 `models/` 与 DeepspeedStrategy，但不走 Ray）

- **SFT**：`SFTLoss`（`loss.py:87-113`）= −logp 的 aggregate_loss 归约——与 RL 共用同一套 token/seq 聚合语义，`test_loss_aggregation.py` 同时覆盖两者。
- **RM 训练**：`PairWiseLoss`（:273-285）`−logσ(r_chosen − r_reject − margin)` 或 `LogExpLoss`（:288-298）；value head 前缀默认 `score`（`--ds.value_head_prefix`）；`--reward.normalize_enable` 训完记录 mean/std 供 PPO 期归一。
- **DPO**：`DPOLoss`（:301-336）。β 典型 0.1-0.5；`ipo=True` 走 IPO 的平方损失（Eq.17）；`label_smoothing` = cDPO；隐式奖励 `β·(logπ_chosen − logπ_ref_chosen)` 作为训练监控。DPO 需要 ref model 常驻（双倍显存）。

### 多轮 Agent：把环境接进训练循环

`MultiTurnAgentExecutor`（`agent.py:31-181`）+ 用户实现的 `AgentInstanceBase` 子类（`--train.agent_func_path` 指向的 py 文件须暴露 `AgentExecutor` 类，`vllm_engine.py:19-30` 动态加载）。协议：

```python
class AgentInstance(AgentInstanceBase):
    async def reset(self, states):      # states = {"observation": prompt, "label": label}
        return {"observation": ...}     # 初始观测文本
    async def step(self, states):       # states = {observation_text, action_text, label, sampling_params}
        return {
            "rewards": tensor,               # 本步奖励（累加成 episode reward）
            "scores": ...,                   # 动态过滤用（默认= 累计reward）
            "environment_feedback": "...",   # 环境反馈文本 → tokenize 后拼进上下文
            "done": bool,
            "sampling_params": ...,          # 可选：下一步覆盖采样参数
            "extra_logs": {...},
        }
```

执行循环（:90-166）：`生成 action → step() → feedback tokenize 拼接 → 直到 done 或 token 预算耗尽`。三个必须讲得出的细节：

1. **每个 prompt 绑定独立 agent 实例**（:38，环境隔离）；
2. `action_ranges` 多段记录，env feedback token 不进 loss（action_mask=0）、rollout_log_prob 补 0；
3. `sampling_params.max_tokens = max_length − len(current_obs)` 每轮重算——多轮的 token 预算是全局共享的。

官方文档的警告（agent_paradigm.rst 结尾）：自定义 agent 时**永远操作 token id 层面提供的字段**，不要自己拼文本再 tokenize——否则重新引入 token-in-token-out 消灭掉的那类 bug。

示例梯度：`agent_func.py`（随机 1-3 步玩具环境）→ `agent_func_gem_multiturn.py`（真实 GEM 环境）→ `agent_func_openai_server_executor.py`（把本地 vLLM 包成 OpenAI 兼容 `/v1/chat/completions`、同时截获 token trace 训练——**工具调用 RL 的标准姿势**）。

### 实战练习（检验一周成果）

**练习 1（算法·手算）**：不看笔记，把 r=[1,0,1,1,0,0,0,0] 这组新数字从原始 reward 一路手算到"进 loss 的逐 token advantage"（reinforce_baseline 与 group_norm 各一遍），然后在 `experience_maker.py` 里逐行验证你的每一步。能指出二者在"全局归一化"上的差异算过关。

**练习 2（代码·可本地跑）**：给 `PolicyLoss` 加一个假想的 `policy_loss_type`（比如把 clip 换成 KL 罚项的 PPO-penalty 变体），在 `tests/` 下仿照 `test_loss_aggregation.py` 写单测并跑通（Mac 纯 CPU 可跑）。这会强迫你吃透 ratio/mask/聚合的全部张量形状。

**练习 3（系统设计）**：给定"2 节点 × 8×A100-80G，训 32B 模型做数学 RLVR"，写出完整启动脚本：部署模式、estimator、KL 方案、长度惩罚、batch 关系、vLLM 参数，每个 flag 一行理由。写完对照 examples 里最接近的脚本互评。

**练习 4（多轮 agent）**：实现一个 `AgentInstanceBase` 子类做简单环境（如 20 questions 或计算器工具调用），接上 `--train.agent_func_path`，画出你这个环境下"S₁ 的十二站旅程"变化了哪几站（提示：①③④站变了，⑤站之后完全不变——这就是解耦）。

### 延伸阅读（源码 ↔ 论文对照表）

| 代码 | 论文 |
|---|---|
| `reinforce`/`reinforce_baseline` | REINFORCE++ (arXiv:2501.03262)；ProRL V2 为其大规模实证 |
| `group_norm` | GRPO（DeepSeekMath, arXiv:2402.03300） |
| `dr_grpo` | Dr. GRPO (arXiv:2503.20783) |
| `dynamic_filtering` + `eps_clip_low_high` + overlong penalty | DAPO (arXiv:2503.14476) |
| `policy_loss_type=gspo` | GSPO (arXiv:2507.18071) |
| `dual_clip` | Dual-clip PPO (arXiv:1912.09729) |
| k1/k2/k3 | Schulman, *Approximating KL Divergence*（joschu.net/blog/kl-approx） |
| TIS/ICEPOP | fengyao.notion.site/off-policy-rl |
| Adaptive KL | Ziegler et al. (arXiv:1909.08593) |
| Karmarkar-Karp 装箱 | verl 的 seqlen_balancing（代码头注释） |

---

## 附录 A：一页公式速查

```
PPO:        L = −E[min(r·A, clip(r, 1−εl, 1+εh)·A)],  r = π_θ/π_old
dual-clip:  A<0 时 L = −max(min(surr1,surr2), c·A)
GSPO:       r_seq = exp(mean_t log r_t)，强制 seq-level 聚合
GAE:        δ_t = r_t + γV_{t+1} − V_t;  A_t = δ_t + γλ·A_{t+1};  ret = A + V
REINFORCE++: A_t = Σ_k γ^k r_{t+k}，全局 (μ,σ) 归一化
RLOO:       A_i = r_i − (Σ_j r_j − r_i)/(n−1)
R++-baseline: A_i = r_i − r̄_group，再全局 (μ,σ) 归一化
GRPO:       A_i = (r_i − r̄)/(σ_group + 1e-9)
Dr.GRPO:    A_i = r_i − r̄（无任何 σ 归一化）
KL:  k1 = Δ;  k2 = Δ²/2;  k3 = e^{−Δ} − 1 + Δ   （Δ = logπ − logπ_ref, clamp ±10）
reward 合成（use_kl_loss=False）: r_t = −β·KL_t + 𝟙[t=EOS]·clip(r, −10, 10)
DAPO overlong: penalty = −min(len − (max_new − buf), buf)/buf × factor
ProRL 截断: coef ≥ 0 ⇒ r ← r·coef;  coef < 0 ⇒ r ← coef
token-level 聚合: Σ(l·m)/N_tokens_global × dp    seq-level: mean_i(mean_t l_it) × dp
IS 校正: w = exp(logπ_old − logπ_rollout)；tis=clamp(w, lo, hi)，icepop=w·𝟙[w∈[lo,hi]]
gas = train_bs × ring × tp / micro_bs / world_size
```

## 附录 B：样本 S₁ 数字总表（案例速查）

| 站 | 位置 | S₁ 的值 |
|---|---|---|
| 输入 | dapo-math-17k | prompt="...divisors of 36...\boxed{}"，label="91" |
| ① template | prompts_dataset.py | 48 token 的 prompt（含 `<|im_start|>`） |
| ③ 生成 | agent.py:240 | 210 个 action token，`\boxed{91}<|im_end|>`，stop |
| ③ 判卷 | math_reward_func.py | reward = scores = 1.0 |
| ④ 张量 | samples_generator.py:247-314 | sequences(1,258)、action_mask(1,257)（索引47..256 为 1） |
| ④ 过滤 | samples_generator.py:169 | 组 scores 均值 0.25 ∈ (0,1) ⇒ 保留 |
| ⑤ 前向 | experience_maker.py:113 | 得 π_old、π_ref 两组 (1,257) logprobs；kl=0 张量 |
| ⑥ baseline | experience_maker.py:267 | 1.0 − 0.25 = 0.75 |
| ⑥ EOS 散播 | models/utils.py:121 | 逐 token reward：仅索引 256 = 0.75 |
| ⑥ return | experience_maker.py:379 | γ=1 ⇒ 210 个 action token 全部 0.75 |
| ⑥ 全局归一 | experience_maker.py:315 | (0.75−μ)/σ ≈ 1.50（μ≈−0.03, σ≈0.52 为示意） |
| ⑨ ratio | loss.py:166 | 首次更新 π_θ=π_old ⇒ ratio≡1，loss=−A |
| ⑨ icepop | loss.py:201 | w=exp(logπ_old−logπ_rollout)≈0.95 ⇒ ×0.95；出界 token 置 0 |
| ⑨ KL loss | ppo_actor.py:319 | k2=(logπ_θ−logπ_ref)²/2 ≈ 0（早期），×1e-5 |
| ⑩ 更新 | ppo_actor.py:354 | 唯一一次 optimizer step：θ₀ → θ₁ |
| ⑪ 同步 | ppo_actor.py:408 | CUDA IPC 零拷贝，θ₁ 进 vLLM |

## 附录 C：学习习惯建议

1. **每读一个模块先跑/写测试**：`tests/` 在 Mac 可跑，是唯一的本地验证手段（训练路径需要多卡 CUDA，别在本机试图启动训练）。
2. **grep 用嵌套名**：`rg "kl\.init_coef" openrlhf/` 而不是搜 `--algo.kl.init_coef`。
3. **文档→源码→脚本三角验证**：文档说语义、源码定真相、examples 脚本给实证组合。有出入以源码为准。
4. **中文文档**在 `zh/latest/`，与英文同名对照读；术语翻译由管线生成，以英文为准。
5. 学完后真正的毕业考：**去 upstream 的 issue 列表挑一个训练稳定性相关的 issue，用这一周的知识写出诊断分析**——能定位到"第几站、哪一行"，才算真的吃透。
