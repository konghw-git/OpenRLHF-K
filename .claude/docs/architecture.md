# OpenRLHF Architecture Deep Dive

Read this when changing training logic, Ray topology, or rollout generation. For option-level semantics, cross-check the official docs pages listed in `official-docs-map.md`.

## Ray actor topology (`openrlhf/trainer/ray/`)

`openrlhf/cli/train_ppo_ray.py` is the orchestrator. It:

1. Creates Ray placement groups (colocated layouts controlled by `args.train.colocate_actor_ref` / `colocate_all`).
2. Spawns vLLM engines via `create_vllm_engines` (`vllm_engine.py`, worker patched by `vllm_worker_wrap.py` for weight sync).
3. Spawns four `RayActorGroup`s from `launcher.py`, each a group of `@ray.remote(num_gpus=1)` actors:
   - `PolicyModelActor` (`ppo_actor.py`) — the trained policy; owns the trainer loop.
   - `CriticModelActor` (`ppo_critic.py`) — value model (absent for critic-free algos like GRPO/REINFORCE++).
   - `ReferenceModelActor` (`launcher.py`) — frozen ref for KL; skipped when `args.algo.kl.init_coef == 0`.
   - `RewardModelActor` (`launcher.py`) — optional; alternatives are a remote RM URL (`serve_rm.py`) or a custom reward function.
4. Calls `fit` on the policy group; other groups serve `forward` RPCs (`execute_batch` fans a batch across the group).

Each actor sets up torch.distributed itself (`BaseDistributedActor`) and wraps its model with `DeepspeedStrategy` (`openrlhf/utils/deepspeed/deepspeed.py`) — ZeRO-2/3, offload, bf16.

## Rollout → experience → train (`openrlhf/trainer/ppo_utils/`)

Per iteration:

1. **`SamplesGenerator.generate_samples`** (`samples_generator.py`) pulls a prompt batch, dispatches to vLLM engines, and builds `Experience` objects (`experience.py`) — packed sequences with `action_mask`/`attention_mask`.
2. **`RemoteExperienceMaker.make_experience_batch`** (`experience_maker.py`) fans the samples to ref/reward/critic actor groups for log-probs, rewards, and values; applies KL (`kl_controller.py`) and computes advantages/returns (GAE or cumulative, `compute_advantages_and_returns`). Advantage estimator variants (group norm for GRPO, dr_grpo, RLOO baselines…) live here.
3. **`ReplayBuffer`** (`replay_buffer.py`) shards experiences across DP ranks (sequence-length balancing via `utils/seqlen_balancing.py`).
4. **`BasePPOTrainer.ppo_train` / `train_step`** (`ppo_trainer.py`) runs the actor (and critic) updates using losses from `openrlhf/models/loss.py` (`PolicyLoss` handles PPO clip and its variants; `aggregate_loss` controls token/seq-level aggregation — covered by `tests/test_loss_aggregation.py`).
5. **`broadcast_to_vllm`** pushes updated weights into the vLLM engines (NCCL or CUDA IPC when colocated).

## Sync vs async vs hybrid engine

- **Sync (`PPOTrainer`, `ppo_trainer.py`)**: generate → train alternate; with `colocate_all` + hybrid engine, vLLM engines sleep/wake so training and inference share the same GPUs (`args.vllm.enable_sleep`, DeepSpeed state offload in `deepspeed_utils.py`).
- **Async (`ppo_trainer_async.py`, `args.train.async_enable`)**: generation and training overlap; rollouts stream in while the policy trains, tolerating off-policy staleness.
- Docs pages: `hybrid_engine.html`, `async_training.html` in the crawled docs.

## Agent abstraction (`openrlhf/utils/agent.py`)

All rollouts go through an `AgentExecutorBase`:

- `SingleTurnAgentExecutor` — classic RLHF / reinforced fine-tuning; reward from RM actor, remote RM URL, or a user `reward_func` python file.
- `MultiTurnAgentExecutor` — wraps a user-provided `AgentInstanceBase` subclass (async `step`/`reset`) for environment-interaction RL. Users pass `--agent_func_path` to their python file; see README "Multi-Turn Agent" section and `agent_paradigm.html` / `agent_training.html` docs.

This is orthogonal to the RL algorithm choice (PPO / REINFORCE++ / GRPO / RLOO are all flags on the same entry point).

## Config system (`openrlhf/utils/config.py`)

CLI flags are flat (`--actor_num_nodes`) but `hierarchize()` regroups them into nested namespaces by prefix: `args.actor.num_nodes`, `args.vllm.num_engines`, `args.algo.kl.init_coef`, `args.train.*`, `args.data.*`, `args.ds.*` (deepspeed), `args.ref.*`, `args.rollout.*`. **When grepping for where a flag is consumed, search the nested form** (e.g. `rollout.temperature`), not the CLI spelling. The full flag reference is `common_options.html` in the crawled docs.

## Non-RL trainers

`train_sft.py` / `train_rm.py` / `train_dpo.py` are single-process-group DeepSpeed trainers (`sft_trainer.py`, `rm_trainer.py`, `dpo_trainer.py`) with datasets from `openrlhf/datasets/` (packing, chat templates). They share `DeepspeedStrategy` and `openrlhf/models/` with the RL path but skip Ray entirely; SLURM/多机 launch via `examples/scripts/train_*_slurm.sh` (docs: `multi-node.html`, `non_rl.html`).
