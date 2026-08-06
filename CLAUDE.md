# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — a Ray + vLLM + DeepSpeed based distributed RLHF training framework. Development happens on the `feat-learn` branch; `main` stays a clean mirror of upstream.

This fork holds framework code only. Project-private material (specs, notes, offline doc snapshots, experiment scripts) lives in the consuming repo, not here.

## Commands

```bash
pip install -e .                       # dev install (deps are heavy: deepspeed, ray, vllm, flash-attn)
pytest tests/                          # run all tests (pyproject testpaths=./tests, verbose by default)
pytest tests/test_loss_aggregation.py -k <expr>   # single test
```

Formatting: black / isort (profile=black) / ruff, all with `line-length = 119`, target Python 3.10.

Training entry points are CLI modules, launched via Ray:

```bash
ray job submit --address="http://127.0.0.1:8265" -- python3 -m openrlhf.cli.train_ppo_ray <args>
python -m openrlhf.cli.train_sft / train_rm / train_dpo / serve_rm / lora_combiner
```

Runnable reference configs live in `examples/scripts/*.sh` (e.g. `train_ppo_ray_hybrid_engine.sh`); treat them as the source of truth for valid flag combinations.

## Architecture in one paragraph

`train_ppo_ray` builds Ray placement groups and spawns four kinds of GPU actors from `openrlhf/trainer/ray/launcher.py` — PolicyModelActor, CriticModelActor, ReferenceModelActor, RewardModelActor (each a DeepSpeed-wrapped `RayActorGroup`) — plus vLLM engines (`vllm_engine.py`) for rollout generation. `trainer/ppo_utils/` turns rollouts into training data: `samples_generator.py` → `experience_maker.py` (rewards, KL, advantages) → `replay_buffer.py` → `ppo_trainer.py` / `ppo_trainer_async.py` train steps. CLI args are hierarchized by `openrlhf/utils/config.py` into nested namespaces (`args.actor.*`, `args.vllm.*`, `args.algo.kl.*`, `args.train.*`, `args.ds.*`) — grep for the nested name, not the raw flag. Multi-turn agent RL goes through `openrlhf/utils/agent.py` executors.

## Repo-specific caveats

- Most code paths require multi-GPU + CUDA (DeepSpeed/vLLM/Ray). Only the pure-logic parts (losses, config, dataset packing, `tests/`) run without one — don't try to "verify" training changes by launching training locally.
- Never commit to `main`; it must stay identical to `upstream/main`. Work on `feat-learn`.
