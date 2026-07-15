# Official Docs Map (local snapshot in docs/official-docs/)

`docs/official-docs/` is a **gitignored local snapshot** of the crawled OpenRLHF and vLLM official docs (EN original + generated ZH translation). The master archive is a separate repo at `~/Desktop/learning-docs` (`konghw-git/learning-docs` on GitHub) — it holds more sites plus the crawl/translation tooling. If the snapshot is missing, copy the two site directories back from the archive (the archive's `vllm.io/en/api/`, 1.1G of auto-generated API reference, is deliberately excluded here).

**Use these as the authoritative reference** when answering questions about features, options, or setup — prefer citing a page over reasoning from memory. Pages are static HTML; readable text sits in the main content div. Sphinx sites also keep plain-text sources under `_sources/*.rst.txt` — often easier to read than HTML.

## OpenRLHF — `docs/official-docs/openrlhf.readthedocs.io/`

English under `en/latest/`, Chinese under `zh/latest/` (same filenames).

| Page | Covers |
|---|---|
| `index.html` | Docs home, feature overview, navigation |
| `quick_start.html` | Installation, dataset preparation, first SFT/RM/PPO runs |
| `common_options.html` | **Full CLI flag reference** for all trainers — check here before adding/renaming a flag |
| `architecture.html` | Official architecture description (Ray actors, vLLM, DeepSpeed) |
| `agent_paradigm.html` | Agent-based execution design (single-turn vs multi-turn) |
| `agent_training.html` | How to write and train custom `AgentInstanceBase` agents |
| `async_training.html` | Async pipeline RL training mode |
| `hybrid_engine.html` | Hybrid engine (colocated training + inference, vLLM sleep/wake) |
| `sequence_parallelism.html` | Ring attention / sequence parallelism for long context |
| `multi-node.html` | Multi-node deployment (Ray cluster, SLURM) |
| `non_rl.html` | SFT / RM / DPO (non-RL) training |
| `checkpoint.html` | Checkpointing, resume, DeepSpeed↔HF conversion |
| `performance.html` | Performance tuning guidance |
| `troubleshooting.html` | Known issues and fixes |
| `nvidia_docker.html` | Docker / NVIDIA container setup |

## vLLM — `docs/official-docs/vllm.io/`

English under `en/`, Chinese under `zh/` (same tree). Crawled from docs.vllm.ai/en/latest — a large MkDocs site; navigate by directory: `getting_started/`, `serving/`, `models/`, `features/` (quantization, LoRA, structured output…), `configuration/`, `design/` (architecture internals). Start from `index.html` or the section's own index page rather than guessing filenames. The auto-generated `api/` reference is not in this snapshot — see the archive repo if needed.

## Conventions

- This snapshot is read-only reference material — don't edit it. Re-crawling, translation, and adding sites all happen in the `~/Desktop/learning-docs` archive repo (see its README and CLAUDE.md), then re-copy the site trees here.
- Docs for other projects (Apodex AI platform, OpenRouter, …) live only in the archive repo, not in this snapshot.
