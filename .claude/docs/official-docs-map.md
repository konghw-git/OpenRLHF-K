# Official Docs Map (crawled readthedocs)

`docs/openrlhf.readthedocs.io/` is a full crawl of the official OpenRLHF documentation:

- `en/latest/*.html` — original English pages
- `zh/latest/*.html` — Chinese translation of the same pages (same filenames, HTML structure identical; translated by the repo owner with `translate_docs.py` + `polish_docs.py`, Gemini-based, run via `uv run` inside that directory)

**Use these as the authoritative reference** when answering questions about features, options, or setup — prefer citing a page here over reasoning from memory. The HTML is Sphinx output; the readable text is inside `<div role="main">`. Plain-text sources also exist under `en/latest/_sources/*.rst.txt` — often easier to read than the HTML.

## Page index (same filenames under `en/latest/` and `zh/latest/`)

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

## Conventions for this directory

- Do not hand-edit `zh/latest/` translations casually — they are generated/refined by the translation pipeline; fix systematic issues by rerunning `translate_docs.py` / `polish_docs.py` instead. Reports: `translation_report.md`, `polish_report.md`.
- `translate_docs.py`, `polish_docs.py`, `pyproject.toml`, `uv.lock` in this directory are the owner's tooling, independent of the OpenRLHF package — don't import them from `openrlhf/` code and don't ship them in the wheel.
- When upstream docs update, re-crawl `en/latest/` first, then rerun the translation pipeline for changed pages.
