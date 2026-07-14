# Git Workflow for this Fork

## Remotes and identity

- `origin` → `git@github-git:konghw-git/OpenRLHF-K.git` (the fork; `github-git` is an SSH host alias in `~/.ssh/config` bound to the konghw-git account key)
- `upstream` → `https://github.com/OpenRLHF/OpenRLHF.git` (official, fetch-only by convention)
- Commit identity is repo-local: `konghw-git <206675831+konghw-git@users.noreply.github.com>`. Do not change global git config.

## Branch rules

- **`main` is a pristine mirror of `upstream/main`. Never commit to it.** Sync it with:
  ```bash
  git checkout main && git fetch upstream && git merge --ff-only upstream/main && git push origin main
  ```
- **`feat-learn`** is the development branch (tracks `origin/feat-learn`). All work, commits, and CLAUDE-managed docs live here.
- To pull upstream updates into development: sync `main` first, then `git merge main` on `feat-learn` (merge, not rebase — the branch is pushed).
- New experiments that might become upstream PRs should branch off `main`, not `feat-learn`, so the diff stays clean.

## CI note

`.github/workflows/python-package.yml` is upstream's PyPI release workflow; it is guarded by `if: github.repository_owner == 'OpenRLHF'` and will not run in this fork. Leave it in place to keep the upstream diff minimal.
