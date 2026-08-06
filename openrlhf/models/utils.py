import os
from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_z3_leaf_modules(model: nn.Module, detect_hybrid: bool = True) -> None:
    """Auto-detect and set DeepSpeed ZeRO3 leaf modules.

    ZeRO3 prefetches submodule parameters assuming a fixed module traversal order.
    This breaks for:
      - MoE: dynamic expert routing makes prefetch unpredictable.
        (https://github.com/microsoft/DeepSpeed/pull/4966)
      - Hybrid architectures (e.g., Qwen3.5): same decoder layer class but different
        child submodules per instance (self_attn vs linear_attn).

    Marking these as z3 leaves forces whole-module allgather instead of per-submodule
    prefetch, at the cost of slightly higher peak memory.

    For hybrid architectures (detect_hybrid=True), the leaf marking works by
    registering a backwards hook that triggers allgather before backward. This
    interacts poorly with Qwen3.5's hybrid decoder layers: the hook steals
    gradient computation for ~390/417 inner parameters, leaving them frozen at
    their initial values. DeepSpeed's native per-instance prefetch handles the
    hybrid layout correctly on its own, so hybrid detection is disabled by
    default for actor/reward model training.

    Set OPENRLHF_Z3_LEAF_HYBRID=1 to re-enable hybrid detection for
    architectures that genuinely need it.
    """
    if os.environ.get("OPENRLHF_Z3_LEAF_HYBRID", "").strip().lower() in ("1", "true", "yes", "on"):
        detect_hybrid = True

    z3_leaf_classes = set()
    child_sigs: dict[type, frozenset[str]] = {}

    for m in model.modules():
        # MoE: dynamic expert routing
        if "SparseMoeBlock" in m.__class__.__name__:
            z3_leaf_classes.add(m.__class__)
            continue

        # Hybrid: same class, different child submodules across instances
        if detect_hybrid:
            cls = m.__class__
            children = frozenset(name for name, _ in m.named_children())
            if not children:
                continue
            if cls in child_sigs:
                if child_sigs[cls] != children:
                    z3_leaf_classes.add(cls)
            else:
                child_sigs[cls] = children

    if z3_leaf_classes:
        deepspeed.utils.set_z3_leaf_modules(model, list(z3_leaf_classes))
        for cls in z3_leaf_classes:
            print(f"Setting zero3 leaf: {cls.__name__}")


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_estimator == "k1":
        pass  # log_ratio is already p - q
    elif kl_estimator == "k2":
        # Non-negative KL approximation: (p - q)^2 / 2
        # http://joschu.net/blog/kl-approx.html
        # Approximately equivalent to one-step KL penalty with k1
        # used in https://arxiv.org/pdf/2310.10505.
        log_ratio = log_ratio**2 / 2.0
    elif kl_estimator == "k3":
        # Non-negative KL approximation: exp(q - p) - 1 - (q - p)
        # http://joschu.net/blog/kl-approx.html
        log_ratio = (-log_ratio).exp() - 1 + log_ratio
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    return log_ratio.clamp(min=-10, max=10)


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    kl_reward = -kl_coef * kl
    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward

    return reward


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def sanitize_mm_token_type_ids(token_type_ids, image_grid_thw, video_grid_thw, spatial_merge_size):
    """Neutralize stray multimodal placeholder markers so ``get_rope_index`` cannot consume a
    phantom grid (silent mRoPE position corruption, or a ``StopIteration`` / ``next(None)`` crash).

    The RL policy can generate image/video placeholder tokens *inside a response*, but the
    processor only guarantees the PROMPT placeholders match the provided grids. Any marker beyond
    the grid-implied capacity (or with no grid at all) is spurious -> re-type it as text (0).
    Handles the single-item-per-sample case exactly; multi-item row->grid mapping is ambiguous
    here so the per-row clamp is skipped (the rollout logit-bias guard remains the primary
    prevention). Image (1) and video (2) markers get the same treatment. ``token_type_ids`` is
    (B, L) with 0=text/1=image/2=video; modified in place.
    """
    merge2 = spatial_merge_size**2

    def _clamp(marker, grid):
        if grid is None:
            # No grid for this modality at all -> every such marker is spurious.
            token_type_ids[token_type_ids == marker] = 0
            return
        # Tokens one grid expands into: prod(t, h, w) / merge**2.  (get_rope_index splits a
        # video grid per frame, but the per-sample total it consumes is the same.)
        per_item = (grid[:, 0] * grid[:, 1] * grid[:, 2] // merge2).tolist()
        # len(per_item) == B alone does NOT imply one item per row: a mixed text/image batch can
        # hit it by coincidence (e.g. 2 image rows x 2 images in a batch of 4), and then row i's
        # grid is not per_item[i] and the clamp would zero legitimate markers. Requiring every
        # row to carry a marker closes that -- if every row has >= 1 item and the item count
        # equals B, the mapping is necessarily 1:1.
        row_has_marker = (token_type_ids == marker).any(dim=-1).all().item()
        if len(per_item) == token_type_ids.size(0) and row_has_marker:
            for i in range(token_type_ids.size(0)):
                idx = (token_type_ids[i] == marker).nonzero(as_tuple=True)[0]
                if idx.numel() > per_item[i]:
                    token_type_ids[i, idx[per_item[i] :]] = 0

    _clamp(1, image_grid_thw)
    _clamp(2, video_grid_thw)
    return token_type_ids


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    # keepdim=True so the per-row mean/var broadcast back over `dim` correctly; masked_mean
    # reduces `dim` without keepdim, so using it here would broadcast along the wrong axis.
    mask_sum = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    mean = (tensor * mask).sum(dim=dim, keepdim=True) / mask_sum
    mean_centered = (tensor - mean) * mask
    var = (mean_centered**2).sum(dim=dim, keepdim=True) / mask_sum
    return mean_centered * var.clamp(min=eps).rsqrt()


@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
