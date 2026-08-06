"""Regression tests for sanitize_mm_token_type_ids.

Root cause (kong C2-4B, 2026-07-24): the RL policy generated a video placeholder token
(id 151656) inside a response; the VLM position builder marked it modality-2 and
transformers get_rope_index did next(grid_iters[2]) with video_grid_thw=None -> crash
`TypeError: 'NoneType' object is not an iterator`. A stray image placeholder likewise makes
get_rope_index consume a phantom grid (StopIteration). sanitize_mm_token_type_ids neutralizes
such stray markers (defense-in-depth behind the rollout logit-bias guard).
"""
import torch

from openrlhf.models.utils import sanitize_mm_token_type_ids


def _grid(t, h, w):
    return torch.tensor([[t, h, w]])


def test_video_marker_dropped_when_no_video_grid():
    # 0=text,2=video ; a generated video token with no video grid must become text.
    tt = torch.tensor([[0, 1, 1, 1, 1, 0, 2, 0]])  # 4 image tokens (matching grid) + 1 stray video
    out = sanitize_mm_token_type_ids(tt, image_grid_thw=_grid(1, 4, 4), video_grid_thw=None, spatial_merge_size=2)
    assert (out == 2).sum().item() == 0  # no video markers remain
    assert (out == 1).sum().item() == 4  # legit image markers preserved


def test_excess_image_markers_clamped_to_grid():
    # grid 1x4x4 / merge^2(4) = 4 image tokens; a 5th (stray) image marker must be dropped.
    tt = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 0]])  # 5 image markers, only 4 legit
    out = sanitize_mm_token_type_ids(tt, image_grid_thw=_grid(1, 4, 4), video_grid_thw=None, spatial_merge_size=2)
    assert (out == 1).sum().item() == 4  # clamped to grid capacity
    # the first 4 (prompt) are kept, the trailing stray one is zeroed
    assert out.tolist() == [[0, 1, 1, 1, 1, 0, 0, 0]]


def test_clean_sequence_unchanged():
    tt = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]])  # exactly 4 image tokens, no strays
    before = tt.clone()
    out = sanitize_mm_token_type_ids(tt, image_grid_thw=_grid(1, 4, 4), video_grid_thw=None, spatial_merge_size=2)
    assert torch.equal(out, before)


def test_no_image_grid_drops_all_image_markers():
    tt = torch.tensor([[0, 1, 0, 2, 0]])
    out = sanitize_mm_token_type_ids(tt, image_grid_thw=None, video_grid_thw=None, spatial_merge_size=2)
    assert out.sum().item() == 0  # every marker neutralized when no grids exist


def test_excess_video_markers_clamped_to_grid():
    # Symmetry with the image clamp: a batch that legitimately carries video must still have a
    # STRAY video marker removed.  The pre-2026-08-06 version only clamped image markers, so a
    # generated <|video_pad|> in a video run survived and shifted every later mRoPE position.
    tt = torch.tensor([[0, 2, 2, 2, 2, 0, 2, 0]])  # 5 video markers, grid only covers 4
    out = sanitize_mm_token_type_ids(
        tt, image_grid_thw=None, video_grid_thw=_grid(1, 4, 4), spatial_merge_size=2
    )
    assert out.tolist() == [[0, 2, 2, 2, 2, 0, 0, 0]]


def test_video_grid_present_leaves_legit_video_markers():
    tt = torch.tensor([[0, 2, 2, 2, 2, 0]])
    before = tt.clone()
    out = sanitize_mm_token_type_ids(
        tt, image_grid_thw=None, video_grid_thw=_grid(1, 4, 4), spatial_merge_size=2
    )
    assert torch.equal(out, before)


def test_image_and_video_clamped_independently_in_one_batch():
    # 4 legit image + 4 legit video tokens, one stray of each.
    tt = torch.tensor([[1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2]])
    out = sanitize_mm_token_type_ids(
        tt, image_grid_thw=_grid(1, 4, 4), video_grid_thw=_grid(1, 4, 4), spatial_merge_size=2
    )
    assert out.tolist() == [[1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0]]


def test_multi_image_batch_skips_row_clamp_but_drops_video():
    # 2 grids but batch size 1 -> per-row mapping ambiguous: image clamp skipped, video still dropped.
    tt = torch.tensor([[1, 1, 1, 1, 2]])
    out = sanitize_mm_token_type_ids(
        tt, image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]), video_grid_thw=None, spatial_merge_size=2
    )
    assert (out == 2).sum().item() == 0  # video always dropped when no video grid
    assert (out == 1).sum().item() == 4  # image markers left intact (guard defers to rollout logit-bias)
