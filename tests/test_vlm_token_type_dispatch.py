"""Regression tests for Actor._build_mm_token_type_ids across VLM families.

2026-08-06 review: the method read `config.vision_config.spatial_merge_size` unconditionally
in order to sanitize stray placeholder markers.  That attribute only exists on Qwen-style
mRoPE vision configs; on Gemma3 (SiglipVisionConfig) and Llava (CLIPVisionConfig) -- both of
which upstream supports, and for which token_type_ids drives the *bidirectional image
attention mask* rather than mRoPE -- it raised AttributeError on the first forward.  Had the
attribute existed, the fallback would have been worse: no `image_grid_thw` means sanitize
zeroes every image marker, silently disabling that mask.

These run on CPU with no weights: only the pure token_type_ids logic is under test.
"""
import torch

from openrlhf.models.actor import Actor


class _VisionCfg:
    """Stand-in for a non-mRoPE vision config: no spatial_merge_size."""


class _MRopeVisionCfg:
    spatial_merge_size = 2


class _Cfg:
    def __init__(self, vision_config, image_token_id=99, video_token_id=None):
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


def _actor(cfg):
    a = Actor.__new__(Actor)  # bypass __init__: no weights needed for this method
    a.is_vlm = True
    a._vlm_config = cfg
    return a


def test_non_mrope_vlm_keeps_image_markers():
    # Gemma/Llava shape: pixel_values but no *_grid_thw, and no spatial_merge_size.
    a = _actor(_Cfg(_VisionCfg()))
    seq = torch.tensor([[1, 99, 99, 5, 6]])
    out = a._build_mm_token_type_ids(seq, {"pixel_values": torch.zeros(1, 3, 4, 4)})
    assert out.tolist() == [[0, 1, 1, 0, 0]], "image markers must survive on a non-mRoPE VLM"


def test_mrope_vlm_still_sanitizes():
    # Qwen shape: grid present -> excess markers beyond grid capacity get re-typed as text.
    a = _actor(_Cfg(_MRopeVisionCfg()))
    seq = torch.tensor([[1, 99, 99, 99, 99, 5, 99, 6]])  # 5 markers, grid covers 4
    mm = {"image_grid_thw": torch.tensor([[1, 4, 4]]), "pixel_values": torch.zeros(16, 8)}
    out = a._build_mm_token_type_ids(seq, mm)
    assert out.tolist() == [[0, 1, 1, 1, 1, 0, 0, 0]]


def test_no_mm_inputs_leaves_markers_untouched():
    a = _actor(_Cfg(_MRopeVisionCfg()))
    seq = torch.tensor([[1, 99, 99, 5]])
    assert a._build_mm_token_type_ids(seq).tolist() == [[0, 1, 1, 0]]


def test_video_token_id_marked_when_configured():
    a = _actor(_Cfg(_MRopeVisionCfg(), video_token_id=98))
    seq = torch.tensor([[1, 99, 98, 5]])
    mm = {"image_grid_thw": torch.tensor([[1, 2, 2]]), "video_grid_thw": torch.tensor([[1, 2, 2]])}
    out = a._build_mm_token_type_ids(seq, mm)
    assert out.tolist() == [[0, 1, 2, 0]]
