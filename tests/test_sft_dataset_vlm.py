"""Tests for the VLM-specific parts of SFTDataset that need no weights or dataset.

`_apply_image_pixel_limits`: the limits are expressed as longest_edge/shortest_edge, which is a
Qwen*-VL convention. Image processors sized by height/width (Siglip on Gemma3, CLIP on Llava)
must be left alone -- replacing their `size` with edge keys makes resize() fail on "height".

`_check_unsupported_combinations`: SFT has no VLM flag on the command line, so combinations that
images break can only be rejected once the dataset is built. Left unchecked, Actor.forward
asserts on the first micro-batch that carries images, which on a mixed corpus can be many steps
into a run.
"""

import pytest

from openrlhf.datasets.sft_dataset import SFTDataset


class _Size(dict):
    """Mimics transformers' SizeDict: attribute access with None for absent keys."""

    def __getattr__(self, name):
        return self.get(name)


class _ImageProcessor:
    def __init__(self, size):
        self.size = _Size(size)


class _Processor:
    def __init__(self, size):
        self.image_processor = _ImageProcessor(size)


class _Strategy:
    def __init__(self, data_args, ds_args):
        self.args = type("Args", (), {"data": type("Data", (), data_args)(), "ds": type("Ds", (), ds_args)()})()
        self.printed = []

    def print(self, msg):
        self.printed.append(msg)


def _dataset(size, ds_args=None, **data_args):
    ds = SFTDataset.__new__(SFTDataset)  # bypass __init__: no tokenizer or dataset needed
    ds.processor = _Processor(size)
    ds.strategy = _Strategy(data_args, ds_args or {"ring_attn_size": 1})
    return ds


QWEN_SIZE = {"longest_edge": 1003520, "shortest_edge": 3136}
SIGLIP_SIZE = {"height": 224, "width": 224}


def test_edge_sized_processor_is_overridden():
    ds = _dataset(QWEN_SIZE, image_max_pixels=2097152, image_min_pixels=1024)
    ds._apply_image_pixel_limits()
    assert ds.processor.image_processor.size == {"longest_edge": 2097152, "shortest_edge": 1024}


def test_hw_sized_processor_is_left_untouched():
    ds = _dataset(SIGLIP_SIZE, image_max_pixels=2097152, image_min_pixels=None)
    ds._apply_image_pixel_limits()
    assert ds.processor.image_processor.size == SIGLIP_SIZE
    assert any("not sized by longest_edge" in m for m in ds.strategy.printed)


@pytest.mark.parametrize("size", [QWEN_SIZE, SIGLIP_SIZE])
def test_no_limits_configured_is_a_noop(size):
    ds = _dataset(size, image_max_pixels=None, image_min_pixels=None)
    ds._apply_image_pixel_limits()
    assert ds.processor.image_processor.size == size


def test_only_max_pixels_keeps_the_processor_shortest_edge():
    ds = _dataset(QWEN_SIZE, image_max_pixels=2097152, image_min_pixels=None)
    ds._apply_image_pixel_limits()
    assert ds.processor.image_processor.size == {"longest_edge": 2097152, "shortest_edge": 3136}


def _guard_dataset(has_images, ring_attn_size=1, multiturn=False):
    ds = _dataset(QWEN_SIZE, ds_args={"ring_attn_size": ring_attn_size})
    ds.has_images = has_images
    ds.multiturn = multiturn
    return ds


def test_images_with_ring_attention_is_rejected():
    with pytest.raises(ValueError, match="does not support ring attention"):
        _guard_dataset(has_images=True, ring_attn_size=4)._check_unsupported_combinations()


def test_images_with_multiturn_is_rejected():
    with pytest.raises(NotImplementedError, match="Multiturn SFT with images"):
        _guard_dataset(has_images=True, multiturn=True)._check_unsupported_combinations()


@pytest.mark.parametrize("ring_attn_size,multiturn", [(1, False), (4, False), (1, True), (4, True)])
def test_text_only_dataset_accepts_every_combination(ring_attn_size, multiturn):
    # No images -> ring attention and multiturn stay available, exactly as upstream.
    ds = _guard_dataset(has_images=False, ring_attn_size=ring_attn_size, multiturn=multiturn)
    ds._check_unsupported_combinations()


def test_images_without_ring_attention_is_accepted():
    _guard_dataset(has_images=True, ring_attn_size=1)._check_unsupported_combinations()
