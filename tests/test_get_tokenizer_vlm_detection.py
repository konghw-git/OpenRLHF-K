"""Tests for VLM detection in ``get_tokenizer``.

A VLM must get an ``AutoProcessor`` (tokenizer + image_processor); a text-only model must get a
plain ``AutoTokenizer``. The distinction matters beyond tokenization: ``save_model`` persists
whatever ``get_tokenizer`` returned, so a VLM that silently receives a plain tokenizer produces
checkpoints without any image-preprocessing config.

``is_vlm`` is set on the ``Actor`` wrapper, not on the HF module it wraps, and callers pass either
one. These tests pin down all three cases: attribute present and true, present and false, and
absent (must fall back to sniffing the config rather than assuming text-only).
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from openrlhf.utils import utils as utils_mod


class _FakeInnerTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 7


class _FakeProcessor:
    """Stand-in for AutoProcessor: exposes ``tokenizer`` and ``image_processor``."""

    def __init__(self):
        self.tokenizer = _FakeInnerTokenizer()
        self.image_processor = object()


class _FakeTokenizer:
    """Stand-in for AutoTokenizer: no ``image_processor``."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 7


@pytest.fixture
def loaders(monkeypatch):
    """Replace both HF loaders and record which one was called."""
    calls = []

    fake_processor_cls = MagicMock()
    fake_processor_cls.from_pretrained.side_effect = lambda *a, **k: (
        calls.append("processor"),
        _FakeProcessor(),
    )[1]
    fake_tokenizer_cls = MagicMock()
    fake_tokenizer_cls.from_pretrained.side_effect = lambda *a, **k: (
        calls.append("tokenizer"),
        _FakeTokenizer(),
    )[1]

    # AutoProcessor is imported inside the function body, AutoTokenizer at module import time.
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = fake_processor_cls
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(utils_mod, "AutoTokenizer", fake_tokenizer_cls)

    return calls


@pytest.fixture
def sniffed(monkeypatch):
    """Record whether the config-sniffing fallback ran, and force it to say 'VLM'."""
    seen = []

    def _fake(pretrain):
        seen.append(pretrain)
        return True

    monkeypatch.setattr(utils_mod, "is_vlm_model", _fake)
    return seen


def test_wrapper_with_is_vlm_true_gets_processor(loaders, sniffed):
    actor = types.SimpleNamespace(is_vlm=True, config=types.SimpleNamespace(pad_token_id=None))

    tok = utils_mod.get_tokenizer("some/vlm", actor)

    assert loaders == ["processor"]
    assert hasattr(tok, "image_processor")
    assert sniffed == [], "attribute was present; must not re-sniff the config"


def test_model_without_is_vlm_falls_back_to_config(loaders, sniffed):
    """The regression: PolicyModelActor passes ``actor.model``, which has no ``is_vlm``.

    A missing attribute means "unknown", not "text-only" — so the config decides.
    """
    hf_model = types.SimpleNamespace(config=types.SimpleNamespace(pad_token_id=None))
    assert not hasattr(hf_model, "is_vlm")

    tok = utils_mod.get_tokenizer("some/vlm", hf_model)

    assert loaders == ["processor"], "missing is_vlm must not silently select the text-only branch"
    assert hasattr(tok, "image_processor")
    assert sniffed == ["some/vlm"]


def test_is_vlm_false_is_honoured_without_sniffing(loaders, sniffed):
    """An explicit False is an answer, not a missing value — the fallback must stay out."""
    text_model = types.SimpleNamespace(is_vlm=False, config=types.SimpleNamespace(pad_token_id=None))

    tok = utils_mod.get_tokenizer("some/llm", text_model)

    assert loaders == ["tokenizer"]
    assert not hasattr(tok, "image_processor")
    assert sniffed == [], "explicit False must win over the config sniff"


def test_model_none_sniffs_the_config(loaders, sniffed):
    tok = utils_mod.get_tokenizer("some/vlm", None)

    assert loaders == ["processor"]
    assert hasattr(tok, "image_processor")
    assert sniffed == ["some/vlm"]


def test_processor_branch_mirrors_pad_token_onto_the_wrapper(loaders, sniffed):
    """AutoProcessor doesn't delegate tokenizer attributes; downstream code reads them off it."""
    actor = types.SimpleNamespace(is_vlm=True, config=types.SimpleNamespace(pad_token_id=None))

    tok = utils_mod.get_tokenizer("some/vlm", actor, padding_side="left")

    assert tok.tokenizer.padding_side == "left"
    assert tok.pad_token == "<eos>"
    assert tok.pad_token_id == 7
    assert actor.config.pad_token_id == 7
