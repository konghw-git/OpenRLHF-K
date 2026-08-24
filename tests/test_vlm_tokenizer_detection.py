import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from datasets import Dataset

EOS_TOKEN = "<eos>"
EOS_ID = 2


def _load_module(relpath, name):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sft_dataset_module():
    utils_module = types.ModuleType("openrlhf.utils.utils")

    def zero_pad_sequences(sequences, side="left", value=0, stack=False):
        max_len = max(sequence.size(-1) for sequence in sequences)
        padded = []
        for sequence in sequences:
            pad_len = max_len - sequence.size(-1)
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
            padded.append(F.pad(sequence, padding, value=value))
        return torch.stack(padded) if stack else torch.cat(padded)

    utils_module.zero_pad_sequences = zero_pad_sequences
    sys.modules["openrlhf.utils.utils"] = utils_module
    return _load_module("openrlhf/datasets/sft_dataset.py", "_sft_dataset_vlm_under_test")


class InnerTokenizer:
    """Whitespace tokenizer standing in for a processor's inner tokenizer."""

    eos_token = EOS_TOKEN
    eos_token_id = EOS_ID
    pad_token = EOS_TOKEN
    pad_token_id = 0
    padding_side = "left"

    def __call__(
        self, text, max_length=None, padding=False, truncation=False, return_tensors=None, add_special_tokens=True
    ):
        ids = [EOS_ID if word == EOS_TOKEN else 1 for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors != "pt":
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<{m['role']}> {m['content']} </{m['role']}> " for m in messages)
        return rendered + "<assistant> " if add_generation_prompt else rendered


class FakeProcessor:
    """Mimics AutoProcessor: `images` is the first positional argument."""

    def __init__(self):
        self.image_processor = object()
        self.tokenizer = InnerTokenizer()

    def __call__(self, images=None, text=None, **kwargs):
        if isinstance(images, str):
            raise ValueError(f"Incorrect image source. Got {images}")
        return self.tokenizer(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)


def test_missing_is_vlm_attribute_falls_back_to_config_sniffing(monkeypatch):
    utils = _load_module("openrlhf/utils/utils.py", "_openrlhf_utils_under_test")
    monkeypatch.setattr(utils, "is_vlm_model", lambda pretrain: True)
    monkeypatch.setattr(utils, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: InnerTokenizer()))
    # get_tokenizer imports AutoProcessor inside the function, so the stub goes in sys.modules
    stub_transformers = types.ModuleType("transformers")
    stub_transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor())
    monkeypatch.setitem(sys.modules, "transformers", stub_transformers)
    # the HF module a caller passes carries no `is_vlm`; the config says VLM
    hf_module = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))

    tokenizer = utils.get_tokenizer("some/vlm", hf_module, "right")
    assert hasattr(tokenizer, "image_processor"), "a VLM must get the processor, not a plain tokenizer"

    # an explicit False is still honoured
    text_only = utils.get_tokenizer("some/vlm", SimpleNamespace(is_vlm=False, config=SimpleNamespace()), "right")
    assert not hasattr(text_only, "image_processor")


def test_sft_dataset_tokenizes_text_through_the_inner_tokenizer():
    SFTDataset = _load_sft_dataset_module().SFTDataset
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(input_key="messages", output_key=None, apply_chat_template=True, multiturn=False)
        ),
        print=lambda *a: None,
    )
    rows = [
        {
            "messages": [
                {"role": "user", "content": "what is six times seven"},
                {"role": "assistant", "content": "forty two"},
            ]
        }
    ]
    dataset = SFTDataset(Dataset.from_list(rows), FakeProcessor(), 64, strategy, num_processors=1)
    assert len(dataset) == 1
    input_ids, attention_mask, loss_mask = dataset[0]
    assert input_ids.shape == attention_mask.shape == loss_mask.shape
    assert loss_mask.sum() > 0
