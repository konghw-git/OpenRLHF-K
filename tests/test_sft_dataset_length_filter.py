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


def _load_sft_dataset_module():
    root = Path(__file__).resolve().parents[1]

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

    spec = importlib.util.spec_from_file_location(
        "_sft_dataset_under_test", root / "openrlhf" / "datasets" / "sft_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SFTDataset = _load_sft_dataset_module().SFTDataset


class WordTokenizer:
    """Whitespace tokenizer: one id per word, EOS_TOKEN -> EOS_ID."""

    eos_token = EOS_TOKEN
    eos_token_id = EOS_ID
    pad_token_id = 0

    def __call__(
        self, text, max_length=None, padding=False, truncation=False, return_tensors=None, add_special_tokens=True
    ):
        ids = [EOS_ID if word == EOS_TOKEN else 1 for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors != "pt":  # same shape contract as a HF tokenizer
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<{m['role']}> {m['content']} </{m['role']}> " for m in messages)
        return rendered + "<assistant> " if add_generation_prompt else rendered


class FakeStrategy:
    def __init__(self):
        self.args = SimpleNamespace(
            data=SimpleNamespace(input_key="messages", output_key=None, apply_chat_template=True, multiturn=False)
        )
        self.printed = []

    def print(self, *args):
        self.printed.append(" ".join(str(a) for a in args))


def _dataset(response_words, max_length):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "what is six times seven"},
                {"role": "assistant", "content": " ".join(["step"] * response_words)},
            ]
        }
    ]
    strategy = FakeStrategy()
    dataset = SFTDataset(Dataset.from_list(rows), WordTokenizer(), max_length, strategy, num_processors=1)
    return dataset, strategy


def test_long_target_with_short_prompt_is_dropped():
    # A prompt-only length check keeps this row and silently truncates the target at
    # max_length, i.e. trains the model to stop mid-derivation.
    dataset, strategy = _dataset(response_words=200, max_length=32)
    assert len(dataset) == 0
    assert any("dropped 1/1" in line for line in strategy.printed)


def test_sample_that_fits_is_kept_and_ends_with_eos():
    dataset, _ = _dataset(response_words=4, max_length=64)
    assert len(dataset) == 1
    input_ids, attention_mask, loss_mask = dataset[0]
    # every kept sample already ends with EOS, so no EOS has to be forced onto it
    assert int(input_ids[0][-1]) == EOS_ID
    assert int(attention_mask[0][-1]) == 1
    # the supervised span is the response, not the prompt
    assert loss_mask[0, 0] == 0
    assert loss_mask.sum() > 0


def test_truncated_sample_keeps_its_real_last_token():
    dataset, _ = _dataset(response_words=4, max_length=64)
    # shrink the budget so __getitem__ has to truncate this row: the last token must stay
    # what the data says, not a forged EOS that teaches the model to stop mid-derivation
    dataset.max_length = 6
    input_ids, _, _ = dataset[0]
    assert input_ids.shape[-1] == 6
    assert int(input_ids[0][-1]) != EOS_ID
