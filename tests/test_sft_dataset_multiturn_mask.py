import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset

ASSISTANT_ID = 3
TURN_END_ID = 4
OTHER_ID = 1


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
        "_sft_dataset_multiturn_under_test", root / "openrlhf" / "datasets" / "sft_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SFTDataset = _load_sft_dataset_module().SFTDataset

TRAJECTORY = [
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "<think> t1 </think> a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "<think> t2 </think> a2"},
]


class WordTokenizer:
    """Whitespace tokenizer; words of an assistant turn get ASSISTANT_ID, everything else OTHER_ID."""

    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, strip_earlier_reasoning):
        self.strip_earlier_reasoning = strip_earlier_reasoning

    def __call__(
        self, text, max_length=None, padding=False, truncation=False, return_tensors=None, add_special_tokens=True
    ):
        ids = []
        inside_assistant = False
        for word in text.split():
            if word == "<assistant>":
                inside_assistant = True
                ids.append(OTHER_ID)
            elif word == "</assistant>":
                inside_assistant = False
                ids.append(TURN_END_ID)
            elif word == self.eos_token:
                ids.append(self.eos_token_id)
            else:
                ids.append(ASSISTANT_ID if inside_assistant else OTHER_ID)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors != "pt":
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for i, message in enumerate(messages):
            content = message["content"]
            # what a thinking template does: reasoning is kept only for the final turn
            if self.strip_earlier_reasoning and message["role"] == "assistant" and i != len(messages) - 1:
                content = re.sub(r"<think>.*?</think>", "", content).strip()
            parts.append(f"<{message['role']}> {content} </{message['role']}> ")
        rendered = "".join(parts)
        return rendered + "<assistant> " if add_generation_prompt else rendered


def _build(strip_earlier_reasoning):
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(input_key="messages", output_key=None, apply_chat_template=True, multiturn=True)
        ),
        print=lambda *a: None,
    )
    tokenizer = WordTokenizer(strip_earlier_reasoning)
    return SFTDataset(
        Dataset.from_list([{"messages": TRAJECTORY}]), tokenizer, 128, strategy, num_processors=1, multiturn=True
    )


def test_template_that_rewrites_earlier_turns_is_rejected():
    # Without this check the spans silently slide onto the user turns and the tool output.
    with pytest.raises(Exception) as excinfo:
        _build(strip_earlier_reasoning=True)
    assert "turn 1" in str(excinfo.value)


def test_stable_template_supervises_exactly_the_assistant_turns():
    dataset = _build(strip_earlier_reasoning=False)
    input_ids, _, loss_mask = dataset[0]
    ids, mask = input_ids[0].tolist(), loss_mask[0].tolist()
    # a mask position supervises the token it predicts, i.e. the next one
    supervised = [ids[i + 1] for i, m in enumerate(mask[:-1]) if m > 0]
    assert set(supervised) == {ASSISTANT_ID, TURN_END_ID}, "user turns must not be supervised"
    assert supervised.count(ASSISTANT_ID) == ids.count(ASSISTANT_ID), "every assistant token is supervised"
    assert supervised.count(TURN_END_ID) == 2, "both assistant turns end with a supervised terminator"
