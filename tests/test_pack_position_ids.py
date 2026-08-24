import importlib.util
import sys
from pathlib import Path

import torch
from flash_attn.bert_padding import unpad_input


def _load_ring_attn_utils():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_ring_attn_utils_under_test", root / "openrlhf" / "models" / "ring_attn_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pack_position_ids = _load_ring_attn_utils().pack_position_ids


def test_packed_rows_stay_token_aligned():
    lens = [5, 3]
    seqlen = max(lens)
    attention_mask = torch.zeros(len(lens), seqlen, dtype=torch.int32)
    for row, length in enumerate(lens):
        attention_mask[row, :length] = 1

    # row 0: text position; rows 1-3: distinguishable mRoPE stand-ins
    text_pos = torch.clip(attention_mask.long().cumsum(-1) - 1, min=0)
    pos4 = torch.stack([text_pos, text_pos + 100, text_pos + 200, text_pos + 300])

    _, indices, _, _, _ = unpad_input(torch.zeros(len(lens), seqlen, 1), attention_mask)
    packed = pack_position_ids(pos4, indices)

    assert packed.shape == (4, 1, sum(lens))
    expected_text = torch.cat([torch.arange(length) for length in lens])
    assert torch.equal(packed[0, 0], expected_text)
    for row in range(1, 4):
        assert torch.equal(packed[row, 0], expected_text + row * 100)
