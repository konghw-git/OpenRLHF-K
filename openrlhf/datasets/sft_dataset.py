import logging
from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences

logger = logging.getLogger(__name__)

# Processor outputs that must NOT be forwarded as multimodal tensors: they are
# sequence-length dependent, so they would break right-padding/batching.  Actor.forward
# rebuilds (mm_)token_type_ids from input_ids for the full prompt+response sequence.
# Mirrors openrlhf/utils/vlm_utils.py::process_prompt_with_images.
_MM_SKIP_KEYS = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}


def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # VLM 下 get_tokenizer 返回的是 AutoProcessor(utils.py:51-68),而 processor.__call__
        # 的第一个位置参数是 images(Qwen3VLProcessor)。本类所有纯文本 tokenize 必须走
        # 内层 tokenizer,否则文本会被当成图片路径 -> ValueError: Incorrect image source。
        # 纯文本模型下 processor is None、text_tokenizer is tokenizer,行为不变。
        self.processor = tokenizer if hasattr(tokenizer, "image_processor") else None
        self.text_tokenizer = tokenizer.tokenizer if self.processor is not None else tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn
        self._truncation_warned = False

        if self.processor is not None and hasattr(self.processor, "image_processor"):
            # Cap the image resolution: uncapped images blow up the placeholder expansion,
            # which OOMs or overruns max_len (C2 crashed on exactly this).  Same cap as the
            # RL side uses.
            self.processor.image_processor.size = {"longest_edge": 2097152, "shortest_edge": 3136}
            self._image_pad_token = "<|image_pad|>"
            pad_id = self.text_tokenizer.convert_tokens_to_ids(self._image_pad_token)
            self._image_pad_id = None if pad_id == self.text_tokenizer.unk_token_id else pad_id
        else:
            self._image_pad_token = None
            self._image_pad_id = None

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.output_key = getattr(self.strategy.args.data, "output_key", None)
        self.image_key = getattr(self.strategy.args.data, "image_key", "images")
        self.apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args.data, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        num_raw = len(processed_dataset)
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        num_dropped = num_raw - len(processed_dataset)
        if num_dropped:
            # Make the length filter visible: dropping is correct, doing it silently is not.
            self.strategy.print(
                f"SFTDataset: dropped {num_dropped}/{num_raw} samples "
                f"(empty, or prompt+response longer than max_len={self.max_length})"
            )

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.images = processed_dataset["images"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

        if self.multiturn and self.processor is not None and any(self.images):
            # response_ranges are computed with the TEXT tokenizer, so they do not account for
            # the image placeholder expansion -> every range would be shifted left. Refuse
            # rather than train on a silently misaligned loss mask.
            raise NotImplementedError(
                "Multiturn SFT with images is not supported yet (response_ranges are text-caliber "
                "and would misalign the loss mask by the image placeholder expansion)."
            )

    def _num_image_placeholder_tokens(self, images):
        """Number of tokens one image placeholder expands into, per image, summed.

        Uses the processor's public size->patch utility (the same one vLLM uses to plan
        placeholders), so only the image *header* is read -- no decode, no resize.  Returns
        None when it cannot be determined, in which case callers fall back to text caliber.
        """
        if self.processor is None or not images:
            return 0
        try:
            from PIL import Image

            sizes = []
            for ref in images if isinstance(images, list) else [images]:
                if ref is None:
                    continue
                if isinstance(ref, Image.Image):
                    w, h = ref.size
                else:
                    with Image.open(ref) as im:  # lazy: reads the header only
                        w, h = im.size
                sizes.append([h, w])
            if not sizes:
                return 0
            mm_data = self.processor._get_num_multimodal_tokens(image_sizes=sizes)
            return int(sum(mm_data.num_image_tokens))
        except Exception as e:  # unknown processor API / unreadable image
            logger.warning(f"Cannot pre-compute image token count ({e}); falling back to text-caliber length")
            return None

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajectory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(data[input_key][:idx], tokenize=False, add_generation_prompt=True)
                    response = apply_chat_template(data[input_key][: idx + 1], tokenize=False)[len(prompt) :]

                    start_idx = (
                        self.text_tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                        start_idx
                        + self.text_tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    response_ranges.append((start_idx, end_idx))  # left close right close

        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        images = data.get(self.image_key, None) if self.image_key else None
        if images is not None and not isinstance(images, list):
            images = [images]

        if not self.pretrain_mode:
            prompt_token = self.text_tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            drop = not prompt or not response or prompt_ids_len >= self.max_length - 2

            # Also filter on the TOTAL length.  Checking only the prompt lets a short prompt
            # with a long (e.g. long-CoT) target through, where it is silently truncated at
            # max_length -- i.e. the model is trained on a derivation that stops mid-way.
            # Dropping is the correct behaviour; truncating is not.
            if not drop:
                response_ids_len = len(
                    self.text_tokenizer(response, padding=False, truncation=False, add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                # VLM: one <|image_pad|> in the prompt expands into grid/merge**2 tokens at
                # __getitem__ time, so the text caliber above underestimates image samples.
                extra = 0
                if images and self.processor is not None:
                    n_expanded = self._num_image_placeholder_tokens(images)
                    if n_expanded is not None and self._image_pad_token is not None:
                        n_pad = prompt.count(self._image_pad_token)
                        extra = n_expanded - n_pad
                # +2: the eos appended in __getitem__, plus one token of slack for the
                # prompt/response boundary re-tokenization.
                if prompt_ids_len + response_ids_len + extra + 2 > self.max_length:
                    drop = True
            if drop:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "images": images,
            "response_ranges": response_ranges if self.multiturn else None,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.text_tokenizer.eos_token):
                text += " " + self.text_tokenizer.eos_token
        else:
            text = prompt

        tokenize_kwargs = dict(
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        images = self.images[idx] if (self.processor is not None and self.images is not None) else None
        if images:
            from openrlhf.utils.vlm_utils import load_images  # same loading path as the RL side

            pil_images = load_images(images)
            enc = self.processor(images=pil_images, text=[text], **tokenize_kwargs)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            mm_inputs = {k: v for k, v in enc.items() if k not in _MM_SKIP_KEYS}
            # ⚠ self.prompt_ids_lens[idx] is TEXT caliber: it does not include the image
            # placeholder expansion, so reusing it here shifts the loss mask ~grid tokens to
            # the left and supervises the image pad tokens.  Silent: no error, no OOM, a
            # perfectly normal-looking loss curve, and 30% of the data learned wrong.
            # (An over-long multimodal sample is at least loud: if truncation cuts into the
            # image block, the processor itself raises "Mismatch in image token count".)
            # Recover the real boundary with a closed form over the grids we already have,
            # instead of a second (CPU-expensive) image_processor call:
            #   prompt_len(processor) = prompt_len(text) - n_placeholder + sum(prod(grid))/merge**2
            prompt_ids_len = self._expanded_prompt_ids_len(idx, mm_inputs.get("image_grid_thw"))
        else:
            input_token = self.text_tokenizer(text, **tokenize_kwargs)
            input_ids = input_token["input_ids"]
            attention_mask = input_token["attention_mask"]
            mm_inputs = {}
            prompt_ids_len = self.prompt_ids_lens[idx]

        loss_mask = self.get_loss_mask(input_ids, idx, prompt_ids_len)

        if not self.pretrain_mode:
            eos_token_id = self.text_tokenizer.eos_token_id
            # `truncation=True` may have cut the sample at max_length.  Forcing EOS onto the
            # last token of a TRUNCATED sample teaches the model to stop mid-derivation, so
            # only do it when the sample actually ended (where it is a no-op anyway, since
            # `text` ends with eos).  Over-long samples are supposed to be dropped by the
            # length filter in process_data; if one still gets here, say so loudly.
            if int(input_ids[0][-1]) != eos_token_id:
                if input_ids.shape[-1] < self.max_length:
                    input_ids[0][-1] = eos_token_id
                    attention_mask[0][-1] = True
                elif not self._truncation_warned:
                    self._truncation_warned = True
                    logger.warning(
                        f"SFTDataset: sample {idx} was truncated at max_length={self.max_length} "
                        "(the length filter underestimated it, e.g. image placeholder expansion). "
                        "Leaving it without a forced EOS -- fix the data instead."
                    )
        return input_ids, attention_mask, loss_mask, mm_inputs

    def _expanded_prompt_ids_len(self, idx, image_grid_thw):
        """Processor-caliber prompt length for a multimodal sample (see __getitem__)."""
        prompt_text_ids = self.text_tokenizer(self.prompts[idx], add_special_tokens=False)["input_ids"]
        if image_grid_thw is None or self._image_pad_id is None:
            return len(prompt_text_ids)
        merge = getattr(self.processor.image_processor, "merge_size", 2)
        grid_tokens = sum(int(g[0]) * int(g[1]) * int(g[2]) // (merge**2) for g in image_grid_thw)
        n_pad = sum(1 for t in prompt_text_ids if t == self._image_pad_id)
        return len(prompt_text_ids) - n_pad + grid_tokens

    def get_loss_mask(self, input_ids, idx, prompt_ids_len=None):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            if prompt_ids_len is None:
                prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            response_ranges = self.response_ranges[idx]
            for start_idx, end_idx in response_ranges:
                loss_mask[0, start_idx - 1 : end_idx] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []
        mm_list = []

        for input_id, attention_mask, loss_mask, mm_inputs in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)
            mm_list.append(mm_inputs)

        input_ids = zero_pad_sequences(input_ids, "right", self.text_tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")

        # Concatenate only the rows that actually carry multimodal tensors, along the batch
        # dim.  The model re-associates them with their rows via the <|image_pad|> runs in
        # input_ids, so a mixed text/image batch works.  All-text batch -> {} (never a None
        # value: that would be forwarded as a kwarg and crash the model).
        mm_inputs = {}
        for mm in mm_list:
            for k, v in mm.items():
                mm_inputs.setdefault(k, []).append(v)
        mm_inputs = {k: torch.cat(v, dim=0) for k, v in mm_inputs.items() if v}
        return input_ids, attention_masks, loss_masks, mm_inputs
